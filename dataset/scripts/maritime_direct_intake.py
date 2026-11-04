"""GCE-only bounded, original-grid EE SDK window intake; no ready markers.

The high-volume endpoint serves 1700² original RGB + producer validity pixels
directly, avoiding whole-scene batch export queues. All outputs remain pending
native semantic/quality review. https://developers.google.com/earth-engine/
apidocs/ee-image-getdownloadurl documents the 32 MB / 10000-pixel bounds.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.windows import Window, bounds, transform

from scripts.maritime_cloud_inventory import (create_only, digest_bytes, file_sha256,
    json_bytes, parse_gs, require_cloud_vm, verify_output)
from scripts.maritime_export_scheduler import export_images, native_projection
from scripts.maritime_pilot import core_identity, core_window, write_geotiff
from scripts.maritime_reuse_candidates import approved_asset_ids

ENDPOINT = 'https://earthengine-highvolume.googleapis.com'
MAX_BYTES = 32 * 1024 * 1024
MAX_ATTEMPTS = 3
WATER_PROXY = 'JRC/GSW1_4/GlobalSurfaceWater'


def catalog_plan(record):
    bands = {band['id']: band for band in record.get('bands', [])}
    if not {'R', 'G', 'B'}.issubset(bands):
        raise ValueError('Original catalog RGB bands required')
    grids = [bands[name].get('grid') for name in ('R', 'G', 'B')]
    if not grids[0] or any(grid != grids[0] for grid in grids[1:]):
        raise ValueError('Original RGB grids differ')
    for name in ('R', 'G', 'B'):
        dtype = bands[name].get('dataType', {})
        limits = dtype.get('range', {})
        if (dtype.get('precision') != 'INT' or limits.get('min', 0) < 0
                or limits.get('max') != 255):
            raise ValueError('Original uint8 RGB catalog pixels required')
    grid = grids[0]
    affine = grid.get('affineTransform', {})
    terms = [affine.get('scaleX'), affine.get('shearX', 0), affine.get('translateX', 0),
             affine.get('shearY', 0), affine.get('scaleY'), affine.get('translateY', 0)]
    definition = grid.get('crsCode') or grid.get('crsWkt')
    crs, terms = native_projection(dict(crs=definition, transform=terms))
    if not (crs.is_projected and math.isclose(crs.linear_units_factor[1], 1., abs_tol=1e-12)
            and abs(terms[0] - .8) < 1e-6 and abs(terms[4] + .8) < 1e-6
            and terms[1] == terms[3] == 0):
        raise ValueError('Original metric north-up 0.8 metre grid required')
    dimensions = grid.get('dimensions', {})
    width, height = dimensions.get('width'), dimensions.get('height')
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (width, height)):
        raise ValueError('Original positive integer raster dimensions required')
    props = record.get('properties', {})
    return dict(asset_id=record['id'], catalog_id=props.get('catalogID') or record['id'],
        crs=definition, normalized_crs=crs.to_string(), crs_transform=list(terms),
        width=width, height=height, grid=grid, properties=props,
        acquisition_utc=props.get('collectionStartTime') or record.get('startTime'),
        catalog_update_time=record.get('updateTime'))


def planned_windows(plan):
    affine = Affine(*plan['crs_transform'])
    extent = bounds(Window(0, 0, plan['width'], plan['height']), affine)
    result = []
    for y in range(math.floor(extent[1] / 1280), math.ceil(extent[3] / 1280)):
        for x in range(math.floor(extent[0] / 1280), math.ceil(extent[2] / 1280)):
            core, shift = core_window(affine, (x, y))
            col, row = int(core.col_off), int(core.row_off)
            if col < 50 or row < 50 or col + 1650 > plan['width'] or row + 1650 > plan['height']:
                continue
            context = Window(col - 50, row - 50, 1700, 1700)
            identity = core_identity(plan['catalog_id'], plan['normalized_crs'], affine, core)
            result.append(dict(asset_id=plan['asset_id'], acquisition_id=plan['catalog_id'],
                sample_id=digest_bytes(json_bytes(identity)), global_core_identity=identity,
                crs=plan['normalized_crs'], world_grid_cell=[x, y], native_snap_metres=shift,
                core_window=[col, row, 1600, 1600], context_window=[col - 50, row - 50, 1700, 1700],
                core_bounds=list(bounds(core, affine)), context_bounds=list(bounds(context, affine)),
                core_transform=list(transform(core, affine))[:6],
                context_transform=list(transform(context, affine))[:6]))
    # Deterministic spatial spreading instead of a north-west-only capped scan.
    return sorted(result, key=lambda item: item['sample_id'])


def spread_windows(plans, sites, limit):
    groups = {}
    for asset in sorted(plans):
        if plans[asset]: groups.setdefault(sites[asset], []).append(asset)
    offsets = {asset: 0 for asset in plans}
    rotations = {site: 0 for site in groups}
    result, seen = [], set()
    while len(result) < limit:
        progressed = False
        for site in sorted(groups):
            assets = groups[site]
            for _ in range(len(assets)):
                asset = assets[rotations[site] % len(assets)]
                rotations[site] += 1
                offset = offsets[asset]
                if offset < len(plans[asset]):
                    window = plans[asset][offset]
                    offsets[asset] += 1
                    progressed = True
                    identity = window.get('sample_id') or (asset, tuple(window['world_grid_cell']))
                    if identity in seen: continue
                    seen.add(identity)
                    result.append(window)
                    break
            if len(result) >= limit: break
        if not progressed: break
    return result


def download_parameters(plan, window):
    if 1700 * 1700 * 4 >= MAX_BYTES:
        raise ValueError('Download exceeds documented 32 MB limit')
    return dict(crs=plan['crs'], crs_transform=window['context_transform'],
                dimensions=[1700, 1700], bands=['R', 'G', 'B', 'producer_mask'],
                format='GEO_TIFF', filePerBand=False)


def coarse_parameters(plan, window=None):
    width, height = (window['context_window'][2:] if window else (plan['width'], plan['height']))
    origin = window['context_window'][:2] if window else [0, 0]
    factor = 32
    while (max(math.ceil(width/factor), math.ceil(height/factor)) > 10000
           or math.ceil(width/factor)*math.ceil(height/factor)*2 > 8*1024*1024):
        factor *= 2
    if factor > 256:
        raise ValueError('Asset too large for bounded exact original-mask coarse screening')
    affine = window['context_transform'] if window else plan['crs_transform']
    return dict(crs=plan['crs'], crs_transform=[affine[0]*factor, 0, affine[2],
        0, affine[4]*factor, affine[5]], dimensions=[math.ceil(width/factor),
        math.ceil(height/factor)], format='GEO_TIFF', filePerBand=False,
        bands=['producer_mask_min', 'water_proxy'], factor=factor, original_pixel_origin=origin)


def coarse_decision(values, spec, window):
    factor = spec['factor']
    origin_x, origin_y = spec.get('original_pixel_origin', [0, 0])
    col, row, width, height = window['context_window']
    col, row = col-origin_x, row-origin_y
    # One original-pixel margin also protects against tiny affine roundoff
    # causing a reducer footprint to touch the neighbouring source pixel.
    x0, y0 = math.ceil((col+1)/factor), math.ceil((row+1)/factor)
    x1, y1 = math.floor((col+width-1)/factor), math.floor((row+height-1)/factor)
    interior = values[0, y0:y1, x0:x1]
    # A min-reduced zero in a fully contained, native-aligned block proves
    # at least one invalid original pixel inside this exact context. Boundary
    # blocks are deliberately excluded because their zero could be outside.
    unsupported = bool(interior.size and np.any(interior == 0))
    invalid_block = None
    if unsupported:
        iy, ix = np.argwhere(interior == 0)[0]
        invalid_block = [origin_x+(x0+int(ix))*factor, origin_y+(y0+int(iy))*factor, factor, factor]
    col, row, width, height = window['core_window']
    col, row = col-origin_x, row-origin_y
    water = values[1, math.ceil(row/factor):math.floor((row+height)/factor),
                   math.ceil(col/factor):math.floor((col+width)/factor)]
    fraction = float(np.mean(water)) if water.size else None
    tier = 0 if fraction is not None and .05 <= fraction <= .95 else (
        1 if fraction is not None and fraction > .95 else 2)
    return dict(proven_unsupported=unsupported, fully_contained_mask_blocks=int(interior.size),
        proven_invalid_block_original_pixel_window=invalid_block,
        proof_block_margin_original_pixels=1,
        coarse_water_fraction=fraction, priority_tier=tier,
        semantic_status='unconfirmed_screening_only', water_proxy_asset=WATER_PROXY,
        water_proxy_limitation='historical surface water; dry or unmapped marine water remains eligible')


def read_coarse(path, spec):
    with rasterio.open(path) as ds:
        if (ds.count != 2 or ds.dtypes != ('uint8', 'uint8')
            or [ds.width, ds.height] != spec['dimensions']
            or ds.crs != CRS.from_user_input(spec['crs'])
            or max(abs(a-b) for a,b in zip(tuple(ds.transform)[:6],spec['crs_transform'])) > 1e-6):
            raise ValueError('Coarse screen returned wrong original-aligned grid')
        values = ds.read()
    if not np.all((values[0] == 0) | (values[0] == 255)) or not np.all((values[1] == 0) | (values[1] == 1)):
        raise ValueError('Coarse original minimum mask / water proxy values invalid')
    return values


def read_download(path, plan, window):
    with rasterio.open(path) as ds:
        if (ds.count != 4 or ds.dtypes != ('uint8',) * 4
            or ds.width != 1700 or ds.height != 1700
            or ds.crs != CRS.from_user_input(plan['crs'])
            or max(abs(a-b) for a,b in zip(tuple(ds.transform)[:6], window['context_transform'])) > 1e-6):
            raise ValueError('Direct source returned wrong original RGB/mask grid or dtype')
        if all(ds.descriptions) and ds.descriptions != ('R', 'G', 'B', 'producer_mask'):
            raise ValueError('Direct source returned conflicting RGB/mask band order')
        values = ds.read()
        if not np.all((values[3] == 0) | (values[3] == 255)):
            raise ValueError('Original minimum RGB producer mask is not binary 0/255')
        if not np.all(values[3] == 255):
            raise ValueError('Original 1700-pixel core and real halo are not fully supported')
        header = dict(crs=ds.crs.to_string(), affine=list(ds.transform)[:6],
                      width=ds.width, height=ds.height, bands=4, dtype='uint8')
    return values[:3], values[3:4], header


def retry_delay(exc, attempt):
    if isinstance(exc, ValueError): return None
    code = getattr(exc, 'code', None)
    response = getattr(exc, 'response', None)
    if code is None and response is not None: code = getattr(response, 'status_code', None)
    if code in (401, 403, 404): return None
    if code == 429 or code in (500, 502, 503, 504) or isinstance(exc, (OSError, URLError, TimeoutError)):
        return min(120, 30 * 2 ** (attempt - 1))
    # EEException may omit an HTTP status attribute. Respect explicit permission
    # errors and rate limiting; never treat a generic execution error as transient.
    message = str(exc).lower()
    if any(token in message for token in ('permission', 'forbidden', 'unauthorized')): return None
    if any(token in message for token in ('429', 'too many requests', 'rate limit')):
        return min(120, 30 * 2 ** (attempt - 1))
    return None


class AttemptLedger:
    def __init__(self, path, configuration_sha256, window_id):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else dict(
            configuration_sha256=configuration_sha256, window_id=window_id, attempts=[])
        if (self.state.get('configuration_sha256') != configuration_sha256
            or self.state.get('window_id') != window_id):
            raise ValueError('Persisted attempt configuration conflict')

    def save(self):
        temporary = self.path.with_suffix('.tmp')
        temporary.write_bytes(json_bytes(self.state))
        temporary.replace(self.path)

    def begin(self):
        if self.state['attempts'] and self.state['attempts'][-1]['state'] in ('REJECTED', 'BLOCKED', 'COMPLETED'):
            raise RuntimeError('Window already terminal; new attempts blocked')
        if len(self.state['attempts']) >= MAX_ATTEMPTS:
            raise RuntimeError('Persisted direct-window attempts exhausted')
        attempt = len(self.state['attempts']) + 1
        self.state['attempts'].append(dict(number=attempt, state='STARTED',
            started_utc=datetime.now(timezone.utc).isoformat()))
        self.save()
        return attempt

    def finish(self, status, message=None):
        self.state['attempts'][-1].update(state=status, finished_utc=datetime.now(timezone.utc).isoformat())
        if message: self.state['attempts'][-1]['reason'] = message
        self.save()


def run(args):
    instance = require_cloud_vm()  # Before credentials, URLs or any raw transport.
    if not 1 <= args.workers <= 8 or args.max_candidates < 1:
        raise ValueError('Workers must be 1..8 and max-candidates positive')
    import ee
    import google.auth
    from google.cloud import storage
    from PIL import Image
    approval = json.loads(Path(args.approved_assets).read_text(encoding='utf-8-sig'))
    catalog = json.loads(Path(args.catalog_metadata).read_text(encoding='utf-8-sig'))
    site_metadata = json.loads(Path(args.site_metadata).read_text(encoding='utf-8-sig'))
    by_asset = {record['id']: record for record in catalog}
    approved = approved_asset_ids(approval)
    groups = {asset: group for group in approval.get('review_metadata', {}).get('groups', [])
              for asset in group.get('asset_ids', [])}
    plans, sites, rejected = {}, {}, []
    for asset in sorted(approved):
        try:
            if asset not in by_asset or asset not in groups:
                raise ValueError('Approved asset missing catalog or geographic approval metadata')
            plans[asset] = catalog_plan(by_asset[asset])
            sites[asset] = groups[asset]['group_id']
        except (ValueError, KeyError, TypeError) as exc:
            rejected.append(dict(asset_id=asset, reason=str(exc)))
    windows = spread_windows({asset: planned_windows(plan) for asset, plan in plans.items()},
                             sites, args.max_candidates)
    if not windows: raise ValueError('No approved original-grid full-halo windows')
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    credentials, _ = google.auth.default(scopes=[
        'https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/earthengine'])
    ee.Initialize(credentials=credentials, project=args.project, opt_url=ENDPOINT)
    client = storage.Client(project=args.project, credentials=credentials)
    bucket_name, prefix = parse_gs(args.output_prefix.rstrip('/'))
    bucket = client.bucket(bucket_name)
    def publish(name, payload):
        return dict(uri=f'gs://{bucket_name}/{prefix}/{name}',
                    **create_only(bucket.blob(f'{prefix}/{name}'), payload))
    locks = {asset: threading.Lock() for asset in plans}
    checked = set()
    backoff_lock = threading.Lock()
    cooldown = [0.]
    coarse_enabled = getattr(args, 'coarse_screen', False)
    coarse_reports, coarse_skipped = [], 0
    if coarse_enabled:
        screen_config = dict(worker_sha256=file_sha256(__file__),
            catalog_sha256=digest_bytes(json_bytes(catalog)), project=args.project,
            initial_window_sha256=digest_bytes(json_bytes(windows)),
            workers=args.workers, max_candidates=args.max_candidates,
            endpoint=ENDPOINT, water_proxy=WATER_PROXY, original_min_reducer=True,
            best_effort=False, base_factor=32, max_attempts=MAX_ATTEMPTS,
            policy='reject only invalid min blocks wholly contained in native 1700 context; water only prioritizes')
        screen_config_sha = digest_bytes(json_bytes(screen_config))
        publish('coarse-configuration.json', json_bytes(screen_config))

        def screen_window(window):
            key, asset = window['sample_id'], window['asset_id']
            plan = plans[asset]
            spec = coarse_parameters(plan, window)
            path = scratch / 'coarse' / f'{key}.tif'
            path.parent.mkdir(parents=True, exist_ok=True)
            receipt_blob = bucket.blob(f'{prefix}/coarse-receipts/{key}.json')
            if receipt_blob.exists():
                receipt_blob.reload()
                receipt = json.loads(receipt_blob.download_as_bytes(
                    if_generation_match=int(receipt_blob.generation), checksum='crc32c'))
                if receipt.get('configuration_sha256') != screen_config_sha:
                    raise ValueError('Coarse screen receipt configuration conflict')
                report = json.loads(verify_output(client, receipt['screen_report']))
                if report['status'] == 'completed':
                    path.write_bytes(verify_output(client, report['coarse_source']))
                    return read_coarse(path, spec), spec, receipt['screen_report'], report
                return None, spec, receipt['screen_report'], report
            ledger = AttemptLedger(scratch / 'coarse-attempts' / f'{key}.json', screen_config_sha, key)
            report = dict(configuration_sha256=screen_config_sha, asset_id=asset,
                window_id=key, parameters=spec, status='pending', candidates=[],
                original_mask_recipe='original min RGB mask, unmask zero global, native-aligned minimum reduction; bestEffort false',
                water_proxy=WATER_PROXY, semantic_status='unconfirmed_screening_only')
            values = None
            staged_screen = path.with_suffix('.completed.json')
            if staged_screen.exists():
                report = json.loads(staged_screen.read_text())
                if report.get('configuration_sha256') != screen_config_sha or report.get('status') != 'completed':
                    raise ValueError('Staged coarse report configuration conflict')
                path.write_bytes(verify_output(client, report['coarse_source']))
                values = read_coarse(path, spec)
                if not ledger.state['attempts']: raise ValueError('Staged coarse report has no started attempt')
                if ledger.state['attempts'][-1]['state'] == 'STARTED': ledger.finish('COMPLETED')
                if ledger.state['attempts'][-1]['state'] != 'COMPLETED':
                    raise ValueError('Staged coarse report conflicts with failed attempt')
            while report['status'] != 'completed':
                try: attempt = ledger.begin()
                except RuntimeError:
                    report['status'] = 'attempts_exhausted_or_terminal'
                    break
                try:
                    with backoff_lock: delay = max(0, cooldown[0] - time.monotonic())
                    if delay: time.sleep(delay)
                    with locks[asset]:
                        if asset not in checked:
                            current = ee.data.getAsset(asset)
                            current['id'] = asset
                            if catalog_plan(current) != plan:
                                raise ValueError('Original EE asset catalog/grid changed after inventory pin')
                            checked.add(asset)
                    blob = bucket.blob(f'{prefix}/coarse-sources/{key}.tif')
                    if blob.exists():
                        blob.reload()
                        blob.download_to_filename(str(path), if_generation_match=int(blob.generation), checksum='crc32c')
                    else:
                        _, original_mask = export_images(ee, asset)
                        mask = (original_mask.unmask(0, sameFootprint=False)
                            .setDefaultProjection(crs=plan['crs'], crsTransform=plan['crs_transform'])
                            .reduceResolution(reducer=ee.Reducer.min(), bestEffort=False,
                                              maxPixels=(spec['factor']+2)**2)
                            .reproject(crs=plan['crs'], crsTransform=spec['crs_transform'])
                            .rename('producer_mask_min').toUint8())
                        water = (ee.Image(WATER_PROXY).select('occurrence').gt(20)
                            .unmask(0, sameFootprint=False).toUint8().rename('water_proxy'))
                        params = {name:value for name,value in spec.items()
                                  if name not in ('factor', 'original_pixel_origin')}
                        url = mask.addBands(water).getDownloadURL(params)
                        with urlopen(url, timeout=180) as response: payload = response.read(MAX_BYTES+1)
                        if len(payload) > MAX_BYTES: raise ValueError('Coarse response exceeds documented 32 MB bound')
                        path.write_bytes(payload)
                    values = read_coarse(path, spec)
                    ref = publish(f'coarse-sources/{key}.tif', path.read_bytes())
                    report.update(status='completed', coarse_source=ref)
                    temporary = staged_screen.with_suffix('.tmp')
                    temporary.write_bytes(json_bytes(report))
                    temporary.replace(staged_screen)
                    ledger.finish('COMPLETED')
                    break
                except Exception as exc:
                    delay = retry_delay(exc, attempt)
                    reason = str(exc) if isinstance(exc, ValueError) else f'{type(exc).__name__} HTTP {getattr(exc, "code", "unknown")}'
                    ledger.finish('BLOCKED' if delay is None else 'FAILED', reason)
                    report.update(status='unavailable_full_native_fallback', reason=reason)
                    if delay is None or attempt >= MAX_ATTEMPTS: break
                    ledger.state['attempts'][-1]['retry_backoff_seconds'] = delay
                    ledger.save()
                    with backoff_lock: cooldown[0] = max(cooldown[0], time.monotonic()+delay)
            report['attempt_history'] = ledger.state['attempts']
            ref = publish(f'coarse-reports/{key}.json', json_bytes(report))
            publish(f'coarse-receipts/{key}.json', json_bytes(dict(
                configuration_sha256=screen_config_sha, screen_report=ref)))
            print(json.dumps(dict(asset_id=asset, coarse_screen_status=report['status'])), flush=True)
            return values, spec, ref, report

        ranked = {asset:[] for asset in plans}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(screen_window, window):window for window in windows}
            for future in as_completed(futures):
                window = futures[future]
                asset = window['asset_id']
                values, spec, ref, report = future.result()
                coarse_reports.append(ref)
                decision = coarse_decision(values, spec, window) if values is not None else dict(
                    proven_unsupported=False, priority_tier=2, coarse_water_fraction=None,
                    semantic_status='unconfirmed_screening_only', status='screen_unavailable_full_native_fallback')
                if decision['proven_unsupported']:
                    coarse_skipped += 1
                else:
                    window['coarse_screen'] = dict(report_object=ref, **decision)
                    ranked[asset].append(window)
                publish(f'coarse-window-decisions/{window["sample_id"]}.json', json_bytes(dict(
                    asset_id=asset, window_id=window['sample_id'], coarse_report=ref,
                    configuration_sha256=screen_config_sha, **decision)))
        for asset, eligible in ranked.items():
            eligible.sort(key=lambda row:(row['coarse_screen']['priority_tier'],
                abs((row['coarse_screen']['coarse_water_fraction'] or 0)-.5), row['sample_id']))
        windows = spread_windows(ranked, sites, args.max_candidates)
    config = dict(worker_sha256=file_sha256(__file__), approval_sha256=digest_bytes(json_bytes(approval)),
        catalog_sha256=digest_bytes(json_bytes(catalog)), site_metadata_sha256=digest_bytes(json_bytes(site_metadata)),
        project=args.project, endpoint=ENDPOINT, workers=args.workers, max_candidates=args.max_candidates,
        selected_window_sha256=digest_bytes(json_bytes(windows)), max_attempts=MAX_ATTEMPTS,
        coarse_screen=coarse_enabled,
        transport='EE SDK getDownloadURL GEO_TIFF original RGB+original min-band mask',
        status='pending_native_review_only')
    config_sha = digest_bytes(json_bytes(config))
    publish('configuration.json', json_bytes(config))

    def process(window):
        key, asset = window['sample_id'], window['asset_id']
        receipt_blob = bucket.blob(f'{prefix}/receipts/{key}.json')
        if receipt_blob.exists():
            receipt_blob.reload()
            receipt = json.loads(receipt_blob.download_as_bytes(
                if_generation_match=int(receipt_blob.generation), checksum='crc32c'))
            if receipt.get('configuration_sha256') != config_sha:
                raise ValueError('Existing direct-window receipt configuration conflict')
            report = json.loads(verify_output(client, receipt['source_report']))
            for candidate in report.get('candidates', []):
                for ref in candidate['artifacts'].values(): verify_output(client, ref)
                verify_output(client, candidate['source_object'])
            return receipt['source_report'], report
        plan = plans[asset]
        ledger = AttemptLedger(scratch / 'attempts' / f'{key}.json', config_sha, key)
        work = scratch / 'windows' / key
        work.mkdir(parents=True, exist_ok=True)
        report = dict(configuration_sha256=config_sha, instance_id=instance, asset_id=asset,
                      candidates=[], exhaustive_source_scan=False, status='pending')
        staged = work / 'completed-report.json'
        if staged.exists():
            report = json.loads(staged.read_text())
            if report.get('configuration_sha256') != config_sha or report.get('status') != 'completed':
                raise ValueError('Staged successful report configuration conflict')
            for candidate in report['candidates']:
                verify_output(client, candidate['source_object'])
                for ref in candidate['artifacts'].values(): verify_output(client, ref)
            if not ledger.state['attempts']:
                raise ValueError('Staged successful report has no persisted started attempt')
            if ledger.state['attempts'][-1]['state'] == 'STARTED': ledger.finish('COMPLETED')
            if ledger.state['attempts'][-1]['state'] != 'COMPLETED':
                raise ValueError('Staged successful report conflicts with failed attempt state')
        while report['status'] != 'completed':
            try: attempt = ledger.begin()
            except RuntimeError:
                report.update(status='attempts_exhausted_or_terminal', attempt_history=ledger.state['attempts'])
                break
            try:
                with backoff_lock: delay = max(0, cooldown[0] - time.monotonic())
                if delay: time.sleep(delay)
                with locks[asset]:
                    if asset not in checked:
                        current = ee.data.getAsset(asset)
                        current['id'] = asset
                        observed = catalog_plan(current)
                        if (observed != plan):
                            raise ValueError('Original EE asset catalog/grid changed after inventory pin')
                        checked.add(asset)
                path = work / 'original-rgb-producer-mask.tif'
                original_blob = bucket.blob(f'{prefix}/original-windows/{key}.tif')
                if original_blob.exists():
                    original_blob.reload()
                    original_blob.download_to_filename(str(path),
                        if_generation_match=int(original_blob.generation), checksum='crc32c')
                else:
                    rgb_image, mask_image = export_images(ee, asset)
                    image = rgb_image.addBands(mask_image.rename('producer_mask'))
                    url = image.getDownloadURL(download_parameters(plan, window))
                    with urlopen(url, timeout=180) as response:
                        payload = response.read(MAX_BYTES + 1)
                    if len(payload) > MAX_BYTES:
                        raise ValueError('Direct response exceeds documented 32 MB bound')
                    path.write_bytes(payload)
                original_ref = publish(f'original-windows/{key}.tif', path.read_bytes())
                # Retain original downloaded windows, including rejected support,
                # rather than losing evidence of the source-mask decision.
                report.update(original_mirror=original_ref, source_sha256=original_ref['sha256'])
                rgb, masks, header = read_download(path, plan, window)
                affine = Affine(*window['context_transform'])
                rgb_path, mask_path = work / 'rgb-context.tif', work / 'validity-context.tif'
                write_geotiff(rgb_path, rgb, affine, plan['crs'])
                with rasterio.open(mask_path, 'w', driver='GTiff', width=1700, height=1700,
                    count=1, dtype='uint8', crs=plan['crs'], transform=affine,
                    compress='DEFLATE', tiled=True, blockxsize=256, blockysize=256) as dst:
                    dst.write(masks)
                native_path, overview_path = work / 'native-core.png', work / 'overview.png'
                native = Image.fromarray(np.moveaxis(rgb[:, 50:1650, 50:1650], 0, -1))
                native.save(native_path)
                native.resize((400, 400), Image.Resampling.BOX).save(overview_path)
                artifacts = {kind: publish(f'candidates/{key}/{value.name}', value.read_bytes())
                    for kind, value in dict(rgb=rgb_path, mask=mask_path,
                        native_preview=native_path, overview=overview_path).items()}
                site = sites[asset]
                source = dict(original_ref, asset_id=asset, source_id=asset, catalog_id=plan['catalog_id'],
                    properties=plan['properties'], acquisition_utc=plan['acquisition_utc'],
                    site_id=site, country=site_metadata.get(site, {}).get('country') or groups[asset].get('country'),
                    grid=plan['grid'], geometry=by_asset[asset].get('geometry'), native_gsd_m=.8,
                    producer_mask=artifacts['mask'], mask_provenance='producer_original_min_rgb_mask',
                    license='CC-BY-4.0', license_url='https://creativecommons.org/licenses/by/4.0/',
                    attribution='Planet Labs PBC', provider='Google Earth Engine',
                    processing_history=dict(original_asset_grid=plan['grid'],
                        original_asset_context_window=window['context_window'],
                        transport='direct original-pixel RGB + original min-band mask subset',
                        download_parameters=download_parameters(plan, window), ephemeral_url_persisted=False))
                candidate = dict(window, artifacts=artifacts, source_object=source,
                    source_sha256=original_ref['sha256'], producer_mask_sha256=artifacts['mask']['sha256'],
                    source_pixel_window=[0, 0, 1700, 1700], source_native_header=header,
                    site_id=site, mask_provenance='producer_original_min_rgb_mask',
                    source_processing='native_pixels_only_no_warp_no_resampling',
                    original_mask_known=True, semantic_accepted=False,
                    status='pending_semantic_and_quality_review', vessel_presence='unconfirmed')
                report.update(source=source, source_sha256=original_ref['sha256'], native_header=header,
                    original_transform=window['context_transform'],
                    producer_mask_sha256=artifacts['mask']['sha256'], candidates=[candidate], status='completed')
                staging = staged.with_suffix('.tmp')
                staging.write_bytes(json_bytes(report))
                staging.replace(staged)
                ledger.finish('COMPLETED')
                break
            except Exception as exc:
                delay = retry_delay(exc, attempt)
                reason = str(exc) if isinstance(exc, ValueError) else f'{type(exc).__name__} HTTP {getattr(exc, "code", "unknown")}'
                status = 'REJECTED' if isinstance(exc, ValueError) else 'BLOCKED' if delay is None else 'FAILED'
                ledger.finish(status, reason)
                if delay is not None:
                    ledger.state['attempts'][-1]['retry_backoff_seconds'] = delay
                    ledger.save()
                report.update(status=status.lower(), reason=reason)
                if delay is None or attempt >= MAX_ATTEMPTS: break
                with backoff_lock: cooldown[0] = max(cooldown[0], time.monotonic() + delay)
        report['attempt_history'] = ledger.state['attempts']
        ref = publish(f'source-reports/{key}.json', json_bytes(report))
        publish(f'receipts/{key}.json', json_bytes(dict(configuration_sha256=config_sha, source_report=ref)))
        print(json.dumps(dict(window_id=key, source_status=report['status'],
                             candidate_count=len(report['candidates']))), flush=True)
        return ref, report

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, window) for window in windows]
        for future in as_completed(futures): results.append(future.result())
    results.sort(key=lambda row: row[0]['uri'])
    candidates = sorted([candidate for _, report in results for candidate in report['candidates']],
                        key=lambda item: item['sample_id'])
    summary = dict(configuration=config, configuration_sha256=config_sha, instance_id=instance,
        candidate_count=len(candidates), attempted_window_count=len(windows),
        initial_capped_window_trials=args.max_candidates,
        source_reports=[ref for ref, _ in results], candidates=candidates,
        coarse_reports=sorted(coarse_reports,key=lambda ref:ref['uri']),
        coarse_proven_unsupported_skipped=coarse_skipped,
        rejected_assets=rejected, rejected_window_count=len(windows) - len(candidates),
        exhaustive=False, semantic_accepted_count=0, vessel_confirmation_count=0,
        status='pending_semantic_and_quality_review')
    ref = publish('candidate-manifest.json', json_bytes(summary))
    print(json.dumps(dict(candidate_count=len(candidates), candidate_manifest=ref)), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-assets', required=True)
    parser.add_argument('--catalog-metadata', required=True)
    parser.add_argument('--site-metadata', required=True)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--max-candidates', type=int, default=64,
                        help='Bounded window trials, never a semantic accepted-core target')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--coarse-screen', action='store_true',
        help='New-prefix opt-in: exact native-aligned minimum-mask screening and historical water prioritization')
    run(parser.parse_args())


if __name__ == '__main__': main()
