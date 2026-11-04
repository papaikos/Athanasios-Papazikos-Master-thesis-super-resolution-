"""Bounded cloud-worker SkySat RGB preparation; no calibration or provisioning.

Read one generation-pinned TIFF at a time. Only completed exports are eligible.
Covered native pixels, not whole files, count as accepted source bytes. This is
preparation evidence only: optimizer consumption must be verified separately.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

import numpy as np
import rasterio
from rasterio.enums import MaskFlags, Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds, Window
from scipy.ndimage import binary_erosion, gaussian_filter


def validate_manifest(manifest):
    payload = {k: v for k, v in manifest.items() if k != 'sha256'}
    checksum = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if manifest.get('sha256') != checksum:
        raise ValueError('Manifest checksum mismatch')
    assets = [row['asset_id'] for row in manifest['sources']]
    if len(assets) != len(set(assets)):
        raise ValueError('Duplicate satellite asset identity')


def degrade_rgb(hr, valid):
    if hr.shape != (3, 512, 512) or valid.shape != (512, 512):
        raise ValueError('Expected three RGB channels and 512-square validity')
    valid = valid & np.isfinite(hr).all(axis=0)
    safe = binary_erosion(valid, structure=np.ones((17, 17)), border_value=0)
    blurred = gaussian_filter(np.where(valid[None], hr, 0), (0, 2, 2),
                              truncate=4, mode='reflect')
    lr = blurred.reshape(3, 128, 4, 128, 4).mean(axis=(2, 4))
    return lr.astype(np.float32), safe.reshape(128, 4, 128, 4).all(axis=(1, 3))


def completed_objects(manifest, state, objects):
    groups, sources, seen = {}, {}, set()
    for row in manifest['sources']:
        split = row['split']
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Unknown split')
        if groups.setdefault(row['group_id'], split) != split:
            raise ValueError('Geographic group crosses split boundary')
        if row['source_id'] in sources:
            raise ValueError('Duplicate source identity')
        sources[row['source_id']] = row
    result = []
    for obj in objects:
        metadata = obj.get('metadata', obj)
        if not isinstance(metadata, dict):
            raise ValueError('Malformed object metadata')
        uri = obj.get('url') or obj.get('storage_url')
        if not uri:
            if not isinstance(metadata.get('name'), str):
                raise ValueError('Missing object identity')
            uri = 'gs://' + metadata.get('bucket', manifest['bucket']) + '/' + metadata['name']
        if not isinstance(uri, str):
            raise ValueError('Malformed object identity')
        uri, separator, url_generation = uri.partition('#')
        if separator and not url_generation.isdigit():
            raise ValueError('Malformed URL generation')
        if not uri.lower().endswith(('.tif', '.tiff')):
            continue
        prefix = f"gs://{manifest['bucket']}/{manifest['prefix']}/"
        if not uri.startswith(prefix):
            continue
        source_id = uri[len(prefix):].split('/')[0]
        row = sources.get(source_id)
        if row is None or state['tasks'].get(source_id, {}).get('state') != 'COMPLETED':
            continue
        generation = str(metadata.get('generation', ''))
        if separator and generation != url_generation:
            raise ValueError('Conflicting object generation')
        if 'generation' in obj and str(obj['generation']) != generation:
            raise ValueError('Conflicting object generation')
        if metadata.get('bucket') and metadata.get('name'):
            if uri != 'gs://' + metadata['bucket'] + '/' + metadata['name']:
                raise ValueError('Conflicting object identity')
        size = int(metadata.get('size', 0))
        if not generation.isdigit() or size <= 0:
            raise ValueError('Generation-pinned positive object size required')
        if uri in seen:
            raise ValueError('Duplicate object')
        seen.add(uri)
        result.append((row, dict(uri=uri, generation=generation, bytes=size)))
    return sorted(result, key=lambda item: item[1]['uri'])


def validate_grid(source, row):
    grid = row['grid']
    affine = grid['affineTransform']
    expected_crs = rasterio.crs.CRS.from_user_input(grid.get('crsCode') or grid['crsWkt'])
    t = source.transform
    if source.crs != expected_crs or not np.allclose(
        [t.a, t.b, t.d, t.e],
        [affine['scaleX'], affine.get('shearX', 0), affine.get('shearY', 0), affine['scaleY']],
        atol=1e-8, rtol=1e-8
    ):
        raise ValueError('Source grid differs from manifest')
    offsets = [(t.c-affine['translateX'])/t.a, (t.f-affine['translateY'])/t.e]
    if not np.allclose(offsets, np.round(offsets), atol=1e-4, rtol=0):
        raise ValueError('Source origin is not aligned with manifest native grid')


def patches(path, row=None):
    with rasterio.open(path) as source:
        require_native_mask(source)
        if row is not None:
            validate_grid(source, row)
        t = source.transform
        if (source.count != 3 or source.dtypes != ('uint8',) * 3
                or not source.crs or not source.crs.is_projected
                or source.crs.linear_units != 'metre'
                or t.b != 0 or t.d != 0 or not 0 < t.a <= 2.5
                or not -2.5 <= t.e < 0):
            raise ValueError('Expected north-up native RGB uint8, metric CRS, <=2.5m')
        b = source.bounds
        # Whole 1280-m cells only: excludes TIFF edges and prevents shard overlap.
        for y in range(math.ceil(b.bottom / 1280), math.floor(b.top / 1280)):
            for x in range(math.ceil(b.left / 1280), math.floor(b.right / 1280)):
                left, bottom = x * 1280, y * 1280
                nw = from_bounds(left, bottom, left + 1280, bottom + 1280, t)
                # Include all native pixels touching the HR footprint for mask proof.
                c0, r0 = math.floor(nw.col_off), math.floor(nw.row_off)
                c1, r1 = math.ceil(nw.col_off + nw.width), math.ceil(nw.row_off + nw.height)
                masks = source.read_masks(window=Window(c0, r0, c1-c0, r1-r0))
                if not (masks == 255).all():
                    continue
                transform = from_origin(left, bottom + 1280, 2.5, 2.5)
                with WarpedVRT(source, crs=source.crs, transform=transform,
                               width=512, height=512, resampling=Resampling.average,
                               dtype='float32') as vrt:
                    hr = vrt.read() / np.float32(255)
                valid = np.isfinite(hr).all(axis=0)
                if not valid.all():
                    continue
                lr, mask = degrade_rgb(hr, valid)
                # Count pixel centers inside each disjoint window exactly once.
                nc = max(0, math.ceil(nw.col_off + nw.width - .5) - math.ceil(nw.col_off - .5))
                nr = max(0, math.ceil(nw.row_off + nw.height - .5) - math.ceil(nw.row_off - .5))
                lr_transform = from_origin(left, bottom + 1280, 10, 10)
                yield dict(hr=hr, lr=lr, mask=mask, transform=list(transform)[:6],
                           lr_transform=list(lr_transform)[:6],
                           crs=str(source.crs), cell=[x, y], native_bytes=nc*nr*3)


def require_native_mask(source):
    if any(MaskFlags.all_valid in flags for flags in source.mask_flag_enums):
        raise ValueError('Native validity mask or explicit nodata required; padding is not evidence')


def valid_native_bytes(path):
    total = 0
    with rasterio.open(path) as ds:
        require_native_mask(ds)
        for top in range(0, ds.height, 512):
            window = Window(0, top, ds.width, min(512, ds.height - top))
            total += int(np.all(ds.read_masks(window=window) == 255, axis=0).sum()) * 3
    return total


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cloud(args, *command, check=True):
    return subprocess.run([args.gcloud, 'storage', *map(str, command)], check=check)


def publish(args, path, destination):
    cloud(args, 'cp', '--if-generation-match=0', path, destination, check=False)
    readback = path.with_suffix('.readback')
    cloud(args, 'cp', destination, readback)
    if digest(path) != digest(readback):
        raise ValueError('Immutable publication differs from readback')
    readback.unlink()


def validate_receipt(receipt, row, obj, config):
    expected = {**config, 'source_object': obj,
                **{k: row[k] for k in ('source_id', 'group_id', 'split')}}
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError('Existing receipt identity/configuration mismatch; use a new output prefix')
    source_hash = receipt.get('source_sha256', '')
    if len(source_hash) != 64 or any(c not in '0123456789abcdef' for c in source_hash):
        raise ValueError('Receipt missing source SHA256')
    if (not isinstance(receipt.get('pairs'), int) or receipt['pairs'] < 0
            or not isinstance(receipt.get('shards'), list)
            or bool(receipt['pairs']) != bool(receipt['shards'])
            or not 0 <= receipt['accepted_source_bytes'] <= receipt['valid_native_rgb_bytes']):
        raise ValueError('Malformed receipt accounting')


def verify_shards(args, receipt, prefix, scratch):
    for index, shard in enumerate(receipt['shards']):
        if shard['uri'] != prefix + f'/pairs-{index:05d}.tar':
            raise ValueError('Receipt shard outside expected output identity')
        local = scratch / 'resume-shard.tar'
        cloud(args, 'cp', shard['uri'], local)
        if digest(local) != shard['sha256']:
            raise ValueError('Resumed shard checksum mismatch')
        local.unlink()


def progress(args, stage, summaries, **details):
    summary = dict(updated_at=datetime.now(timezone.utc).isoformat(), stage=stage,
                   completed_objects=len(summaries), pairs=sum(s['pairs'] for s in summaries),
                   accepted_train_source_bytes=sum(s['accepted_source_bytes'] for s in summaries
                                                   if s['split'] == 'train'),
                   raw_bytes=sum(s['raw_bytes'] for s in summaries),
                   valid_train_native_rgb_bytes=sum(s['valid_native_rgb_bytes'] for s in summaries
                                                    if s['split'] == 'train'),
                   split_pairs={split: sum(s['pairs'] for s in summaries if s['split'] == split)
                                for split in ('train', 'validation', 'test')},
                   optimizer_consumed_bytes=0, **details)
    print(json.dumps(summary), flush=True)
    if getattr(args, 'progress_file', None):
        path = Path(args.progress_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.partial')
        temporary.write_text(json.dumps(summary, indent=2))
        temporary.replace(path)


def prepare(args):
    manifest = json.loads(Path(args.manifest).read_text())
    validate_manifest(manifest)
    state = json.loads(Path(args.task_state).read_text())
    objects = json.loads(Path(args.objects).read_text())
    if state.get('manifest_sha256') != manifest.get('sha256'):
        raise ValueError('Task state does not match manifest')
    items = completed_objects(manifest, state, objects)
    config = dict(recipe='rgb-v2', manifest_sha256=manifest['sha256'],
                  max_pairs_per_object=args.max_pairs_per_source)
    selected, summaries = set(), []
    progress(args, 'starting', summaries, eligible_objects=len(items))
    for row, obj in items:
        if row['source_id'] not in selected and len(selected) >= args.max_sources:
            continue
        selected.add(row['source_id'])
        identity = hashlib.sha256((obj['uri'] + '#' + obj['generation']).encode()).hexdigest()[:24]
        prefix = args.output_prefix.rstrip('/') + '/' + identity
        with tempfile.TemporaryDirectory(dir=args.scratch) as tmp:
            scratch = Path(tmp)
            report = scratch / 'receipt.json'
            progress(args, 'checking_resume', summaries, source_object=obj)
            if cloud(args, 'cp', prefix + '/receipt.json', report, check=False).returncode == 0:
                old = json.loads(report.read_text())
                validate_receipt(old, row, obj, config)
                verify_shards(args, old, prefix, scratch)
                summaries.append(old)
                progress(args, 'resumed_verified', summaries, source_object=obj)
                continue
            if shutil.disk_usage(scratch).free < obj['bytes'] + 512_000_000:
                raise ValueError('Insufficient scratch disk for one source and pair batch')
            source = scratch / 'source.tif'
            progress(args, 'downloading', summaries, source_object=obj)
            cloud(args, 'cp', obj['uri'] + '#' + obj['generation'], source)
            if source.stat().st_size != obj['bytes']:
                raise ValueError('Pinned source byte count mismatch')
            with rasterio.open(source) as raster:
                validate_grid(raster, row)
            progress(args, 'counting_native_validity', summaries, source_object=obj)
            result = dict(**config, source_id=row['source_id'], group_id=row['group_id'],
                          split=row['split'], source_object=obj, raw_bytes=obj['bytes'],
                          source_sha256=digest(source),
                          valid_native_rgb_bytes=valid_native_bytes(source),
                          accepted_source_bytes=0, pairs=0, shards=[],
                          optimizer_consumed_bytes=0)
            progress(args, 'preparing_pairs', summaries, source_object=obj)
            batch = []
            def flush():
                if not batch:
                    return
                tarpath = scratch / f"pairs-{len(result['shards']):05d}.tar"
                with tarfile.open(tarpath, 'w') as tar:
                    for name, content in batch:
                        info = tarfile.TarInfo(name)
                        info.size = len(content)
                        tar.addfile(info, io.BytesIO(content))
                uri = prefix + '/' + tarpath.name
                publish(args, tarpath, uri)
                result['shards'].append(dict(uri=uri, sha256=digest(tarpath)))
                progress(args, 'shard_verified', summaries, source_object=obj,
                         current_object_pairs=result['pairs'], shards=len(result['shards']))
                tarpath.unlink()
                batch.clear()
            for patch in patches(source, row):
                provenance = {**row, 'source_object': obj, 'crs': patch['crs'],
                              'transform': patch['transform'], 'lr_transform': patch['lr_transform'],
                              'native_covered_bytes': patch['native_bytes'],
                              'cell': patch['cell'], 'recipe': config['recipe'],
                              'manifest_sha256': manifest['sha256']}
                buffer = io.BytesIO()
                np.savez_compressed(buffer, hr=patch['hr'], lr=patch['lr'], mask=patch['mask'],
                                    metadata=json.dumps(provenance, sort_keys=True))
                batch.append((f"{result['pairs']:06d}.npz", buffer.getvalue()))
                result['pairs'] += 1
                result['accepted_source_bytes'] += patch['native_bytes']
                if len(batch) >= 32:
                    flush()
                if result['pairs'] >= args.max_pairs_per_source:
                    break
            flush()
            validate_receipt(result, row, obj, config)
            report.write_text(json.dumps(result, sort_keys=True, indent=2))
            publish(args, report, prefix + '/receipt.json')
            summaries.append(result)
            progress(args, 'object_complete', summaries, source_object=obj)
    progress(args, 'complete', summaries)
    if getattr(args, 'summary_file', None):
        path = Path(args.summary_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.partial')
        temporary.write_text(json.dumps(dict(config=config, objects=summaries), indent=2))
        temporary.replace(path)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'task-state', 'objects', 'output-prefix'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--gcloud', default='gcloud')
    p.add_argument('--scratch')
    p.add_argument('--progress-file', default='skysat-preparation-progress.json')
    p.add_argument('--summary-file')
    p.add_argument('--max-sources', type=int, default=1)
    p.add_argument('--max-pairs-per-source', type=int, default=8,
                   help='Bound per TIFF object; use a new output prefix for a larger run')
    args = p.parse_args()
    if args.max_sources < 1 or args.max_pairs_per_source < 1:
        p.error('Bounds must be positive')
    prepare(args)


if __name__ == '__main__':
    main()
