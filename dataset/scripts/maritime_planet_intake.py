"""GCE-only, generation-pinned intake of free Planet SkySat original assets.

Outputs are review candidates, never semantic acceptances or final releases.
Run: python -m scripts.maritime_planet_intake --sources /app/inputs/planet-sources.json
     --output-prefix gs://BUCKET/control/RUN/planet-intake --scratch /mnt/work/planet
"""
import argparse
import io
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy.sparse import csr_matrix

from scripts.maritime_cloud_inventory import (
    create_only, digest_bytes, file_sha256, json_bytes, parse_gs, require_cloud_vm)


def validate_source(row):
    parse_gs(row['uri'])
    if row['license'] != 'CC-BY-4.0':
        raise ValueError('Unverified free imagery license')
    if not str(row['generation']).isdigit() or int(row['generation']) <= 0:
        raise ValueError('Invalid pinned generation')
    if not isinstance(row['bytes'], int) or row['bytes'] <= 0 or not row['crc32c']:
        raise ValueError('Missing original object size or CRC32C')
    gsd = float(row['native_gsd_m'])
    if not math.isfinite(gsd) or not 0 < gsd <= .8:
        raise ValueError('Original native GSD exceeds 0.8 m')
    props = row['stac']['properties']
    if abs(float(props['gsd']) - gsd) > 1e-9 or float(props['pl:pixel_resolution']) != .5:
        raise ValueError('Pinned native GSD/product spacing metadata disagreement')


def validate_header(ds):
    t = ds.transform
    if (ds.crs is None or not ds.crs.is_projected or ds.crs.linear_units != 'metre'
            or not all(math.isfinite(v) for v in tuple(t)[:6])
            or abs(t.a - .5) > 1e-9 or abs(t.e + .5) > 1e-9 or t.b != 0 or t.d != 0):
        raise ValueError('Original requires north-up metric 0.5 m product grid')
    if ds.count not in (3, 4) or ds.dtypes[:3] != ('uint8',) * 3 or ds.colorinterp[:3] != (
            ColorInterp.red, ColorInterp.green, ColorInterp.blue):
        raise ValueError('Explicit three uint8 RGB bands required')
    if ds.count == 4 and (ds.colorinterp[3] != ColorInterp.alpha or ds.dtypes[3] != 'uint8'):
        raise ValueError('Only explicit uint8 alpha may accompany RGB')
    flags = ds.mask_flag_enums[:3]
    if all(MaskFlags.alpha in f for f in flags):
        return 'original_embedded_alpha_mask'
    if all(MaskFlags.per_dataset in f and MaskFlags.nodata not in f for f in flags):
        return 'original_embedded_dataset_mask'
    raise ValueError('Original validity unresolved: nodata or brightness cannot prove support')


def area_matrix(start, count, spacing=.5, target=.8):
    """Exact physical overlap fractions; start is metres from input left edge."""
    if start < 0 or count <= 0 or spacing <= 0 or target < spacing:
        raise ValueError('Invalid area resampling geometry')
    end = start + count * target
    width = math.ceil(end / spacing - 1e-10)
    rows, cols, data = [], [], []
    for out in range(count):
        lo, hi = start + out * target, start + (out + 1) * target
        for pixel in range(math.floor(lo / spacing + 1e-10), math.ceil(hi / spacing - 1e-10)):
            overlap = min(hi, (pixel + 1) * spacing) - max(lo, pixel * spacing)
            if overlap > 1e-10:
                rows.append(out)
                cols.append(pixel)
                data.append(overlap / target)
    return csr_matrix((data, (rows, cols)), shape=(count, width), dtype=np.float64)


def downsample(rgb, valid, x0, y0, count, spacing=.5):
    wx, wy = area_matrix(x0, count, spacing), area_matrix(y0, count, spacing)
    if rgb.dtype != np.uint8 or rgb.shape != (3, wy.shape[1], wx.shape[1]) or valid.shape != rgb.shape[1:]:
        raise ValueError('Area input window shape or RGB dtype mismatch')
    result = np.stack([np.asarray((wy @ channel.astype(np.float64)) @ wx.T) for channel in rgb])
    # Integer incidence counts make a single invalid contributor invalidate output,
    # independently of floating-point sum tolerances and RGB brightness.
    iy, ix = wy.copy(), wx.copy()
    iy.data[:] = 1
    ix.data[:] = 1
    holes = (iy @ (~valid).astype(np.float64)) @ ix.T
    support = np.asarray(holes) == 0
    return np.clip(np.floor(result + .5), 0, 255).astype(np.uint8), support


def overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) + 1e-6 and min(a[3], b[3]) > max(a[1], b[1]) + 1e-6


def write_candidate(path, rgb, support, crs, transform):
    # Publish one self-contained TIFF, including validity; never a lost .msk sidecar.
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path, 'w', driver='GTiff', width=1700, height=1700,
                count=3, dtype='uint8', crs=crs, transform=transform,
                photometric='RGB', tiled=True, blockxsize=256, blockysize=256,
                compress='DEFLATE') as output:
            output.write(rgb)
            output.write_mask(support.astype(np.uint8) * 255)
    with rasterio.open(path) as check:
        if not np.array_equal(check.read(), rgb) or not np.array_equal(check.dataset_mask(), support.astype(np.uint8) * 255):
            raise ValueError('Published mother candidate TIFF readback mismatch')


def run(args):
    instance = require_cloud_vm()
    from google.cloud import storage
    from PIL import Image
    rows = json.loads(Path(args.sources).read_text(encoding='utf-8-sig'))
    if args.max_cores_per_source < 1 or not rows:
        raise ValueError('Nonempty sources and positive maximum cores required')
    identities = set()
    for row in rows:
        validate_source(row)
        identity = (row['uri'], str(row['generation']))
        if identity in identities:
            raise ValueError('Repeated pinned source object')
        identities.add(identity)
    client = storage.Client(project=args.project)
    bucket_name, prefix = parse_gs(args.output_prefix)
    bucket = client.bucket(bucket_name)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    seen_strip = {}
    reports = []

    def publish(name, payload):
        return dict(uri=f'gs://{bucket_name}/{prefix}/{name}',
                    **create_only(bucket.blob(f'{prefix}/{name}'), payload))

    for source in sorted(rows, key=lambda r: r['asset_id']):
        report = dict(asset_id=source['asset_id'], source=source, candidates=[],
                      status='review_pending', semantic_accepted_count=0, rejections={})
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            path = Path(temporary) / 'original.tif'
            source_bucket, object_name = parse_gs(source['uri'])
            blob = client.bucket(source_bucket).blob(object_name, generation=int(source['generation']))
            blob.reload()
            if str(blob.generation) != str(source['generation']) or blob.size != source['bytes'] or blob.crc32c != source['crc32c']:
                raise ValueError('Pinned original metadata mismatch')
            blob.download_to_filename(str(path), if_generation_match=int(source['generation']), checksum='crc32c')
            report['source_sha256'] = file_sha256(path)
            with rasterio.open(path) as ds:
                try:
                    provenance = validate_header(ds)
                except ValueError as exc:
                    report.update(status='rejected_header_or_original_validity', reason=str(exc))
                else:
                    props = source['stac']['properties']
                    strip = props.get('pl:strip_id')
                    if not strip:
                        raise ValueError('Original acquisition strip identity missing')
                    report.update(mask_provenance=provenance, crs=ds.crs.to_string(),
                                  original_transform=list(ds.transform)[:6],
                                  resampling='one_time_exact_area_0.5_to_0.8_float64_round_half_up_uint8',
                                  processing_history='Planet ortho_visual; preserve original pansharpening/color correction history')
                    width, height = math.floor(ds.width * .5 / .8), math.floor(ds.height * .5 / .8)
                    target = Affine(.8, 0, ds.transform.c, 0, -.8, ds.transform.f)
                    key = (strip, ds.crs.to_string())
                    existing = seen_strip.setdefault(key, [])
                    for row in range(50, height - 1650 + 1, 1600):
                        for col in range(50, width - 1650 + 1, 1600):
                            if len(report['candidates']) >= args.max_cores_per_source:
                                break
                            left, top = target * (col, row)
                            core_bounds = [left, top - 1280, left + 1280, top]
                            if any(overlap(core_bounds, prev) for prev in existing):
                                report['rejections']['same_strip_spatial_overlap'] = report['rejections'].get('same_strip_spatial_overlap', 0) + 1
                                continue
                            x, y = (col - 50) * .8, (row - 50) * .8
                            sx, sy = math.floor(x / .5 + 1e-10), math.floor(y / .5 + 1e-10)
                            phase_x, phase_y = max(0, x - sx * .5), max(0, y - sy * .5)
                            wx, wy = area_matrix(phase_x, 1700), area_matrix(phase_y, 1700)
                            window = Window(sx, sy, wx.shape[1], wy.shape[1])
                            rgb = ds.read([1, 2, 3], window=window)
                            valid = np.all(ds.read_masks([1, 2, 3], window=window) == 255, axis=0)
                            out, support = downsample(rgb, valid, phase_x, phase_y, 1700)
                            if not support.all():
                                report['rejections']['incomplete_original_contributors'] = report['rejections'].get('incomplete_original_contributors', 0) + 1
                                continue
                            identity = dict(asset=source['asset_id'], strip=strip, crs=ds.crs.to_string(), core_bounds=core_bounds)
                            sid = digest_bytes(json_bytes(identity))[:24]
                            candidate_path = Path(temporary) / 'candidate.tif'
                            write_candidate(candidate_path, out, support, ds.crs,
                                            target * Affine.translation(col - 50, row - 50))
                            mother = publish(f'candidates/{sid}.tif', candidate_path.read_bytes())
                            image = io.BytesIO()
                            Image.fromarray(np.moveaxis(out[:, 50:1650, 50:1650], 0, -1)).save(image, format='PNG')
                            preview = publish(f'review/native/{sid}.png', image.getvalue())
                            report['candidates'].append(dict(sample_id=sid, **identity, mother=mother, preview=preview,
                                requires_semantic_review=True, requires_native_quality_review=True,
                                near_stereo_review_pending=source['site_id'] == 'san_francisco',
                                final_eligible=False, original_mask_provenance=provenance,
                                context_validity='every_positive_area_contributor_original_mask_255',
                                source_pixel_window=[sx, sy, wx.shape[1], wy.shape[1]],
                                native_gsd_m=source['native_gsd_m'], mother_spacing_m=.8))
                            existing.append(core_bounds)
            ref = publish(f'sources/{source["asset_id"]}.json', json_bytes(report))
            reports.append(ref)
            (scratch / 'checkpoint.json').write_bytes(json_bytes(dict(reports=reports)))
            print(json.dumps(dict(asset_id=source['asset_id'], status=report['status'], candidates=len(report['candidates']))), flush=True)
    return publish('summary.json', json_bytes(dict(instance=instance, sources=reports,
        worker_sha256=file_sha256(__file__), source_manifest_sha256=digest_bytes(json_bytes(rows)),
        max_cores_per_source=args.max_cores_per_source, accepted_count=0,
        status='provisional_candidates_no_training_ready_marker')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--sources', required=True)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--max-cores-per-source', type=int, default=20)
    print(json.dumps(run(parser.parse_args())))


if __name__ == '__main__':
    main()
