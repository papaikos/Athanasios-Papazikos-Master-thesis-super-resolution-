"""Four provisional cloud proof datasets; final release gates remain mandatory."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import zipfile

import numpy as np
import rasterio
from rasterio.windows import Window, transform as window_transform

from scripts.maritime_cloud_inventory import (create_only, digest_bytes, file_sha256,
    json_bytes, parse_gs, require_cloud_vm, validate_header, validate_manifest)
from src.maritime_recipes import array_hash, canonical_hr, degrade
from src.cloud_config import GCP_PROJECT


def core_window(affine, cell):
    left, top = cell[0] * 1280, (cell[1] + 1) * 1280
    col, row = ~affine * (left, top)
    win = Window(round(col), round(row), 1600, 1600)
    native_left, native_top = affine * (win.col_off, win.row_off)
    shift = [native_left - left, native_top - top]
    if max(abs(v) for v in shift) > .4001:
        raise ValueError('Native snapping exceeds half a source pixel')
    return win, shift


def core_identity(asset_id, crs, affine, window):
    left, top = affine * (window.col_off, window.row_off)
    return [asset_id, crs, round(left, 6), round(top, 6), 1600, 1600]


def deterministic_npz(arrays):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, array in sorted(arrays.items()):
            raw = io.BytesIO()
            array = np.asarray(array)
            np.lib.format.write_array(raw, array if array.ndim == 0 else np.ascontiguousarray(array), allow_pickle=False)
            entry = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, raw.getvalue(), compresslevel=6)
    return output.getvalue()


def validate_pair(pair):
    for name, shape in [('hr', (3, 512, 512)), ('lr', (3, 128, 128))]:
        array = pair[name]
        if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all() or (array < 0).any() or (array > 1).any():
            raise ValueError('Invalid training array: ' + name)
    for name, shape in [('hr_mask', (512, 512)), ('lr_mask', (128, 128))]:
        if pair[name].shape != shape or not np.isin(pair[name], [0, 1]).all() or not pair[name].all():
            raise ValueError('Invalid full-support pilot mask: ' + name)


def write_geotiff(path, rgb, affine, crs):
    with rasterio.open(path, 'w', driver='GTiff', width=1700, height=1700,
            count=3, dtype='uint8', crs=crs, transform=affine,
            photometric='RGB', tiled=True, blockxsize=256, blockysize=256,
            compress='DEFLATE', predictor=2) as ds:
        ds.write(rgb)


def run(args):
    instance = require_cloud_vm()
    from google.cloud import storage
    from PIL import Image, ImageDraw
    import pyarrow as pa
    import pyarrow.parquet as pq

    sources = validate_manifest(json.loads(Path(args.sources).read_text()))
    reviews = json.loads(Path(args.review_references).read_text())
    by_source = defaultdict(list)
    for review in reviews:
        by_source[review['source_object']['uri']].append(review)
    client = storage.Client(project=args.project)
    bucket_name, prefix = parse_gs(args.output_prefix)
    bucket = client.bucket(bucket_name)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    config = dict(source_manifest_sha256=digest_bytes(json_bytes(sources)),
                  reviews_sha256=digest_bytes(json_bytes(reviews)), max_samples=args.max_samples,
                  seed=20261006, worker_sha256=file_sha256(__file__), instance=instance,
                  status='provisional_development_only_no_final_ready_markers')
    samples, rejections, all_outputs = [], [], []
    shard_members = {recipe: [] for recipe in ['d1', 'd2', 'd3']}
    shard_refs = {recipe: [] for recipe in shard_members}
    loader_receipts = {recipe: [] for recipe in shard_members}

    def publish(name, payload):
        ref = dict(uri=f'gs://{bucket_name}/{prefix}/{name}',
                   **create_only(bucket.blob(f'{prefix}/{name}'), payload))
        all_outputs.append(ref)
        return ref

    def flush_shards():
        for recipe, members in shard_members.items():
            if not members:
                continue
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode='w', format=tarfile.USTAR_FORMAT) as tar:
                for sample_id, payload in sorted(members):
                    info = tarfile.TarInfo('development/' + sample_id + '.npz')
                    info.size = len(payload)
                    info.mode = 0o644
                    tar.addfile(info, io.BytesIO(payload))
            data = buf.getvalue()
            from src.maritime_dataset import MaritimeDataset, smoke_batch
            with tempfile.TemporaryDirectory(dir=scratch, prefix='loader-') as temporary:
                shard = Path(temporary) / 'pairs.tar'
                shard.write_bytes(data)
                loader_receipts[recipe].append(smoke_batch(MaritimeDataset([shard], recipe)))
            name = f'children/{recipe}/shards/development-{len(shard_refs[recipe]):05d}.tar'
            ref = publish(name, data)
            ref['sample_ids'] = [sample_id for sample_id, _ in sorted(members)]
            shard_refs[recipe].append(ref)
            members.clear()

    seen = set()
    for source in sources:
        refs = sorted(by_source.get(source['uri'], []), key=lambda r: r['pair_id'])
        if not refs or len(samples) >= args.max_samples:
            continue
        source_bucket, name = parse_gs(source['uri'])
        blob = client.bucket(source_bucket).blob(name, generation=int(source['generation']))
        blob.reload()
        if int(blob.size) != source['bytes'] or blob.crc32c != source['crc32c']:
            raise ValueError('Pinned source integrity mismatch')
        with tempfile.TemporaryDirectory(dir=scratch, prefix='source-') as temporary:
            path = Path(temporary) / 'original.tif'
            blob.download_to_filename(str(path), if_generation_match=int(source['generation']), checksum='crc32c')
            source_sha = file_sha256(path)
            with rasterio.open(path) as ds:
                validate_header(ds)
                if not (source.get('export_recipe', {}).get('unmask') == 0
                        and source['export_recipe'].get('noData') == 0 and ds.nodata == 0):
                    raise ValueError('Documented conservative native support required for historical pilot')
                for review in refs:
                    if len(samples) >= args.max_samples:
                        break
                    if str(review['source_object']['generation']) != source['generation']:
                        raise ValueError('Historical review references a different source generation')
                    win, shift = core_window(ds.transform, review['cell'])
                    ctx = Window(win.col_off - 50, win.row_off - 50, 1700, 1700)
                    reject = None
                    if ctx.col_off < 0 or ctx.row_off < 0 or ctx.col_off + 1700 > ds.width or ctx.row_off + 1700 > ds.height:
                        reject = 'real_context_outside_source'
                    elif not (ds.read_masks(window=ctx) == 255).all():
                        reject = 'incomplete_original_core_or_halo_support'
                    if reject:
                        rejections.append(dict(historical_pair_id=review['pair_id'], reason=reject))
                        continue
                    identity = core_identity(source['asset_id'], ds.crs.to_string(), ds.transform, win)
                    sample_id = digest_bytes(json_bytes(identity))[:24]
                    if sample_id in seen:
                        raise ValueError('Duplicate acquisition/core window')
                    seen.add(sample_id)
                    rgb = ds.read(window=ctx)
                    mask = np.ones((1700, 1700), bool)
                    context, hr = canonical_hr(rgb, mask)
                    ctx_affine = window_transform(ctx, ds.transform)
                    affine = window_transform(win, ds.transform)
                    provenance = dict(sample_id=sample_id, mother_id=sample_id, split='development',
                        site_id=source['site_id'], geographic_group_id=source['site_id'],
                        source=source, source_sha256=source_sha, acquisition_utc=source['acquisition_utc'],
                        crs=ds.crs.to_string(), mother_context_transform=list(ctx_affine)[:6],
                        mother_core_transform=list(affine)[:6], core_window=list(win.flatten()),
                        hr_transform=[2.5, 0, affine.c, 0, -2.5, affine.f],
                        lr_transform=[10, 0, affine.c, 0, -10, affine.f],
                        halo_pixels=50, rgb_order=['R', 'G', 'B'], core_bytes=7680000,
                        gsd=[.8, 2.5, 10], original_native_snapping_shift_m=shift,
                        hr_sha256=array_hash(hr), context_sha256=array_hash(context),
                        historical_review=review['historical_review'], historical_pair_id=review['pair_id'],
                        requires_new_native_scale_review=True, vessel_presence='unconfirmed',
                        small_craft_presence='unconfirmed', semantic_status='historical_reference_pending_current_review',
                        source_support='documented_unmask_zero_nodata_zero_conservative',
                        container_image_id=args.container_image_id,
                        code_sha256=config['worker_sha256'], lr_origin='synthetic_hr_only')
                    mother_file = Path(temporary) / 'mother.tif'
                    write_geotiff(mother_file, rgb, ctx_affine, ds.crs)
                    provenance['mother'] = publish(f'mother/imagery/{sample_id}.tif', mother_file.read_bytes())
                    mask_file = Path(temporary) / 'mask.tif'
                    with rasterio.open(mask_file, 'w', driver='GTiff', width=1700, height=1700,
                            count=1, dtype='uint8', transform=ctx_affine, crs=ds.crs, compress='DEFLATE') as out:
                        out.write(np.full((1, 1700, 1700), 255, np.uint8))
                    provenance['mother_mask'] = publish(f'mother/masks/{sample_id}.tif', mask_file.read_bytes())
                    shared = deterministic_npz(dict(context=context, hr=hr,
                        hr_mask=np.ones((512, 512), bool), provenance=np.array(json.dumps(provenance, sort_keys=True))))
                    provenance['shared'] = publish(f'shared/hr-context/{sample_id}.npz', shared)
                    results = {}
                    for recipe in shard_members:
                        result = degrade(context, sample_id, recipe=recipe)
                        pair = dict(hr=hr, lr=result['lr'], hr_mask=result['hr_mask'], lr_mask=result['lr_mask'])
                        validate_pair(pair)
                        if result['metadata']['hr_sha256'] != provenance['hr_sha256']:
                            raise ValueError('Canonical HR mismatch between recipes')
                        # Repeat the identical recipe to establish deterministic decoded arrays.
                        repeated = degrade(context, sample_id, recipe=recipe)
                        if array_hash(repeated['lr']) != array_hash(result['lr']):
                            raise ValueError('Nondeterministic recipe retry')
                        metadata = dict(provenance, recipe=result['metadata'],
                                        pair_id=digest_bytes(json_bytes([sample_id, result['metadata']['recipe'], '0'])))
                        pair['provenance'] = np.array(json.dumps(metadata, sort_keys=True))
                        shard_members[recipe].append((sample_id, deterministic_npz(pair)))
                        results[recipe] = result['lr']
                    # Native source detail remains available at original pixel dimensions.
                    native = Image.fromarray(np.moveaxis(rgb[:, 50:-50, 50:-50], 0, -1))
                    buf = io.BytesIO()
                    native.save(buf, format='PNG')
                    provenance['native_review_preview'] = publish(f'review/native/{sample_id}.png', buf.getvalue())
                    sheet = Image.new('RGB', (400 * 5, 430), 'white')
                    panels = [('mother', np.asarray(native.resize((400, 400)))), ('HR', np.moveaxis(hr, 0, -1) * 255)]
                    panels.extend((recipe, np.moveaxis(results[recipe], 0, -1) * 255) for recipe in ['d1', 'd2', 'd3'])
                    draw = ImageDraw.Draw(sheet)
                    for i, (label, pixels) in enumerate(panels):
                        panel = Image.fromarray(np.asarray(np.clip(pixels, 0, 255), np.uint8)).resize((400, 400), Image.Resampling.NEAREST)
                        sheet.paste(panel, (400 * i, 30))
                        draw.text((400 * i + 6, 8), label, fill='black')
                    buf = io.BytesIO()
                    sheet.save(buf, format='PNG')
                    publish(f'review/comparison/{sample_id}.png', buf.getvalue())
                    samples.append(provenance)
                    if len(shard_members['d1']) == 32:
                        flush_shards()
                    (scratch / 'pilot-progress.json').write_text(json.dumps(dict(samples=len(samples), rejections=len(rejections))))
                    print(json.dumps(dict(samples=len(samples), rejections=len(rejections), sample_id=sample_id)), flush=True)
    flush_shards()
    if not samples:
        raise ValueError('No fully supported native pilot samples')
    sample_ref = publish('shared/common-samples.json', json_bytes(samples))
    # Structured common index retains full JSON provenance as one string column.
    table = pa.Table.from_pylist([dict(sample_id=s['sample_id'], site_id=s['site_id'], split=s['split'],
        hr_sha256=s['hr_sha256'], core_bytes=s['core_bytes'], provenance=json.dumps(s, sort_keys=True)) for s in samples])
    buf = io.BytesIO()
    pq.write_table(table, buf, compression='zstd')
    publish('shared/common-samples.parquet', buf.getvalue())
    for recipe in shard_refs:
        publish(f'children/{recipe}/manifest.json', json_bytes(dict(status='provisional_pilot',
            recipe=recipe, sample_count=len(samples), common_samples=sample_ref, shards=shard_refs[recipe])))
        publish(f'children/{recipe}/DATA_CARD.md', (f'# {recipe} provisional development proof set\n\n'
            f'{len(samples)} paired samples, synthetic 10 m LR and common 2.5 m HR. '
            'Numerically checked with loader smoke checks and deterministic recomputation. '
            'Final maritime, source diversity, size and split gates are pending; this is not the requested final release.\n').encode())
    report = dict(status='numerically_verified_provisional_pending_mother_and_semantic_gates',
        sample_count=len(samples), useful_core_bytes=len(samples) * 7680000,
        semantic_accepted_core_bytes=0, sites=len({s['site_id'] for s in samples}),
        configuration=config, common_samples=sample_ref, child_shards=shard_refs,
        rejections=rejections, pilot_samples_target=args.max_samples,
        mother_gate_passed=False, final_child_jobs_started=False,
        decoded_arrays_identical_on_retry=True, every_pair_loader_verified=True,
        training_loader_receipts=loader_receipts,
        all_outputs=all_outputs)
    publish('mother/manifest.json', json_bytes(dict(status='provisional_pilot', common_samples=sample_ref,
        imagery=[s['mother'] for s in samples], masks=[s['mother_mask'] for s in samples])))
    publish('mother/DATA_CARD.md', (f'# 0.8 m provisional mother proof set\n\n{len(samples)} '
        'native RGB windows with real 50 pixel halos. Historical maritime references require current '
        'native scale review. No final mother acceptance marker has been issued.\n').encode())
    publish('pilot-report.json', json_bytes(report))
    (scratch / 'pilot-report.json').write_bytes(json_bytes(report))
    print(json.dumps({k: v for k, v in report.items() if k not in ('all_outputs', 'rejections', 'child_shards')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default=GCP_PROJECT)
    parser.add_argument('--sources', required=True)
    parser.add_argument('--review-references', required=True)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--max-samples', type=int, default=100)
    parser.add_argument('--container-image-id', required=True)
    args = parser.parse_args()
    if args.max_samples <= 0:
        parser.error('max-samples must be positive')
    run(args)


if __name__ == '__main__':
    main()
