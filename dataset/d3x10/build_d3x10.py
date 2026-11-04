"""Build D3x10: 1 m HR / 10 m LR (x10) maritime pairs with the D3 recipe.

Inputs are only read: Mother TARs (2.5 m context + provenance), D3 TARs (each
sample's D3 draws and 10 m LR), Mother/native/objects (original GeoTIFFs) and
the Jev master table (sample list + flags). For every sample:

1. verify the native GeoTIFF size and SHA-256 against Mother's pin;
2. integrate the 1360 m context by exact physical overlap to 2.5 m and require
   it to reproduce Mother's stored context (geometry check), then to 1.0 m;
3. apply D3 with the PSF fixed in metres: sigma 2.5-7.5 m (D3: 1-3 px at
   2.5 m), radius 30 m (61x61 at 1 m), reflect boundary, exact 10x10 area mean,
   4 LR px halo crop, D3's own noise sigma and noise seed, clip once;
4. require the re-derived draws to equal D3's stored draws, and record how far
   the new LR is from D3's LR;
5. quantize to uint8 (half-up) and write deterministic NPZs into TAR shards.

  python build_d3x10.py build  --out DIR [--workers 8] [--sample-ids a,b,...] [--resume RUN_DIR]
  python build_d3x10.py verify --out DIR --run-dir RUN_DIR [--workers 8]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import glob
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import time
import traceback

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import scipy
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parent
# Expanded maritime dataset (Mother, D1-D3) and the Jev master table made by
# quality_check/thesis_report.py. Set both in the environment (see .env.example).
DATA = Path(os.environ.get('MARITIME_DATA_ROOT', 'Maritime_RGB_Expanded_20261008'))
JEV_MASTER = Path(os.environ.get('JEV_MASTER_TABLE', ROOT.parents[1] / 'quality_check/thesis_data/'
                                 '2026-10-19_095519Z_rules_v4_full_mother/master_table.csv'))
CODE = ROOT.parent  # the dataset package of this repository (src/, scripts/)
sys.path.insert(0, str(CODE))
from src.maritime_recipes import RECIPE_VERSIONS, _seed, array_hash  # noqa: E402
from src.maritime_uint8 import QUANTIZATION, quantize_rgb, uint8_array_hash  # noqa: E402
from scripts.maritime_planet_intake import area_matrix  # noqa: E402
from scripts.maritime_pilot import deterministic_npz  # noqa: E402
from scripts.maritime_earthview_intake import apply_toa_mapping  # noqa: E402

RECIPE = 'd3x10-variable-psf-noise-v1'
STORAGE = 'maritime-rgb-uint8-1m10m-v1'
HR_M, FACTOR = 1.0, 10
CONTEXT_PX, HALO_HR, HALO_LR = 1360, 40, 4          # 1360 m context = 1280 m core + 40 m halo
HR_PX, LR_PX = 1280, 128
KERNEL_RADIUS_M = 30.0                               # D3: radius 12 px at 2.5 m
SHARD_SIZE = 64
GEOMETRY_TOL = 1e-6
SPLITS = ('train', 'validation', 'test')
CODE_FILES = ['src/maritime_recipes.py', 'src/maritime_uint8.py', 'scripts/maritime_planet_intake.py',
              'scripts/maritime_pilot.py', 'scripts/maritime_earthview_intake.py']
OPERATORS = ['native exact area 1360 m -> 1.0 m', 'gaussian-psf (metres)', 'area10', 'crop 4 LR px',
             'noise', 'clip', 'uint8 quantization']


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, default=str), encoding='utf-8')
    os.replace(tmp, path)


def load_member(tar_path, member):
    with tarfile.open(tar_path) as tar:
        payload = tar.extractfile(member).read()
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


# ---------------------------------------------------------------- per sample

_VERIFIED = set()


def read_native(prov):
    """Original RGB over the 1360 m context, with its spacing and area phase."""
    kind = prov['source_kind']
    if kind == 'skysat_raw_window':
        pin, window = prov['native_window_ref']['object'], prov['native_window_ref']['context_window']
    else:
        pin, window = prov['native_context'], None
    path = DATA / 'Mother/native/objects' / (pin['sha256'] + '.tif')
    if path not in _VERIFIED:
        if path.stat().st_size != pin['bytes'] or sha256_file(path) != pin['sha256']:
            raise ValueError('native object size/SHA-256 differs from Mother pin')
        _VERIFIED.add(path)
    with rasterio.open(path) as ds:
        win = Window(*window) if window else None
        rgb = ds.read([1, 2, 3], window=win)
        valid = np.all(ds.read_masks([1, 2, 3], window=win) == 255, axis=0)
        pixel, original_dtype = ds.transform.a, str(ds.dtypes[0])
    if kind == 'skysat_raw_window':
        spacing, phase = 0.8, (0.0, 0.0)      # Mother's nominal 0.8 m grid, 1700 px = 1360 m
    elif kind == 'earthview':
        if rgb.shape[1:] != (CONTEXT_PX, CONTEXT_PX) or pixel != 1:
            raise ValueError('EarthView native context is not 1360 px at 1 m')
        spacing, phase = 1.0, (0.0, 0.0)
    else:
        source = prov['source_provenance']
        spacing, phase = float(source['product_spacing_m']), tuple(map(float, source['area_phase_m']))
    mapping = None
    if rgb.dtype == np.uint16:
        if kind != 'earthview' or prov.get('input_product') != 'TOA':
            raise ValueError('uint16 native outside EarthView TOA')
        mapping = prov['rgb_mapping']
        rgb = apply_toa_mapping(rgb, mapping)
    elif rgb.dtype != np.uint8:
        raise ValueError('unsupported native dtype')
    native = dict(file=path.name, sha256=pin['sha256'], bytes=pin['bytes'], uri=pin.get('uri'),
                  window=window, product_spacing_m=spacing, encoded_pixel_m=pixel,
                  area_phase_m=list(phase), original_dtype=original_dtype,
                  rgb_mapping=mapping)
    return rgb, valid, spacing, phase, native


def area(rgb, valid, spacing, phase, count, target):
    """Exact physical-overlap integration, float64, of uint8 RGB / 255."""
    wx = area_matrix(phase[0], count, spacing, target)
    wy = area_matrix(phase[1], count, spacing, target)
    h, w = wy.shape[1], wx.shape[1]
    if rgb.shape[1] < h or rgb.shape[2] < w:
        raise ValueError('native window smaller than the 1360 m context')
    if not valid[:h, :w].all():
        raise ValueError('incomplete native support')
    out = np.stack([np.asarray((wy @ ch[:h, :w].astype(np.float64)) @ wx.T) for ch in rgb])
    return (out / 255.).astype(np.float32)


def d3_draws(sample_id, release_seed):
    """Exactly the draws of src.maritime_recipes.degrade(..., 'd3')."""
    version = RECIPE_VERSIONS['d3']
    parameter_seed = _seed(release_seed, sample_id, version, '0', 'parameters')
    noise_seed = _seed(release_seed, sample_id, version, '0', 'noise')
    rng = np.random.Generator(np.random.PCG64(parameter_seed))
    isotropic = rng.random() < 0.5
    sigma_x = float(rng.uniform(1, 3))
    sigma_y = sigma_x if isotropic else float(rng.uniform(1, 3))
    angle = 0.0 if isotropic else float(rng.uniform(0, np.pi))
    noise = 0.0 if rng.random() < 0.2 else float(rng.uniform(0, 2) / 255)
    return dict(kernel_type='isotropic' if isotropic else 'anisotropic', sigma_x_hr_pixels=sigma_x,
                sigma_y_hr_pixels=sigma_y, angle_radians=angle, noise_sigma_normalized=noise,
                parameter_seed=parameter_seed, noise_seed=noise_seed)


def d3x10(context, draws):
    radius = int(KERNEL_RADIUS_M / HR_M)
    sigma_x_m, sigma_y_m = draws['sigma_x_hr_pixels'] * 2.5, draws['sigma_y_hr_pixels'] * 2.5
    sx, sy, a = sigma_x_m / HR_M, sigma_y_m / HR_M, draws['angle_radians']
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    rx, ry = np.cos(a) * x + np.sin(a) * y, -np.sin(a) * x + np.cos(a) * y
    kernel = np.exp(-0.5 * ((rx / sx) ** 2 + (ry / sy) ** 2))
    kernel = (kernel / kernel.sum()).astype(np.float32)
    # scipy.ndimage 'reflect' == numpy 'symmetric'; the kernel is point-symmetric.
    padded = np.pad(context, ((0, 0), (radius, radius), (radius, radius)), mode='symmetric')
    blurred = np.stack([fftconvolve(ch, kernel, mode='valid') for ch in padded]).astype(np.float32)
    side = CONTEXT_PX // FACTOR
    lr = blurred.reshape(3, side, FACTOR, side, FACTOR).mean(axis=(2, 4), dtype=np.float32)
    lr = lr[:, HALO_LR:-HALO_LR, HALO_LR:-HALO_LR]
    if draws['noise_sigma_normalized']:
        rng = np.random.Generator(np.random.PCG64(draws['noise_seed']))
        lr = lr + rng.normal(0, draws['noise_sigma_normalized'], lr.shape).astype(np.float32)
    clipping = float(np.count_nonzero((lr < 0) | (lr > 1)) / lr.size)
    lr = np.ascontiguousarray(np.clip(lr, 0, 1), dtype=np.float32)
    return lr, dict(sigma_x_m=sigma_x_m, sigma_y_m=sigma_y_m, kernel_radius_m=KERNEL_RADIUS_M,
                    kernel_shape=[2 * radius + 1] * 2, kernel_sha256=array_hash(kernel),
                    clipping_fraction=clipping)


def process_sample(s):
    sid = s['sample_id']
    mother = load_member(DATA / 'Mother' / s['split'] / s['mother_tar'], s['mother_member'])
    prov = json.loads(str(mother['provenance']))
    if prov['sample_id'] != sid or prov['split'] != s['split']:
        raise ValueError('Mother identity/split mismatch')
    if array_hash(mother['context']) != prov['context_sha256']:
        raise ValueError('Mother context hash mismatch')
    d3 = load_member(DATA / 'D3' / s['d3_split'] / s['d3_tar'], sid + '.npz')
    d3meta = json.loads(str(d3['provenance']))
    d3r = d3meta['recipe']
    if d3r['recipe'] != RECIPE_VERSIONS['d3'] or d3r['sample_id'] != sid or d3r['variant_id'] != '0':
        raise ValueError('D3 identity mismatch')
    draws = d3_draws(sid, d3r['release_seed'])
    for key, value in draws.items():
        if d3r[key] != value:
            raise ValueError('re-derived D3 draw differs from D3 shard: ' + key)

    rgb, valid, spacing, phase, native = read_native(prov)
    if native['file'] != s['native_tif']:
        raise ValueError('native file differs from master table')
    geometry = float(np.abs(area(rgb, valid, spacing, phase, 544, 2.5) - mother['context']).max())
    if geometry > GEOMETRY_TOL:
        raise ValueError(f'2.5 m re-integration differs from Mother context by {geometry}')
    context = area(rgb, valid, spacing, phase, CONTEXT_PX, HR_M)
    hr = np.ascontiguousarray(context[:, HALO_HR:-HALO_HR, HALO_HR:-HALO_HR])
    lr, recipe_extra = d3x10(context, draws)
    hr_u8, lr_u8 = quantize_rgb(hr), quantize_rgb(lr)

    diff = np.abs(lr_u8.astype(np.int16) - d3['lr'].astype(np.int16))
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    psnr = float('inf') if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    hr_native = None
    if spacing == HR_M and phase == (0.0, 0.0):
        hr_native = bool(np.array_equal(hr_u8, rgb[:, HALO_HR:HALO_HR + HR_PX, HALO_HR:HALO_HR + HR_PX]))

    x0, y0 = prov['hr_transform'][2], prov['hr_transform'][5]
    checks = dict(native_sha256_verified=True, mother_context_sha256_verified=True,
                  geometry_2_5m_max_abs_vs_mother=geometry, d3_draws_rederived=True,
                  lr_vs_d3_psnr_db=psnr, lr_vs_d3_mean_abs_dn=float(diff.mean()),
                  lr_vs_d3_max_abs_dn=int(diff.max()), hr_equals_native_pixels=hr_native)
    meta = dict(
        schema=STORAGE, sample_id=sid, split=s['split'], source_kind=prov['source_kind'],
        native_gsd_m=prov['native_gsd_m'], native_gsd_ge_1m=bool(prov['native_gsd_m'] >= 1.0),
        lr_origin='synthetic_hr_only', crs=prov['crs'], core_bounds=prov['core_bounds'],
        hr_transform=[HR_M, 0, x0, 0, -HR_M, y0], lr_transform=[HR_M * FACTOR, 0, x0, 0, -HR_M * FACTOR, y0],
        hr_shape=[3, HR_PX, HR_PX], lr_shape=[3, LR_PX, LR_PX], scale=FACTOR, native=native,
        hr_method=dict(method='exact physical-overlap area integration of original RGB/255',
                       implementation='scripts.maritime_planet_intake.area_matrix, float64 -> float32',
                       context_px=CONTEXT_PX, core_px=HR_PX, halo_px=HALO_HR,
                       earthview_toa_display_mapping='scripts.maritime_earthview_intake.apply_toa_mapping'
                       if native['rgb_mapping'] else None),
        recipe=dict(recipe=RECIPE, derived_from=RECIPE_VERSIONS['d3'], release_seed=d3r['release_seed'],
                    variant_id='0', **draws, **recipe_extra, sigma_semantics='D3 sigma (px at 2.5 m) x 2.5 = metres',
                    boundary='reflect (numpy symmetric pad) of the real 1360 m context',
                    convolution='scipy.signal.fftconvolve valid, float32', reduction='exact 10x10 area mean',
                    crop_lr_pixels=HALO_LR, noise_distribution='independent zero-mean Gaussian per channel/pixel',
                    rng='numpy.random.PCG64', operator_order=OPERATORS, numpy_version=np.__version__,
                    scipy_version=scipy.__version__, hr_sha256=array_hash(hr), lr_sha256=array_hash(lr)),
        storage=dict(version=STORAGE, dtype='uint8', quantization=QUANTIZATION,
                     array_hash_encoding='C-contiguous uint8 bytes',
                     hr_sha256=uint8_array_hash(hr_u8), lr_sha256=uint8_array_hash(lr_u8)),
        checks=checks,
        references=dict(mother=dict(split=s['split'], tar=s['mother_tar'], member=s['mother_member'],
                                    context_sha256=prov['context_sha256']),
                        d3=dict(split=s['d3_split'], tar=s['d3_tar'], member=sid + '.npz',
                                lr_float_sha256=d3r['lr_sha256'],
                                lr_uint8_sha256=d3meta.get('storage', {}).get('lr_sha256'))),
        mother_provenance=prov)
    payload = deterministic_npz(dict(hr=hr_u8, lr=lr_u8, hr_mask=np.ones((HR_PX, HR_PX), bool),
                                     lr_mask=np.ones((LR_PX, LR_PX), bool),
                                     provenance=np.asarray(json.dumps(meta, sort_keys=True))))
    record = dict(sample_id=sid, x10_native_spacing_m=spacing,
                  x10_native_original_dtype=native['original_dtype'],
                  x10_geometry_max_abs=geometry, x10_lr_vs_d3_psnr_db=psnr,
                  x10_lr_vs_d3_mean_abs_dn=checks['lr_vs_d3_mean_abs_dn'],
                  x10_lr_vs_d3_max_abs_dn=checks['lr_vs_d3_max_abs_dn'],
                  x10_hr_equals_native_pixels=hr_native, x10_clipping_fraction=recipe_extra['clipping_fraction'],
                  x10_kernel_type=draws['kernel_type'], x10_sigma_x_m=recipe_extra['sigma_x_m'],
                  x10_sigma_y_m=recipe_extra['sigma_y_m'], x10_angle_radians=draws['angle_radians'],
                  x10_noise_sigma_dn=draws['noise_sigma_normalized'] * 255,
                  x10_hr_uint8_sha256=meta['storage']['hr_sha256'],
                  x10_lr_uint8_sha256=meta['storage']['lr_sha256'], x10_npz_bytes=len(payload))
    return payload, record


def build_shard(job):
    started = time.time()
    out = Path(job['out']) / job['split'] / (job['name'] + '.tar')
    partial = Path(str(out) + '.partial')
    records = []
    try:
        with open(partial, 'wb') as fh, tarfile.open(fileobj=fh, mode='w', format=tarfile.PAX_FORMAT) as tar:
            for s in job['samples']:
                try:
                    payload, record = process_sample(s)
                except Exception as error:
                    raise RuntimeError(f"{s['sample_id']}: {error!r}") from error
                info = tarfile.TarInfo(s['sample_id'] + '.npz')
                info.size, info.mode, info.mtime = len(payload), 0o644, 0
                tar.addfile(info, io.BytesIO(payload))
                record.update(x10_split=job['split'], x10_shard=out.name, x10_member=info.name)
                records.append(record)
        digest, size = sha256_file(partial), partial.stat().st_size
        os.replace(partial, out)
    except Exception:
        partial.unlink(missing_ok=True)
        return dict(status='error', shard=job['name'], utc=utc(), error=traceback.format_exc())
    result = dict(status='ok', shard=out.name, split=job['split'], sha256=digest, bytes=size,
                  samples=len(records), seconds=round(time.time() - started, 1), utc=utc())
    write_json(Path(job['records']) / (job['name'] + '.json'), dict(**result, records=records))
    return result


# ---------------------------------------------------------------- orchestration

def d3_index():
    index = {}
    for split in SPLITS:
        for path in sorted(glob.glob(str(DATA / 'D3' / split / '*.tar'))):
            with tarfile.open(path) as tar:
                for name in tar.getnames():
                    index[name[:-4]] = (split, Path(path).name)
    return index


def make_plan(sample_ids=None):
    master = pd.read_csv(JEV_MASTER, low_memory=False)
    index = d3_index()
    if set(master.sample_id) != set(index) or len(master) != len(index):
        raise ValueError('Jev master table and D3 shards do not hold the same samples')
    if any(index[r.sample_id][0] != r.split for r in master.itertuples()):
        raise ValueError('split differs between Mother and D3')
    if sample_ids:
        master = master[master.sample_id.isin(sample_ids)]
        if len(master) != len(set(sample_ids)):
            raise ValueError('unknown sample id requested')
    jobs = []
    for split in SPLITS:
        rows = master[master.split == split].sort_values('sample_id')
        for start in range(0, len(rows), SHARD_SIZE):
            jobs.append(dict(split=split, name=f'd3x10-{split}-{start // SHARD_SIZE:05d}', samples=[
                dict(sample_id=r.sample_id, split=r.split, mother_tar=r.tar, mother_member=r.image,
                     d3_split=index[r.sample_id][0], d3_tar=index[r.sample_id][1], native_tif=r.native_tif)
                for r in rows.iloc[start:start + SHARD_SIZE].itertuples()]))
    return jobs


def environment():
    return dict(host=platform.node(), platform=platform.platform(), python=sys.version.split()[0],
                numpy=np.__version__, scipy=scipy.__version__, rasterio=rasterio.__version__,
                pandas=pd.__version__, cpu_count=os.cpu_count())


def input_pins():
    return dict(jev_master_table=dict(path=str(JEV_MASTER), sha256=sha256_file(JEV_MASTER)),
                mother_manifest_sha256=sha256_file(DATA / 'Mother/manifest.json'),
                d3_manifest_sha256=sha256_file(DATA / 'D3/manifest.json'),
                code_files={f: sha256_file(CODE / f) for f in CODE_FILES},
                build_script_sha256=sha256_file(Path(__file__)))


def log(run, line):
    print(line, flush=True)
    with open(run / 'progress.log', 'a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def build(args):
    out = Path(args.out)
    if args.resume:
        run = Path(args.resume)
        plan = json.loads((run / 'shard_plan.json').read_text())
        log(run, f'{utc()} resume run {run.name}')
    else:
        label = 'dryrun' if args.sample_ids else 'full'
        run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%SZ') + '_' + label)
        run.mkdir(parents=True)
        (run / 'records').mkdir()
        plan = make_plan(args.sample_ids.split(',') if args.sample_ids else None)
        write_json(run / 'shard_plan.json', plan)
        shutil.copy2(__file__, run / 'script_snapshot.py')
        write_json(run / 'run.json', dict(started_utc=utc(), label=label, args=vars(args), out=str(out),
                                          recipe=RECIPE, storage=STORAGE, shard_size=SHARD_SIZE,
                                          shards=len(plan), samples=sum(len(j['samples']) for j in plan),
                                          environment=environment(), inputs=input_pins()))
        log(run, f'{utc()} start {label}: {len(plan)} shards, out={out}')
    for split in SPLITS:
        (out / split).mkdir(parents=True, exist_ok=True)
    todo = [j for j in plan if not ((out / j['split'] / (j['name'] + '.tar')).exists()
                                    and (run / 'records' / (j['name'] + '.json')).exists())]
    for job in todo:
        job.update(out=str(out), records=str(run / 'records'))
    log(run, f'{utc()} {len(plan) - len(todo)} shards already done, {len(todo)} to build, {args.workers} workers')
    started, done, failed = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(build_shard, job) for job in todo]):
            result = future.result()
            done += 1
            if result['status'] != 'ok':
                failed += 1
                with open(run / 'errors.jsonl', 'a', encoding='utf-8') as fh:
                    fh.write(json.dumps(result) + '\n')
                log(run, f"{utc()} ERROR {result['shard']}: {result['error'].strip().splitlines()[-1]}")
                continue
            rate = (time.time() - started) / done
            log(run, f"{utc()} ok {result['shard']} {result['samples']} samples {result['bytes'] / 1e6:.0f} MB "
                     f"{result['seconds']}s [{done}/{len(todo)}, eta {rate * (len(todo) - done) / 60:.0f} min]")
    if failed:
        log(run, f'{utc()} {failed} shard(s) failed; fix and rerun with --resume {run}')
        return 1
    finalize(run, out, plan)
    return 0


def finalize(run, out, plan):
    shards, records = [], []
    for job in plan:
        rec = json.loads((run / 'records' / (job['name'] + '.json')).read_text())
        if [r['sample_id'] for r in rec['records']] != [s['sample_id'] for s in job['samples']]:
            raise ValueError('records differ from plan: ' + job['name'])
        shards.append(dict(split=rec['split'], file=f"{rec['split']}/{rec['shard']}", sha256=rec['sha256'],
                           bytes=rec['bytes'], samples=rec['samples'],
                           sample_ids=[r['sample_id'] for r in rec['records']]))
        records.extend(rec['records'])
    master = pd.read_csv(JEV_MASTER, low_memory=False)
    built = pd.DataFrame(records)
    table = master.merge(built, on='sample_id', how='inner', validate='one_to_one')
    table.insert(table.columns.get_loc('native_gsd_m') + 1, 'native_gsd_ge_1m', table.native_gsd_m >= 1.0)
    table.to_csv(out / 'samples.csv', index=False)
    counts = {split: int((table.split == split).sum()) for split in SPLITS}
    manifest = dict(
        schema='maritime-d3x10-manifest-v1', status='BUILT', created_utc=utc(), build_run=run.name,
        recipe=RECIPE, derived_from=RECIPE_VERSIONS['d3'], storage=STORAGE, scale=FACTOR,
        hr=dict(gsd_m=HR_M, shape=[3, HR_PX, HR_PX], dtype='uint8'),
        lr=dict(gsd_m=HR_M * FACTOR, shape=[3, LR_PX, LR_PX], dtype='uint8', origin='synthetic_hr_only'),
        npz_keys=['hr', 'lr', 'hr_mask', 'lr_mask', 'provenance'], counts=counts,
        samples_csv=dict(file='samples.csv', sha256=sha256_file(out / 'samples.csv'),
                         note='Jev master table (all metadata + every Jev answer + decision) joined with x10_* build '
                              'columns; native_gsd_ge_1m flags HR taken at native resolution (no downsampling margin)'),
        shards=shards, inputs=json.loads((run / 'run.json').read_text())['inputs'],
        environment=environment())
    write_json(out / 'manifest.json', manifest)
    shutil.copy2(out / 'samples.csv', run / 'samples.csv')
    shutil.copy2(out / 'manifest.json', run / 'manifest.json')
    summary = dict(finished_utc=utc(), counts=counts, shards=len(shards),
                   total_bytes=int(sum(s['bytes'] for s in shards)),
                   native_gsd_ge_1m=int(table.native_gsd_ge_1m.sum()),
                   jev_decision=table.decision.value_counts(dropna=False).to_dict(),
                   geometry_max_abs=float(built.x10_geometry_max_abs.max()),
                   lr_vs_d3_psnr_db=built.x10_lr_vs_d3_psnr_db.replace(np.inf, np.nan).describe().to_dict(),
                   lr_vs_d3_max_abs_dn=built.x10_lr_vs_d3_max_abs_dn.value_counts().sort_index().to_dict(),
                   hr_equals_native_pixels=built.x10_hr_equals_native_pixels.value_counts(dropna=False).to_dict(),
                   clipping_fraction_max=float(built.x10_clipping_fraction.max()))
    write_json(run / 'summary.json', summary)
    log(run, f"{utc()} finalized: {counts}, {summary['total_bytes'] / 1e9:.1f} GB -> {out / 'manifest.json'}")


# ---------------------------------------------------------------- verification

def verify_shard(path, entry):
    digest = sha256_file(path)
    if digest != entry['sha256']:
        return dict(file=entry['file'], ok=False, error='shard sha256 mismatch')
    ids = []
    with tarfile.open(path) as tar:
        for member in tar:
            with np.load(io.BytesIO(tar.extractfile(member).read()), allow_pickle=False) as data:
                meta = json.loads(str(data['provenance']))
                hr, lr = data['hr'], data['lr']
                ok = (hr.shape == (3, HR_PX, HR_PX) and lr.shape == (3, LR_PX, LR_PX)
                      and hr.dtype == np.uint8 and lr.dtype == np.uint8
                      and data['hr_mask'].all() and data['lr_mask'].all()
                      and uint8_array_hash(hr) == meta['storage']['hr_sha256']
                      and uint8_array_hash(lr) == meta['storage']['lr_sha256']
                      and member.name == meta['sample_id'] + '.npz' and meta['recipe']['recipe'] == RECIPE
                      and meta['hr_transform'][0] == HR_M and meta['lr_transform'][0] == HR_M * FACTOR
                      and meta['hr_transform'][2] == meta['lr_transform'][2]
                      and meta['hr_transform'][5] == meta['lr_transform'][5])
            if not ok:
                return dict(file=entry['file'], ok=False, error='sample check failed: ' + member.name)
            ids.append(meta['sample_id'])
    if ids != entry['sample_ids']:
        return dict(file=entry['file'], ok=False, error='member list differs from manifest')
    return dict(file=entry['file'], ok=True, samples=len(ids))


def verify(args):
    out, run = Path(args.out), Path(args.run_dir)
    manifest = json.loads((out / 'manifest.json').read_text())
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(verify_shard, [out / s['file'] for s in manifest['shards']], manifest['shards']))
    bad = [r for r in results if not r['ok']]
    report = dict(utc=utc(), shards=len(results), samples=sum(r.get('samples', 0) for r in results),
                  failed=bad, seconds=round(time.time() - started, 1),
                  samples_csv_sha256_ok=sha256_file(out / 'samples.csv') == manifest['samples_csv']['sha256'])
    write_json(run / f"verify_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%SZ')}.json", report)
    log(run, f"{utc()} verify: {report['shards']} shards, {report['samples']} samples, {len(bad)} failed, "
             f"samples.csv ok={report['samples_csv_sha256_ok']}")
    return 1 if bad or not report['samples_csv_sha256_ok'] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', required=True)
    b.add_argument('--workers', type=int, default=8)
    b.add_argument('--sample-ids', help='comma-separated subset (dry run)')
    b.add_argument('--resume', help='existing run directory to continue')
    v = sub.add_parser('verify')
    v.add_argument('--out', required=True)
    v.add_argument('--run-dir', required=True)
    v.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    for forbidden in (DATA, JEV_MASTER.parents[2]):
        if Path(args.out).resolve().is_relative_to(forbidden.resolve()):
            parser.error('output must be outside the datasets')
    raise SystemExit(build(args) if args.command == 'build' else verify(args))


if __name__ == '__main__':
    main()
