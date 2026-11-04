"""Build the x10 SkySat synthetic demonstration inputs (1 m HR / 10 m LR) from the preserved New Bedford scene.

Same scene and the same 10 spots as the D1-D3 x4 SkySat demonstration (centres of the original 256x256 crops).
Each spot: a 400x400 native (0.8 m) context = 320 m, exact area integration to 1 m (320 px), the D3x10
recipe (PSF in metres, 10x10 area mean, 4 LR px halo crop, D3 draws/noise from seeds) -> LR 24x24 at 10 m;
HR core 240x240 at 1 m; plus a 2.5 m reference of the same core (96x96) so the D3 x4 model can be scored on
the same 10 m input. HR/LR are stored as uint8 like the training data. Run with the system Python (rasterio).
"""
import hashlib
import json
import os
import math
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
from scipy.signal import fftconvolve

ROOT = Path(os.environ.get('RUN_ARCHIVE', Path(__file__).resolve().parents[1]))  # archive folder of one run, e.g. 'D1 run'
D3 = Path(os.environ.get('D3_ARCHIVE', ROOT.parent / 'D3 run'))
SRC = D3 / 'skysat-comparisons/inputs'
OUT = ROOT / 'skysat-comparisons/inputs'
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'dataset'))  # dataset package of this repo
from scripts.maritime_planet_intake import area_matrix  # noqa: E402  (the D3x10 build's exact area integrator)
from src.maritime_recipes import RECIPE_VERSIONS, _seed, array_hash  # noqa: E402

# Original D3 crop origins (native px) and labels, from D3 run/skysat-comparisons/inputs/prepare_sources.py
POSITIONS = [(512, 0, 'Industrial port tanks and quay'), (768, 256, 'Working quay and harbor'), (768, 512, 'Commercial docks'),
             (768, 768, 'Waterfront pier'), (768, 1024, 'Port shoreline'), (1024, 0, 'Floating dock and harbor'),
             (1024, 256, 'Harbor craft'), (1024, 512, 'Moored boats'), (1280, 512, 'Shore and moorings'),
             (1280, 768, 'Breakwater and shoreline')]
NATIVE_M, CONTEXT_NATIVE, CONTEXT_M, HALO_M, FACTOR = 0.8, 400, 320, 40, 10
RELEASE_SEED = 20261006


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def area(rgb, count, target, phase=0.0):
    w = area_matrix(phase, count, NATIVE_M, target)
    h = w.shape[1]
    out = np.stack([np.asarray((w @ ch[:h, :h].astype(np.float64)) @ w.T) for ch in rgb])
    return (out / 255.).astype(np.float32)


def draws(sample_id):
    """Exactly the D3 draw procedure (src.maritime_recipes.degrade / build_d3x10.d3_draws)."""
    version = RECIPE_VERSIONS['d3']
    ps, ns = _seed(RELEASE_SEED, sample_id, version, '0', 'parameters'), _seed(RELEASE_SEED, sample_id, version, '0', 'noise')
    rng = np.random.Generator(np.random.PCG64(ps))
    iso = rng.random() < 0.5
    sx = float(rng.uniform(1, 3)); sy = sx if iso else float(rng.uniform(1, 3))
    angle = 0.0 if iso else float(rng.uniform(0, np.pi))
    noise = 0.0 if rng.random() < 0.2 else float(rng.uniform(0, 2) / 255)
    return dict(kernel_type='isotropic' if iso else 'anisotropic', sigma_x_hr_pixels=sx, sigma_y_hr_pixels=sy, angle_radians=angle,
                noise_sigma_normalized=noise, parameter_seed=ps, noise_seed=ns)


def d3x10(context, d):
    """build_d3x10.d3x10 generalized to any context side (PSF in metres, area10, 4 LR px crop, noise, clip)."""
    radius = 30
    sx, sy, a = d['sigma_x_hr_pixels'] * 2.5, d['sigma_y_hr_pixels'] * 2.5, d['angle_radians']
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    rx, ry = np.cos(a) * x + np.sin(a) * y, -np.sin(a) * x + np.cos(a) * y
    k = np.exp(-0.5 * ((rx / sx) ** 2 + (ry / sy) ** 2)); k = (k / k.sum()).astype(np.float32)
    padded = np.pad(context, ((0, 0), (radius, radius), (radius, radius)), mode='symmetric')
    blurred = np.stack([fftconvolve(c, k, mode='valid') for c in padded]).astype(np.float32)
    s = context.shape[1] // FACTOR
    lr = blurred.reshape(3, s, FACTOR, s, FACTOR).mean(axis=(2, 4), dtype=np.float32)[:, 4:-4, 4:-4]
    if d['noise_sigma_normalized']:
        lr = lr + np.random.Generator(np.random.PCG64(d['noise_seed'])).normal(0, d['noise_sigma_normalized'], lr.shape).astype(np.float32)
    return np.ascontiguousarray(np.clip(lr, 0, 1), dtype=np.float32), array_hash(k)


def q8(x):
    return np.floor(np.clip(x.astype(np.float64), 0, 1) * 255 + .5).astype(np.uint8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ('new-bedford-native-1536.tif', 'original-scene-metadata.json', 'ATTRIBUTION-LICENSE.txt', 'acquisition-request.json',
                 'acquire_skysat.py', 'prepare_sources.py', 'source-overview.png'):
        if (SRC / name).exists():
            shutil.copy2(SRC / name, OUT / ('original-D3-' + name if name.endswith('.py') else name))
            assert sha(SRC / name) == sha(OUT / ('original-D3-' + name if name.endswith('.py') else name))
    tif = OUT / 'new-bedford-native-1536.tif'
    rows = []
    with rasterio.open(tif) as ds:
        assert abs(ds.transform.a - NATIVE_M) < 1e-9 and ds.shape == (1536, 1536)
        for i, (x0, y0, label) in enumerate(POSITIONS, 1):
            sid = f'skysat-x10-{i:02d}'
            cx, cy = x0 + 128, y0 + 128
            col = min(max(cx - CONTEXT_NATIVE // 2, 0), 1536 - CONTEXT_NATIVE)
            row = min(max(cy - CONTEXT_NATIVE // 2, 0), 1536 - CONTEXT_NATIVE)
            win = Window(col, row, CONTEXT_NATIVE, CONTEXT_NATIVE)
            rgb = ds.read([1, 2, 3], window=win)
            if np.any(np.all(rgb == 0, axis=0)):
                raise ValueError(f'{sid}: nodata in context')
            context = area(rgb, CONTEXT_M, 1.0)                                # 320x320 at 1 m
            hr = context[:, HALO_M:-HALO_M, HALO_M:-HALO_M]                    # 240x240 core
            d = draws(sid)
            lr, kernel_sha = d3x10(context, d)                                 # 24x24 at 10 m
            ref25 = area(rgb[:, 50:350, 50:350], 96, 2.5)                      # same core, 2.5 m, for the x4 model
            hr8, lr8, r8 = q8(hr), q8(lr), q8(ref25)
            folder = OUT / sid
            folder.mkdir(exist_ok=True)
            Image.fromarray(hr8.transpose(1, 2, 0)).save(folder / 'HR-reference-1m.png')
            Image.fromarray(r8.transpose(1, 2, 0)).save(folder / 'HR-reference-2.5m.png')
            Image.fromarray(lr8.transpose(1, 2, 0)).save(folder / 'LR-10m.png')
            np.save(folder / 'LR-10m-fp32.npy', (lr8.astype(np.float32) / 255).astype(np.float32))
            transform = ds.window_transform(win)
            core_left, core_top = transform.c + HALO_M, transform.f - HALO_M
            record = {'id': sid, 'content': label, 'original_D3_crop_xywh': [x0, y0, 256, 256],
                      'context_window_native_xywh': [col, row, CONTEXT_NATIVE, CONTEXT_NATIVE],
                      'context_shift_from_centre_px': [col - (cx - 200), row - (cy - 200)],
                      'crs': str(ds.crs), 'core_bounds': [core_left, core_top - 240, core_left + 240, core_top],
                      'hr_transform': [1.0, 0, core_left, 0, -1.0, core_top], 'lr_transform': [10.0, 0, core_left, 0, -10.0, core_top],
                      'ref_2_5m_transform': [2.5, 0, core_left, 0, -2.5, core_top],
                      'hr_method': 'exact physical-overlap area integration of native 0.8 m uint8 RGB/255 (area_matrix), quantized uint8',
                      'lr_recipe': 'd3x10-variable-psf-noise-v1 (generalized context size)', 'release_seed': RELEASE_SEED,
                      'draws': d, 'kernel_sha256': kernel_sha, 'clipping': 'once', 'storage': 'uint8 half-up, as the D3x10 dataset',
                      'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in folder.iterdir() if p.is_file() and p.suffix != '.json'},
                      'source': {'file': tif.name, 'sha256': sha(tif), 'acquisition': 'SkySat New Bedford harbor (preserved D3 demonstration download)',
                                 'license': 'CC-BY-4.0, (c) 2014 Planet Labs Inc.'},
                      'scientific_limitation': 'Same acquisition / region as known training sources (as recorded for the D3 demonstration). '
                                               'Synthetic LR; not independent held-out validation.'}
            (folder / 'provenance.json').write_text(json.dumps(record, indent=1), encoding='utf-8')
            rows.append(record)
            print(sid, label, 'window', col, row, 'shift', record['context_shift_from_centre_px'])
    (OUT / 'sources.json').write_text(json.dumps({'schema': 'd3x10-skysat-x10-demonstration-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                                                  'count': len(rows), 'hr_m': 1.0, 'lr_m': 10.0, 'hr_core_px': 240, 'lr_core_px': 24,
                                                  'samples': rows}, indent=1), encoding='utf-8')
    shutil.copy2(Path(__file__), OUT / 'prepare_skysat_x10.py')


if __name__ == '__main__':
    main()
