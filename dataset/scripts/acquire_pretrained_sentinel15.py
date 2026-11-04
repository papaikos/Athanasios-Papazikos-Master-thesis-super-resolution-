"""Fetch bounded spectral inputs on the preserved 15 Sentinel-2 example grids.

Small synchronous Earth Engine downloads only; no VM, export or billing changes.
"""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from urllib.request import urlopen
from src.cloud_config import GCP_PROJECT

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / '.venv-satellite/Lib/site-packages'))
import numpy as np
from scripts import skysat_acquire

SOURCE = ROOT / 'outputs/sentinel-examples'
OUT = ROOT / 'outputs/pretrained-comparison-20260929/inputs'
REPORT = ROOT / 'reports/pretrained-comparison-20260929/acquisition.json'
BANDS = ['B4', 'B3', 'B2', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((SOURCE / 'manifest.json').read_text('utf-8'))
    assert len(source_manifest['examples']) == 15
    skysat_acquire.GCLOUD = 'C:/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
    ee = skysat_acquire.initialize(GCP_PROJECT, True)
    rows = []
    for source in source_manifest['examples']:
        key = source['id']
        saved = OUT / f'{key}.json'
        if saved.exists():
            row = json.loads(saved.read_text('utf-8'))
            assert sha(OUT / row['path']) == row['sha256']
            rows.append(row)
            continue
        prov = source['provenance']
        assert sha(SOURCE / source['path']) == source['sha256']
        im = ee.Image(prov['scene_id'])
        # Native10m RGBN copied without resampling; native20m bands bilinear.
        spectral = im.select(BANDS[:4]).addBands(im.select(BANDS[4:]).resample('bilinear'))
        masks = im.select(BANDS).mask().rename(['valid_' + b for b in BANDS])
        image = spectral.addBands(masks).addBands(im.select('SCL')).unmask(0)
        url = image.getDownloadURL({'format': 'NPY', 'crs': prov['crs'],
                                   'crs_transform': prov['transform'], 'dimensions': [128, 128]})
        with urlopen(url, timeout=120) as response:
            raw = response.read()
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        assert arr.shape == (128, 128)
        reflectance = np.stack([arr[b] for b in BANDS]).astype(np.float32) / np.float32(10000)
        valid = np.stack([arr['valid_' + b] > 0 for b in BANDS])
        scl = arr['SCL'].astype(np.uint8)
        assert np.isfinite(reflectance).all()
        old = np.load(SOURCE / source['path'], allow_pickle=False)
        assert np.array_equal(reflectance[:3], old['reflectance']), 'RGB/grid changed'
        assert np.array_equal(scl, old['scl']), 'SCL/grid changed'
        tci = old['lr'].copy()
        metadata = {
            **prov, 'bands': BANDS, 'native_resolution_m': [10] * 4 + [20] * 6,
            'input_grid_m': 10, 'native20m_resampling': 'bilinear',
            'mask_and_scl_resampling': 'nearest', 'normalization': 'L2A surface reflectance DN / 10000; no clipping or image-specific stretch',
            'all_band_valid_fraction': float(valid.all(axis=0).mean()),
            'scl_clear_fraction': float((~np.isin(scl, [0, 1, 3, 8, 9, 10, 11])).mean()),
            'original_tci_preserved': True, 'source_manifest_sha256': sha(SOURCE / 'manifest.json'),
            'acquired_utc': datetime.now(timezone.utc).isoformat(),
        }
        path = OUT / f'{key}.npz'
        np.savez_compressed(path, reflectance=reflectance, valid_mask=valid,
                            scl=scl, tci=tci, bands=np.array(BANDS),
                            provenance=np.array(json.dumps(metadata)))
        shutil.copy2(SOURCE / f'{key}-original.png', OUT / f'{key}-original.png')
        row = {'id': key, 'title': source['title'], 'path': path.name,
               'sha256': sha(path), 'download_sha256': hashlib.sha256(raw).hexdigest(),
               'original_png_sha256': sha(OUT / f'{key}-original.png'),
               'provenance': metadata}
        write_json(saved, row)
        rows.append(row)
        write_json(REPORT, {'status': 'in_progress', 'completed': len(rows), 'examples': rows})
        print(json.dumps({'completed': len(rows), 'id': key,
                          'valid_fraction': metadata['all_band_valid_fraction']}), flush=True)
    manifest = {'status': 'complete', 'completed': len(rows), 'bands': BANDS,
                'created_utc': datetime.now(timezone.utc).isoformat(), 'examples': rows,
                'input_shape': [10, 128, 128], 'reflectance_type': 'L2A surface reflectance',
                'no_cloud_compute_or_bulk_exports': True}
    write_json(OUT / 'manifest.json', manifest)
    write_json(REPORT, manifest)


if __name__ == '__main__':
    main()
