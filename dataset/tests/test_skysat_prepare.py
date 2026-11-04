import numpy as np
import pytest
import rasterio
import hashlib
import json
from types import SimpleNamespace
from rasterio.transform import from_origin

from src.skysat_prepare import (degrade_rgb, completed_objects, patches,
                               validate_manifest, validate_receipt, verify_shards, validate_grid)


def test_rgb_degradation_masks_kernel_support():
    hr = np.full((3, 512, 512), 0.5, np.float32)
    valid = np.ones((512, 512), bool)
    valid[250, 250] = False
    lr, mask = degrade_rgb(hr, valid)
    assert lr.shape == (3, 128, 128)
    assert not mask[62, 62]
    assert not mask[0, 0]
    assert mask[20, 20]
    assert lr[:, 20, 20] == pytest.approx([0.5] * 3)


def test_completed_objects_enforces_geographic_split():
    manifest = {'bucket': 'b', 'prefix': 'raw', 'sources': [
        {'source_id': 'a', 'group_id': 'g', 'split': 'train'},
        {'source_id': 'b', 'group_id': 'g', 'split': 'test'}]}
    with pytest.raises(ValueError, match='split'):
        completed_objects(manifest, {'tasks': {}}, [])
    manifest['sources'].pop()
    obj = {'name': 'raw/a/rgb-000.tif', 'bucket': 'b', 'generation': '1', 'size': '20'}
    assert not completed_objects(manifest, {'tasks': {'a': {'state': 'RUNNING'}}}, [obj])
    assert len(completed_objects(manifest, {'tasks': {'a': {'state': 'COMPLETED'}}}, [obj])) == 1


def test_patches_preserve_rgb_and_mask_holes(tmp_path):
    path = tmp_path / 'rgb.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=1024, height=512,
                       count=3, dtype='uint8', crs='EPSG:32610',
                       transform=from_origin(0, 1280, 2.5, 2.5)) as ds:
        ds.write(np.full((3, 512, 1024), 128, np.uint8))
        mask = np.full((512, 1024), 255, np.uint8)
        mask[20, 20] = 0
        ds.write_mask(mask)
    result = list(patches(path))
    assert len(result) == 1
    assert result[0]['hr'].shape == (3, 512, 512)
    assert result[0]['native_bytes'] == 512 * 512 * 3
    assert result[0]['hr'].mean() == pytest.approx(128 / 255)


def test_cloud_sdk_nested_objects_and_generation_conflicts():
    manifest = {'bucket': 'b', 'prefix': 'raw', 'sources': [
        {'source_id': 'a', 'group_id': 'g', 'split': 'train'}]}
    state = {'tasks': {'a': {'state': 'COMPLETED'}}}
    obj = {'url': 'gs://b/raw/a/rgb.tif#123', 'type': 'cloud_object',
           'metadata': {'bucket': 'b', 'name': 'raw/a/rgb.tif',
                        'generation': '123', 'size': '45'}}
    assert completed_objects(manifest, state, [obj])[0][1] == {
        'uri': 'gs://b/raw/a/rgb.tif', 'generation': '123', 'bytes': 45}
    obj['metadata']['generation'] = '124'
    with pytest.raises(ValueError, match='generation'):
        completed_objects(manifest, state, [obj])
    obj['metadata']['generation'] = '123'
    obj['metadata']['size'] = 'broken'
    with pytest.raises(ValueError):
        completed_objects(manifest, state, [obj])
    obj['metadata']['size'] = '45'
    obj['metadata']['name'] = 'raw/a/wrong.tif'
    with pytest.raises(ValueError, match='identity'):
        completed_objects(manifest, state, [obj])


def test_missing_native_mask_fails_closed(tmp_path):
    path = tmp_path / 'unmasked.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=512, height=512,
                       count=3, dtype='uint8', crs='EPSG:32610',
                       transform=from_origin(0, 1280, 2.5, 2.5)) as ds:
        ds.write(np.zeros((3, 512, 512), np.uint8))
    with pytest.raises(ValueError, match='mask'):
        list(patches(path))


def test_manifest_checksum_and_asset_duplicates():
    m = {'sources': [{'source_id': 'a', 'asset_id': 'asset'}]}
    m['sha256'] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
    validate_manifest(m)
    m['sources'][0]['source_id'] = 'changed'
    with pytest.raises(ValueError, match='checksum'):
        validate_manifest(m)
    m.pop('sha256')
    m['sources'].append({'source_id': 'b', 'asset_id': 'asset'})
    m['sha256'] = hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError, match='asset'):
        validate_manifest(m)


def test_native_average_and_hr_lr_pixel_alignment(tmp_path):
    path = tmp_path / 'checker.tif'
    pixels = np.indices((1024, 1024)).sum(axis=0) % 2 * 200 + 20
    with rasterio.open(path, 'w', driver='GTiff', width=1024, height=1024,
                       count=3, dtype='uint8', crs='EPSG:32610',
                       transform=from_origin(0, 1280, 1.25, 1.25)) as ds:
        ds.write(np.stack([pixels] * 3).astype(np.uint8))
        ds.write_mask(np.full((1024, 1024), 255, np.uint8))
    patch = next(patches(path))
    assert patch['hr'] == pytest.approx(np.full((3, 512, 512), 120 / 255), abs=1e-6)
    assert patch['lr'][:, 20, 20] == pytest.approx([120 / 255] * 3)
    assert patch['transform'] == [2.5, 0, 0, 0, -2.5, 1280]
    assert patch['lr_transform'] == [10, 0, 0, 0, -10, 1280]
    assert patch['native_bytes'] == 1024 * 1024 * 3


def test_receipt_rejects_changed_manifest_config_and_identity():
    row = {'source_id': 'a', 'group_id': 'g', 'split': 'train'}
    obj = {'uri': 'gs://b/a', 'generation': '1', 'bytes': 20}
    config = {'recipe': 'rgb-v2', 'manifest_sha256': 'hash', 'max_pairs_per_object': 8}
    receipt = {**row, **config, 'source_object': obj, 'pairs': 1,
               'source_sha256': 'f' * 64,
               'accepted_source_bytes': 3, 'valid_native_rgb_bytes': 6,
               'shards': [{'uri': 'gs://b/out/pairs-00000.tar', 'sha256': 'a' * 64}]}
    validate_receipt(receipt, row, obj, config)
    for key in ['split', 'manifest_sha256', 'max_pairs_per_object']:
        bad = {**receipt, key: 'changed'}
        with pytest.raises(ValueError):
            validate_receipt(bad, row, obj, config)


def test_resume_reads_and_checks_each_tar(tmp_path, monkeypatch):
    import src.skysat_prepare as module
    content = b'validated tar'
    calls = []
    def fake_cloud(args, *command, **kwargs):
        calls.append(command)
        command[-1].write_bytes(content)
    monkeypatch.setattr(module, 'cloud', fake_cloud)
    receipt = {'shards': [{'uri': 'gs://b/out/pairs-00000.tar',
                           'sha256': hashlib.sha256(content).hexdigest()}]}
    verify_shards(SimpleNamespace(), receipt, 'gs://b/out', tmp_path)
    assert len(calls) == 1
    receipt['shards'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checksum'):
        verify_shards(SimpleNamespace(), receipt, 'gs://b/out', tmp_path)


def test_lr_area_alignment_for_linear_rgb_ramps():
    y, x = np.indices((512, 512), dtype=np.float32)
    hr = np.stack([x/512, y/512, (x+y)/1024])
    lr, mask = degrade_rgb(hr, np.ones((512, 512), bool))
    # A symmetric blur preserves a linear ramp away from edges. Area averaging
    # puts each LR center at HR center index 4*j+1.5, without any pixel shift.
    assert mask[30, 40]
    assert lr[:, 30, 40] == pytest.approx(
        [(4*40+1.5)/512, (4*30+1.5)/512, (4*40+4*30+3)/1024], abs=1e-6)


def test_manifest_grid_allows_shard_offsets_but_rejects_fractional_shift():
    row = {'grid': {'crsCode': 'EPSG:32610', 'affineTransform': {
        'scaleX': 1.25, 'scaleY': -1.25, 'translateX': 0, 'translateY': 1280}}}
    source = SimpleNamespace(crs=rasterio.crs.CRS.from_epsg(32610),
                             transform=from_origin(10240, 1280, 1.25, 1.25))
    validate_grid(source, row)
    source.transform = from_origin(10240.1, 1280, 1.25, 1.25)
    with pytest.raises(ValueError, match='origin'):
        validate_grid(source, row)
