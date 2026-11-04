from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from scripts import maritime_direct_intake as direct
from scripts.maritime_candidate_manifest import normalize_candidate


def catalog():
    grid = dict(crsCode='EPSG:32644', dimensions=dict(width=6400, height=6400),
                affineTransform=dict(scaleX=.8, scaleY=-.8, translateX=0, translateY=5120))
    return dict(id='SKYSAT/example', updateTime='2026-01-01',
                properties=dict(catalogID='example'),
                bands=[dict(id=name, dataType=dict(precision='INT', range=dict(max=255)),
                            grid=deepcopy(grid)) for name in ('R', 'G', 'B')])


def test_windows_preserve_real_halo_native_pixels_and_nonoverlap():
    plan = direct.catalog_plan(catalog())
    windows = direct.planned_windows(plan)
    assert len(windows) == 4  # Boundary cells have no real 50-pixel halo.
    for win in windows:
        col, row, width, height = win['context_window']
        assert min(col, row) >= 0 and col + width <= 6400 and row + height <= 6400
        assert (width, height) == (1700, 1700)
        assert win['context_transform'][2] == col * .8
        assert win['context_transform'][5] == 5120 - row * .8
    cells = {tuple(row['world_grid_cell']) for row in windows}
    assert len(cells) == len(windows)


@pytest.mark.parametrize('mutation', ['grid', 'float', 'scale', 'rotation'])
def test_original_grid_or_dtype_mismatch_rejected(mutation):
    row = catalog()
    if mutation == 'grid': row['bands'][1]['grid']['dimensions']['width'] = 6500
    if mutation == 'float': row['bands'][0]['dataType']['precision'] = 'FLOAT'
    if mutation == 'scale': row['bands'][0]['grid']['affineTransform']['scaleX'] = 1
    if mutation == 'rotation': row['bands'][0]['grid']['affineTransform']['shearX'] = .1
    with pytest.raises(ValueError): direct.catalog_plan(row)


def test_round_robin_spreads_sites_then_assets_before_second_windows():
    plans = {name: [dict(asset_id=name, world_grid_cell=[i, 0]) for i in range(3)]
             for name in ('a', 'b', 'c')}
    rows = direct.spread_windows(plans, {'a': 'site1', 'b': 'site1', 'c': 'site2'}, 6)
    assert [row['asset_id'] for row in rows[:3]] == ['a', 'c', 'b']
    assert len({(row['asset_id'], tuple(row['world_grid_cell'])) for row in rows}) == 6


def test_download_request_has_exact_grid_and_no_region_scale_resampling():
    window = direct.planned_windows(direct.catalog_plan(catalog()))[0]
    params = direct.download_parameters(direct.catalog_plan(catalog()), window)
    assert params['dimensions'] == [1700, 1700] and params['format'] == 'GEO_TIFF'
    assert params['crs_transform'] == window['context_transform']
    assert params['filePerBand'] is False
    assert params['bands'] == ['R', 'G', 'B', 'producer_mask']
    assert not {'region', 'scale', 'resampling'} & params.keys()


def test_spreading_deduplicates_physical_windows_across_asset_aliases():
    plans = {'a': [dict(asset_id='a', world_grid_cell=[0, 0], sample_id='same')],
             'b': [dict(asset_id='b', world_grid_cell=[0, 0], sample_id='same')]}
    assert len(direct.spread_windows(plans, {'a': 'site1', 'b': 'site1'}, 10)) == 1


def test_coarse_min_mask_rejects_only_wholly_contained_invalid_blocks():
    plan = direct.catalog_plan(catalog())
    spec = direct.coarse_parameters(plan)
    window = direct.planned_windows(plan)[0]
    values = np.ones((2, spec['dimensions'][1], spec['dimensions'][0]), dtype=np.uint8)
    values[0] *= 255
    col, row, _, _ = window['context_window']
    factor = spec['factor']
    # This block intersects the context boundary but includes outside pixels:
    # its zero cannot prove an invalid pixel inside the downloaded window.
    values[0, row // factor, col // factor] = 0
    decision = direct.coarse_decision(values, spec, window)
    assert not decision['proven_unsupported']
    values[0, math.ceil(row / factor) + 1, math.ceil(col / factor) + 1] = 0
    assert direct.coarse_decision(values, spec, window)['proven_unsupported']


def test_coarse_mixed_water_priority_never_confirms_semantics_or_discards_dry():
    plan = direct.catalog_plan(catalog())
    spec = direct.coarse_parameters(plan)
    windows = direct.planned_windows(plan)
    values = np.zeros((2, spec['dimensions'][1], spec['dimensions'][0]), dtype=np.uint8)
    values[0] = 255
    dry = direct.coarse_decision(values, spec, windows[0])
    assert not dry['proven_unsupported'] and dry['priority_tier'] == 2
    values[1, :, ::2] = 1
    mixed = direct.coarse_decision(values, spec, windows[0])
    assert mixed['priority_tier'] == 0 and .4 < mixed['coarse_water_fraction'] < .6
    assert mixed['semantic_status'] == 'unconfirmed_screening_only'


def test_coarse_request_keeps_native_origin_and_documents_exact_minimum():
    spec = direct.coarse_parameters(direct.catalog_plan(catalog()))
    assert spec['factor'] == 32
    assert spec['crs_transform'] == [25.6, 0, 0, 0, -25.6, 5120]
    assert spec['dimensions'] == [200, 200]
    assert spec['bands'] == ['producer_mask_min', 'water_proxy']
    assert not {'scale', 'region'} & spec.keys()


def test_coarse_context_request_is_tiny_and_ignores_padded_outside_tail():
    plan = direct.catalog_plan(catalog())
    window = direct.planned_windows(plan)[0]
    spec = direct.coarse_parameters(plan, window)
    assert spec['dimensions'] == [54, 54]
    assert spec['original_pixel_origin'] == window['context_window'][:2]
    assert spec['crs_transform'][2] == window['context_transform'][2]
    assert spec['crs_transform'][5] == window['context_transform'][5]
    values = np.ones((2, 54, 54), dtype=np.uint8)
    values[0] = 255
    values[0, -1, :] = 0
    values[0, :, -1] = 0
    assert not direct.coarse_decision(values, spec, window)['proven_unsupported']
    values[0, 5, 7] = 0
    decision = direct.coarse_decision(values, spec, window)
    assert decision['proven_unsupported']
    assert decision['proven_invalid_block_original_pixel_window'] == [
        window['context_window'][0] + 7*32, window['context_window'][1] + 5*32, 32, 32]


def raster(tmp_path, mask=255, shifted=False):
    win = direct.planned_windows(direct.catalog_plan(catalog()))[0]
    affine = Affine(*win['context_transform'])
    if shifted: affine = affine * Affine.translation(1, 0)
    path = tmp_path / 'source.tif'
    with rasterio.open(path, 'w', driver='GTiff', width=1700, height=1700,
                       count=4, dtype='uint8', crs='EPSG:32644', transform=affine) as ds:
        ds.write(np.zeros((3, 1700, 1700), dtype=np.uint8), indexes=[1, 2, 3])
        ds.write(np.full((1700, 1700), mask, dtype=np.uint8), 4)
    return path, win


def test_black_rgb_is_valid_using_original_mask(tmp_path):
    path, win = raster(tmp_path)
    rgb, mask, header = direct.read_download(path, direct.catalog_plan(catalog()), win)
    assert not rgb.any() and mask.all() and header['width'] == 1700


@pytest.mark.parametrize('mask,shifted', [(1, False), (0, False), (255, True)])
def test_invalid_mask_or_shifted_return_rejected(tmp_path, mask, shifted):
    path, win = raster(tmp_path, mask, shifted)
    with pytest.raises(ValueError): direct.read_download(path, direct.catalog_plan(catalog()), win)


def test_retry_policy_does_not_retry_permissions_or_invalid_pixels():
    assert direct.retry_delay(SimpleNamespace(code=429), 1) == 30
    assert direct.retry_delay(SimpleNamespace(code=503), 2) == 60
    assert direct.retry_delay(SimpleNamespace(code=403), 1) is None
    assert direct.retry_delay(ValueError('invalid native grid'), 1) is None


def test_attempt_ledger_consumes_uncertain_interrupted_attempts(tmp_path):
    ledger = direct.AttemptLedger(tmp_path / 'attempt.json', 'configuration', 'window')
    assert ledger.begin() == 1
    recovered = direct.AttemptLedger(tmp_path / 'attempt.json', 'configuration', 'window')
    assert recovered.begin() == 2
    recovered.finish('FAILED', 'transport error')
    assert recovered.begin() == 3
    with pytest.raises(RuntimeError, match='exhausted'): recovered.begin()
    with pytest.raises(ValueError, match='configuration'):
        direct.AttemptLedger(tmp_path / 'attempt.json', 'different', 'window')


def test_gce_guard_runs_before_credentials_or_pixels():
    # No operational parameters or SDK imports are available: the guard must
    # fail first and prohibit all local pixel transport.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(direct, 'require_cloud_vm', lambda: (_ for _ in ()).throw(RuntimeError('GCE required')))
        with pytest.raises(RuntimeError, match='GCE required'): direct.run(SimpleNamespace())


@pytest.mark.parametrize('coarse,unsupported', [(False, False), (True, False), (True, True)])
def test_direct_intake_end_to_end_normalizes_and_resumes_without_redownload(tmp_path, monkeypatch, coarse, unsupported):
    record = catalog()
    window = direct.planned_windows(direct.catalog_plan(record))[0]
    fixture, _ = raster(tmp_path)
    spec = direct.coarse_parameters(direct.catalog_plan(record), window)
    coarse_fixture = tmp_path / 'coarse.tif'
    with rasterio.open(coarse_fixture, 'w', driver='GTiff', width=54, height=54,
        count=2, dtype='uint8', crs='EPSG:32644', transform=Affine(*spec['crs_transform'])) as ds:
        mask_values = np.full((54, 54), 255, dtype=np.uint8)
        if unsupported: mask_values[5, 7] = 0
        ds.write(mask_values, 1)
        ds.write(np.zeros((54, 54), dtype=np.uint8), 2)
    # planned_windows is deterministic; raster() writes the first planned grid.
    calls, objects = [], {}
    class Blob:
        def __init__(self, name): self.name = name
        def exists(self): return self.name in objects
        def reload(self): self.generation = 1
        def download_as_bytes(self, **kwargs): return objects[self.name]
        def download_to_filename(self, path, **kwargs): Path(path).write_bytes(objects[self.name])
    class Bucket:
        def blob(self, name): return Blob(name)
    client = SimpleNamespace(bucket=lambda name: Bucket())
    def publish(blob, payload):
        if blob.name in objects: assert objects[blob.name] == payload
        objects[blob.name] = payload
        return dict(generation='1', bytes=len(payload), crc32c='synthetic-crc', sha256=direct.digest_bytes(payload))
    class Image:
        def __getattr__(self, name): return lambda *args, **kwargs: self
        def rename(self, name): return self
        def addBands(self, other): return self
        def getDownloadURL(self, params):
            calls.append(params)
            return (coarse_fixture if params['bands'] == ['producer_mask_min', 'water_proxy'] else fixture).as_uri()
    ee = ModuleType('ee')
    ee.Initialize = lambda **kwargs: calls.append(kwargs)
    ee.data = SimpleNamespace(getAsset=lambda asset: deepcopy(record))
    ee.Image = lambda *args: Image()
    ee.Reducer = SimpleNamespace(min=lambda: 'minimum')
    auth = ModuleType('google.auth')
    auth.default = lambda **kwargs: ('credentials', 'project')
    storage = ModuleType('google.cloud.storage')
    storage.Client = lambda **kwargs: client
    google = ModuleType('google')
    google.auth = auth
    cloud = ModuleType('google.cloud')
    cloud.storage = storage
    google.cloud = cloud
    for name, module in [('ee', ee), ('google', google), ('google.auth', auth),
                         ('google.cloud', cloud), ('google.cloud.storage', storage)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(direct, 'require_cloud_vm', lambda: '12345')
    monkeypatch.setattr(direct, 'export_images', lambda ee, asset: (Image(), Image()))
    monkeypatch.setattr(direct, 'create_only', publish)
    monkeypatch.setattr(direct, 'verify_output', lambda client, ref: objects[direct.parse_gs(ref['uri'])[1]])
    approval = dict(asset_ids=[record['id']], review_metadata=dict(groups=[dict(
        group_id='site1', asset_ids=[record['id']], country='India')]))
    args = SimpleNamespace(project='project', workers=1, max_candidates=1,
        output_prefix='gs://bucket/direct-test', scratch=str(tmp_path / 'scratch'), coarse_screen=coarse)
    for key, value in [('approved_assets', approval), ('catalog_metadata', [record]),
                       ('site_metadata', {'site1': {'country': 'India'}})]:
        path = tmp_path / (key + '.json')
        path.write_text(json.dumps(value))
        setattr(args, key, str(path))
    first = direct.run(args)
    if unsupported:
        assert first['candidate_count'] == 0
        assert first['coarse_proven_unsupported_skipped'] == 1
        assert first['attempted_window_count'] == 0
        assert not any(call.get('dimensions') == [1700, 1700] for call in calls)
        assert direct.run(args)['candidate_count'] == 0
        assert len([call for call in calls if call.get('dimensions') == [54, 54]]) == 1
        return
    assert first['candidate_count'] == 1
    candidate = first['candidates'][0]
    report_ref = first['source_reports'][0]
    report = json.loads(objects[direct.parse_gs(report_ref['uri'])[1]])
    normalized = normalize_candidate(candidate, report, report_ref)
    assert normalized['core_bounds'] == window['core_bounds']
    assert normalized['original_mask_object']['sha256'] == candidate['artifacts']['mask']['sha256']
    assert normalized['source_sha256'] == candidate['source_object']['sha256']
    assert normalized['acquisition_id'] == 'ee_catalog:example'
    assert normalized['final_eligible'] is False and normalized['semantic_status'] == 'pending'
    assert normalized['country'] == 'India'
    second = direct.run(args)
    assert second['candidates'] == first['candidates']
    # A crash after materialization but before receipt publication must adopt
    # the successful, pinned staged report without consuming another request.
    del objects['direct-test/receipts/' + candidate['sample_id'] + '.json']
    del objects[direct.parse_gs(report_ref['uri'])[1]]
    third = direct.run(args)
    assert third['candidates'] == first['candidates']
    assert len([call for call in calls if call.get('dimensions') == [1700, 1700]]) == 1
    assert len([call for call in calls if call.get('dimensions') == [54, 54]]) == int(coarse)
    assert all('file:' not in payload.decode('utf-8', errors='ignore')
               for name, payload in objects.items() if name.endswith('.json'))
