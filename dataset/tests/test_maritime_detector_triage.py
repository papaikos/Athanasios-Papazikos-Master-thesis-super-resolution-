import hashlib

import numpy as np
import pytest


def candidate():
    pin = dict(uri='gs://bucket/object', generation='1', bytes=1, crc32c='AAAAAA==', sha256='a' * 64)
    return dict(sample_id='s', schema='maxar-mixed-native-v1', native_context=pin, hr=pin,
        product_spacing_m=0.5, area_phase_m=[0, 0], physical_core_bounds=[0, 0, 1280, 1280],
        physical_context_bounds=[-40, -40, 1320, 1320],
        source=dict(native_gsd_m=0.5, license='CC-BY-NC-4.0'))


def test_tile_edges_cover_context_and_preserve_full_image_coordinates():
    from scripts.maritime_detector_triage import infer_tiles, tile_origins
    assert tile_origins(1360) == [0, 336]
    image = np.zeros((1360, 1360, 3), np.uint8)
    def predict(tile):
        assert tile.shape == (1024, 1024, 3)
        return [dict(class_id=1, confidence=0.9,
                     polygon=[[1, 2], [101, 2], [101, 12], [1, 12]])]
    records, evidence = infer_tiles(image, predict)
    assert records[-1]['polygon'][0] == [337, 338]
    assert len(records) == evidence['tile_count'] == 4
    assert evidence['tile_windows'][-1] == [336, 336, 1024, 1024]


def test_native_and_one_metre_core_coordinates_account_for_fractional_halo():
    from scripts.maritime_detector_triage import prepare_triage_image
    row = candidate()
    row['area_phase_m'] = [0.25, 0.25]
    rgb = np.full((3, 2721, 2721), 100, np.uint8)
    native, geometry = prepare_triage_image(rgb, row, mode='native')
    assert native.shape == (2721, 2721, 3)
    assert geometry['core_bounds_pixels'] == [80.5, 80.5, 2640.5, 2640.5]
    reduced, geometry = prepare_triage_image(rgb, row, mode='one-metre')
    assert reduced.shape == (1360, 1360, 3)
    assert geometry['core_bounds_pixels'] == [40, 40, 1320, 1320]
    assert geometry['pixel_spacing_m'] == 1
    assert (reduced == 100).all()
    assert (rgb == 100).all()


def test_rgb_is_passed_to_detector_as_contiguous_bgr():
    from scripts.maritime_detector_triage import prepare_triage_image
    rgb = np.empty((3, 2720, 2720), np.uint8)
    rgb[0], rgb[1], rgb[2] = 10, 20, 30
    bgr, _ = prepare_triage_image(rgb, candidate(), mode='native')
    assert bgr.flags.c_contiguous
    assert bgr[0, 0].tolist() == [30, 20, 10]


def test_manifest_license_provenance_and_unsupported_skysat_fail_closed():
    from scripts.maritime_detector_triage import candidate_rows
    row = candidate()
    selected = candidate_rows(dict(candidates=[row]), 1)
    assert selected[0]['source']['license'] == 'CC-BY-NC-4.0'
    with pytest.raises(ValueError, match='Maxar'):
        candidate_rows(dict(candidates=[dict(sample_id='sky', source_object=row['native_context'])]), 1)
    with pytest.raises(ValueError, match='Duplicate'):
        candidate_rows(dict(candidates=[row, row]), 2)


def test_weight_sha256_and_size_are_verified(tmp_path):
    from scripts.maritime_detector_triage import verify_weights
    weights = tmp_path / 'model.pt'
    payload = bytes(23267238)
    weights.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert verify_weights(weights, expected)['sha256'] == expected
    with pytest.raises(ValueError, match='SHA'):
        verify_weights(weights, 'f' * 64)
    weights.write_bytes(b'x')
    with pytest.raises(ValueError, match='size'):
        verify_weights(weights, expected)


def test_cross_tile_duplicate_is_removed_after_coordinate_mapping():
    from scripts.maritime_detector_triage import infer_tiles
    from src.maritime_quality import summarize_ship_detections
    calls = iter([0, 336])
    def predict(tile):
        offset = next(calls)
        return [dict(class_id=1, confidence=0.9,
            polygon=[[400-offset, 50], [500-offset, 50], [500-offset, 60], [400-offset, 60]])]
    predictions, _ = infer_tiles(np.zeros((1024, 1360, 3), np.uint8), predict)
    result = summarize_ship_detections(predictions, pixel_spacing_m=1,
        core_bounds=[40, 40, 1320, 984])
    assert result['duplicates_removed'] == 1
    assert result['ship_counts']['medium'] == 1


def test_native_triage_rejects_missing_context_contributors():
    from scripts.maritime_detector_triage import prepare_triage_image
    with pytest.raises(ValueError, match='dimensions'):
        prepare_triage_image(np.zeros((3, 2719, 2720), np.uint8), candidate(), mode='native')


def test_cloud_guard_runs_before_input_access(monkeypatch):
    from scripts.maritime_detector_triage import run
    from scripts import maritime_cloud_inventory
    def deny_local():
        raise RuntimeError('GCE required before input access')
    monkeypatch.setattr(maritime_cloud_inventory, 'require_cloud_vm', deny_local)
    with pytest.raises(RuntimeError, match='GCE required'):
        run(object())

def test_raw_skysat_selection_is_normalized_with_pinned_source():
    from scripts.maritime_detector_triage import candidate_rows
    pin = candidate()['native_context']
    source = dict(source_object=pin, mask=dict(provenance='original_embedded_dataset_mask'),
                  core_bounds=[0, 0, 1280, 1280], context_transform=[.8,0,-40,0,-.8,1320])
    row = dict(sample_id='sky', source_kind='skysat_raw_window', source=source)
    normalized = candidate_rows(dict(candidates=[row]), 1)[0]
    assert normalized['sample_id'] == 'sky'
    assert normalized['_triage_source_kind'] == 'skysat_raw_window'
    assert source.get('sample_id') is None


def test_earthview_candidate_manifest_support_and_unknown_kind_rejection():
    from scripts.maritime_detector_triage import candidate_rows
    row = dict(candidate(), schema='earthview-candidate-v1', source_kind='earthview')
    assert candidate_rows(dict(candidates=[row]), 1)[0]['_triage_source_kind'] == 'earthview'
    with pytest.raises(ValueError, match='Unsupported'):
        candidate_rows(dict(selection=[dict(sample_id='x', source_kind='unknown', source={})]), 1)


def test_skysat_one_metre_derivative_uses_original_context():
    from scripts.maritime_detector_triage import prepare_triage_image
    rgb = np.full((3, 1700, 1700), 123, np.uint8)
    source = dict(_triage_source_kind='skysat_raw_window', core_bounds=[0,0,1280,1280])
    bgr, geometry = prepare_triage_image(rgb, source)
    assert bgr.shape == (1360,1360,3)
    assert (bgr == 123).all()
    assert geometry['core_bounds_pixels'] == [40,40,1320,1320]
    assert geometry['native_product_spacing_m'] == .8


def test_worker_limit_is_validated():
    from scripts.maritime_detector_triage import validate_workers
    assert validate_workers(8) == 8
    for invalid in (0,9,True,1.5):
        with pytest.raises(ValueError, match='workers'):
            validate_workers(invalid)

def test_skysat_encoded_affine_rounding_preserves_nominal_recipe_basis():
    from scripts.maritime_detector_triage import prepare_triage_image
    row = dict(_triage_source_kind='skysat_raw_window',core_bounds=[0,0,1279.99995,1280.00005])
    _, geometry = prepare_triage_image(np.zeros((3,1700,1700),np.uint8),row,mode='native')
    assert geometry['core_bounds_pixels'] == [50,50,1650,1650]
    assert 'nominal' in geometry['coordinate_basis']


def test_earthview_one_metre_original_display_context_is_not_upsampled():
    from scripts.maritime_detector_triage import prepare_triage_image
    row = dict(candidate(),_triage_source_kind='earthview',product_spacing_m=1)
    rgb = np.zeros((3,1360,1360),np.uint8)
    rgb[0,:,::2] = 255
    bgr, geometry = prepare_triage_image(rgb,row)
    np.testing.assert_array_equal(bgr[:,:,2],rgb[0])
    assert geometry['pixel_spacing_m'] == 1


def test_skysat_separate_mask_requires_full_pin():
    from scripts.maritime_detector_triage import candidate_rows
    row = dict(sample_id='sky',source_kind='skysat_raw_window',source=dict(
        source_object=candidate()['native_context'],mask=dict(object=dict(uri='gs://missing'))))
    with pytest.raises(ValueError,match='pin'):
        candidate_rows(dict(selection=[row]),1)

def test_earthview_loader_reproduces_original_assembly_before_display(monkeypatch):
    import sys
    from types import SimpleNamespace
    import rasterio.io
    from scripts.maritime_detector_triage import load_original_context
    calls=[]
    original=np.full((3,1360,1360),1000,np.uint16)
    context=np.full((3,544,544),.25,np.float32)
    class Dataset:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self): return original
    class Memory:
        def __init__(self,payload): assert payload == b'native'
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def open(self): return Dataset()
    class Store:
        def get(self,pin): return pin
        def trim_cache(self): calls.append('trim')
    monkeypatch.setattr(rasterio.io,'MemoryFile',Memory)
    monkeypatch.setitem(sys.modules,'scripts.maritime_ultimate_release',SimpleNamespace(read_maxar=None,read_skysat=None))
    monkeypatch.setitem(sys.modules,'src.maritime_recipes',SimpleNamespace(canonical_hr=None))
    def verify(store,row):
        calls.append('original_assembly'); return context
    def canonical(ds,row,stored):
        calls.append('canonical_native'); assert stored == b'hr'; return context,1,'a','b'
    monkeypatch.setitem(sys.modules,'src.maritime_earthview_source',SimpleNamespace(
        verify_original_assembly=verify,canonical_from_native=canonical))
    def mapping(rgb,saved):
        calls.append('saved_mapping'); assert saved == {'version':'frozen'}
        np.testing.assert_array_equal(rgb,original)
        return np.full(rgb.shape,64,np.uint8)
    monkeypatch.setitem(sys.modules,'scripts.maritime_earthview_intake',SimpleNamespace(apply_toa_mapping=mapping))
    row=dict(_triage_source_kind='earthview',native_context=b'native',hr=b'hr',
             input_product='TOA',rgb_mapping={'version':'frozen'})
    rgb,returned=load_original_context(Store(),row)
    assert calls == ['original_assembly','canonical_native','saved_mapping','trim']
    assert rgb.dtype == np.uint8 and (rgb == 64).all()
    assert returned is context

def test_original_object_order_keeps_same_source_windows_adjacent_and_default_stable():
    from scripts.maritime_detector_triage import candidate_rows
    def sky(identity,uri):
        reference=dict(candidate()['native_context'],uri=uri)
        return dict(sample_id=identity,source_kind='skysat_raw_window',source=dict(source_object=reference,mask={}))
    document=dict(candidates=[sky('a','gs://bucket/z.tif'),sky('b','gs://bucket/a.tif'),sky('c','gs://bucket/z.tif')])
    assert [row['sample_id'] for row in candidate_rows(document,3)]==['a','b','c']
    assert [row['sample_id'] for row in candidate_rows(document,3,source_order='original-object')]==['b','a','c']
    assert [row['sample_id'] for row in candidate_rows(document,2,source_order='original-object')]==['b','a']
    with pytest.raises(ValueError,match='order'):
        candidate_rows(document,3,source_order='unknown')

@pytest.mark.parametrize('workers,source_order,expected',[
    (1,'identity',{}),(2,'original-object',{'parallel_source_workers':2,'source_order':'original-object'}),
    (2,'original-object',{'parallel_source_workers':2,'source_order':'original-object','task_grouping':'original-object','worker_source_cache_target_bytes':2000000000,'aggregate_source_cache_target_bytes':4000000000}),
])
def test_configuration_only_records_nondefault_execution_options(tmp_path,monkeypatch,workers,source_order,expected):
    import json
    from types import SimpleNamespace
    from scripts import maritime_detector_triage as triage,maritime_cloud_inventory as cloud
    from google.cloud import storage
    manifest=tmp_path/'candidates.json';manifest.write_text(json.dumps(dict(candidates=[candidate()])))
    captured=[]
    class StopAfterConfiguration(Exception):pass
    class Bucket:
        def blob(self,name):return SimpleNamespace(name=name)
    monkeypatch.setattr(storage,'Client',lambda **kwargs:SimpleNamespace(bucket=lambda name:Bucket()))
    monkeypatch.setattr(cloud,'require_cloud_vm',lambda:'synthetic-instance')
    monkeypatch.setattr(triage,'verify_weights',lambda *args:dict(sha256='a'*64))
    def publish(blob,payload):
        captured.append(json.loads(payload));raise StopAfterConfiguration
    monkeypatch.setattr(cloud,'create_only',publish)
    args=SimpleNamespace(manifest=str(manifest),weights='synthetic',weights_sha256='a'*64,
        project='synthetic',confidence=.25,iou_threshold=.5,scratch=tmp_path/'cache',limit=1,
        workers=workers,source_order=source_order,task_grouping=expected.get('task_grouping','candidate'),triage_mode='one-metre',output_prefix='gs://synthetic/triage')
    with pytest.raises(StopAfterConfiguration):triage.run(args)
    config=captured[0]
    for key in ('parallel_source_workers','source_order','task_grouping','worker_source_cache_target_bytes','aggregate_source_cache_target_bytes'):
        if key in expected:assert config[key]==expected[key]
        else:assert key not in config

def test_explicit_gpu_device_validation_and_cpu_default():
    from scripts.maritime_detector_triage import validate_device
    assert validate_device('cpu',2)=='cpu'
    assert validate_device('cuda:0',1)=='cuda:0'
    for device,workers in [('cuda:1',1),('cuda:0',2),('auto',1)]:
        with pytest.raises(ValueError,match='device|GPU'):
            validate_device(device,workers)

def test_earthview_pinned_context_requires_source_validator_and_skips_reassembly(monkeypatch):
    import sys
    from types import SimpleNamespace
    import rasterio.io
    from scripts.maritime_detector_triage import load_original_context
    calls=[]
    class Dataset:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return np.zeros((3,1360,1360),np.uint8)
    class Memory:
        def __init__(self,payload):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def open(self):return Dataset()
    class Store:
        def get(self,pin):return pin
        def trim_cache(self):pass
    monkeypatch.setattr(rasterio.io,'MemoryFile',Memory)
    monkeypatch.setitem(sys.modules,'scripts.maritime_ultimate_release',SimpleNamespace(read_maxar=None,read_skysat=None))
    monkeypatch.setitem(sys.modules,'src.maritime_recipes',SimpleNamespace(canonical_hr=None))
    def validate(row):calls.append('fully_pinned_source_metadata')
    def assembly(*args):pytest.fail('Original files must not be downloaded/reassembled in pinned-context mode')
    def canonical(*args):calls.append('native_context_hash_mask_mapping_hr');return np.zeros((3,544,544),np.float32),1,'a','b'
    monkeypatch.setitem(sys.modules,'src.maritime_earthview_source',SimpleNamespace(
        verify_original_assembly=assembly,canonical_from_native=canonical,validate_source=validate))
    monkeypatch.setitem(sys.modules,'scripts.maritime_earthview_intake',SimpleNamespace(apply_toa_mapping=None))
    row=dict(_triage_source_kind='earthview',native_context=b'original',hr=b'hr',input_product='VISUAL')
    rgb,context=load_original_context(Store(),row,earthview_verification='pinned-context')
    assert calls==['fully_pinned_source_metadata','native_context_hash_mask_mapping_hr']
    assert rgb.shape==(3,1360,1360) and context.shape==(3,544,544)


def test_explicit_cpu_thread_count_is_bounded():
    from scripts.maritime_detector_triage import validate_cpu_threads
    for count in (1,2,4,8):assert validate_cpu_threads(count)==count
    for count in (True,0,9,1.5):
        with pytest.raises(ValueError,match="threads"):validate_cpu_threads(count)


def test_requested_parallel_cpu_threads_survive_predictor_device_setup(monkeypatch,tmp_path):
    import sys
    from types import SimpleNamespace
    from scripts.maritime_detector_triage import initialize_detector
    state={'threads':0}
    class Device:
        def __init__(self,name):self.type=name
    def predict(**kwargs):
        if isinstance(kwargs['device'],str):state['threads']=8
        return []
    torch=SimpleNamespace(__version__='synthetic',device=Device,set_num_threads=lambda n:state.update(threads=n),
        get_num_threads=lambda:state['threads'],set_num_interop_threads=lambda n:None,get_num_interop_threads=lambda:1)
    model=SimpleNamespace(names={1:'ship',7:'harbor'},predict=predict)
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(__version__='8.3.0',YOLO=lambda *args,**kwargs:model))
    predictor=initialize_detector(tmp_path/'synthetic.pt',.25,cpu_threads=4)
    predictor(np.zeros((8,8,3),np.uint8))
    assert state['threads']==4 and predictor.runtime['effective_inference_cpu_threads']==4


def test_original_task_groups_preserve_rows_and_default_tasks():
    from scripts.maritime_detector_triage import candidate_task_groups
    def sky(i,uri):return dict(sample_id=i,_triage_source_kind='skysat_raw_window',source_object=dict(uri=uri,generation='7',sha256='a'*64,bytes=10,crc32c='x'))
    rows=[sky('a','gs://b/a'),sky('b','gs://b/a'),sky('c','gs://b/b'),dict(sample_id='d')]
    assert candidate_task_groups(rows)==[[row] for row in rows]
    assert candidate_task_groups(rows,'original-object','original-object')==[rows[:2],rows[2:3],rows[3:]]
    with pytest.raises(ValueError,match='order'):candidate_task_groups(rows,'original-object','identity')
    bad=[rows[0],dict(rows[1],source_object=dict(rows[1]['source_object'],sha256='b'*64))]
    with pytest.raises(ValueError,match='pins'):candidate_task_groups(bad,'original-object','original-object')
