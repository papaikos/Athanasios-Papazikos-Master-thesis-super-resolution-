import copy
import io
import json
import tarfile
import sys
import types

import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.transform import array_bounds

from scripts import maritime_ultimate_release as release


PIN = dict(uri='gs://test/native.tif', generation='1', bytes=12,
           crc32c='AAAAAA==', sha256='a'*64)


def selection():
    row = dict(sample_id='one', source_kind='skysat', site_id='port',
               geographic_group_id='group', split='train', acquisition_id='capture',
               country='Greece', source=dict(mother=PIN.copy(), native_gsd_m=.8,
                   crs='EPSG:32634', core_bounds=[500000, 4000000, 501280, 4001280],
                   context_transform=[.8,0,499960,0,-.8,4001320],
                   mask=dict(provenance='original_embedded_dataset_mask')))
    return dict(schema='ultimate-selection-v1', review_policy={'basis':'user_inspected_baseline'},
                splits={'assignment':{'group':'train'}}, gates={}, selection=[row])


def test_frozen_identity_gsd_and_split_are_enforced():
    rows, gaps = release.validate_selection(selection())
    assert rows[0]['sample_id'] == 'one'
    assert 'native_60gb_target' in gaps
    for mutate, match in [
        (lambda d:d['selection'].append(copy.deepcopy(d['selection'][0])), 'Duplicate'),
        (lambda d:d['selection'][0]['source'].update(native_gsd_m=1.07), 'GSD'),
        (lambda d:d['selection'][0].update(split='test'), 'Frozen'),
        (lambda d:d['selection'][0]['source']['mother'].pop('sha256'), 'pin'),
    ]:
        document=selection();mutate(document)
        with pytest.raises(ValueError,match=match):release.validate_selection(document)


def test_same_acquisition_overlap_is_rejected():
    document=selection();other=copy.deepcopy(document['selection'][0]);other['sample_id']='two'
    document['selection'].append(other)
    with pytest.raises(ValueError,match='Overlapping'):release.validate_selection(document)


def test_raw_window_geometry_cannot_count_halo():
    document=selection();row=document['selection'][0];row['source_kind']='skysat_raw_window'
    source=row['source'];source['source_object']=source.pop('mother')
    source.update(core_window=[50,50,1600,1600],context_window=[0,0,1700,1700])
    assert release.validate_selection(document)[0][0]['source_kind']=='skysat_raw_window'
    source['core_window'][2]=1700
    with pytest.raises(ValueError,match='1600'):release.validate_selection(document)


def test_native_black_is_valid_and_invalid_mask_rejected():
    row=selection()['selection'][0]
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',height=1700,width=1700,count=3,dtype='uint8',
                         crs=row['source']['crs'],transform=Affine(*row['source']['context_transform'])) as ds:
            ds.write(np.zeros((3,1700,1700),np.uint8))
            ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
            ds.write_mask(np.full((1700,1700),255,np.uint8))
            rgb,valid=release.read_skysat(ds,row['source'])
            assert valid.all() and rgb.max()==0
            invalid=np.full((1700,1700),255,np.uint8);invalid[0,0]=0;ds.write_mask(invalid)
            with pytest.raises(ValueError,match='support'):release.read_skysat(ds,row['source'])


def test_serialized_pair_loads_real_torch_and_recomputes(tmp_path):
    context=np.full((3,544,544),.25,np.float32)
    row=selection()['selection'][0]
    common=release.common_sample(row,context,7680000,'b'*64,'c'*64)
    for recipe in ('d1','d2','d3'):
        payload,metadata=release.pair_payload(context,common,recipe,20261006)
        path=tmp_path/(recipe+'.tar')
        with tarfile.open(path,'w') as archive:release.tar_member(archive,'one.npz',payload)
        report=release.verify_shard(path,recipe,{'one':context},20261006)
        assert report['verified_samples']==1
        assert report['numeric_recomputed_samples']==1
        assert metadata['hr_sha256']==common['hr_sha256']


def test_tar_and_npz_are_deterministic():
    context=np.zeros((3,544,544),np.float32)
    common=release.common_sample(selection()['selection'][0],context,7680000,'b'*64,'c'*64)
    assert release.pair_payload(context,common,'d3',7)[0]==release.pair_payload(context,common,'d3',7)[0]


def test_no_false_full_ready_from_training_format():
    report=release.release_state(True,['native_60gb_target'])
    assert report['training_format_verified']
    assert report['status']=='PROVISIONAL_TRAINING_READY'
    assert not report['full_target_passed']


def test_maxar_native_geometry_recomputed_not_resized_or_halo_counted():
    rgb=np.zeros((3,1360,1360),np.uint8);rgb[:,40:1320,40:1320]=64
    geometry=dict(product_spacing_m=1.,area_phase_m=[0.,0.],
        source={'epsg':32634,'event_license':'CC-BY-NC-4.0','native_gsd_m':.8},
        original_transform=[1,0,500000,0,-1,4001360],source_pixel_window=[0,0,1360,1360],
        physical_core_bounds=[500040,4000040,501320,4001320],
        physical_context_bounds=[500000,4000000,501360,4001360],
        owned_core_pixel_window=[40,40,1280,1280],native_gsd_m=.8,
        useful_native_rgb_bytes=1280*1280*3,
        decoded_native_context_sha256=release.digest_bytes(rgb.tobytes()),
        decoded_useful_core_sha256=release.digest_bytes(rgb[:,40:1320,40:1320].tobytes()))
    valid=np.ones((1360,1360),bool)
    context=release.area_hr(rgb,valid,geometry)
    stored=release.deterministic_npz(dict(hr_context=context,hr=context[:,16:-16,16:-16]))
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',height=1360,width=1360,count=3,dtype='uint8',
            crs='EPSG:32634',transform=Affine(*geometry['original_transform'])) as ds:
            ds.write(rgb);ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
            ds.write_mask(np.full((1360,1360),255,np.uint8))
            result,count,_,_=release.read_maxar(ds,geometry,stored)
            assert count==1280*1280*3 and np.array_equal(result,context)
            broken=copy.deepcopy(geometry);broken['owned_core_pixel_window'][2]+=1
            with pytest.raises(ValueError,match='accounting'):release.read_maxar(ds,broken,stored)
            bad=release.deterministic_npz(dict(hr_context=context+.001,hr=context[:,16:-16,16:-16]))
            with pytest.raises(ValueError,match='disagrees'):release.read_maxar(ds,geometry,bad)


def test_raw_reads_exact_original_window_and_separate_mask():
    row=selection()['selection'][0];source=row['source']
    source.update(core_window=[100,100,1600,1600],context_window=[50,50,1700,1700])
    transform=Affine(*source['context_transform'])*Affine.translation(-50,-50)
    with MemoryFile() as rgb_file,MemoryFile() as mask_file:
        with rgb_file.open(driver='GTiff',height=1800,width=1800,count=3,dtype='uint8',crs=source['crs'],transform=transform) as ds,mask_file.open(
            driver='GTiff',height=1800,width=1800,count=1,dtype='uint8',crs=source['crs'],transform=transform) as mask:
            ds.write(np.full((3,1800,1800),31,np.uint8));ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
            values=np.full((1800,1800),255,np.uint8);values[0,0]=0;mask.write(values,1)
            rgb,valid=release.read_skysat(ds,source,mask,raw=True)
            assert rgb.shape==(3,1700,1700) and valid.all() and rgb.min()==31
            values[50,50]=0;mask.write(values,1)
            with pytest.raises(ValueError,match='support'):release.read_skysat(ds,source,mask,raw=True)


def test_complete_synthetic_release_publishes_four_consistent_manifests(tmp_path,monkeypatch):
    document=selection();source=document['selection'][0]['source']
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',height=1700,width=1700,count=3,dtype='uint8',
            crs=source['crs'],transform=Affine(*source['context_transform'])) as ds:
            ds.write(np.full((3,1700,1700),45,np.uint8))
            ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
            ds.write_mask(np.full((1700,1700),255,np.uint8))
        native=memory.read()
    source['mother'].update(bytes=len(native),sha256=release.digest_bytes(native))
    selection_path=tmp_path/'selection.json';selection_path.write_text(json.dumps(document))
    written={}
    class Store:
        scratch=tmp_path
        def get(self,ref):
            assert ref['sha256']==release.digest_bytes(native)
            return native
        def publish(self,name,data):
            if name in written:assert written[name]==data
            written[name]=data
            return dict(uri='gs://test/release/'+name,generation='1',bytes=len(data),
                        sha256=release.digest_bytes(data),crc32c='AAAAAA==')
        def existing(self,name):return None
        def trim_cache(self):pass
    storage=types.ModuleType('google.cloud.storage');storage.Client=lambda **kwargs:object()
    cloud=types.ModuleType('google.cloud');cloud.storage=storage
    monkeypatch.setitem(sys.modules,'google.cloud',cloud)
    monkeypatch.setitem(sys.modules,'google.cloud.storage',storage)
    monkeypatch.setattr(release,'require_cloud_vm',lambda:'9001')
    monkeypatch.setattr(release,'CloudStore',lambda *args:Store())
    result=release.run(types.SimpleNamespace(selection=selection_path,output_prefix='gs://test/release',
        scratch=tmp_path,image_id='sha256:'+'d'*64,code_sha256='e'*64,
        release_seed=20261006,shard_samples=32,workers=2))
    assert result['useful_native_rgb_bytes']==7680000
    assert result['status']=='PROVISIONAL_TRAINING_READY' and not result['full_target_passed']
    assert set(result['manifests'])=={'mother','d1','d2','d3'}
    manifests=[json.loads(written[name+'/manifest.json']) for name in ('mother','d1','d2','d3')]
    assert all(m['samples'][0]['hr_sha256']==manifests[0]['samples'][0]['hr_sha256'] for m in manifests)
    assert all(m['shards'][0]['split']=='train' for m in manifests)
    assert manifests[0]['samples'][0]['native_context']==source['mother']
    assert result['verification']['numeric_recomputed_samples_per_recipe']==1


def test_original_float32_point8_coordinate_tags_preserve_native_pixels():
    document=selection();row=document['selection'][0];row['source_kind']='skysat_raw_window'
    source=row['source'];source['source_object']=source.pop('mother')
    source.update(core_window=[50,50,1600,1600],context_window=[0,0,1700,1700])
    spacing=float(np.float32(.8))
    transform=Affine(spacing,0,499960,0,-spacing,4001320)
    source['context_transform']=list(transform)[:6]
    source['core_bounds']=list(array_bounds(1600,1600,transform*Affine.translation(50,50)))
    rows,gaps=release.validate_selection(document)
    assert rows[0]['source']['context_transform'][0]==spacing
    assert source['core_bounds'][2]-source['core_bounds'][0]-1280==pytest.approx(.0000190734863)
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',height=1700,width=1700,count=3,dtype='uint8',
            crs=source['crs'],transform=transform) as ds:
            native=np.full((3,1700,1700),47,np.uint8);native[:,50,50]=0
            ds.write(native);ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
            ds.write_mask(np.full((1700,1700),255,np.uint8))
            decoded,valid=release.read_skysat(ds,source,raw=True)
            assert np.array_equal(decoded,native) and valid.all()
            context,_=release.canonical_hr(decoded,valid)
            common=release.common_sample(row,context,7680000,'b'*64,'c'*64)
            assert common['coordinate_encoding']['encoded_native_spacing_m']==[spacing,spacing]
            assert common['coordinate_encoding']['native_context_transform']==list(transform)[:6]
            assert common['hr_transform'][0]==2.5
            assert common['useful_native_rgb_bytes']==7680000


def test_quantized_core_edges_are_tolerated_but_material_overlap_rejected():
    document=selection();row=document['selection'][0]
    row['source']['core_bounds'][2]+=.0000190734863
    row['source']['core_bounds'][3]+=.0000190734863
    other=copy.deepcopy(row);other['sample_id']='two'
    other['source']['core_bounds']=[501280,4000000,502560.0000190735,4001280.0000190735]
    document['selection'].append(other)
    assert len(release.validate_selection(document)[0])==2
    other['source']['core_bounds'][0]-=.4;other['source']['core_bounds'][2]-=.4
    with pytest.raises(ValueError,match='Overlapping'):release.validate_selection(document)
