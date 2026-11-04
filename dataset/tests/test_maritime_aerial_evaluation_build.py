import io
import json
import hashlib
import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.enums import ColorInterp
from rasterio.transform import Affine
from tests.test_maritime_aerial_source import fixture,raster
from tests.test_maritime_naip_source import naip_fixture
from scripts.maritime_aerial_evaluation import site_allocation
from scripts.maritime_aerial_evaluation_release import load_evaluation_inputs
from scripts import maritime_aerial_evaluation_build as build
from src import maritime_aerial_source,maritime_naip_source

class Store:
    def __init__(self,tmp_path):self.scratch=tmp_path;self.data={};self.original=None
    def publish(self,name,payload):
        pin=dict(uri='gs://evaluation/'+name,generation='1',bytes=len(payload),crc32c='ABCD',sha256=hashlib.sha256(payload).hexdigest());self.data[pin['uri']]=payload;return pin
    def get(self,pin):return self.data[pin['uri']]
    def cached_source(self,pin):assert pin==self.original_pin;return self.original
    def trim_cache(self):pass

def inputs(store,row):
    entries=[]
    for scope in ('preserved_baseline','expanded_training','source_development'):
        exposure=dict(sample_id=scope,source_kind='maxar',site_id=scope,acquisition_id=scope,split='train',crs='EPSG:6340',core_bounds=[100000,0,101280,1280])
        pin=store.publish(scope+'.json',json.dumps(dict(schema='aerial-evaluation-exposure-population-v1',scope=scope,complete=True,rows=[exposure])).encode())
        entries.append(dict(scope=scope,pin=pin))
    manifest=store.publish('exposure.json',json.dumps(dict(schema='aerial-evaluation-exposure-evidence-v1',complete=True,populations=entries)).encode())
    return dict(schema='aerial-evaluation-selection-v1',selection=[row],exposure_evidence_pin=manifest)

def evaluation_row(family):
    row,rgb=(fixture() if family=='noaa' else naip_fixture());row['split']='test'
    site=next(str(i) for i in range(1000) if site_allocation(str(i))=='evaluation')
    row['site_id']='aerial-'+site;row['geographic_group_id']=row['site_id'];row['source']['source']['port_AOI_matches']=[{'port_id':site}]
    return row,rgb

def test_complete_exposure_manifest_required_and_pinned(tmp_path):
    store=Store(tmp_path);row,_=evaluation_row('noaa');doc=inputs(store,row)
    rows,exposure=load_evaluation_inputs(store,doc);assert rows==[row] and len(exposure)==3
    bad=dict(doc);bad.pop('exposure_evidence_pin')
    with pytest.raises(ValueError):load_evaluation_inputs(store,bad)
    manifest=json.loads(store.get(doc['exposure_evidence_pin']));manifest['populations'].pop();bad['exposure_evidence_pin']=store.publish('missing.json',json.dumps(manifest).encode())
    with pytest.raises(ValueError,match='complete'):load_evaluation_inputs(store,bad)

@pytest.mark.parametrize('family',['noaa','naip'])
@pytest.mark.parametrize('native_only',[True,False])
def test_full_source_replay_builds_true_test_four_formats(tmp_path,family,native_only,monkeypatch):
    if native_only:monkeypatch.setattr(build,'pair_payload',lambda *args:(b'recipe-not-tested-here',{}))
    else:pytest.importorskip('torch',reason='Unchanged D1 and real loader need pinned PyTorch runtime')
    row,rgb=evaluation_row(family);source=row['source'];adapter=maritime_aerial_source if family=='noaa' else maritime_naip_source;store=Store(tmp_path)
    with raster(rgb,source) as native,MemoryFile() as original:
        with native.open() as ds:context,*_=adapter.canonical_from_native(ds,source)
        out=io.BytesIO();np.savez_compressed(out,hr_context=context,hr=context[:,16:-16,16:-16]);store.data[source['hr']['uri']]=out.getvalue();store.data[source['native_context']['uri']]=native.read()
        with original.open(driver='GTiff',width=1360,height=1360,count=4,dtype='uint8',crs=source['source_crs'],transform=Affine(*source['original_transform'])) as ds:
            ds.write(rgb,indexes=[1,2,3]);ds.write(np.full((1360,1360),255,np.uint8),4);ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha if family=='noaa' else ColorInterp.undefined)
        evidence=dict(source=source['source']);keys=('original_item_metadata_object','official_dataset_metadata_object') if family=='noaa' else ('original_item_metadata_object','original_collection_metadata_object','official_usda_policy_object','official_usgs_metadata_object')
        for k in keys:evidence[k]=store.publish(k+'.json',b'{}')
        store.data[source['source_evidence_pin']['uri']]=json.dumps(evidence).encode();store.original=original.name;store.original_pin=source['source_original_object']
        common,ctx,shared,children=build.process_evaluation_sample(store,row,20261006)
        assert common['split']=='test' and common['original_parent_pixels_replayed'] and common['separate_auxiliary_evaluation']
        assert set(children)=={'d1','d2','d3'}
        if native_only:
            with np.load(io.BytesIO(shared),allow_pickle=False) as arrays:
                assert json.loads(str(arrays['provenance']))['split']=='test' and arrays['hr'].shape==(3,512,512)
            with pytest.raises(ValueError,match='evaluation'):adapter.process_sample(store,row,20261006)
            import rasterio
            with rasterio.open(original.name,'r+') as ds:
                changed=rgb[0].copy();changed[80,80]=0;ds.write(changed,1)
            with pytest.raises(ValueError,match='original|parent'):build.process_evaluation_sample(store,row,20261006)
            row['split']='train'
            with pytest.raises(ValueError):build.process_evaluation_sample(store,row,20261006)
            return
        doc=inputs(store,row);result=build.build_evaluation(store,doc,20261006,shard_samples=1)
        assert result['status']=='AUXILIARY_EVALUATION_FORMAT_VERIFIED' and result['train_sample_count']==0
        assert set(result['manifests'])=={'mother','d1','d2','d3'}
        for name,pin in result['manifests'].items():
            manifest=json.loads(store.get(pin));assert manifest['samples'][0]['split']=='test' and manifest['shards'][0]['split']=='test'
            assert manifest['sample_count']==len(manifest['samples'])==1
            assert manifest['evaluation_format_verified'] and manifest['train_sample_count']==0
            if name=='mother':assert manifest['training_format_verified']
        with pytest.raises(ValueError,match='evaluation'):adapter.process_sample(store,row,20261006)
        row['split']='train'
        with pytest.raises(ValueError):build.process_evaluation_sample(store,row,20261006)

def test_opposite_evaluation_splits_cannot_share_geography(tmp_path):
    store=Store(tmp_path);row,_=evaluation_row('noaa');other=dict(row,sample_id='other',split='validation',geographic_group_id='other')
    doc=inputs(store,row);doc['selection'].append(other)
    with pytest.raises(ValueError,match='evaluation'):load_evaluation_inputs(store,doc)
