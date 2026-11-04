import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

from scripts import maritime_release_build as build


def selection():
    return dict(selection=[dict(canonical_core_id='sample-a', candidate_id='a',
        review_status='accepted_native_review', review_decision=dict(decision='accept',
        reviewer='human', native_scale_reviewed=True, quality_status='pass',
        duplicate_status='clear', support_status='verified', primary_category='coastline',
        vessel_presence='absent', small_craft_presence='absent', hard_negative=False),
        site_id='site', geographic_group_id='group', country='GR', acquisition_id='asset',
        split='train', crs='EPSG:32634', core_bounds=[40,40,1320,1320],
        core_transform=[.8,0,40,0,-.8,1320], context_transform=[.8,0,0,0,-.8,1360],
        mask=dict(provenance='producer_original_min_rgb_mask'), source_sha256='a'*64,
        native_preview=dict(uri='gs://input/preview'), mother=dict(uri='gs://input/rgb'))],
        gates=dict(metadata_selection_passed=False, missing=['size']),
        splits=dict(assignment={'group':'train'}))


def test_positive_review_and_strict_gate():
    doc=selection()
    with pytest.raises(ValueError, match='provisional'):
        build.validate_selection(doc, False)
    rows, limitations=build.validate_selection(doc, True)
    assert len(rows)==1 and limitations
    doc['selection'][0]['review_decision']['support_status']='pending'
    with pytest.raises(ValueError, match='review'):
        build.validate_selection(doc, True)
    doc['selection']=[]
    with pytest.raises(ValueError, match='positive'):
        build.validate_selection(doc, True)


def test_strict_requires_independent_pinned_audits():
    doc=selection(); doc['gates']['metadata_selection_passed']=True
    with pytest.raises(ValueError, match='provisional'):
        build.validate_selection(doc, False)
    pin=dict(uri='gs://audit/object',generation='1',bytes=1,crc32c='crc',sha256='a'*64)
    doc['independent_audits']={name:dict(passed=True,independent=True,reviewer='auditor',
        evidence=pin) for name in ('numeric','source','size')}
    assert build.validate_selection(doc,True)[1]==['size','unique_native_core_size_target']


def test_source_bounds_and_duplicate_rejection():
    doc=selection(); doc['selection'].append(copy.deepcopy(doc['selection'][0]))
    with pytest.raises(ValueError, match='Duplicate'):
        build.validate_selection(doc,True)
    doc=selection(); doc['selection'][0]['core_bounds'][0]=41
    with pytest.raises(ValueError, match='bounds'):
        build.validate_selection(doc,True)


def test_native_mask_reject_and_valid_black(tmp_path):
    row=selection()['selection'][0]
    rgb=tmp_path/'rgb.tif'; mask=tmp_path/'mask.tif'
    profile=dict(driver='GTiff',height=1700,width=1700,transform=Affine(*row['context_transform']),
        crs=row['crs'],dtype='uint8')
    with rasterio.open(rgb,'w',count=3,photometric='RGB',nodata=0,**profile) as ds:
        ds.write(np.zeros((3,1700,1700),dtype='uint8'))
    values=np.full((1,1700,1700),255,dtype='uint8')
    with rasterio.open(mask,'w',count=1,**profile) as ds: ds.write(values)
    with rasterio.open(rgb) as ds, rasterio.open(mask) as ms:
        array, support=build.read_native(ds,row,ms)
        assert not array.any() and support.all()
    values[0,1,1]=0
    with rasterio.open(mask,'w',count=1,**profile) as ds: ds.write(values)
    with rasterio.open(rgb) as ds, rasterio.open(mask) as ms:
        with pytest.raises(ValueError,match='support'): build.read_native(ds,row,ms)


def test_cross_children_identity_mismatch():
    base=dict(sample_id='a',hr_sha256='h',context_sha256='c',site_id='s',
        geographic_group_id='g',split='train',core_bounds=[0,0,1280,1280],
        crs='EPSG:32634',hr_transform=[2.5,0,0,0,-2.5,1280])
    mother=dict(samples=[base])
    children={name:dict(recipe=name,samples=[copy.deepcopy(base)],state='COMPLETE')
        for name in ('d1','d2','d3')}
    build.validate_children(mother,children)
    children['d3']['samples'][0]['hr_sha256']='different'
    with pytest.raises(ValueError,match='Common'): build.validate_children(mother,children)


def test_stratified_recompute_includes_small_strata_and_pilot():
    rows=[dict(sample_id=str(i),split='train',primary_category='coastline') for i in range(40)]
    rows.append(dict(sample_id='holdout',split='test',primary_category='open_water'))
    chosen=build.recompute_ids(rows)
    assert 'holdout' in chosen and len(chosen)==3
    assert len(build.recompute_ids(rows,pilot=True))==41


class Blob:
    def __init__(self):
        self.data=None; self.generation=1; self.size=0; self.crc32c='crc'; self.uploads=0
    def exists(self): return self.data is not None
    def reload(self): pass
    def upload_from_string(self,data,**kwargs):
        assert kwargs==dict(if_generation_match=0,checksum='crc32c')
        assert not self.exists()
        self.data=data; self.size=len(data); self.uploads+=1
    def download_as_bytes(self,**kwargs):
        assert kwargs==dict(if_generation_match=int(self.generation),checksum='crc32c')
        return self.data


@pytest.fixture
def release_fixture(tmp_path,monkeypatch):
    import sys
    from types import ModuleType
    pytest.importorskip('pyarrow'); pytest.importorskip('torch')
    objects={}
    def blob(bucket,name,generation=None):
        obj=objects.setdefault((bucket,name),Blob())
        # Real storage may return missing metadata for a requested generation;
        # verify_output must reject either mismatch before processing bytes.
        return obj
    client=SimpleNamespace(bucket=lambda bucket:SimpleNamespace(blob=lambda name,generation=None:blob(bucket,name,generation)))
    client.list_blobs=lambda bucket,prefix:[SimpleNamespace(name=name) for (owner,name),obj in objects.items()
        if owner==bucket and name.startswith(prefix) and obj.exists()]
    cloud=ModuleType('google.cloud'); storage=ModuleType('google.cloud.storage'); storage.Client=lambda:client
    monkeypatch.setitem(sys.modules,'google.cloud',cloud)
    monkeypatch.setitem(sys.modules,'google.cloud.storage',storage)
    monkeypatch.setattr(build,'require_cloud_vm',lambda:'123')
    doc=selection(); row=doc['selection'][0]
    store=build.Store(client,'gs://input/base')
    profile=dict(driver='GTiff',height=1700,width=1700,crs=row['crs'],
        transform=Affine(*row['context_transform']),dtype='uint8',compress='DEFLATE')
    rgb=tmp_path/'rgb.tif'; mask=tmp_path/'mask.tif'
    with rasterio.open(rgb,'w',count=3,photometric='RGB',nodata=0,**profile) as ds:
        array=np.zeros((3,1700,1700),np.uint8); array[:,300:900,300:900]=75; ds.write(array)
    with rasterio.open(mask,'w',count=1,**profile) as ds: ds.write(np.full((1,1700,1700),255,np.uint8))
    row['mother']=store.put('rgb.tif',rgb.read_bytes())
    row['mask'].update(kind='separate_validity',object=store.put('mask.tif',mask.read_bytes()))
    path=tmp_path/'selection.json'; path.write_text(json.dumps(doc))
    args=SimpleNamespace(selection=str(path),output_prefix='gs://output/build',scratch=str(tmp_path/'scratch'),
        container_image_id='image@sha256:'+'1'*64,phase='mother',provisional=True,pilot=True)
    return args,objects,client


def test_four_release_materialization_and_verified_resume(release_fixture):
    args,objects,client=release_fixture
    result=build.run(args)
    assert result['state']=='COMPLETE'
    uploads=sum(obj.uploads for obj in objects.values())
    assert build.run(args)==result
    assert sum(obj.uploads for obj in objects.values())==uploads
    for phase in ('d1','d2','d3','finalize'):
        args.phase=phase; result=build.run(args)
    assert result['status']=='PROVISIONAL_READY' and not result['full_training_ready']
    assert result['common_sample_count']==1
    assert result['no_active_or_uncertain_build_tasks'] is False
    assert 'terminal_supervisor_evidence_unverified' in result['limitations']
    assert ('output','build/release.json') in objects
    assert not any('MOTHER_READY' in name or 'DATASET_READY' in name for _,name in objects)
    for name in build.CHILDREN.values():
        manifest=json.loads(objects['output','build/'+name+'/manifest.json'].data)
        assert manifest['loader_reports'][0]['verified_samples']==1
        assert manifest['recomputed_sample_ids']==['sample-a']


def test_resumed_object_generation_mismatch_fails(release_fixture):
    args,objects,_=release_fixture
    build.run(args)
    objects['output','build/shared/sample-a.npz'].generation=2
    with pytest.raises(ValueError,match='metadata mismatch'): build.run(args)


def test_finalize_fails_missing_phase_and_configuration_change(release_fixture):
    args,_,_=release_fixture
    build.run(args)
    args.phase='finalize'
    with pytest.raises(ValueError,match='Missing completed'): build.run(args)
    args.phase='mother'; args.pilot=False
    with pytest.raises(ValueError,match='conflict'): build.run(args)


def test_child_resume_and_failed_upload_recovery(release_fixture,monkeypatch):
    args,objects,_=release_fixture
    build.run(args)
    args.phase='d2'
    real_put=build.Store.put
    failed=[False]
    def fail_after_pair(self,name,payload):
        if name=='control/d2/sample-a.json' and not failed[0]:
            failed[0]=True
            raise OSError('injected process loss before durable sample receipt')
        return real_put(self,name,payload)
    monkeypatch.setattr(build.Store,'put',fail_after_pair)
    with pytest.raises(OSError,match='process loss'): build.run(args)
    pair=objects['output','build/'+build.CHILDREN['d2']+'/pairs/train/sample-a.npz']
    assert pair.uploads==1
    first=build.run(args); uploads=sum(obj.uploads for obj in objects.values())
    assert pair.uploads==1
    assert build.run(args)==first
    assert sum(obj.uploads for obj in objects.values())==uploads


def test_finalizer_rejects_missing_sample_receipt(release_fixture):
    args,objects,_=release_fixture
    for phase in ('mother','d1','d2','d3'):
        args.phase=phase; build.run(args)
    objects['output','build/control/d3/sample-a.json'].data=None
    args.phase='finalize'
    with pytest.raises(ValueError,match='per-sample'): build.run(args)


def test_cross_split_geographic_overlap_rejected():
    doc=selection(); other=copy.deepcopy(doc['selection'][0])
    other.update(canonical_core_id='sample-b',candidate_id='b',acquisition_id='other',
        site_id='other-site',geographic_group_id='other-group',split='test')
    doc['selection'].append(other); doc['splits']['assignment']['other-group']='test'
    with pytest.raises(ValueError,match='Cross-split geographic'): build.validate_selection(doc,True)


def test_shared_hash_tamper_fails(release_fixture):
    args,_,client=release_fixture
    build.run(args)
    store=build.Store(client,args.output_prefix)
    mother,_=store.document(build.MOTHER+'/manifest.json')
    sample=mother['samples'][0]; sample['context_sha256']='f'*64
    with pytest.raises(ValueError,match='shared input hash'): build.read_shared(store,sample)


def test_binary_and_exact_mask_grid_fail(tmp_path):
    row=selection()['selection'][0]
    rgb=tmp_path/'rgb.tif'; mask=tmp_path/'mask.tif'
    profile=dict(driver='GTiff',height=1700,width=1700,transform=Affine(*row['context_transform']),
        crs=row['crs'],dtype='uint8')
    with rasterio.open(rgb,'w',count=3,photometric='RGB',**profile) as ds:
        ds.write(np.zeros((3,1700,1700),np.uint8))
    values=np.full((1,1700,1700),255,np.uint8); values[0,0,0]=254
    with rasterio.open(mask,'w',count=1,**profile) as ds: ds.write(values)
    with rasterio.open(rgb) as ds,rasterio.open(mask) as ms:
        with pytest.raises(ValueError,match='Nonbinary'): build.read_native(ds,row,ms)
    profile['transform']=Affine(.8,0,.01,0,-.8,1360)
    with rasterio.open(mask,'w',count=1,**profile) as ds: ds.write(values)
    with rasterio.open(rgb) as ds,rasterio.open(mask) as ms:
        with pytest.raises(ValueError,match='grid mismatch'): build.read_native(ds,row,ms)


def terminal_fixture(store):
    starts=[]; refs=[]
    for phase in ('mother','d1','d2','d3'):
        start=dict(phase=phase,job_attempt_id=phase+'-attempt',instance_id='123',
            container_image_id='image@sha256:'+'1'*64,config_sha256='config')
        ref=store.put('control/attempts/'+phase+'/'+phase+'-attempt-started.json',build.json_bytes(start))
        starts.append(ref)
        log=store.put('logs/'+phase+'.txt',b'closed container log')
        terminal=dict(start,producer='sdk_supervisor',supervisor_id='controller',state='EXITED',
            exit_code=0,worker_process_exited=True,logs_closed=True,closed_log=log,start_record=ref)
        refs.append(store.put('supervisor/'+phase+'.json',build.json_bytes(terminal)))
    return dict(receipts=refs),starts


def test_terminal_supervisor_requires_closed_logs_all_attempts(release_fixture):
    args,_,client=release_fixture
    store=build.Store(client,args.output_prefix)
    evidence,starts=terminal_fixture(store)
    result=build.verify_terminal_evidence(store,evidence,'config',args.container_image_id)
    assert result['verified'] and result['successful_phases']==['d1','d2','d3','mother']
    incomplete=copy.deepcopy(evidence); incomplete['receipts'].pop()
    with pytest.raises(ValueError,match='attempt'): build.verify_terminal_evidence(store,incomplete,'config',args.container_image_id)
    payload=json.loads(store.get(evidence['receipts'][0])); payload['logs_closed']=False
    bad=store.put('supervisor/not-closed.json',build.json_bytes(payload))
    incomplete=copy.deepcopy(evidence); incomplete['receipts'][0]=bad
    with pytest.raises(ValueError,match='closed'): build.verify_terminal_evidence(store,incomplete,'config',args.container_image_id)


def test_terminal_supervisor_rejects_config_and_log_pin_mismatch(release_fixture):
    args,objects,client=release_fixture
    store=build.Store(client,args.output_prefix)
    evidence,_=terminal_fixture(store)
    with pytest.raises(ValueError,match='configuration'):build.verify_terminal_evidence(store,evidence,'different',args.container_image_id)
    objects['output','build/logs/mother.txt'].generation=2
    with pytest.raises(ValueError,match='metadata mismatch'):build.verify_terminal_evidence(store,evidence,'config',args.container_image_id)


def test_vendor_gsd_original_raster_spacing_and_mother_grid_are_distinct(release_fixture):
    args,_,client=release_fixture
    path=build.Path(args.selection); doc=json.loads(path.read_text())
    doc['selection'][0].update(native_gsd_m=.7,source_raster_transform=[.5,0,0,0,-.5,1360],
        source_grid=dict(crs='EPSG:32634'),acquisition_utc='2020-01-01T00:00:00Z')
    path.write_text(json.dumps(doc))
    build.run(args)
    store=build.Store(client,args.output_prefix)
    mother,_=store.document(build.MOTHER+'/manifest.json')
    row=mother['samples'][0]
    assert row['native_gsd_m']==row['source_gsd_m']==.7
    assert row['source_raster_spacing_m']==[.5,.5]
    assert row['mother_grid_gsd_m']==.8
    assert row['acquisition_utc']=='2020-01-01T00:00:00Z'
    assert build.source_raster_spacing(dict(crs='EPSG:32634',source_grid={})) is None
