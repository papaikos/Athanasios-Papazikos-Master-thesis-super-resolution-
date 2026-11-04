import copy
import hashlib
import io
import json
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.enums import ColorInterp
from rasterio.transform import Affine

from scripts import maritime_release_audit as audit
from scripts.maritime_cloud_inventory import json_bytes
from src.maritime_recipes import array_hash, canonical_hr, degrade


def row():
    return dict(sample_id='sample',acquisition_id='capture',site_id='site',geographic_group_id='group',
        split='train',crs='EPSG:32634',core_bounds=[40,40,1320,1320],
        core_transform=[.8,0,40,0,-.8,1320],context_bounds=[0,0,1360,1360],
        context_transform=[.8,0,0,0,-.8,1360],
        hr_transform=[2.5,0,40,0,-2.5,1320],lr_transform=[10,0,40,0,-10,1320])


def tif(values,color_order=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)):
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',height=1700,width=1700,count=values.shape[0],
                dtype=values.dtype,crs='EPSG:32634',transform=Affine(.8,0,0,0,-.8,1360),
                compress='DEFLATE',photometric='RGB' if values.shape[0]==3 and color_order==(
                    ColorInterp.red,ColorInterp.green,ColorInterp.blue) else 'MINISBLACK') as dataset:
            dataset.write(values)
            if values.shape[0]==3: dataset.colorinterp=color_order
        return memory.read()


def npz(**values):
    output=io.BytesIO(); np.savez(output,**values); return output.getvalue()


class Store:
    def __init__(self): self.objects={}; self.start_rows=[]; self.verified={}
    def put(self,name,payload):
        ref=dict(uri='gs://fixture/'+name,generation='1',bytes=len(payload),crc32c='crc',
                 sha256=hashlib.sha256(payload).hexdigest())
        self.objects[name]=(payload,ref); return ref
    def get(self,ref):
        audit.validate_pin(ref); name=ref['uri'].removeprefix('gs://fixture/')
        payload,stored=self.objects[name]
        assert ref==stored
        self.verified[name]=ref
        return payload
    def discover(self,name):
        payload,ref=self.objects[name]; return json.loads(self.get(ref)),ref
    def starts(self): return self.start_rows


def terminal_fixture():
    store=Store(); receipts=[]
    for phase in audit.NAMES:
        start=dict(phase=phase,job_attempt_id=phase+'-attempt',instance_id='123',
                   config_sha256='config',container_image_id='image')
        pin=store.put(phase+'-started.json',json_bytes(start)); store.start_rows.append((start,pin))
        log=store.put(phase+'.log',b'exit 0\n')
        terminal=dict(start,producer='sdk_supervisor',supervisor_id='supervisor',state='EXITED',
            worker_process_exited=True,logs_closed=True,exit_code=0,start_record=pin,closed_log=log)
        receipts.append(store.put(phase+'-terminal.json',json_bytes(terminal)))
    return store,dict(receipts=receipts)


def test_accounting_excludes_development_and_rejects_physical_alias():
    a=row(); b=dict(a,sample_id='dev',acquisition_id='different',site_id='dev',
                    geographic_group_id='dev',split='development')
    result=audit.native_accounting([a,b])
    assert result['useful_native_core_bytes']==7680000
    assert result['unique_native_core_count']==1
    with pytest.raises(ValueError,match='physical'):
        audit.native_accounting([a,dict(a,sample_id='alias')])
    with pytest.raises(ValueError,match='Cross-split'):
        audit.native_accounting([a,dict(a,sample_id='test',acquisition_id='second',split='test')])


def test_native_valid_black_and_invalid_real_halo():
    rgb=tif(np.zeros((3,1700,1700),np.uint8)); mask=np.full((1,1700,1700),255,np.uint8)
    decoded,support=audit.native_arrays(rgb,tif(mask),row())
    assert not decoded.any() and support.all()
    mask[0,0,0]=0
    with pytest.raises(ValueError,match='halo'):
        audit.native_arrays(rgb,tif(mask),row())
    wrong=row(); wrong['core_bounds'][0]+=1
    with pytest.raises(ValueError,match='bounds'):
        audit.native_arrays(rgb,tif(np.full_like(mask,255)),wrong)


def test_terminal_requires_every_started_attempt_and_closed_logs():
    store,evidence=terminal_fixture()
    assert audit.terminal_audit(store,evidence,'config','image')['attempts']==4
    with pytest.raises(ValueError,match='All started'):
        audit.terminal_audit(store,dict(receipts=evidence['receipts'][:-1]),'config','image')
    ref=evidence['receipts'][0]; terminal=json.loads(store.get(ref)); terminal['logs_closed']=False
    evidence['receipts'][0]=store.put('bad.json',json_bytes(terminal))
    with pytest.raises(ValueError,match='exit evidence'):
        audit.terminal_audit(store,evidence,'config','image')


def test_pin_reader_checks_metadata_and_decoded_sha():
    payload=b'object'; pin=dict(uri='gs://bucket/key',generation='7',bytes=6,crc32c='crc',
                               sha256=hashlib.sha256(payload).hexdigest())
    blob=SimpleNamespace(generation='7',size=6,crc32c='crc',
        reload=lambda **kwargs:None,download_as_bytes=lambda **kwargs:payload)
    client=SimpleNamespace(bucket=lambda name:SimpleNamespace(blob=lambda *args,**kwargs:blob))
    store=audit.AuditStore(client,'gs://bucket/root')
    assert store.get(pin)==payload
    with pytest.raises(ValueError,match='SHA'):
        store.get(dict(pin,sha256='a'*64))
    blob.crc32c='changed'
    with pytest.raises(ValueError,match='CRC'):
        store.get(pin)


def release_fixture(corrupt_lr=False,shift_training_origin=False):
    store=Store(); config=dict(provisional=True,container_image_id='image')
    configref=store.put('control/config.json',json_bytes(config)); sha=hashlib.sha256(json_bytes(config)).hexdigest()
    native=np.zeros((3,1700,1700),np.uint8); native[:,:,800:]=127
    context,hr=canonical_hr(native,np.ones((1700,1700),bool)); sample=row()
    if shift_training_origin:
        sample['hr_transform'][2]+=10000
        sample['lr_transform'][2]+=10000
    sample.update(hr_sha256=array_hash(hr),context_sha256=array_hash(context),lr_origin='synthetic_hr_only')
    sample['mother']=store.put('native.tif',tif(native))
    sample['mask']=dict(object=store.put('mask.tif',tif(np.full((1,1700,1700),255,np.uint8))))
    sample['shared']=store.put('shared.npz',npz(context=context,hr=hr,hr_mask=np.ones((512,512),bool)))
    refs={}; cards={}; terminals=[]
    for phase in audit.NAMES:
        current=copy.deepcopy(sample)
        manifest=dict(release=audit.NAMES[phase],state='COMPLETE',config_sha256=sha,config=configref,
                      container_image_id='image',samples=[current])
        objects={key:current[key] for key in ('mother','shared')}; objects['mask']=current['mask']['object']
        if phase=='mother': manifest['useful_native_core_bytes']=7680000
        else:
            result=degrade(context,'sample',phase)
            if corrupt_lr and phase=='d3':
                result['lr'][0,0,0]=np.float32(.5)
                result['metadata']['lr_sha256']=array_hash(result['lr'])
            current['recipe']=result['metadata']
            metadata=dict(current,recipe=result['metadata'])
            payload=npz(hr=hr,lr=result['lr'],hr_mask=result['hr_mask'],lr_mask=result['lr_mask'],
                        provenance=np.array(json.dumps(metadata,sort_keys=True)))
            current['pair']=store.put(phase+'.npz',payload); objects['pair']=current['pair']
            buffer=io.BytesIO()
            with tarfile.open(fileobj=buffer,mode='w') as archive:
                info=tarfile.TarInfo('train/sample.npz'); info.size=len(payload); archive.addfile(info,io.BytesIO(payload))
            shard=store.put(phase+'.tar',buffer.getvalue())
            manifest.update(recipe=phase,shards=[dict(split='train',sample_ids=['sample'],object=shard)])
        store.put('control/'+phase+'/sample.json',json_bytes(dict(state='COMPLETE',config_sha256=sha,
                    sample=current,objects=objects)))
        refs[phase]=store.put(audit.NAMES[phase]+'/manifest.json',json_bytes(manifest))
        refs[phase+'_ready']=store.put(phase+'-ready.json',json_bytes(dict(status='PROVISIONAL_READY',
            manifest=refs[phase],config=configref)))
        cards[phase]=store.put(phase+'-card.md',b'provisional\n')
        store.put('control/phases/'+phase+'-complete.json',json_bytes(dict(state='COMPLETE',
                  config_sha256=sha,manifest=refs[phase])))
        start=dict(phase=phase,job_attempt_id=phase,instance_id='123',config_sha256=sha,container_image_id='image')
        startref=store.put(phase+'-start.json',json_bytes(start)); store.start_rows.append((start,startref))
        terminal=dict(start,state='EXITED',producer='sdk_supervisor',supervisor_id='supervisor',
            worker_process_exited=True,logs_closed=True,exit_code=0,start_record=startref,
            closed_log=store.put(phase+'.log',b'done\n'))
        terminals.append(store.put(phase+'-terminal.json',json_bytes(terminal)))
    release=dict(config=configref,releases=refs,data_cards=cards,status='PROVISIONAL_READY',
                 full_training_ready=False,terminal_supervisor_audit=dict(terminal_receipts=terminals))
    store.put('release.json',json_bytes(release)); return store


def test_all_three_real_torch_loaders_and_every_pair_recompute(tmp_path):
    pytest.importorskip('torch')
    report=audit.audit_release(release_fixture(),tmp_path)
    assert report['passed'] and report['native_accounting']['useful_native_core_bytes']==7680000
    assert all(r['verified_samples']==1 for r in report['loader_reports'].values())
    assert not report['full_training_ready_attested']
    assert not report['original_raw_source_window_pixel_comparison_verified']


def test_provisional_cannot_attest_full_ready(tmp_path):
    pytest.importorskip('torch'); store=release_fixture()
    release,_=store.discover('release.json'); release['full_training_ready']=True
    store.put('release.json',json_bytes(release))
    with pytest.raises(ValueError,match='falsely'):
        audit.audit_release(store,tmp_path)


def test_self_consistent_wrong_lr_is_rejected_by_recomputation(tmp_path):
    pytest.importorskip('torch')
    with pytest.raises(ValueError,match='recomputation'):
        audit.audit_release(release_fixture(corrupt_lr=True),tmp_path)


def test_self_consistent_shifted_training_origins_fail_native_binding(tmp_path):
    pytest.importorskip('torch')
    # All manifests, receipts and NPZ provenance share the shifted origins.
    with pytest.raises(ValueError,match='Training grid.*native core'):
        audit.audit_release(release_fixture(shift_training_origin=True),tmp_path)


def test_native_requires_declared_rgb_order_and_real_context_footprint():
    values=np.zeros((3,1700,1700),np.uint8)
    mask=tif(np.full((1,1700,1700),255,np.uint8))
    with pytest.raises(ValueError,match='RGB'):
        audit.native_arrays(tif(values,(ColorInterp.blue,ColorInterp.green,ColorInterp.red)),mask,row())
    shifted=row(); shifted['context_bounds'][0]+=1
    with pytest.raises(ValueError,match='Context footprint'):
        audit.native_arrays(tif(values),mask,shifted)
