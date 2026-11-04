import base64
import hashlib
import io
import json
import tarfile

import google_crc32c
import numpy as np
import pytest


def pin(payload=b'payload', uri='gs://bucket/file.tar'):
    return dict(uri=uri,generation='7',bytes=len(payload),
        crc32c=base64.b64encode(google_crc32c.Checksum(payload).digest()).decode(),
        sha256=hashlib.sha256(payload).hexdigest())


class Blob:
    def __init__(self,payload):
        self.payload=payload;self.generation=7;self.size=len(payload)
        self.crc32c=pin(payload)['crc32c'];self.calls=0
    def reload(self): pass
    def download_to_filename(self,path,*,if_generation_match,checksum):
        assert if_generation_match==7 and checksum=='crc32c'
        self.calls+=1
        from pathlib import Path
        Path(path).write_bytes(self.payload)


class Client:
    def __init__(self,blob): self.object=blob
    def bucket(self,name): return self
    def blob(self,name,generation): return self.object


def test_pinned_atomic_download_reuses_verified_file_and_rejects_wrong_hash(tmp_path):
    from scripts.maritime_uint8_download import fetch
    payload=b'new uint8 archive';reference=pin(payload);blob=Blob(payload)
    target=tmp_path/'shard.tar'
    fetch(Client(blob),reference,target)
    assert target.read_bytes()==payload and blob.calls==1
    fetch(Client(blob),reference,target)
    assert blob.calls==1
    target.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='Existing'): fetch(Client(blob),reference,target)
    assert target.read_bytes()==b'wrong'


def test_failed_integrity_never_publishes_partial_file(tmp_path):
    from scripts.maritime_uint8_download import fetch
    reference=pin(b'expected');target=tmp_path/'shard.tar'
    blob=Blob(b'expected');blob.payload=b'corrupted'
    with pytest.raises(ValueError): fetch(Client(blob),reference,target)
    assert not target.exists()
    assert not target.with_suffix('.tar.partial').exists()


def manifest(name='d1'):
    from src.maritime_recipes import RECIPE_VERSIONS
    from src.maritime_uint8 import STORAGE_VERSION
    return dict(dataset=name,training_format_verified=True,sample_count=2,
        storage_version=STORAGE_VERSION,recipe=RECIPE_VERSIONS.get(name),
        samples=[dict(sample_id='train-id',split='train'),dict(sample_id='test-id',split='test')],
        shards=[dict(split='train',sample_ids=['train-id'],object=pin(uri='gs://b/train.tar')),
                dict(split='test',sample_ids=['test-id'],object=pin(b'test',uri='gs://b/test.tar'))])


def test_download_jobs_use_only_requested_recipe_and_split():
    from scripts.maritime_uint8_download import build_jobs
    jobs=build_jobs({'d1':manifest()},recipes=['d1'],splits=['train'])
    assert len(jobs)==1
    assert jobs[0][1].parts[:2]==('D1','train')
    corrupted=manifest();corrupted['shards'][0]['sample_ids']=['test-id']
    with pytest.raises(ValueError,match='shard'): build_jobs({'d1':corrupted},recipes=['d1'],splits=['train'])


def auxiliary_manifest(name='d1'):
    document=manifest(name)
    document.update(schema='maritime-aerial-auxiliary-evaluation-manifest-v1',
        evaluation_format_verified=True,separate_auxiliary_evaluation=True,train_sample_count=0)
    document['samples'][0]['split']='validation'
    document['shards'][0]['split']='validation'
    return document


def auxiliary_audit():
    return dict(schema='marine-selected-addition-visual-coverage-v1',passed=True,
        required_review_fraction=1,reviewed_added_ids=['train-id','test-id'],gaps=[])


def test_auxiliary_download_keeps_evaluation_separate_from_frozen_training():
    from scripts.maritime_uint8_download import build_auxiliary_jobs
    documents={name:auxiliary_manifest(name) for name in ('mother','d1','d2','d3')}
    jobs=build_auxiliary_jobs(documents,auxiliary_audit(),name='aerial-v1',recipes=['d1'],splits=['test'])
    assert len(jobs)==1
    assert jobs[0][1].parts[:4]==('D1','evaluation','aerial-v1','test')
    assert not build_auxiliary_jobs(documents,auxiliary_audit(),name='aerial-v1',recipes=['d1'],splits=['train'])


@pytest.mark.parametrize('change',[{'train_sample_count':1},{'evaluation_format_verified':False},
    {'separate_auxiliary_evaluation':False}])
def test_auxiliary_download_rejects_training_or_unverified_population(change):
    from scripts.maritime_uint8_download import build_auxiliary_jobs
    document=auxiliary_manifest();document.update(change)
    with pytest.raises(ValueError):
        build_auxiliary_jobs({'d1':document},auxiliary_audit(),name='aerial-v1',recipes=['d1'])


def test_auxiliary_download_requires_exact_complete_visual_coverage_and_safe_name():
    from scripts.maritime_uint8_download import build_auxiliary_jobs
    document=auxiliary_manifest();audit=auxiliary_audit();audit['reviewed_added_ids']=['test-id']
    with pytest.raises(ValueError,match='review'):
        build_auxiliary_jobs({'d1':document},audit,name='aerial-v1',recipes=['d1'])
    with pytest.raises(ValueError,match='name'):
        build_auxiliary_jobs({'d1':document},auxiliary_audit(),name='../escape',recipes=['d1'])


def test_mother_native_uint16_reference_remains_opaque_and_deduplicated():
    from scripts.maritime_uint8_download import build_jobs
    mother=manifest('mother')
    original=pin(b'uint16 TOA bytes',uri='gs://b/uint16.tif')
    for sample in mother['samples']:
        sample.update(native_context=original,source_provenance=dict(source_rgb_dtype='uint16'))
    jobs=build_jobs({'mother':mother},recipes=['mother'],splits=['train','test'],include_native=True)
    native=[(ref,path) for ref,path in jobs if 'native' in path.parts]
    assert len(native)==1 and native[0][0]==original
    assert native[0][1].suffix=='.tif'


def test_manifest_split_loader_does_not_include_test_shards(tmp_path):
    from scripts.maritime_uint8_download import FOLDERS
    from scripts.maritime_uint8_training_smoke import local_shards
    folder=tmp_path/FOLDERS['d1'];folder.mkdir()
    document=manifest()
    (folder/'manifest.json').write_text(json.dumps(document))
    target=folder/'train'/(document['shards'][0]['object']['sha256']+'.tar')
    target.parent.mkdir();target.write_bytes(b'placeholder')
    selected=local_shards(tmp_path,'d1','train')
    assert selected==[target]


def test_local_auxiliary_loader_uses_only_separate_evaluation_population(tmp_path):
    from scripts.maritime_uint8_training_smoke import local_shards
    document=auxiliary_manifest()
    folder=tmp_path/'D1'/'evaluation'/'aerial-v1';folder.mkdir(parents=True)
    (folder/'manifest.json').write_text(json.dumps(document))
    audit_folder=tmp_path/'control'/'evaluation'/'aerial-v1';audit_folder.mkdir(parents=True)
    (audit_folder/'quality-audit.json').write_text(json.dumps(auxiliary_audit()))
    reference=document['shards'][1]['object']
    target=folder/'test'/(reference['sha256']+'.tar');target.parent.mkdir();target.write_bytes(b'test')
    assert local_shards(tmp_path,'d1','test',evaluation='aerial-v1')==[target]
    with pytest.raises(ValueError,match='training'):
        local_shards(tmp_path,'d1','train',evaluation='aerial-v1')


def test_masked_supervised_loss_has_no_recipe_specific_consistency_term():
    torch=pytest.importorskip('torch')
    from scripts.maritime_uint8_training_smoke import masked_supervised_loss
    prediction=torch.zeros((1,3,2,2),requires_grad=True)
    target=torch.ones_like(prediction);target.data[:,:,0,0]=100
    mask=torch.ones((1,2,2),dtype=torch.bool);mask[:,0,0]=False
    loss=masked_supervised_loss(prediction,target,mask,'l1')
    assert loss.item()==1
    loss.backward()
    assert torch.count_nonzero(prediction.grad[:,:,0,0]).item()==0
    assert masked_supervised_loss(prediction.detach(),target,mask,'mse').item()==1


def test_engineering_smoke_updates_parameters_on_validated_uint8_pairs(tmp_path):
    pytest.importorskip('torch')
    from scripts.maritime_uint8_training_smoke import training_smoke
    from src.maritime_recipes import RECIPE_VERSIONS,array_hash
    from src.maritime_uint8 import encode_uint8_pair
    hr=np.full((3,512,512),0.5,np.float32);lr=np.full((3,128,128),0.25,np.float32)
    metadata=dict(sample_id='train-id',split='train',lr_origin='synthetic_hr_only',hr_sha256=array_hash(hr),
        hr_transform=[2.5,0,0,0,-2.5,1280],lr_transform=[10,0,0,0,-10,1280],
        recipe=dict(recipe=RECIPE_VERSIONS['d1'],sample_id='train-id',hr_sha256=array_hash(hr),lr_sha256=array_hash(lr)))
    pair,metadata=encode_uint8_pair(hr,lr,metadata)
    stream=io.BytesIO();np.savez_compressed(stream,**pair,provenance=np.array(json.dumps(metadata)))
    document=manifest();document['samples']=document['samples'][:1];document['sample_count']=1;document['shards']=document['shards'][:1]
    directory=tmp_path/'D1';(directory/'train').mkdir(parents=True)
    archive_path=directory/'train'/(document['shards'][0]['object']['sha256']+'.tar')
    with tarfile.open(archive_path,'w') as archive:
        member=tarfile.TarInfo('train-id.npz');member.size=len(stream.getvalue())
        archive.addfile(member,io.BytesIO(stream.getvalue()))
    (directory/'manifest.json').write_text(json.dumps(document))
    result=training_smoke(tmp_path,recipe='d1',steps=1)
    assert result['optimizer_updated_parameters']
    assert result['hr_shape']==[1,3,512,512]
    assert result['lr_shape']==[1,3,128,128]
    assert result['scope']=='engineering format smoke; not a trained thesis model'
