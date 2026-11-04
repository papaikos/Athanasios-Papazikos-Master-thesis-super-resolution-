import io
import json
import tarfile
from types import SimpleNamespace
import sys
import types

import numpy as np
import pytest
from PIL import Image

from scripts import maritime_curated_release as release
from src.maritime_curated_dataset import CuratedMaritimeDataset


def xml(gsd='.3',width=930):
    return (f'<annotation><Img_Resolution>{gsd}</Img_Resolution><filename>sample.bmp</filename>'
        f'<size><width>{width}</width><height>930</height><depth>3</depth></size>'
        '<object><name>Motorboat</name><level_3>49</level_3><bndbox>'
        '<xmin>450</xmin><ymin>450</ymin><xmax>480</xmax><ymax>480</ymax></bndbox></object>'
        '<object><name>Fishing Vessel</name><bndbox><xmin>0</xmin><ymin>0</ymin>'
        '<xmax>40</xmax><ymax>40</ymax></bndbox></object></annotation>').encode()


def record():
    pin=dict(uri='gs://test/image.bmp',generation='1',bytes=10,crc32c='AAAAAA==',sha256='a'*64)
    return dict(status='QUALIFIED_GSD_RGB_LABELED',native_gsd_m=.3,width=930,height=930,
        image_original=pin.copy(),annotation_original=pin.copy(),decoded_pixel_sha256='b'*64,
        source_family='xView',official_splits=['test'],repository_commit='c'*40)


def test_qualified_filter_never_accepts_unknown_or_coarse_gsd():
    good=record()
    doc=dict(records=[good,dict(status='REJECTED_GSD_UNKNOWN'),dict(status='REJECTED_GSD_COARSER_THAN_1M')])
    assert release.qualified_records(doc)==[good]
    for gsd in (None,1.07):
        bad=record();bad['native_gsd_m']=gsd
        with pytest.raises(ValueError,match='0.3'):release.qualified_records(dict(records=[bad]))


def test_centered_context_has_real_support_and_original_annotations():
    native=np.full((930,930,3),51,np.uint8)
    context,geometry,annotations=release.prepare_context(native,xml())
    assert context.shape==(3,96,96) and np.allclose(context,.2,atol=1e-7)
    assert geometry['native_context_window']==[65,65,800,800]
    assert geometry['hr_core_source_bounds']==pytest.approx([198.3333333333]*2+[731.6666666667]*2)
    assert len(annotations)==1 and annotations[0]['class_name']=='Motorboat'
    assert annotations[0]['source_bbox']==[450,450,480,480]
    assert annotations[0]['fully_inside_hr_core']
    with pytest.raises(ValueError,match='GSD'):release.prepare_context(native,xml('1.07'))
    with pytest.raises(ValueError,match='dimensions'):release.prepare_context(native,xml(width=800))


def test_quarantined_pair_loader_has_real_64_16_torch_batches(tmp_path):
    context,geometry,annotations=release.prepare_context(np.full((930,930,3),73,np.uint8),xml())
    common=release.sample_metadata(record(),context,geometry,annotations)
    for recipe in ('d1','d2','d3'):
        payload=release.pair_payload(common,context,recipe,20261006)
        path=tmp_path/(recipe+'.tar')
        with tarfile.open(path,'w') as archive:release.tar_member(archive,common['sample_id']+'.npz',payload)
        dataset=CuratedMaritimeDataset([path],recipe)
        pair,meta=dataset.read_arrays(0)
        assert pair['hr'].shape==(3,64,64) and pair['lr'].shape==(3,16,16)
        assert meta['split']=='train' and meta['georeferencing'] is None
        report=release.verify_shard(path,recipe,{common['sample_id']:context},20261006)
        assert report['verified_samples']==1 and report['numeric_recomputed_samples']==1


@pytest.mark.parametrize('field,value',[('georeferencing','EPSG:4326'),('split','test')])
def test_loader_rejects_invented_georeferencing_and_wrong_split(tmp_path,field,value):
    context,geometry,annotations=release.prepare_context(np.zeros((930,930,3),np.uint8),xml())
    common=release.sample_metadata(record(),context,geometry,annotations)
    payload=release.pair_payload(common,context,'d2',1)
    with np.load(io.BytesIO(payload),allow_pickle=False) as data:
        arrays={key:np.array(data[key]) for key in data.files}
    meta=json.loads(str(arrays['provenance']));meta[field]=value
    arrays['provenance']=np.asarray(json.dumps(meta))
    path=tmp_path/'bad.tar'
    with tarfile.open(path,'w') as archive:release.tar_member(archive,common['sample_id']+'.npz',release.deterministic_npz(arrays))
    with pytest.raises(ValueError,match='quarantine'):CuratedMaritimeDataset([path],'d2').read_arrays(0)


def test_end_to_end_supplement_preserves_originals_and_never_claims_quota(tmp_path,monkeypatch):
    buffer=io.BytesIO();Image.fromarray(np.full((930,930,3),89,np.uint8)).save(buffer,format='BMP')
    original=buffer.getvalue();annotation=xml();sample=record()
    sample['decoded_pixel_sha256']=release.pixel_identity(original)
    sample['image_original'].update(bytes=len(original),sha256=release.digest_bytes(original))
    sample['annotation_original'].update(uri='gs://test/source.xml',bytes=len(annotation),sha256=release.digest_bytes(annotation))
    document=dict(records=[sample],license='academic-only; constituent terms apply',license_evidence_url='https://example.test/license')
    path=tmp_path/'qualified.json';path.write_text(json.dumps(document))
    written={}
    class Store:
        scratch=tmp_path
        def get(self,ref):
            value=annotation if ref['uri'].endswith('.xml') else original
            assert len(value)==ref['bytes'] and release.digest_bytes(value)==ref['sha256']
            return value
        def publish(self,name,data):
            written[name]=data
            return dict(uri='gs://test/release/'+name,generation='1',bytes=len(data),
                        crc32c='AAAAAA==',sha256=release.digest_bytes(data))
        def existing(self,name):return None
    storage=types.ModuleType('google.cloud.storage');storage.Client=lambda **kwargs:object()
    cloud=types.ModuleType('google.cloud');cloud.storage=storage
    monkeypatch.setitem(sys.modules,'google.cloud',cloud);monkeypatch.setitem(sys.modules,'google.cloud.storage',storage)
    monkeypatch.setattr(release,'require_cloud_vm',lambda:'9001')
    monkeypatch.setattr(release,'CloudStore',lambda *args:Store())
    result=release.run(SimpleNamespace(manifest=path,output_prefix='gs://test/release',scratch=tmp_path,
        image_id='sha256:'+'d'*64,code_sha256='e'*64,release_seed=20261006,shard_samples=32,max_samples=None))
    assert result['sample_count']==1 and result['unique_native_quota_contribution_bytes']==0
    assert result['status']=='TRAINING_FORMAT_VERIFIED_SUPPLEMENT'
    assert set(result['manifests'])=={'mother','d1','d2','d3'}
    manifests=[json.loads(written[name+'/manifest.json']) for name in ('mother','d1','d2','d3')]
    assert all(m['split']=='train' and not m['geographic_holdout_isolation_verified'] for m in manifests)
    assert all(m['samples'][0]['hr_sha256']==manifests[0]['samples'][0]['hr_sha256'] for m in manifests)
    assert manifests[0]['samples'][0]['source_provenance']['official_splits']==['test']
    assert manifests[0]['samples'][0]['native_context_ref']['object']==sample['image_original']
