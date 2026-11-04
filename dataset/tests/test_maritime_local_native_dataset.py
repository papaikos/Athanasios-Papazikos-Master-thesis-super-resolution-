import base64,hashlib,json
from pathlib import Path
import pytest


def pin(payload,uri='gs://bucket/image.tif'):
    import google_crc32c
    return dict(uri=uri,generation='1',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
                crc32c=base64.b64encode(google_crc32c.Checksum(payload).digest()).decode())


def test_local_store_checks_once_and_detects_changed_stat(tmp_path,monkeypatch):
    from src.maritime_local_native_dataset import LocalNativeStore
    from scripts import maritime_uint8_download
    data=b'observed original';reference=pin(data)
    path=tmp_path/(reference['sha256']+'.tif');path.write_bytes(data)
    calls=[];original=maritime_uint8_download._checksums
    def checksums(path): calls.append(path);return original(path)
    monkeypatch.setattr(maritime_uint8_download,'_checksums',checksums)
    store=LocalNativeStore(tmp_path)
    assert store.cached_source(reference)==path
    assert store.cached_source(reference)==path
    assert len(calls)==1
    path.write_bytes(b'conflicting file')
    with pytest.raises(ValueError,match='changed'):store.cached_source(reference)


def test_local_store_missing_conflicting_and_normalized_suffix(tmp_path):
    from src.maritime_local_native_dataset import LocalNativeStore,native_filename
    reference=pin(b'original',uri='gs://bucket/image.TIFF')
    assert native_filename(reference).endswith('.tif')
    store=LocalNativeStore(tmp_path)
    with pytest.raises(FileNotFoundError):store.cached_source(reference)
    (tmp_path/native_filename(reference)).write_bytes(b'corrupt!')
    with pytest.raises(ValueError,match='verification'):store.cached_source(reference)


def manifest(tmp_path,status='READY',kind='skysat_raw_window'):
    reference=pin(b'original')
    sample=dict(sample_id='one',split='train',source_kind=kind,source_provenance={},
        native_window_ref=dict(object=reference,mask={}),native_gsd_m=.8)
    doc=dict(dataset='mother',status=status,training_format_verified=True,sample_count=1,samples=[sample],
        shards=[dict(split='train',sample_ids=['one'],object=reference)])
    path=tmp_path/'manifest.json';path.write_text(json.dumps(doc));return path


def test_ready_manifest_and_explicit_format_only_gate(tmp_path):
    from src.maritime_local_native_dataset import LocalNativeMaritimeDataset
    path=manifest(tmp_path,status='REVIEW_PENDING')
    with pytest.raises(ValueError,match='READY'):LocalNativeMaritimeDataset(path)
    dataset=LocalNativeMaritimeDataset(path,allow_format_only=True)
    assert len(dataset)==1 and dataset.epoch==0
    dataset.set_epoch(2);assert dataset.epoch==2
    with pytest.raises(ValueError):dataset.set_epoch(-1)


def test_unknown_source_and_invalid_patch_size_fail(tmp_path):
    from src.maritime_local_native_dataset import LocalNativeMaritimeDataset
    with pytest.raises(ValueError,match='Unsupported'):LocalNativeMaritimeDataset(manifest(tmp_path,kind='unknown'))
    with pytest.raises(ValueError,match='patch_size'):LocalNativeMaritimeDataset(manifest(tmp_path),patch_size=True)


def test_no_cloud_guard_or_sdk_is_needed(tmp_path,monkeypatch):
    from src.maritime_local_native_dataset import LocalNativeMaritimeDataset
    from scripts import maritime_cloud_inventory
    monkeypatch.setattr(maritime_cloud_inventory,'require_cloud_vm',lambda:pytest.fail('No GCE guard allowed'))
    assert len(LocalNativeMaritimeDataset(manifest(tmp_path)))==1

def test_offline_tensor_batch_reuses_native_pixels_and_epoch_crop(tmp_path):
    torch=pytest.importorskip('torch')
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from src.maritime_local_native_dataset import LocalNativeMaritimeDataset,native_filename
    from src.maritime_native_dataset import native_patch
    rgb=np.arange(3*80*80,dtype=np.uint8).reshape(3,80,80)
    original=tmp_path/'source.tif'
    with rasterio.open(original,'w',driver='GTiff',width=80,height=80,count=3,dtype='uint8',
        crs='EPSG:32631',transform=from_origin(0,64,.8,.8),photometric='RGB') as ds:ds.write(rgb)
    reference=pin(original.read_bytes())
    path=manifest(tmp_path)
    document=json.loads(path.read_text())
    sample=document['samples'][0];sample['native_window_ref'].update(object=reference,core_window=[10,10,60,60])
    path.write_text(json.dumps(document))
    objects=tmp_path/'native'/'objects';objects.mkdir(parents=True)
    local=objects/native_filename(reference);local.write_bytes(original.read_bytes())
    dataset=LocalNativeMaritimeDataset(path,patch_size=16)
    dataset.set_epoch(3)
    with rasterio.open(local) as ds:expected,_=native_patch(ds,[10,10,60,60],16,'one',3,20261006)
    batch=next(iter(torch.utils.data.DataLoader(dataset,batch_size=1,num_workers=0)))
    assert batch['image'].shape==(1,3,16,16)
    np.testing.assert_array_equal(batch['image'][0].numpy(),expected)
    assert local.read_bytes()==original.read_bytes()


def test_local_store_pickles_without_lock_and_reverifies(tmp_path):
    import pickle
    from src.maritime_local_native_dataset import LocalNativeStore,native_filename
    reference=pin(b'original');(tmp_path/native_filename(reference)).write_bytes(b'original')
    store=LocalNativeStore(tmp_path);store.cached_source(reference)
    restored=pickle.loads(pickle.dumps(store))
    assert not restored.verified
    assert restored.cached_source(reference)==tmp_path/native_filename(reference)
