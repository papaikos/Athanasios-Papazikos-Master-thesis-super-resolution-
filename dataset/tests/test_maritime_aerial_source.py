import copy
import io
from types import SimpleNamespace

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from scripts.maritime_cloud_inventory import digest_bytes
from scripts.maritime_maxar_intake import core_geometry
from scripts.maritime_noaa_intake import candidate_metadata
from src import maritime_aerial_source as adapter


def fixture():
    grid=SimpleNamespace(crs=CRS.from_epsg(6340),transform=Affine(1,0,-40,0,-1,1320),width=1360,height=1360)
    geometry=core_geometry(grid,0,0)
    pin=dict(uri='gs://test/original.tif',generation='1',bytes=42,crc32c='ABCD',sha256='a'*64)
    src=dict(source_kind='aerial_rgb',source_id='10317:tile',source_type='noaa_aerial',epsg=6340,
        href='https://coastalimagery.blob.core.windows.net/digitalcoast/tile.tif',
        dataset={'id':'10317','license':'CC0-1.0','license_evidence_url':'https://www.fisheries.noaa.gov/inport/item/75459'},
        original_http_identity={'etag':'observed','bytes':42},transform_affine=list(grid.transform)[:6],
        shape=[1360,1360],native_gsd_m=1.,delivered_gsd_m=1.,original_camera_capture_gsd_m=None,
        acquisition_day='2025-02-01',stac_datetime='2025-04-02',parent_tile_id='tile',
        capture_identity_type='dataset_acquisition_day_and_original_orthomosaic_tile',
        rgb_indexes=[1,2,3],alpha_index=4,original_product='orthorectified_aerial_mosaic')
    record=candidate_metadata(src,geometry,grid.transform,pin,dict(pin,uri='gs://test/evidence.json'))
    rgb=np.full((3,1360,1360),100,np.uint8)
    record.update(native_context=dict(pin,uri='gs://test/native.tif'),hr=dict(pin,uri='gs://test/hr.npz'),
        canonical_hr_method='canonical_sparse_exact_area_v1',
        decoded_native_context_sha256=digest_bytes(rgb.tobytes()),
        decoded_useful_core_sha256=digest_bytes(rgb[:,40:-40,40:-40].tobytes()))
    row=dict(source=record,source_kind='aerial_rgb',sample_id=record['sample_id'],split='train',
        site_id='port',geographic_group_id='port',acquisition_id='10317:2025-02-01')
    return row,rgb


def raster(rgb,record):
    memory=MemoryFile()
    with memory.open(driver='GTiff',width=1360,height=1360,count=3,dtype='uint8',
        crs=record['source_crs'],transform=Affine(*record['native_window_transform'])) as ds:
        ds.write(rgb);ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
    return memory


def test_truthful_aerial_source_geometry_accounting_and_dispatch():
    from scripts import maritime_ultimate_release as release
    row,_=fixture();adapter.validate_source(row['source'])
    assert release.source_geometry(row)==('EPSG:6340',[0,0,1280,1280])
    assert release.native_bytes(row)==1280*1280*3
    doc=dict(schema='ultimate-selection-v1',review_policy='explicit aerial provenance',
        selection=[row],splits={'assignment':{'port':'train'}})
    release.validate_selection(doc)
    assert row['source']['original_camera_capture_gsd_m'] is None


@pytest.mark.parametrize('key,value', [('delivered_gsd_m',1.1),('original_camera_capture_gsd_m',.25),
    ('physical_core_bounds',[0,0,1000,1000]),('native_window_transform',[1,0,0,0,-1,0]),
    ('source_href','https://unverified.test/tile.tif'),('original_mask_provenance','inferred'),
    ('source_radiometry','histogram_stretched'),('canonical_hr_method','bilinear')])
def test_forged_or_inconsistent_aerial_metadata_rejected(key,value):
    row,_=fixture();row['source'][key]=value
    with pytest.raises(ValueError):adapter.validate_source(row['source'])


def test_original_pixels_reproduce_canonical_and_verify_stored_arrays():
    row,rgb=fixture();record=row['source']
    with raster(rgb,record) as memory,memory.open() as ds:
        context,useful,*_=adapter.canonical_from_native(ds,record)
        assert context.shape==(3,544,544) and context.dtype==np.float32
        assert useful==1280*1280*3
        np.testing.assert_allclose(context,100/255,atol=1e-7)
        buffer=io.BytesIO();np.savez_compressed(buffer,hr_context=context,hr=context[:,16:-16,16:-16])
        adapter.canonical_from_native(ds,record,buffer.getvalue())
        corrupt=copy.deepcopy(record);corrupt['decoded_native_context_sha256']='b'*64
        with pytest.raises(ValueError,match='hash'):adapter.canonical_from_native(ds,corrupt)


def test_original_parent_replay_requires_alpha_and_exact_rgb():
    row,rgb=fixture();record=row['source']
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',width=1360,height=1360,count=4,dtype='uint8',
            crs=record['source_crs'],transform=Affine(*record['original_transform'])) as ds:
            ds.write(rgb,indexes=[1,2,3]);ds.write(np.full((1360,1360),255,np.uint8),4)
            ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
        with memory.open() as ds:adapter.verify_original_parent(ds,record,rgb)
        changed=rgb.copy();changed[0,100,100]+=1
        with memory.open() as ds:
            with pytest.raises(ValueError,match='original'):adapter.verify_original_parent(ds,record,changed)


def test_native_loader_validates_aerial_context_without_camera_gsd_claim():
    from src.maritime_native_dataset import validate_aerial_native
    row,rgb=fixture()
    with raster(rgb,row['source']) as memory,memory.open() as ds:
        validate_aerial_native(ds,row['source'],[40,40,1280,1280])
        with pytest.raises(ValueError):validate_aerial_native(ds,row['source'],[0,0,1280,1280])


def test_process_replays_parent_pins_and_keeps_aerial_provenance(monkeypatch):
    import json
    from scripts import maritime_ultimate_release as release
    row,rgb=fixture();record=row['source']
    with raster(rgb,record) as native:
        with native.open() as ds:context,*_=adapter.canonical_from_native(ds,record)
        buffer=io.BytesIO();np.savez_compressed(buffer,hr_context=context,hr=context[:,16:-16,16:-16])
        with MemoryFile() as original:
            with original.open(driver='GTiff',width=1360,height=1360,count=4,dtype='uint8',
                crs=record['source_crs'],transform=Affine(*record['original_transform'])) as ds:
                ds.write(rgb,indexes=[1,2,3]);ds.write(np.full((1360,1360),255,np.uint8),4)
                ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
            evidence=dict(source=record['source'],original_item_metadata_object=record['source_evidence_pin'],
                          official_dataset_metadata_object=record['source_evidence_pin'])
            class Store:
                def get(self,pin):
                    if pin==record['native_context']:return native.read()
                    if pin==record['hr']:return buffer.getvalue()
                    return json.dumps(evidence).encode()
                def cached_source(self,pin):
                    assert pin==record['source_original_object'];return original.name
            monkeypatch.setattr(release,'pair_payload',lambda *args:(b'recipe',{}))
            common,_,shared,children=release.process_sample(Store(),row,20261006)
            assert common['source_platform']=='aerial' and common['original_parent_pixels_replayed']
            assert common['original_camera_capture_gsd_m'] is None and common['delivered_gsd_m']==1.
            assert set(children)=={'d1','d2','d3'}
            with np.load(io.BytesIO(shared),allow_pickle=False) as arrays:
                assert arrays['hr'].shape==(3,512,512) and arrays['hr_mask'].all()
            row['split']='test'
            with pytest.raises(ValueError,match='evaluation'):release.process_sample(Store(),row,20261006)


def test_original_alpha_hole_cannot_be_hidden_by_captured_all_valid_mask():
    row,rgb=fixture();record=row['source'];alpha=np.full((1360,1360),255,np.uint8);alpha[80,80]=0
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',width=1360,height=1360,count=4,dtype='uint8',
            crs=record['source_crs'],transform=Affine(*record['original_transform'])) as ds:
            ds.write(rgb,indexes=[1,2,3]);ds.write(alpha,4)
            ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
        with memory.open() as ds:
            with pytest.raises(ValueError,match='support'):adapter.verify_original_parent(ds,record,rgb)
