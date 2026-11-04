"""NAIP evidence and replay contracts, separate from NOAA alpha products."""
import copy

import numpy as np
import pytest
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.crs import CRS
from types import SimpleNamespace

from tests.test_maritime_aerial_source import fixture, raster
from src import maritime_naip_source as adapter
from scripts.maritime_cloud_inventory import digest_bytes
from scripts.maritime_maxar_intake import core_geometry


def naip_fixture():
    row,rgb=fixture();record=row['source'];source=record['source']
    record['source_type']=source['source_type']='naip_aerial'
    source['dataset']=dict(id='naip',license='Public-Domain',
        license_evidence_url='https://www.usgs.gov/media/images/tnmcorps-naip-imagery-layer',
        stac_license='proprietary',license_conflict=True)
    record['source_href']=source['href']='https://naipblobs.blob.core.windows.net/naip/tile.tif'
    record['original_mask_provenance']='all_original_RGB_masks255'
    record['capture_identity_type']=source['capture_identity_type']='filename_acquisition_day_and_original_orthomosaic_tile'
    source.pop('alpha_index')
    source['original_raster_header']=dict(count=4,dtypes=['uint8']*4,
        colorinterp=['red','green','blue','undefined'],mask_flags=[['all_valid']]*4,nodata=None)
    source['original_support_policy']=record['original_support_policy']='all_original_RGB_masks255_and_no_RGB000'
    source['original_void_mask_available']=record['original_void_mask_available']=False
    source['stac_datetime_role']='capture_calendar_day_hour_unverified'
    record['stac_datetime']=source['stac_datetime'];record['publication_datetime']=None
    return row,rgb


def test_naip_contract_dispatch_preserves_license_conflict_and_unknown_camera_gsd():
    from scripts import maritime_ultimate_release as release
    row,_=naip_fixture();adapter.validate_source(row['source'])
    assert release.native_bytes(row)==1280*1280*3
    assert row['source']['original_camera_capture_gsd_m'] is None
    assert row['source']['source']['dataset']['license_conflict'] is True
    release.validate_selection(dict(schema='ultimate-selection-v1',review_policy='original evidence',
        selection=[row],splits={'assignment':{'port':'train'}}))


def test_explicit_v2_context_version_is_verified_and_unknown_geometry_version_rejected():
    row,_=naip_fixture();record=row['source']
    record['geometry_version']='original-contributor-window-v2'
    adapter.validate_source(record)
    record['geometry_version']='silently_resized'
    with pytest.raises(ValueError,match='geometry version'):
        adapter.validate_source(record)


@pytest.mark.parametrize('key,value',[
    ('original_camera_capture_gsd_m',.6),('delivered_gsd_m',1.1),
    ('original_mask_provenance','original_alpha255_and_all_original_RGB_masks255'),
    ('source_href','https://unverified.test/image.tif'),('canonical_hr_method','bilinear'),
    ('physical_core_bounds',[0,0,1000,1000])])
def test_forged_naip_contract_rejected(key,value):
    row,_=naip_fixture();row['source'][key]=value
    with pytest.raises(ValueError):adapter.validate_source(row['source'])


def test_proprietary_stac_conflict_cannot_be_silently_removed():
    row,_=naip_fixture();row['source']['source']['dataset']['license_conflict']=False
    with pytest.raises(ValueError,match='license'):adapter.validate_source(row['source'])


def test_canonical_naip_uses_original_rgb_without_stretch():
    row,rgb=naip_fixture();record=row['source']
    with raster(rgb,record) as memory,memory.open() as ds:
        context,useful,*_=adapter.canonical_from_native(ds,record)
        assert context.shape==(3,544,544) and context.dtype==np.float32
        assert useful==1280*1280*3
        np.testing.assert_allclose(context,100/255,atol=1e-7)
        from src.maritime_native_dataset import validate_aerial_native
        validate_aerial_native(ds,record,[40,40,1280,1280])


def test_naip_original_replay_checks_header_and_exact_rgb_without_fake_alpha():
    row,rgb=naip_fixture();record=row['source']
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',width=1360,height=1360,count=4,dtype='uint8',
                crs=record['source_crs'],transform=Affine(*record['original_transform'])) as ds:
            ds.write(rgb,indexes=[1,2,3]);ds.write(np.full((1360,1360),80,np.uint8),4)
            ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.undefined)
        with memory.open() as ds:adapter.verify_original_parent(ds,record,rgb)
        corrupt=rgb.copy();corrupt[0,50,50]+=1
        with memory.open() as ds:
            with pytest.raises(ValueError,match='original'):adapter.verify_original_parent(ds,record,corrupt)
        inconsistent=copy.deepcopy(record)
        inconsistent['source']['original_raster_header']['mask_flags'][0]=['nodata']
        with memory.open() as ds:
            with pytest.raises(ValueError,match='header'):adapter.verify_original_parent(ds,inconsistent,rgb)


def test_fractional_point_six_metre_area_geometry_and_saved_hr_are_exact():
    import io
    row,_=naip_fixture();record=row['source'];source=record['source']
    transform=Affine(.6,0,-40,0,-.6,1320)
    grid=SimpleNamespace(crs=CRS.from_epsg(6340),transform=transform,width=2267,height=2267)
    record.update(core_geometry(grid,0,0));record['original_transform']=source['transform_affine']=list(transform)[:6]
    source['shape']=[2267,2267];source['native_gsd_m']=source['delivered_gsd_m']=.6
    record['native_gsd_m']=record['delivered_gsd_m']=.6
    record['native_window_transform']=list(transform)[:6]
    rgb=np.full((3,2267,2267),100,np.uint8)
    x,y,w,h=record['owned_core_pixel_window']
    record['decoded_native_context_sha256']=digest_bytes(rgb.tobytes())
    record['decoded_useful_core_sha256']=digest_bytes(rgb[:,y:y+h,x:x+w].tobytes())
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',width=2267,height=2267,count=3,dtype='uint8',
                crs=record['source_crs'],transform=transform) as ds:
            ds.write(rgb);ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
        with memory.open() as ds:
            context,useful,*_=adapter.canonical_from_native(ds,record)
            assert useful==2133*2133*3
            np.testing.assert_allclose(context,100/255,atol=1e-7)
            payload=io.BytesIO();np.savez_compressed(payload,hr_context=context,hr=context[:,16:-16,16:-16])
            reproduced,*_=adapter.canonical_from_native(ds,record,payload.getvalue())
            assert np.array_equal(context,reproduced)


def test_original_nodata_hole_cannot_be_hidden_by_captured_valid_rgb():
    row,rgb=naip_fixture();record=row['source'];header=record['source']['original_raster_header']
    header.update(nodata=0,mask_flags=[['nodata']]*4)
    record['original_void_mask_available']=record['source']['original_void_mask_available']=True
    original=rgb.copy();original[:,80,80]=0
    with MemoryFile() as memory:
        with memory.open(driver='GTiff',width=1360,height=1360,count=4,dtype='uint8',nodata=0,
                crs=record['source_crs'],transform=Affine(*record['original_transform'])) as ds:
            ds.write(original,indexes=[1,2,3]);ds.write(np.full((1360,1360),80,np.uint8),4)
            ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.undefined)
        with memory.open() as ds:
            with pytest.raises(ValueError,match='support'):adapter.verify_original_parent(ds,record,rgb)


def test_all_valid_masks_cannot_rescue_original_zero_padding():
    row,rgb=naip_fixture();record=row['source'];rgb[:,80,80]=0
    record['decoded_native_context_sha256']=digest_bytes(rgb.tobytes())
    record['decoded_useful_core_sha256']=digest_bytes(rgb[:,40:-40,40:-40].tobytes())
    with raster(rgb,record) as memory,memory.open() as ds:
        assert np.all(ds.read_masks()==255)
        with pytest.raises(ValueError,match='RGB000'):adapter.canonical_from_native(ds,record)


def test_naip_stac_capture_calendar_cannot_be_called_publication():
    row,_=naip_fixture();row['source']['publication_datetime']=row['source']['stac_datetime']
    with pytest.raises(ValueError,match='calendar'):adapter.validate_source(row['source'])
