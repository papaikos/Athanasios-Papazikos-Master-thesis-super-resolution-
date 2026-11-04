import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from scripts.maritime_cloud_inventory import digest_bytes
from src.maritime_native_dataset import native_patch


def test_original_native_patch_values_and_position_are_preserved(tmp_path):
    rgb=np.arange(3*80*80,dtype=np.uint8).reshape(3,80,80)
    p=tmp_path/'original.tif'
    with rasterio.open(p,'w',driver='GTiff',width=80,height=80,count=3,dtype='uint8',
        crs='EPSG:32631',transform=from_origin(0,64,.8,.8),photometric='RGB') as ds:ds.write(rgb)
    with rasterio.open(p) as ds:
        image,meta=native_patch(ds,[10,10,60,60],16,'a',0,1)
        x,y=meta['original_pixel_offset']
        np.testing.assert_array_equal(image,rgb[:,y:y+16,x:x+16].astype(np.float32)/255)
        assert 10<=x<=54 and 10<=y<=54
        assert meta['product_spacing_m']==.8


def test_original_mask_hole_rejected(tmp_path):
    p=tmp_path/'original.tif'
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),rasterio.open(p,'w',driver='GTiff',width=60,height=60,
        count=3,dtype='uint8',crs='EPSG:32631',transform=from_origin(0,60,.5,.5),photometric='RGB') as ds:
        ds.write(np.ones((3,60,60),np.uint8));ds.write_mask(np.zeros((60,60),np.uint8))
    with rasterio.open(p) as ds:
        with pytest.raises(ValueError,match='mask'):native_patch(ds,[0,0,60,60],32,'a',0,1)


def test_never_upscales_small_native_cores(tmp_path):
    p=tmp_path/'original.tif'
    with rasterio.open(p,'w',driver='GTiff',width=20,height=20,count=3,dtype='uint8',
        crs='EPSG:32631',transform=from_origin(0,20,.5,.5),photometric='RGB') as ds:ds.write(np.ones((3,20,20),np.uint8))
    with rasterio.open(p) as ds:
        with pytest.raises(ValueError,match='fit'):native_patch(ds,[0,0,20,20],32,'a',0,1)


@pytest.mark.parametrize('mapping_kind',['fixed','whole-mosaic'])
def test_uint16_native_patch_uses_saved_mapping_without_per_patch_fit(tmp_path,mapping_kind):
    from scripts.maritime_earthview_intake import TOA_DISPLAY_MAPPING,TOA_ROBUST_RECIPE,apply_toa_mapping
    import copy
    dn=np.arange(3*80*80,dtype=np.uint16).reshape(3,80,80)
    mapping=copy.deepcopy(TOA_DISPLAY_MAPPING if mapping_kind=='fixed' else TOA_ROBUST_RECIPE)
    if mapping_kind=='whole-mosaic': mapping.update(low_dn=[100,200,300],high_dn=[12000,14000,16000])
    saved=copy.deepcopy(mapping)
    path=tmp_path/'original-toa.tif'
    with rasterio.open(path,'w',driver='GTiff',width=80,height=80,count=3,dtype='uint16',
        crs='EPSG:32631',transform=from_origin(0,80,1,1),photometric='RGB') as ds:
        ds.write(dn)
    before=path.read_bytes()
    with rasterio.open(path) as ds:
        image,metadata=native_patch(ds,[10,10,60,60],16,'toa',0,1,rgb_mapping=mapping)
        x,y=metadata['original_pixel_offset']
        expected=apply_toa_mapping(dn[:,y:y+16,x:x+16],saved).astype(np.float32)/255
        np.testing.assert_array_equal(image,expected)
        assert metadata['original_rgb_dtype']=='uint16'
        assert metadata['output_representation']=='normalized_uint8_display_rgb_derivative'
        assert metadata['rgb_mapping']==saved
    assert mapping==saved and path.read_bytes()==before


def test_uint16_without_saved_mapping_is_rejected(tmp_path):
    path=tmp_path/'original-toa.tif'
    with rasterio.open(path,'w',driver='GTiff',width=40,height=40,count=3,dtype='uint16',
        crs='EPSG:32631',transform=from_origin(0,40,1,1),photometric='RGB') as ds:
        ds.write(np.ones((3,40,40),np.uint16))
    with rasterio.open(path) as ds:
        with pytest.raises(ValueError,match='mapping'):
            native_patch(ds,[0,0,40,40],16,'toa',0,1)


def test_earthview_native_context_geometry_is_checked_against_raster(tmp_path):
    from src.maritime_native_dataset import validate_earthview_native
    from scripts.maritime_earthview_intake import TOA_DISPLAY_MAPPING
    path=tmp_path/'synthetic-earthview.tif'
    affine=from_origin(100,2000,1,1)
    with rasterio.open(path,'w',driver='GTiff',width=1360,height=1360,count=3,dtype='uint16',
        crs='EPSG:32631',transform=affine,photometric='RGB') as ds:
        ds.write(np.ones((3,1360,1360),np.uint16))
    source=dict(schema='earthview-candidate-v1',source_kind='earthview',source_rgb_dtype='uint16',
        product_spacing_m=1.,native_context_transform=list(affine)[:6],source_crs='EPSG:32631',
        physical_context_bounds=[100,640,1460,2000],physical_core_bounds=[140,680,1420,1960],
        original_mask_provenance='original_declared_nodata_zero_rgb_band_masks',rgb_mapping=TOA_DISPLAY_MAPPING)
    with rasterio.open(path) as ds:
        validate_earthview_native(ds,source,[40,40,1280,1280])
        source['physical_core_bounds'][0]=141
        with pytest.raises(ValueError,match='physical'):
            validate_earthview_native(ds,source,[40,40,1280,1280])
