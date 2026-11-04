import numpy as np
import hashlib
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.enums import ColorInterp
from scripts.maritime_naip_intake import original_header

def test_original_reuse_requires_same_parent_http_identity_and_full_byte_hash(tmp_path):
    from scripts.maritime_naip_intake import verify_reused_original
    path=tmp_path/'original.tif';path.write_bytes(b'original byte fixture')
    source=dict(source_id='naip:parent',original_http_identity={'etag':'original-etag','bytes':path.stat().st_size})
    receipt=dict(source_id=source['source_id'],original_http_identity=source['original_http_identity'],
        object=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    verify_reused_original(path,source,receipt)
    path.write_bytes(b'changed byte fixture!')
    with pytest.raises(ValueError):verify_reused_original(path,source,receipt)
    path.write_bytes(b'original byte fixture');receipt['source_id']='naip:other'
    with pytest.raises(ValueError):verify_reused_original(path,source,receipt)

def test_only_explicit_v2_plans_replay_actual_original_contributor_window():
    from types import SimpleNamespace
    from rasterio.crs import CRS
    from rasterio.coords import BoundingBox
    from rasterio.transform import Affine
    from rasterio.warp import transform
    from scripts.maritime_maxar_intake import planned_cores
    from scripts.maritime_naip_intake import verified_naip_core_plans
    from src.maritime_aerial_geometry import original_contributor_geometry
    t=Affine(.3,0,335037.9,0,-.3,4700590.8)
    ds=SimpleNamespace(transform=t,width=20000,height=6000,crs=CRS.from_epsg(26919),
        bounds=BoundingBox(t.c,t.f+6000*t.e,t.c+20000*t.a,t.f))
    lon,lat=transform(ds.crs,'EPSG:4326',[339840],[4699520])
    source=dict(source_id='naip:synthetic',port_AOI_matches=[dict(longitude=lon[0],latitude=lat[0])])
    old=next(g for g in planned_cores(ds,source) if g['world_grid_cell']==[265,3671])
    assert verified_naip_core_plans(ds,source,{'core_plans':{source['source_id']:[old]}})==[old]
    new=dict(old,**original_contributor_geometry(ds,265,3671))
    assert verified_naip_core_plans(ds,source,{'core_plans':{source['source_id']:[new]}})==[new]
    new['geometry_version']='unknown'
    with pytest.raises(ValueError):verified_naip_core_plans(ds,source,{'core_plans':{source['source_id']:[new]}})

def test_bulk_support_rejects_zero_rgb_even_when_original_masks_all_valid():
    from scripts.maritime_naip_intake import original_support
    rgb=np.ones((3,2,2),dtype=np.uint8); masks=np.full_like(rgb,255)
    rgb[:,0,0]=0
    valid=original_support(rgb,masks)
    assert not valid[0,0] and valid[1,1]
    rgb[0,0,0]=1
    assert original_support(rgb,masks).all()
    masks[2,1,1]=0
    assert not original_support(rgb,masks)[1,1]

def test_unsupported_frozen_contributor_geometry_is_rejected_before_pixel_processing():
    from scripts.maritime_naip_intake import contributor_geometry_supported
    geometry=dict(product_spacing_m=.6,area_phase_m=[.2,.20000000018626451],
        source_pixel_window=[483,4803,2267,2267])
    assert not contributor_geometry_supported(geometry)
    geometry['source_pixel_window'][-1]=2268
    assert contributor_geometry_supported(geometry)
    geometry['area_phase_m'][1]=-4.656612873077393e-10
    assert not contributor_geometry_supported(geometry)

def test_header_requires_original_uint8_rgb_and_records_real_mask_flags():
    source={'epsg':26911,'transform_affine':[.6,0,395526,0,-.6,3735204],'shape':[20,20]}
    with MemoryFile() as mem:
        with mem.open(driver='GTiff',width=20,height=20,count=4,dtype='uint8',crs='EPSG:26911',
            transform=from_origin(395526,3735204,.6,.6),nodata=0) as ds:
            ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.undefined)
            ds.write(np.ones((4,20,20),dtype=np.uint8));header=original_header(ds,source)
            assert header['nodata']==0 and header['count']==4
            assert header['mask_flags']==[['nodata']]*4
            assert header['colorinterp'][3]=='undefined'
