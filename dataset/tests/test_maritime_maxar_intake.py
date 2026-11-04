from types import SimpleNamespace
import math
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.transform import Affine
from scripts import maritime_maxar_intake as intake


def header(spacing=.8):
    return SimpleNamespace(crs=CRS.from_epsg(32619), transform=Affine(spacing, 0, 0, 0, -spacing, 7680),
        count=3, dtypes=('uint8',)*3, colorinterp=(ColorInterp.red, ColorInterp.green, ColorInterp.blue),
        mask_flag_enums=([MaskFlags.per_dataset],)*3, width=20000, height=20000)


@pytest.mark.parametrize('gsd', [None, float('nan'), 1.01, 0, -1])
def test_reject_unknown_coarse_native_gsd(gsd):
    with pytest.raises(ValueError, match='GSD'):
        intake.validate_header(header(), gsd)


def test_reject_gray_unresolved_masks_and_product_upsampling():
    ds=header(); ds.colorinterp=(ColorInterp.gray,)*3
    with pytest.raises(ValueError, match='RGB'):
        intake.validate_header(ds,.8)
    ds=header(); ds.mask_flag_enums=([MaskFlags.nodata],)*3
    with pytest.raises(ValueError, match='mask'):
        intake.validate_header(ds,.8)
    with pytest.raises(ValueError, match='spacing'):
        intake.validate_header(header(2),.8)


def test_native_fractional_context_and_halfopen_core_pixel_ownership():
    ds=header(.30517578125)
    a=intake.core_geometry(ds, 1, 4)
    b=intake.core_geometry(ds, 2, 4)
    assert a['physical_core_bounds'] == [1280,5120,2560,6400]
    assert a['physical_context_bounds'] == [1240,5080,2600,6440]
    owned=a['owned_core_pixel_window']; nxt=b['owned_core_pixel_window']
    assert owned[0]+owned[2] == nxt[0]
    assert a['useful_native_rgb_bytes'] == owned[2]*owned[3]*3
    assert a['source_pixel_window'][2] > owned[2]
    assert 0<=a['area_phase_m'][0]<.30517578125
    assert 0<=a['area_phase_m'][1]<.30517578125
    assert a['hr_context_transform'][0] == 2.5
    assert a['hr_context_transform'][2] == 1240


def test_exact_area_preserves_real_black_and_rejects_one_invalid_contributor():
    geometry={'area_phase_m':[.1,.2],'product_spacing_m':.8}
    rgb=np.zeros((3,13,13),np.uint8)
    valid=np.ones((13,13),bool)
    result=intake.area_hr(rgb,valid,geometry,count=4)
    assert result.shape==(3,4,4) and result.dtype==np.float32 and not result.any()
    valid[12,12]=False
    with pytest.raises(ValueError, match='support'):
        intake.area_hr(rgb,valid,geometry,count=4)


def test_exact_area_uses_fractional_physical_overlap_without_uint8_rounding():
    geometry={'area_phase_m':[.1,.2],'product_spacing_m':.8}
    rgb=np.zeros((3,13,13),np.uint8); rgb[:,0,0]=255
    result=intake.area_hr(rgb,np.ones((13,13),bool),geometry,count=4)
    assert result[0,0,0] == pytest.approx(.7*.6/(2.5*2.5), abs=1e-8)
    assert result[0,0,0]*255 != round(float(result[0,0,0]*255))


def test_native_tiff_preserves_rgb_pixels_transform_and_original_black_mask(tmp_path):
    pixels=np.zeros((3,32,32),np.uint8);pixels[0,5,5]=97;pixels[1,5,5]=3
    valid=np.ones((32,32),bool)
    transform=Affine(.30517578125,0,1234,0,-.30517578125,5678)
    path=tmp_path/'native.tif'
    intake.write_native(path,pixels,valid,CRS.from_epsg(32619),transform)
    with rasterio.open(path) as ds:
        assert intake.validate_header(ds,.36)==.30517578125
        np.testing.assert_array_equal(ds.read(),pixels)
        assert ds.transform==transform
        assert ds.dataset_mask().min()==255
    assert not path.with_suffix('.tif.msk').exists()


def test_header_must_match_pinned_stac_origin_and_dimensions():
    ds=header()
    source={'epsg':32619,'visual_transform':list(ds.transform),'visual_shape':[ds.height,ds.width]}
    intake.validate_source_grid(ds,source)
    source['visual_transform'][2]+=10000
    with pytest.raises(ValueError,match='STAC'):
        intake.validate_source_grid(ds,source)
    source['visual_transform']=list(ds.transform)
    source['visual_shape'][1]-=1
    with pytest.raises(ValueError,match='STAC'):
        intake.validate_source_grid(ds,source)


def test_coarse_water_rejects_land_only_when_worldcover_coverage_known():
    values=np.zeros((17,17),dtype=[('water','u1'),('coverage','u1')])
    values['coverage']=1
    result=intake.water_screen_result(values)
    assert result['status']=='INLAND_REJECTED' and result['discard'] is True
    values['coverage'][:]=0
    result=intake.water_screen_result(values)
    assert result['status']=='UNKNOWN_COVERAGE' and result['discard'] is False
    assert result['maritime_support_known'] is False


def test_coarse_water_proxy_pass_is_not_a_vessel_or_stratum_label():
    values=np.zeros((17,17),dtype=[('water','u1'),('coverage','u1')]);values['coverage']=1
    values['water'][:4,:]=1
    result=intake.water_screen_result(values)
    assert result['status']=='WATER_PROXY_PASSED' and result['discard'] is False
    assert result['water_fraction']==pytest.approx(4/17)
    assert result['vessel_presence']=='unconfirmed' and result['maritime_stratum']=='unconfirmed'


def test_coarse_water_invalid_values_are_not_interpreted_as_land():
    values=np.zeros((17,17),dtype=[('water','u1'),('coverage','u1')]);values['coverage']=255
    with pytest.raises(ValueError,match='binary'):
        intake.water_screen_result(values)


def exclusion_manifest(bounds=None):
    bounds=bounds or [1280.,5120.,2560.,6400.]
    physical={'acquisition':'capture-a','crs':'EPSG:32619','physical_core_bounds':bounds}
    identifier=intake.digest_bytes(intake.json_bytes(physical))
    row={'source':{'catalog_id':'capture-a','epsg':32619},'physical_core_bounds':bounds,
         'physical_core_id':identifier,'sample_id':identifier}
    return {'schema':'maxar-mixed-native-v1','candidate_count':1,'candidates':[row]}


def test_exclusion_unions_prior_acquisition_cells_without_counting_aliases():
    a=exclusion_manifest();b=exclusion_manifest([2560.,5120.,3840.,6400.])
    identifiers=intake.excluded_core_ids([a,a,b])
    assert len(identifiers)==2
    assert a['candidates'][0]['physical_core_id'] in identifiers


def test_exclusion_rejects_tampered_identity_and_nonmaxar_schema():
    a=exclusion_manifest();a['candidates'][0]['physical_core_bounds'][0]+=10
    with pytest.raises(ValueError,match='identity'):
        intake.excluded_core_ids([a])
    a=exclusion_manifest();a['schema']='unknown-native'
    with pytest.raises(ValueError,match='schema'):
        intake.excluded_core_ids([a])


@pytest.mark.parametrize('seen_basis',['excluded','checkpointed'])
def test_intake_continues_after_fully_seen_ordinal_and_stops_at_exhaustion(tmp_path, monkeypatch,seen_basis):
    from contextlib import nullcontext
    from google.cloud import storage
    import json

    source={'source_id':'s','catalog_id':'capture-a','epsg':32619,'native_gsd_m':.8,
            'visual_href':'https://example.org/image.tif'}
    for key in ('item_metadata','event_metadata','acquisition_metadata'):
        source[key]={'sha256':intake.digest_bytes(b'{}'),'bytes':2}
    (tmp_path/(intake.digest_bytes(b'{}')+'.json')).write_bytes(b'{}')
    inventory=tmp_path/'inventory.json';inventory.write_text(json.dumps({'sources':[source]}))
    prior=exclusion_manifest()

    class Blob:
        generation=1;crc32c='crc'
        def reload(self): pass
        def exists(self): return False
        def download_as_bytes(self, **kwargs): return intake.json_bytes(prior)
    class Bucket:
        def blob(self, name): return Blob()
    monkeypatch.setattr(storage,'Client',lambda **kwargs:SimpleNamespace(bucket=lambda name:Bucket()))
    monkeypatch.setattr(intake,'require_cloud_vm',lambda:'vm')
    def create(blob,data):
        document=json.loads(data) if data.startswith(b'{') else {}
        if seen_basis=='checkpointed' and 'source_inventory_sha256' in document:
            row=dict(prior['candidates'][0],native_context={},hr={},overview={},useful_native_rgb_bytes=0)
            scratch=tmp_path/'scratch'
            (scratch/'checkpoint.json').write_bytes(intake.json_bytes(dict(
                configuration=document,records=[row],attempts=[],rejections=[])))
        return {}
    monkeypatch.setattr(intake,'create_only',create)
    monkeypatch.setattr(intake,'verify_output',lambda *args:None)
    monkeypatch.setattr(intake,'validate_source',lambda *args:None)
    monkeypatch.setattr(intake,'validate_header',lambda *args:None)
    monkeypatch.setattr(intake,'validate_source_grid',lambda *args:None)
    monkeypatch.setattr(intake,'pinned_http',lambda *args:{'etag':'etag'})
    monkeypatch.setattr(intake.rasterio,'Env',lambda **kwargs:nullcontext())
    monkeypatch.setattr(intake.rasterio,'open',lambda *args:nullcontext(object()))
    geometries=[{'physical_core_bounds':row['physical_core_bounds']} for row in
                (prior['candidates'][0],exclusion_manifest([2560.,5120.,3840.,6400.])['candidates'][0])]
    monkeypatch.setattr(intake,'planned_cores',lambda *args:geometries)
    screened=[]
    def reject(ee, expression, geometry, epsg):
        screened.append(geometry)
        return {'discard':True}
    monkeypatch.setattr(intake,'water_screen',reject)
    # Initialization failure must not prevent reaching the second ordinal.
    # Store a completed water screen for that cell, bypassing any EE dependency.
    second_id=exclusion_manifest([2560.,5120.,3840.,6400.])['candidates'][0]['physical_core_id']
    original_blob=Bucket.blob
    def blob(self, name):
        result=original_blob(self,name)
        if name.endswith(second_id+'.json'):
            result.exists=lambda:True
            def cached(**kwargs):
                screened.append(geometries[1]);return b'{"discard":true}'
            result.download_as_bytes=cached
        return result
    monkeypatch.setattr(Bucket,'blob',blob)
    args=SimpleNamespace(sources=str(inventory),metadata_dir=str(tmp_path),scratch=str(tmp_path/'scratch'),
                         output_prefix='gs://bucket/new',max_sources=1,max_cores=10,
                         coarse_water_screen=True,
                         exclude_manifest=['gs://bucket/prior'] if seen_basis=='excluded' else [])
    result=intake.run(args)
    assert screened==[geometries[1]]
    assert len(result['attempts'])==1 and result['attempts'][0]['status']=='REJECTED'
    assert result['candidate_count']==(1 if seen_basis=='checkpointed' else 0)
