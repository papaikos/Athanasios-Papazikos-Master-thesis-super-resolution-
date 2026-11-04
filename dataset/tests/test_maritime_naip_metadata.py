import pytest
from scripts.maritime_naip_source import normalize_item

def fixture():
    return dict(id='ca_m_3311824_nw_11_060_20220512',collection='naip',properties={
        'gsd':.6,'datetime':'2022-05-12T16:00:00Z','naip:year':'2022','naip:state':'ca',
        'proj:epsg':26911,'proj:shape':[12320,10440],
        'proj:transform':[.6,0,395526,0,-.6,3735204,0,0,1]},assets={'image':{
        'href':'https://naipeuwest.blob.core.windows.net/naip/v002/ca/2022/tile_20220512.tif',
        'eo:bands':[{'common_name':c} for c in ('red','green','blue','nir')]}}),dict(id='USDA-NAIP',license='Public-Domain',
        license_evidence_url='https://www.fsa.usda.gov/help/policies-and-links',
        corroborating_license_url='https://www.usgs.gov/media/images/tnmcorps-naip-imagery-layer',
        collection_license='proprietary',collection_license_link_title='Public Domain',
        stac_license='proprietary',license_conflict=True)

def test_original_naip_grid_day_license_conflict_and_unknown_camera_gsd():
    item,evidence=fixture();row=normalize_item(item,evidence)
    assert row['acquisition_day']=='2022-05-12' and row['alpha_index'] is None
    assert row['transform_affine']==[.6,0,395526,0,-.6,3735204]
    assert row['projected_bounds']==[395526,3727812,401790,3735204]
    assert row['license_conflict'] and row['stac_license']=='proprietary'
    assert row['original_camera_capture_gsd_m'] is None
    assert row['rgb_dtype_declaration']=='not_in_STAC_verified_at_intake'

@pytest.mark.parametrize('key,value',[('gsd',2),('proj:epsg',4326),
    ('proj:transform',[.6,.1,395526,0,-.6,3735204,0,0,1]),('proj:shape',[1000,1000]),
    ('datetime','2022-05-13T16:00:00Z')])
def test_invalid_original_grid_or_capture_conflict_rejected(key,value):
    item,evidence=fixture();item['properties'][key]=value
    with pytest.raises(ValueError):normalize_item(item,evidence)

def test_license_and_unsigned_original_required():
    item,evidence=fixture();evidence['license']='proprietary'
    with pytest.raises(ValueError):normalize_item(item,evidence)

def test_second_filename_date_is_preserved_separately_from_capture():
    item,evidence=fixture();item['id']+='_20240209'
    row=normalize_item(item,evidence)
    assert row['acquisition_day']=='2022-05-12'
    assert row['filename_second_date']=='2024-02-09'
    assert row['filename_second_date_semantics']=='additional_filename_date_not_camera_capture'
    item,evidence=fixture();item['assets']['image']['href']+='?sig=temporary'
    with pytest.raises(ValueError):normalize_item(item,evidence)
