import copy
import pytest
from scripts.maritime_noaa_source import normalize_item

def fixture():
    return {'id':'tile','properties':{'datetime':'2025-04-02T16:45:15Z','license':'NLPL'},
        'assets':{'tile':{'href':'https://coastalimagery.blob.core.windows.net/digitalcoast/tile.tif',
        'proj:transform':[395995,.25,0,3735005,0,-.25], 'proj:epsg':6340,'proj:shape':[12040,12040],
        'raster:bands':[{'data_type':'uint8'}]*5,
        'eo:bands':[{'common_name':c} for c in ('red','green','blue')]+[{'description':'gray'},{'description':'alpha'}]}}}, \
        {'id':'10317','license':'CC0-1.0','license_evidence_url':'https://www.fisheries.noaa.gov/inport/item/75459',
         'acquisition_day':'2025-02-01','sensor':'Applanix DSS'}

def test_real_noaa_transform_and_provenance():
    item,dataset=fixture();r=normalize_item(item,dataset)
    assert r['projected_bounds']==[395995,3731995,399005,3735005]
    assert r['transform_affine']==[.25,0,395995,0,-.25,3735005]
    assert r['alpha_index']==5 and r['license_conflict']
    assert r['acquisition_day']=='2025-02-01' and r['stac_datetime'].startswith('2025-04-02')
    assert not r['final_eligible'] and r['source_type']=='noaa_aerial'

@pytest.mark.parametrize('field,value', [('proj:transform',[0,2,0,0,0,-2]),
    ('proj:transform',[0,.25,.1,0,0,-.25]),('proj:shape',[1000,1000]),
    ('href','https://example.org/tile.tif')])
def test_ineligible_original_rejected(field,value):
    item,dataset=fixture();item['assets']['tile'][field]=value
    with pytest.raises(ValueError):normalize_item(item,dataset)

def test_missing_affirmative_license_rejected():
    item,dataset=fixture();dataset['license']='NLPL'
    with pytest.raises(ValueError):normalize_item(item,dataset)

def test_geographic_units_cannot_be_called_metres():
    item,dataset=fixture();item['assets']['tile']['proj:epsg']=4326
    with pytest.raises(ValueError):normalize_item(item,dataset)

def test_official_noaa_explicit_color_descriptions_without_common_name():
    item,dataset=fixture()
    for band,name in zip(item['assets']['tile']['eo:bands'][:3],('red','green','blue')):
        band.clear();band['description']=name
    assert normalize_item(item,dataset)['rgb_indexes']==[1,2,3]
