import pytest
from scripts.maritime_aerial_evaluation import site_allocation
from scripts.maritime_aerial_evaluation_release import validate_evaluation_selection


def evaluation():
    site=next(str(i) for i in range(1000) if site_allocation(str(i))=='evaluation')
    return dict(sample_id='eval',source_kind='aerial_rgb',site_id='aerial-'+site,
        acquisition_id='separate-flight',split='test',source=dict(output_crs='EPSG:32631',
        physical_core_bounds=[500000,0,501280,1280],
        source={'epsg':32631,'projected_bounds':[500000,0,503000,3000],
                'port_AOI_matches':[{'port_id':site}]},source_original_object={'sha256':'a'*64}))


def training(bounds=(600000,0,601280,1280)):
    return dict(sample_id='train',source_kind='maxar',site_id='prior',acquisition_id='prior-flight',split='train',
                crs='EPSG:32631',core_bounds=list(bounds))


def test_evaluation_remains_separate_and_does_not_count_toward_train_quota():
    rows=validate_evaluation_selection([evaluation()],[training()])
    assert rows[0]['split']=='test' and rows[0]['source_kind']=='aerial_rgb'


def test_geography_four_km_guard_even_across_source_platforms():
    with pytest.raises(ValueError,match='geography'):
        validate_evaluation_selection([evaluation()],[training((502000,0,503280,1280))])


def test_shared_parent_or_capture_cannot_enter_separate_evaluation():
    prior=training();prior['acquisition_id']='separate-flight'
    with pytest.raises(ValueError,match='capture'):validate_evaluation_selection([evaluation()],[prior])
    prior=training();prior['source']={'source_original_object':{'sha256':'a'*64}}
    with pytest.raises(ValueError,match='parent'):validate_evaluation_selection([evaluation()],[prior])


def test_training_role_or_duplicate_evaluation_is_rejected():
    row=evaluation();row['source']['source']['port_AOI_matches'][0]['port_id']=next(str(i) for i in range(1000) if site_allocation(str(i))=='train')
    with pytest.raises(ValueError,match='reservation'):validate_evaluation_selection([row],[training()])
    with pytest.raises(ValueError,match='Duplicate'):validate_evaluation_selection([evaluation(),evaluation()],[training()])


def test_normalized_parent_url_and_site_aliases_cannot_evade_exposure_gate():
    row=evaluation();row['source']['source_href']='https://original'
    prior=training();prior['source_parent_urls']=['https://original']
    with pytest.raises(ValueError,match='parent'):validate_evaluation_selection([row],[prior])
    prior=training();port=row['source']['source']['port_AOI_matches'][0]['port_id'];prior['site_aliases']=['maxar-port-'+port]
    with pytest.raises(ValueError,match='site'):validate_evaluation_selection([row],[prior])


def test_indexed_geography_keeps_authoritative_four_km_guard(monkeypatch):
    from shapely.geometry import box
    from shapely.strtree import STRtree
    from scripts import maritime_aerial_evaluation_release as release
    shapes=[box(1,i*3+10,2,i*3+11) for i in range(20)]+[box(0,0,.01,.01)]
    tree=STRtree(shapes);calls=[]
    def exact(a,b,d):
        calls.append(b)
        return b==shapes[-1]
    monkeypatch.setattr(release,'near',exact)
    assert release._near_exposure(box(0,0,.005,.005),shapes,tree)
    assert calls==[shapes[-1]]
    calls.clear();assert not release._near_exposure(box(0,5,.005,5.005),shapes,tree)
    assert calls==[]


@pytest.mark.parametrize('longitude,latitude',[(0,0),(179.995,20),(10,85.5)])
def test_indexed_guard_matches_exhaustive_at_dateline_and_poles(longitude,latitude):
    from shapely.geometry import box
    from shapely.strtree import STRtree
    from scripts.maritime_aerial_evaluation_release import _near_exposure
    from scripts.maritime_release_select import near
    a=box(longitude,latitude,longitude+.005,latitude+.005)
    b=box(longitude+.02,latitude,longitude+.025,latitude+.005)
    if longitude>170:b=box(-180,latitude,-179.995,latitude+.005)
    shapes=[b,box(longitude,latitude-3,longitude+.005,latitude-2.99)]
    assert _near_exposure(a,shapes,STRtree(shapes))==any(near(a,b,4000) for b in shapes)


def test_evaluation_context_halo_is_included_in_exposure_guard():
    row=evaluation();row['source']['physical_context_bounds']=[499960,-40,501320,1320]
    with pytest.raises(ValueError,match='geography'):
        validate_evaluation_selection([row],[training((505300,0,506580,1280))])
