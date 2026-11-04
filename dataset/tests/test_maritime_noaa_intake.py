from affine import Affine
from scripts.maritime_noaa_intake import candidate_metadata
import numpy as np
import pytest
from scripts.maritime_maxar_intake import area_hr
from scripts.maritime_noaa_intake import aerial_area_hr
from scripts.maritime_noaa_intake import sources_for_role
from scripts.maritime_noaa_intake import planned_source_geometry, verified_core_plans

def test_metadata_core_preflight_matches_original_grid_and_rejects_changed_plan():
    source=dict(source_id='fixture',epsg=6340,transform_affine=[.25,0,384000,0,-.25,3737600],
        shape=[12040,12040],projected_bounds=[384000,3734590,387010,3737600],
        port_AOI_matches=[dict(port_id='fixture',longitude=-118.25,latitude=33.78)])
    ds,plans=planned_source_geometry(source)
    assert plans and all(p['source_pixel_window'][2:]==[5440,5440] for p in plans)
    doc={'core_plans':{'fixture':plans}}
    assert verified_core_plans(ds,source,doc)==plans
    import copy
    changed=copy.deepcopy(doc);changed['core_plans']['fixture'][0]['source_pixel_window'][0]+=1
    with pytest.raises(ValueError):verified_core_plans(ds,source,changed)
    assert verified_core_plans(ds,source,{'core_plans':{'fixture':[]}})==[]

def test_role_selection_defaults_training_and_never_consumes_held_parents():
    sources=[{'source_id':role} for role in ('train','evaluation','hold')]
    allocation={role:{'role':role} for role in ('train','evaluation','hold')}
    assert sources_for_role(sources,allocation)==[{'source_id':'train'}]
    assert sources_for_role(sources,allocation,'evaluation')==[{'source_id':'evaluation'}]
    with pytest.raises(ValueError):sources_for_role(sources,allocation,'hold')

@pytest.mark.parametrize('spacing,phase',[(.25,[0,0]),(.5,[0,0]),(.25,[.125,.125])])
def test_aerial_area_matches_frozen_canonical_bit_exact(spacing,phase):
    count=8;size=int(np.ceil((count*2.5+max(phase))/spacing))
    rgb=np.random.default_rng(619).integers(0,256,size=(3,size,size),dtype=np.uint8)
    valid=np.ones((size,size),dtype=bool)
    geometry=dict(product_spacing_m=spacing,area_phase_m=phase)
    actual,method=aerial_area_hr(rgb,valid,geometry,count)
    np.testing.assert_array_equal(actual,area_hr(rgb,valid,geometry,count))
    assert method==('integer_aligned_box_sum_uint32_v1' if phase==[0,0] else 'canonical_sparse_exact_area_v1')

def test_aerial_area_never_accepts_invalid_support():
    rgb=np.zeros((3,80,80),dtype=np.uint8);valid=np.ones((80,80),dtype=bool);valid[0,0]=False
    with pytest.raises(ValueError):aerial_area_hr(rgb,valid,dict(product_spacing_m=.25,area_phase_m=[0,0]),8)

def test_aerial_identity_geometry_and_radiometry_are_explicit():
    source=dict(source_id='10317:tile',epsg=6340,source_type='noaa_aerial',href='https://example.test',
        original_http_identity={'etag':'x','bytes':10},transform_affine=[.25,0,1,0,-.25,2],
        native_gsd_m=.25,delivered_gsd_m=.25,original_camera_capture_gsd_m=None,
        acquisition_day='2025-02-01',stac_datetime='2025-04-02',parent_tile_id='tile',
        capture_identity_type='dataset_acquisition_day_and_original_orthomosaic_tile')
    geometry=dict(world_grid_cell=[303,2918],physical_core_bounds=[387840,3735040,389120,3736320],
        physical_context_bounds=[387800,3735000,389160,3736360],source_pixel_window=[1,2,5440,5440],
        owned_core_pixel_window=[161,162,5120,5120],useful_native_rgb_bytes=78643200)
    row=candidate_metadata(source,geometry,Affine(.25,0,387800,0,-.25,3736360),{'uri':'original'}, {'uri':'evidence'})
    assert row['schema']=='aerial-rgb-candidate-v1' and row['source_kind']=='aerial_rgb'
    assert row['source_crs']==row['output_crs']=='EPSG:6340'
    assert row['original_camera_capture_gsd_m'] is None
    assert row['acquisition_day'] != row['publication_datetime']
    assert row['physical_core_bounds']==geometry['physical_core_bounds']
    assert not row['final_eligible'] and row['requires_semantic_review']
