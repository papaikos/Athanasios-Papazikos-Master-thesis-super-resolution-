import numpy as np
import pytest
from scripts.maritime_earthview_quality import quality_diagnostics


def test_flat_image_has_no_structure_and_is_only_flagged():
    result = quality_diagnostics(np.full((3,512,512), .5, dtype=np.float32))
    assert result['luminance_std'] == 0
    assert 'LOW_SPATIAL_STRUCTURE_REVIEW' in result['flags']
    assert result['quality_accepted'] is False


def test_chromatic_noise_exceeds_coherent_gray_structure():
    rng=np.random.default_rng(19)
    noise=np.clip(.5+rng.normal(0,.1,(3,512,512)),0,1).astype('float32')
    x=np.arange(512)[None,:]
    gray=np.broadcast_to(.5+.2*np.sin(x/20),(512,512))
    structure=np.stack([gray]*3).astype('float32')
    before=noise.copy()
    n=quality_diagnostics(noise);s=quality_diagnostics(structure)
    assert n['chromatic_highpass_rms'] > s['chromatic_highpass_rms']+.03
    assert s['structure_variance_fraction_sigma4'] > n['structure_variance_fraction_sigma4']
    assert 'CHROMATIC_FINE_NOISE_REVIEW' in n['flags']
    np.testing.assert_array_equal(noise,before)


@pytest.mark.parametrize('bad',[np.zeros((3,400,400)),np.full((3,512,512),np.nan),np.ones((3,512,512))*2])
def test_only_canonical_hr_supported(bad):
    with pytest.raises(ValueError):quality_diagnostics(bad)


def test_spectral_flatness_and_chromatic_directionality_are_descriptive():
    rng=np.random.default_rng(21)
    noise=np.clip(.5+rng.normal(0,.07,(3,512,512)),0,1).astype('float32')
    stripes=np.full((3,512,512),.5,dtype='float32')
    stripes[0]+=.1*np.sin(np.arange(512)[None,:]/2)
    n=quality_diagnostics(noise);s=quality_diagnostics(stripes)
    assert n['chromatic_gradient_direction_fraction'] < .6
    assert s['chromatic_gradient_direction_fraction'] > .99
    assert n['luminance_high_frequency_spectral_flatness'] > s['luminance_high_frequency_spectral_flatness']
    assert set(n['structure_variance_fractions'])=={'sigma1','sigma2','sigma4','sigma8'}


def test_population_review_policy_never_accepts_and_preserves_explicit_exclusion():
    from scripts.maritime_earthview_quality import review_policy
    m=quality_diagnostics(np.full((3,512,512),.5,dtype=np.float32))
    assert review_policy(m,True)['status']=='EXCLUDE_OBSERVED_QUALITY_DEFECT'
    assert review_policy(m,False)['status']=='EXCLUDE_CONSERVATIVE_DIAGNOSTIC_FLAGS'
    clear=dict(m,flags=[],luminance_high_frequency_spectral_flatness=.1)
    assert review_policy(clear,False)['status']=='REQUIRES_FINAL_STRATIFIED_REVIEW'
    assert review_policy(clear,False)['quality_accepted'] is False


def test_actual_hr_review_selection_includes_all_defects_and_pending_plus_ten_controls():
    from scripts.maritime_earthview_quality import select_actual_hr_review
    audit=dict(rows=[dict(sample_id=str(i),status=s,annotation_id=str(i)) for i,s in enumerate(
        ['EXCLUDE_OBSERVED_QUALITY_DEFECT']*17+['QUALITY_REVIEW_PENDING']*14+['NO_OBVIOUS_OVERVIEW_DEFECT']*49)])
    selected=select_actual_hr_review(audit)
    assert len(selected)==41
    assert {r['sample_id'] for r in selected}.issuperset({str(i) for i in range(31)})
    assert selected==select_actual_hr_review(audit)


@pytest.mark.parametrize('kind',['skysat','skysat_raw_window','maxar'])
def test_other_sources_use_existing_canonical_dispatch_without_acceptance(monkeypatch,kind):
    from scripts import maritime_ultimate_release as release
    from scripts.maritime_earthview_quality import canonical_source_diagnostics
    context=np.full((3,544,544),.5,dtype='float32');context[:,0,:]=.7
    before=context.copy();calls=[]
    def process(store,row,seed):
        calls.append((store,row,seed));return dict(sample_id='sample'),context,b'ignored',{}
    monkeypatch.setattr(release,'process_sample',process)
    row=dict(sample_id='sample',source_kind=kind);store=object()
    hr,metrics,evidence=canonical_source_diagnostics(store,row,20261006)
    assert calls==[(store,row,20261006)]
    assert hr.shape==(3,512,512)
    assert metrics['luminance_std']==0
    assert evidence['quality_accepted'] is False
    assert evidence['status']=='DESCRIPTIVE_SOURCE_CALIBRATION_PENDING'
    np.testing.assert_array_equal(context,before)


def test_policy_revision_preserves_observations_and_prior_measurement():
    from scripts.maritime_earthview_quality import revise_quality_result
    import copy
    old=dict(schema='earthview-provisional-quality-result-v1',sample_id='observed',
             metrics=dict(flags=[],luminance_high_frequency_spectral_flatness=.1),
             candidate_pin={'sha256':'candidate'},hr_pin={'sha256':'hr'},
             decoded_canonical_hr_float32_sha256='pixels',config_pin={'sha256':'old'})
    before=copy.deepcopy(old);measurement_pin={'sha256':'measurement'}
    new=revise_quality_result(old,measurement_pin,{'sha256':'new'},{'observed'})
    assert old==before
    assert new['metrics']==old['metrics'] and new['hr_pin']==old['hr_pin']
    assert new['decoded_canonical_hr_float32_sha256']=='pixels'
    assert new['prior_measurement_pin']==measurement_pin
    assert new['prior_config_pin']==old['config_pin']
    assert new['policy']['status']=='EXCLUDE_OBSERVED_QUALITY_DEFECT'
    assert new['policy']['final_eligible'] is False


def test_explicit_revision_requires_all_prior_exclusions_and_unique_31_ids():
    from scripts.maritime_earthview_quality import validate_explicit_revision
    prior={str(i) for i in range(17)}
    audit=dict(explicit_excluded_sample_ids=[str(i) for i in range(31)])
    assert validate_explicit_revision(audit,prior)=={str(i) for i in range(31)}
    with pytest.raises(ValueError):validate_explicit_revision(audit,{'missing'})
    with pytest.raises(ValueError):validate_explicit_revision(dict(explicit_excluded_sample_ids=['x']*31),prior)


def test_bright_neutral_spatial_metrics_flag_white_regions_not_dark_water():
    from scripts.maritime_earthview_quality import bright_neutral_diagnostics
    image=np.full((3,512,512),.1,dtype='float32');image[:,:256,:]=.9
    before=image.copy();result=bright_neutral_diagnostics(image)
    assert result['bright_neutral_fraction']==.5
    assert result['largest_bright_neutral_component_fraction']==.5
    assert result['bright_neutral_patch_fraction']==.5
    assert result['flags']==['EXTENSIVE_BRIGHT_NEUTRAL_REGION_REVIEW']
    assert bright_neutral_diagnostics(before*.1)['flags']==[]
    np.testing.assert_array_equal(image,before)


def test_bright_colored_land_is_not_neutral_and_small_white_target_is_not_extensive():
    from scripts.maritime_earthview_quality import bright_neutral_diagnostics
    image=np.zeros((3,512,512),dtype='float32');image[0]=.95;image[1:]=.6
    assert bright_neutral_diagnostics(image)['bright_neutral_fraction']==0
    image[:]=.1;image[:,:8,:8]=.95
    result=bright_neutral_diagnostics(image)
    assert result['largest_bright_neutral_component_fraction']==64/(512*512)
    assert result['flags']==[]


def test_quality_candidate_projection_preserves_target_pins_and_drops_large_source_payload():
    from scripts.maritime_earthview_quality import quality_candidate_projection
    row=dict(sample_id='sample',source_kind='earthview',final_eligible=False,
             hr={'sha256':'hr'},port={'name':'port'},acquisition_identity=['time'],
             source_items=[{'original_header':'large'}]*16)
    projected=quality_candidate_projection(row)
    assert projected['hr']==row['hr'] and projected['acquisition_identity']==row['acquisition_identity']
    assert 'source_items' not in projected and len(row['source_items'])==16
    with pytest.raises(ValueError):quality_candidate_projection(dict(row,source_kind='maxar'))


def test_development_selection_has_fixed_strata_and_never_reintroduces_known_rejects():
    from scripts.maritime_earthview_quality import select_development_quality_review
    rows=[]
    for i in range(251):
        status='REQUIRES_FINAL_STRATIFIED_REVIEW' if i<160 else 'EXCLUDE_CONSERVATIVE_DIAGNOSTIC_FLAGS'
        rows.append(dict(sample_id=str(i),candidate_pin={'sha256':str(i)},policy={'status':status},
                         port={'country':str(i%7),'id':str(i%17),'size':'L' if i%2 else 'S'},
                         acquisition_identity=[f'2022-01-{i%28+1:02d}T00:00:00Z'],
                         metrics=dict(chromatic_highpass_rms=.001+i*.0001,
                                      luminance_high_frequency_spectral_flatness=.1,
                                      structure_variance_fraction_sigma4=.5)))
    excluded={str(i) for i in range(220,251)}
    selected=select_development_quality_review(rows,excluded)
    assert len(selected)==160 and len({r['sample_id'] for r in selected})==160
    assert not excluded.intersection(r['sample_id'] for r in selected)
    from collections import Counter
    assert Counter(r['development_review_stratum'] for r in selected)=={'unflagged':120,'flagged_borderline':40}
    assert Counter(r['selection_reason'].get('risk_quantile_bin') for r in selected if r['development_review_stratum']=='unflagged')=={0:30,1:30,2:30,3:30}
    assert selected==select_development_quality_review(list(reversed(rows)),excluded)


def test_development_model_hint_requires_exact_candidate_pin():
    from scripts.maritime_earthview_quality import select_development_quality_review
    row=dict(sample_id='one',candidate_pin={'sha256':'correct'})
    with pytest.raises(ValueError,match='candidate'):
        select_development_quality_review([row],set(),{'one':dict(candidate_pin={'sha256':'wrong'},predicted_medium_large=True)})

def test_bounded_ordered_map_keeps_lazy_input_and_releases_full_payloads():
    from scripts.maritime_earthview_quality import bounded_ordered_map
    produced=0
    consumed=0
    def source():
        nonlocal produced
        for i in range(100):
            produced+=1
            assert produced-consumed<=6
            yield i
    outputs=[]
    for result in bounded_ordered_map(lambda i:i*2,source(),workers=2,max_pending=6):
        consumed+=1;outputs.append(result)
    assert outputs==list(range(0,200,2))
    with pytest.raises(ValueError):list(bounded_ordered_map(lambda x:x,[],workers=2,max_pending=1))


def test_measurement_function_fingerprint_ignores_other_module_edits_but_rejects_changed_science():
    from scripts.maritime_earthview_quality import measurement_function_fingerprint
    a='def quality_diagnostics(hr):\n    return hr * 2\n\ndef other():\n    pass\n'
    b='def quality_diagnostics(hr):\n    return hr * 2\n\ndef other():\n    return 42\n'
    assert measurement_function_fingerprint(a)==measurement_function_fingerprint(b)
    assert measurement_function_fingerprint(a)!=measurement_function_fingerprint(a.replace('* 2','* 3'))
    with pytest.raises(ValueError):measurement_function_fingerprint('def other():\n    pass\n')

def test_bounded_projection_does_not_accumulate_full_source_json_payloads():
    import tracemalloc
    from scripts.maritime_earthview_quality import bounded_ordered_map,quality_candidate_projection
    def freeze(i):
        full=dict(sample_id=str(i),source_kind='earthview',final_eligible=False,hr={'sha256':str(i)},port={'name':'port'},acquisition_identity=['time'],source_items=bytearray(250000))
        return {'sha256':'full-source-pin'},quality_candidate_projection(full)
    tracemalloc.start()
    projections=list(bounded_ordered_map(freeze,range(1000),workers=4,max_pending=8))
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    assert len(projections)==1000 and peak<10_000_000
    assert all('source_items' not in r for _,r in projections)


def development_audit_fixture():
    prior=['old'+str(i) for i in range(31)]
    new=['new'+str(i) for i in range(44)]
    return dict(schema='root-earthview-development-actual-hr-visual-audit-v1',previous_excluded_sample_ids=prior,newly_excluded_sample_ids=new,explicit_excluded_sample_ids=prior+new,marine_content_unresolved_sample_ids=['hold'+str(i) for i in range(25)],quality_accepted_count=0,final_eligible=False),prior


def test_development_revision_preserves75_locked_and25_disjoint_marine_holds():
    from scripts.maritime_earthview_quality import validate_development_revision
    audit,prior=development_audit_fixture();excluded,held=validate_development_revision(audit,prior)
    assert len(excluded)==75 and len(held)==25 and not excluded&held
    audit['explicit_excluded_sample_ids'][0]='replacement'
    with pytest.raises(ValueError):validate_development_revision(audit,prior)


def test_development_revision_rejects_hold_overlap_and_false_acceptance():
    from scripts.maritime_earthview_quality import validate_development_revision
    audit,prior=development_audit_fixture();audit['marine_content_unresolved_sample_ids'][0]='old0'
    with pytest.raises(ValueError):validate_development_revision(audit,prior)
    audit,prior=development_audit_fixture();audit['quality_accepted_count']=1
    with pytest.raises(ValueError):validate_development_revision(audit,prior)


def test_development_policy_holds_marine_without_accepting_unflagged():
    from scripts.maritime_earthview_quality import development_review_policy
    metrics=dict(flags=[],luminance_high_frequency_spectral_flatness=.1)
    assert development_review_policy(metrics,'old0',{'old0'},{'hold0'})['status']=='EXCLUDE_OBSERVED_QUALITY_DEFECT'
    policy=development_review_policy(metrics,'hold0',{'old0'},{'hold0'})
    assert policy['status']=='HOLD_MARINE_CONTENT_UNRESOLVED' and not policy['quality_accepted'] and not policy['final_eligible']
    assert development_review_policy(metrics,'clear',{'old0'},{'hold0'})['status']=='REQUIRES_FINAL_STRATIFIED_REVIEW'
