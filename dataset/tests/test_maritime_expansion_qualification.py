import copy
import pytest

from scripts.maritime_expansion_qualification import normalize_review, qualify_reviewed, locked_review_ids


def witness(**changes):
    row = dict(sample_id='patch', source='naip_aerial', candidate_sha256='a'*64,
               hr_float32_sha256='b'*64, verdict='pass', actual_image_review=True,
               reviewer='Codex assistant', observation='Visible useful shoreline',
               useful_marine_context_observed=True, physical_ship_size_verified=False)
    row.update(changes)
    return row


def candidate():
    return dict(sample_id='patch', source_kind='aerial_rgb', source={'immutable': 'pixels'},
                screening={'eligible': False})


def test_exact_observed_pass_qualifies_quality_without_granting_release_readiness():
    row = candidate(); old = copy.deepcopy(row)
    out = qualify_reviewed(row, 'a'*64, 'b'*64, [witness()], source_family='naip_aerial')
    assert row == old
    assert out['screening']['eligible'] is True
    assert out['screening']['priority'] == 3
    assert out['screening']['semantic_evidence'] == 'assistant_observed_marine_context'
    assert out['qualification']['dataset_ready'] is False
    assert out['qualification']['native_replay_required_at_release'] is True
    assert out['qualification']['certified_medium_large_ship'] is False


@pytest.mark.parametrize('change', [dict(verdict='hold'), dict(verdict='reject'),
    dict(actual_image_review=False), dict(useful_marine_context_observed=False)])
def test_missing_or_negative_image_evidence_never_qualifies(change):
    out = qualify_reviewed(candidate(), 'a'*64, 'b'*64, [witness(**change)], source_family='naip_aerial')
    assert out['screening']['eligible'] is False


def test_conflicting_hold_and_prior_rejection_override_new_pass():
    rows = [witness(), witness(verdict='hold')]
    assert not qualify_reviewed(candidate(), 'a'*64, 'b'*64, rows, source_family='naip_aerial')['screening']['eligible']
    assert not qualify_reviewed(candidate(), 'a'*64, 'b'*64, [witness()], source_family='naip_aerial', locked_ids={'patch'})['screening']['eligible']


@pytest.mark.parametrize('change', [dict(candidate_sha256='c'*64), dict(hr_float32_sha256='c'*64), dict(source='maxar')])
def test_identity_or_raster_mismatch_fails_closed(change):
    with pytest.raises(ValueError, match='mismatch'):
        qualify_reviewed(candidate(), 'a'*64, 'b'*64, [witness(**change)], source_family='naip_aerial')


def test_normalization_of_earth_assistant_preserves_real_source_and_no_fake_ship_truth():
    record = dict(sample_id='patch', candidate_pin={'sha256':'a'*64}, hr_float32_sha256='b'*64,
                  decision='PASS', pixel_quality_pass=True, useful_marine_context_observed=True,
                  observation='Clear coast', minor_cloud_global_policy_pending=True)
    out = normalize_review(record, source_family='earthview', reviewer='reviewer', actual_image_document=True)
    assert out['verdict'] == 'pass'
    assert out['actual_image_review'] is True
    assert out['minor_cloud_observed'] is True
    assert out['physical_ship_size_verified'] is False


def test_no_prediction_only_doc_can_become_actual_image_evidence():
    row = witness(actual_image_review=False)
    assert normalize_review(row, source_family='naip_aerial', reviewer='model', actual_image_document=False)['actual_image_review'] is False


def test_saved_satellite_descriptive_categories_keep_visual_ship_priority_and_minor_cloud():
    row = normalize_review(witness(category='vessel_form_visible', quality_code='pass_minor_cloud'),
                           source_family='naip_aerial', reviewer='assistant')
    assert row['minor_cloud_observed'] is True
    out = qualify_reviewed(candidate(), 'a'*64, 'b'*64, [row], source_family='naip_aerial')
    assert out['screening']['priority'] == 1
    assert out['qualification']['certified_medium_large_ship'] is False


@pytest.mark.parametrize('decision,verdict', [('PASS_VESSEL_UNVERIFIED_SIZE','pass'),
    ('PASS_SMALL_MARINE','pass'), ('PASS_MINOR_CLOUD_PENDING_GLOBAL_RULE','pass'),
    ('REJECT_CLOUD_LAND','reject'), ('REJECT_CHROMATIC_ARTIFACT','reject')])
def test_earth_saved_reason_codes_are_normalized_without_losing_original_decision(decision, verdict):
    row = witness(); row.pop('verdict'); row['decision'] = decision
    out = normalize_review(row, source_family='naip_aerial', reviewer='assistant')
    assert out['decision'] == decision
    assert out['verdict'] == verdict


def test_original_and_alternate_export_identities_both_remain_locked():
    assert locked_review_ids([{'sample_id':'old', 'alternate_candidate_ids':['other']}]) == {'old','other'}
    assert locked_review_ids([{'candidate_id':'legacy', 'canonical_core_id':'core',
                              'alternate_candidate_ids':['other']}]) == {'legacy','core','other'}


def test_direct_earth_vessel_reason_gets_vessel_priority_without_size_certification():
    record=witness();record.pop('verdict');record['decision']='PASS_VESSEL_UNVERIFIED_SIZE'
    review=normalize_review(record,source_family='naip_aerial',reviewer='assistant')
    result=qualify_reviewed(candidate(),'a'*64,'b'*64,[review],source_family='naip_aerial')
    assert result['screening']['priority']==1
    assert result['qualification']['certified_medium_large_ship'] is False


def scoring(**changes):
    row=dict(schema='marine-operational-screening-result-v1',sample_id='patch',source='naip_aerial',
        candidate_sha256='a'*64,hr_float32_sha256='b'*64,hr_uint8_chw_sha256='c'*64,
        historical_water_fraction=.4,estimated_cloud_fraction=.05,
        finite_rgb=True,nonconstant_rgb=True,original_support_verified=True)
    row.update(changes)
    return row


def evidence_pin():
    return dict(uri='gs://evidence/result.json',generation='1',bytes=10,sha256='d'*64,crc32c='AAAAAA==')


def screened(**kwargs):
    from scripts.maritime_expansion_qualification import qualify_screened
    return qualify_screened(candidate(),'a'*64,'b'*64,kwargs.pop('score',scoring()),
        evidence_pin=evidence_pin(),policy_pin=evidence_pin(),source_family='naip_aerial',**kwargs)


def test_mixed_marine_nomination_remains_an_explicit_uncalibrated_screening_tier():
    result=screened()
    assert result['screening']['eligible'] is True
    assert result['qualification']['tier']=='automatic_operational_screening'
    assert result['qualification']['dataset_ready'] is False
    assert result['qualification']['actual_image_review'] is False
    assert result['qualification']['population_accuracy_estimate'] is False
    assert result['qualification']['selected_stratified_actual_image_audit_required'] is True


@pytest.mark.parametrize('change',[dict(historical_water_fraction=.04),dict(historical_water_fraction=.96),
    dict(estimated_cloud_fraction=.11),dict(finite_rgb=False),dict(nonconstant_rgb=False),
    dict(original_support_verified=False)])
def test_screened_tier_excludes_unqualified_proxy_quality_or_support(change):
    assert not screened(score=scoring(**change))['screening']['eligible']


@pytest.mark.parametrize('change',[dict(estimated_cloud_fraction=float('nan')),
    dict(historical_water_fraction=1.4),dict(hr_float32_sha256='e'*64)])
def test_screening_invalid_fraction_or_different_pixels_fail_closed(change):
    with pytest.raises(ValueError):screened(score=scoring(**change))


def test_explicit_relaxed_nomination_keeps_high_water_in_negative_quota():
    result=screened(score=scoring(historical_water_fraction=.98,estimated_cloud_fraction=.18),
        nomination_thresholds=dict(historical_water_fraction_min=.01,historical_water_fraction_max=.99,
                                  estimated_cloud_fraction_max=.20))
    assert result['screening']['eligible']
    assert result['screening']['composition']=='open_water_negative'
    assert result['screening']['priority']==5
    assert result['qualification']['nomination_thresholds']['estimated_cloud_fraction_max']==.20
    assert not result['qualification']['actual_image_review']


def test_relaxing_proxy_limits_never_overrides_actual_rejection():
    result=screened(score=scoring(historical_water_fraction=.02,estimated_cloud_fraction=.18),
        reviews=[witness(verdict='reject')],
        nomination_thresholds=dict(historical_water_fraction_min=.01,historical_water_fraction_max=.99,
                                  estimated_cloud_fraction_max=.20))
    assert not result['screening']['eligible']


def test_actual_reject_hold_and_prior_lock_override_automatic_nomination():
    assert not screened(locked_ids=['patch'])['screening']['eligible']
    assert not screened(reviews=[witness(verdict='hold')])['screening']['eligible']
    assert not screened(reviews=[witness(verdict='reject')])['screening']['eligible']


def test_exact_actual_positive_keeps_its_priority_when_model_cloud_estimate_disagrees():
    result=screened(score=scoring(estimated_cloud_fraction=.2),reviews=[witness(category='V')])
    assert result['screening']['eligible'] is True
    assert result['screening']['priority']==1
    assert result['qualification']['tier']=='assistant_actual_image_review'
