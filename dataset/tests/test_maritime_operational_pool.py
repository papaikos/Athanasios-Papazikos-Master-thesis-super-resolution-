import copy
import hashlib
import pytest

from scripts.maritime_cloud_inventory import json_bytes
from scripts.maritime_operational_pool import bind_screened_aerial


def pin(digest='a'*64):
    return dict(uri='gs://bucket/object-'+digest,generation='1',bytes=10,crc32c='AAAAAA==',sha256=digest)


def inputs():
    raw=dict(sample_id='patch',source_type='naip_aerial',parent_tile_id='tile',acquisition_day='2020-01-01',
        nearest_port=dict(port_id='1',country_code_as_published='US'),output_crs='EPSG:32610',
        physical_core_bounds=[0,0,1280,1280],
        split_exclusion_evidence=dict(allocation=dict(role='train')))
    digest=hashlib.sha256(json_bytes(raw)).hexdigest()
    source=dict(raw,candidate_record=pin(digest))
    policy=dict(schema='operational-naip-screening-policy-v1',source_families=['naip_aerial'],
        thresholds=dict(historical_water_fraction_min=.05,historical_water_fraction_max=.95,
                        estimated_cloud_fraction_max=.10))
    score=dict(schema='marine-operational-screening-result-v1',sample_id='patch',source='naip_aerial',
        source_family='naip_aerial',source_candidate=source,candidate_record_pin=pin(digest),
        candidate_sha256=digest,hr_float32_sha256='b'*64,hr_uint8_chw_sha256='c'*64,
        historical_water_fraction=.4,estimated_cloud_fraction=.03,finite_rgb=True,
        nonconstant_rgb=True,original_support_verified=True,policy_pin=pin())
    return score,policy


def bind(score,policy,**kw):
    return bind_screened_aerial(score,evidence_pin=pin('d'*64),policy=policy,policy_pin=pin(),**kw)


def test_exact_pinned_train_source_wraps_without_mutation_or_ready_grant():
    score,policy=inputs();original=copy.deepcopy(score)
    row=bind(score,policy)
    assert score==original
    assert row['source']==score['source_candidate']
    assert row['site_id']=='aerial-port-1'
    assert row['hr_uint8_chw_sha256']=='c'*64
    assert row['screening']['eligible'] is True
    assert row['qualification']['dataset_ready'] is False


@pytest.mark.parametrize('change',[lambda s,p:p.update(source_families=['noaa_aerial']),
    lambda s,p:p['thresholds'].update(estimated_cloud_fraction_max=.9),
    lambda s,p:s.update(policy_pin=pin('e'*64)),
    lambda s,p:s.update(source_family='earthview'),
    lambda s,p:s['source_candidate'].update(acquisition_day='invented'),
    lambda s,p:s.update(candidate_record_pin=pin()),
    lambda s,p:s['source_candidate']['split_exclusion_evidence']['allocation'].update(role='test')])
def test_conflicting_policy_identity_provenance_or_eval_role_cannot_nominate(change):
    score,policy=inputs();change(score,policy)
    with pytest.raises(ValueError):bind(score,policy)


def test_observed_hold_cannot_be_overridden_by_model_nomination():
    score,policy=inputs()
    hold=dict(sample_id='patch',source='naip_aerial',candidate_sha256=score['candidate_sha256'],
        hr_float32_sha256='b'*64,actual_image_review=True,verdict='hold')
    assert bind(score,policy,reviews=[hold])['screening']['eligible'] is False


def test_explicit_relaxed_policy_reuses_measured_scores_without_rewriting_them():
    score,policy=inputs();score.update(historical_water_fraction=.02,estimated_cloud_fraction=.18)
    relaxed=dict(schema='marine-expanded-nomination-policy-v2',source_families=['naip_aerial'],
        score_policy_pins=[pin()],preserved_lock_evidence_pins=[pin('f'*64)],
        thresholds=dict(historical_water_fraction_min=.01,historical_water_fraction_max=.99,
                        estimated_cloud_fraction_max=.20))
    assert not bind(score,policy)['screening']['eligible']
    before=copy.deepcopy(score)
    result=bind(score,policy,nomination_policy=relaxed,nomination_policy_pin=pin('9'*64))
    assert result['screening']['eligible'] and score==before
    assert result['qualification']['scoring_policy_pin']==pin()
    assert result['qualification']['policy_pin']==pin('9'*64)
