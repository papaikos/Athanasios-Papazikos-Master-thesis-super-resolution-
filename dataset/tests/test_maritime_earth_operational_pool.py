import copy
import hashlib
import pytest
from scripts.maritime_cloud_inventory import json_bytes
from scripts.maritime_operational_pool import bind_screened_earth


def pin(sha='a'*64):
    return dict(uri='gs://bucket/'+sha,generation='1',bytes=10,crc32c='AAAAAA==',sha256=sha)


def inputs():
    source=dict(sample_id='earth',source_kind='earthview',schema='earthview-candidate-v1',
        native_gsd_m=1.,license='CC-BY-4.0',hr=pin('c'*64),port=dict(id='12',country='GR'),
        acquisition_identity=['2020-01-01','sensor',32634,1.,'L1'],output_crs='EPSG:32634',
        physical_core_bounds=[0,0,1280,1280],split_exclusion_evidence=dict(
            protected_geometry_pass=True,buffer_m=4000,locks_sha256='d'*64))
    digest=hashlib.sha256(json_bytes(source)).hexdigest()
    score=dict(schema='marine-operational-screening-result-v1',sample_id='earth',source='earthview',
        source_family='earthview',candidate_record_pin=pin(digest),candidate_sha256=digest,
        hr_pin=source['hr'],hr_float32_sha256='b'*64,hr_uint8_chw_sha256='e'*64,
        policy_pin=pin(),historical_water_fraction=.02,estimated_cloud_fraction=.18,
        finite_rgb=True,nonconstant_rgb=True,original_support_verified=True,
        original_assembly_replayed=False,dataset_ready=False,final_eligible=False)
    measured=dict(schema='earthview-canonical-operational-screening-policy-v1',source_family='earthview',
                  final_original_replay_required=True,protected_locks_pin=pin('d'*64))
    policy=dict(schema='marine-expanded-nomination-policy-v2',source_families=['earthview'],
        score_policy_pins=[pin()],preserved_lock_evidence_pins=[pin('d'*64)],
        thresholds=dict(historical_water_fraction_min=.01,historical_water_fraction_max=.99,
                        estimated_cloud_fraction_max=.20))
    return source,score,measured,policy


def bind(source,score,measured,policy,**kwargs):
    return bind_screened_earth(score,source,evidence_pin=pin('f'*64),
        scoring_policy=measured,scoring_policy_pin=pin(),nomination_policy=policy,
        nomination_policy_pin=pin('9'*64),source_validator=lambda s:None,**kwargs)


def test_distinct_measurement_and_relaxed_nomination_policies_are_retained():
    source,score,measured,policy=inputs();before=copy.deepcopy(source)
    result=bind(source,score,measured,policy)
    assert source==before and result['source']==source
    assert result['screening']['eligible'] and result['split']=='train'
    assert result['site_id']=='earthview-port-12'
    assert result['qualification']['policy_pin']==pin('9'*64)
    assert result['qualification']['scoring_policy_pin']==pin()
    assert not result['qualification']['dataset_ready']


@pytest.mark.parametrize('change',[lambda s,r,m,p:s.update(sample_id='changed'),
    lambda s,r,m,p:r.update(hr_pin=pin()),lambda s,r,m,p:p.update(score_policy_pins=[]),
    lambda s,r,m,p:p.update(source_families=['naip_aerial']),
    lambda s,r,m,p:p['thresholds'].update(estimated_cloud_fraction_max=.99),
    lambda s,r,m,p:s['split_exclusion_evidence'].update(protected_geometry_pass=False)])
def test_wrong_source_policy_or_protection_cannot_nominate(change):
    source,score,measured,policy=inputs();change(source,score,measured,policy)
    with pytest.raises(ValueError):bind(source,score,measured,policy)


def test_known_hold_remains_locked_under_relaxed_requirements():
    source,score,measured,policy=inputs()
    hold=dict(sample_id='earth',source='earthview',candidate_sha256=score['candidate_sha256'],
        hr_float32_sha256='b'*64,actual_image_review=True,verdict='hold')
    assert not bind(source,score,measured,policy,reviews=[hold])['screening']['eligible']


def test_frozen_population_contract_resolves_its_preserved_lock_pin():
    source,score,measured,policy=inputs()
    measured.pop('protected_locks_pin')
    measured.update(preserved_lock_evidence_pins=[pin('d'*64)],
        protected_geometry_contract=dict(locks_sha256='d'*64,buffer_m=4000,protected_geometry_pass=True))
    assert bind(source,score,measured,policy)['screening']['eligible']
