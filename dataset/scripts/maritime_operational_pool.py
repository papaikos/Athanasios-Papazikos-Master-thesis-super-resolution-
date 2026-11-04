"""Join verified operational scoring witnesses to unchanged aerial candidates.

Callers first verify GCS generation/CRC/SHA for score, policy and candidate
objects. This metadata join cannot prove image truth, original replay or READY.
"""
import copy
import hashlib

from scripts.maritime_cloud_inventory import json_bytes
from scripts.maritime_expansion_qualification import qualify_screened
from scripts.maritime_uint8_download import validate_pin


def expanded_nomination_limits(policy, policy_pin, scoring_policy_pin, family):
    validate_pin(policy_pin);validate_pin(scoring_policy_pin)
    expected=dict(historical_water_fraction_min=.01,historical_water_fraction_max=.99,
                  estimated_cloud_fraction_max=.20)
    if (policy.get('schema')!='marine-expanded-nomination-policy-v2'
            or family not in policy.get('source_families',[])
            or scoring_policy_pin not in policy.get('score_policy_pins',[])
            or policy.get('thresholds')!=expected
            or not policy.get('preserved_lock_evidence_pins')):
        raise ValueError('Exact expanded nomination policy and scoring evidence required')
    return expected


def bind_screened_earth(score, source, *, evidence_pin, scoring_policy, scoring_policy_pin,
                        nomination_policy, nomination_policy_pin, reviews=(), locked_ids=(),
                        source_validator=None):
    """Bind exact Earth metadata to separately frozen measurement and nomination policies."""
    validate_pin(evidence_pin)
    limits=expanded_nomination_limits(nomination_policy,nomination_policy_pin,scoring_policy_pin,'earthview')
    record=score.get('candidate_record_pin',{})
    validate_pin(record)
    digest=hashlib.sha256(json_bytes(source)).hexdigest()
    protection=source.get('split_exclusion_evidence',{})
    protected_pin=scoring_policy.get('protected_locks_pin',{})
    if not protected_pin:
        contract=scoring_policy.get('protected_geometry_contract',{})
        matching=[p for p in scoring_policy.get('preserved_lock_evidence_pins',[])
                  if p.get('sha256')==contract.get('locks_sha256')]
        if (len(matching)!=1 or contract.get('buffer_m')!=4000
                or contract.get('protected_geometry_pass') is not True):
            raise ValueError('Frozen population protected geometry contract required')
        protected_pin=matching[0]
    if (scoring_policy.get('schema')!='earthview-canonical-operational-screening-policy-v1'
            or scoring_policy.get('source_family')!='earthview'
            or scoring_policy.get('final_original_replay_required') is not True
            or score.get('policy_pin')!=scoring_policy_pin
            or score.get('source_family')!='earthview'
            or digest!=record['sha256'] or digest!=score.get('candidate_sha256')
            or source.get('sample_id')!=score.get('sample_id')
            or source.get('source_kind')!='earthview' or source.get('hr')!=score.get('hr_pin')
            or protection.get('protected_geometry_pass') is not True
            or protection.get('buffer_m')!=4000
            or protection.get('locks_sha256')!=protected_pin.get('sha256')
            or protected_pin not in nomination_policy['preserved_lock_evidence_pins']
            or score.get('original_assembly_replayed') is not False
            or score.get('dataset_ready') is not False or score.get('final_eligible') is not False):
        raise ValueError('Exact Earth source, scoring policy and protected geometry required')
    validate_pin(protected_pin)
    if source_validator is None:
        from src.maritime_earthview_source import validate_source
        source_validator=validate_source
    source_validator(source)
    capture=hashlib.sha256(json_bytes(source['acquisition_identity'])).hexdigest()[:24]
    site='earthview-port-'+str(source['port']['id'])
    candidate=dict(sample_id=source['sample_id'],source_kind='earthview',source=copy.deepcopy(source),
        acquisition_id='earthview-'+capture,site_id=site,split='train',
        geographic_group_id='provisional-expansion-'+site,country=source['port']['country'],
        crs=source['output_crs'],core_bounds=list(source['physical_core_bounds']),candidate_pin=record)
    result=qualify_screened(candidate,digest,score['hr_float32_sha256'],score,
        evidence_pin=evidence_pin,policy_pin=nomination_policy_pin,source_family='earthview',
        reviews=reviews,locked_ids=locked_ids,nomination_thresholds=limits)
    result['qualification'].update(scoring_policy_pin=copy.deepcopy(scoring_policy_pin))
    result.update(candidate_sha256=digest,hr_float32_sha256=score['hr_float32_sha256'],
                  hr_uint8_chw_sha256=score['hr_uint8_chw_sha256'])
    return result


def bind_screened_aerial(score, *, evidence_pin, policy, policy_pin,
                        reviews=(), locked_ids=(), nomination_policy=None, nomination_policy_pin=None):
    validate_pin(evidence_pin);validate_pin(policy_pin)
    family=score.get('source')
    thresholds=dict(historical_water_fraction_min=.05,historical_water_fraction_max=.95,
                    estimated_cloud_fraction_max=.10)
    if (policy.get('schema')!='operational-naip-screening-policy-v1'
            or family!='naip_aerial' or family not in policy.get('source_families',[])
            or any(policy.get('thresholds',{}).get(k)!=v for k,v in thresholds.items())
            or score.get('policy_pin')!=policy_pin
            or score.get('source_family',family)!=family):
        raise ValueError('Exact operational policy, threshold and source scope required')
    source=score.get('source_candidate',{})
    record=source.get('candidate_record',{})
    validate_pin(record)
    raw={k:v for k,v in source.items() if k!='candidate_record'}
    digest=hashlib.sha256(json_bytes(raw)).hexdigest()
    if (score.get('candidate_record_pin')!=record or digest!=record['sha256']
            or digest!=score.get('candidate_sha256') or source.get('source_type')!=family
            or source.get('sample_id')!=score.get('sample_id')
            or source.get('split_exclusion_evidence',{}).get('allocation',{}).get('role')!='train'):
        raise ValueError('Exact unchanged training source candidate required')
    site='aerial-port-'+str(source['nearest_port']['port_id'])
    capture=hashlib.sha256(json_bytes(dict(source_type=family,parent_tile_id=source['parent_tile_id'],
                                         acquisition_day=source['acquisition_day']))).hexdigest()
    candidate=dict(sample_id=source['sample_id'],source_kind='aerial_rgb',source=copy.deepcopy(source),
        acquisition_id='aerial-'+capture,site_id=site,split='train',
        geographic_group_id='provisional-expansion-'+site,
        country=source['nearest_port']['country_code_as_published'],
        crs=source['output_crs'],core_bounds=list(source['physical_core_bounds']),candidate_pin=record)
    limits=None
    if nomination_policy is not None or nomination_policy_pin is not None:
        limits=expanded_nomination_limits(nomination_policy or {},nomination_policy_pin,policy_pin,family)
    result=qualify_screened(candidate,digest,score['hr_float32_sha256'],score,
        evidence_pin=evidence_pin,policy_pin=nomination_policy_pin if limits else policy_pin,source_family=family,
        reviews=reviews,locked_ids=locked_ids,nomination_thresholds=limits)
    if limits:
        result['qualification']['scoring_policy_pin']=copy.deepcopy(policy_pin)
    result.update(candidate_sha256=digest,hr_float32_sha256=score['hr_float32_sha256'],
                  hr_uint8_chw_sha256=score['hr_uint8_chw_sha256'])
    return result
