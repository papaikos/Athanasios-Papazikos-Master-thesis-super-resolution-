"""Attribute reviewed image quality without claiming a completed release.

Inputs must be read from generation/CRC/SHA-verified audit documents. This
module performs no image processing and grants no source, replay, deduplication,
split, payload-size or READY gate. Assistant observations are not ship labels.
"""
import copy
import math
import re


def locked_review_ids(records):
    """Keep original identities and byte-equivalent alternate export identities."""
    identities = set()
    for row in records:
        primary = row.get('sample_id', row.get('candidate_id'))
        if not isinstance(primary, str) or not primary:
            raise ValueError('Locked sample or legacy candidate identity required')
        identities.add(primary)
        identities.update(row.get('alternate_candidate_ids', []))
        if row.get('canonical_core_id'):
            identities.add(row['canonical_core_id'])
    return identities


def normalize_review(row, *, source_family, reviewer, actual_image_document=False):
    """Unify the two saved actual-image audit schemas, preserving uncertainty."""
    out = copy.deepcopy(row)
    out['source'] = row.get('source', source_family)
    out['candidate_sha256'] = row.get('candidate_sha256', row.get('candidate_pin', {}).get('sha256'))
    decision = str(row.get('decision', '')).lower()
    if decision in {'pass_vessel_unverified_size', 'pass_small_marine',
                    'pass_minor_cloud_pending_global_rule'}:
        decision = 'pass'
    elif decision in {'reject_land_context', 'reject_cloud', 'reject_haze',
                      'reject_water_context', 'reject_chromatic_artifact',
                      'reject_cloud_land', 'reject_glint_water'}:
        decision = 'reject'
    out['verdict'] = row.get('verdict', decision)
    if out['verdict'] not in {'pass', 'reject', 'hold'}:
        raise ValueError('Explicit actual-image pass/reject/hold required')
    out['actual_image_review'] = row.get('actual_image_review', actual_image_document) is True
    out['reviewer'] = row.get('reviewer', reviewer)
    out['observed_vessel_form'] = (out['actual_image_review'] is True
        and str(row.get('decision','')).lower() == 'pass_vessel_unverified_size')
    # These category codes occur only in the direct visual audit, not detections.
    observed = row.get('category') in {'S', 'P', 'V', 'R', 'A', 'T', 'shoreline',
        'built_port_marina', 'built_waterfront', 'vessel_form_visible',
        'coastal_reef_structure', 'coastal_settlement'}
    out['useful_marine_context_observed'] = row.get('useful_marine_context_observed', observed) is True
    out['minor_cloud_observed'] = (row.get('minor_cloud_global_policy_pending') is True
                                  or row.get('category') == 'A'
                                  or row.get('quality_code') in {'minor_cloud', 'pass_minor_cloud'})
    out['physical_ship_size_verified'] = row.get('physical_ship_size_verified', False) is True
    return out


def qualify_reviewed(candidate, candidate_sha256, hr_float32_sha256, reviews, *,
                     source_family, locked_ids=()):
    """Qualify exact visually passed candidates for further release checks.

    A hold/reject from any supplied actual-image record wins. Model-only records
    never constitute visual observations. Callers must supply all known audit
    locks and keep reviewed and automatically screened quality tiers distinct.
    """
    for digest in (candidate_sha256, hr_float32_sha256):
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Exact candidate and canonical HR SHA-256 required')
    out = copy.deepcopy(candidate)
    identity = candidate['sample_id']
    matching = [r for r in reviews if r['sample_id'] == identity and r.get('actual_image_review') is True]
    for row in matching:
        if (row.get('candidate_sha256') != candidate_sha256
                or row.get('hr_float32_sha256') != hr_float32_sha256
                or row.get('source') != source_family):
            raise ValueError('Reviewed candidate/source/HR identity mismatch: ' + identity)
        if row.get('verdict') not in {'pass', 'hold', 'reject'}:
            raise ValueError('Explicit review verdict required')
    positive = [r for r in matching if r['verdict'] == 'pass'
                and r.get('useful_marine_context_observed') is True
                and r.get('reviewer') and r.get('observation')]
    blocked = identity in set(locked_ids) or any(r['verdict'] != 'pass' for r in matching)
    eligible = bool(positive) and not blocked
    categories = {r.get('category') for r in positive}
    vessel = (bool(categories & {'V', 'vessel_form_visible'})
              or any(r.get('observed_vessel_form') is True for r in positive))
    port = bool(categories & {'P', 'built_port_marina', 'built_waterfront'})
    reef = bool(categories & {'R', 'coastal_reef_structure'})
    priority = 1 if vessel else (2 if port else (4 if reef else 3))
    screening = dict(out.get('screening', {}))
    screening.update(eligible=eligible, priority=priority,
                     semantic_evidence='assistant_observed_marine_context' if eligible else 'unqualified',
                     composition='reef_context' if reef and not vessel and not port else 'observed_marine_context')
    out['screening'] = screening
    out['qualification'] = dict(schema='reviewed-marine-quality-qualification-v1',
        tier='assistant_actual_image_review' if eligible else 'unqualified',
        source_family=source_family, candidate_sha256=candidate_sha256,
        hr_float32_sha256=hr_float32_sha256, image_quality_eligible=eligible,
        reviewer_names=sorted({r['reviewer'] for r in positive}),
        minor_cloud_observed=any(r.get('minor_cloud_observed') is True for r in positive),
        certified_medium_large_ship=False, independent_human_review=False,
        dataset_ready=False, native_replay_required_at_release=True,
        remaining_checks=['license_and_resolution', 'original_full_context_replay',
                          'deduplication_and_frozen_split', 'selected_stratified_audit',
                          'decoded_uint8_payload', 'numeric_recipes_and_training_smoke'])
    return out


def qualify_screened(candidate, candidate_sha256, hr_float32_sha256, score, *,
                     evidence_pin, policy_pin, source_family, reviews=(), locked_ids=(),
                     nomination_thresholds=None):
    """Nominate mixed marine imagery under an explicit operational proxy rule.

    Callers must generation/CRC/SHA-verify the score and policy documents first.
    Historical water geometry and estimated cloud masks are uncalibrated
    proxies. This is a separate automatic tier requiring a selected-image
    audit, never a claim that each nominated image was visually approved.
    """
    from scripts.maritime_uint8_download import validate_pin
    validate_pin(evidence_pin);validate_pin(policy_pin)
    limits=nomination_thresholds or dict(historical_water_fraction_min=.05,
        historical_water_fraction_max=.95,estimated_cloud_fraction_max=.10)
    lower,upper,cloud_limit=[limits[k] for k in ('historical_water_fraction_min',
        'historical_water_fraction_max','estimated_cloud_fraction_max')]
    if (any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v)
            for v in (lower,upper,cloud_limit)) or not 0<=lower<upper<=1 or not 0<=cloud_limit<=.5):
        raise ValueError('Finite bounded nomination thresholds required')
    reviewed=qualify_reviewed(candidate,candidate_sha256,hr_float32_sha256,reviews,
                              source_family=source_family,locked_ids=locked_ids)
    if (score.get('schema')!='marine-operational-screening-result-v1'
            or score.get('sample_id')!=candidate['sample_id'] or score.get('source')!=source_family
            or score.get('candidate_sha256')!=candidate_sha256
            or score.get('hr_float32_sha256')!=hr_float32_sha256
            or not re.fullmatch('[0-9a-f]{64}',str(score.get('hr_uint8_chw_sha256','')))):
        raise ValueError('Exact screening candidate/source/HR identity required')
    values=[score.get('historical_water_fraction'),score.get('estimated_cloud_fraction')]
    if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v)
           or not 0<=v<=1 for v in values):
        raise ValueError('Finite screening fractions in [0,1] required')
    if reviewed['screening']['eligible']:
        return reviewed
    matching=[r for r in reviews if r['sample_id']==candidate['sample_id']
              and r.get('actual_image_review') is True]
    blocked=candidate['sample_id'] in set(locked_ids) or bool(matching)
    water,cloud=values
    eligible=(not blocked and lower<=water<=upper and cloud<=cloud_limit
              and all(score.get(key) is True for key in
                      ('finite_rgb','nonconstant_rgb','original_support_verified')))
    high_water=water>.95
    reviewed['screening'].update(eligible=eligible,priority=5 if high_water else 3,
        semantic_evidence='historical_mixed_marine_and_operational_cloud_screen' if eligible else 'unqualified',
        composition='open_water_negative' if high_water else 'mixed_marine_proxy')
    reviewed['qualification'].update(schema='screened-marine-quality-qualification-v1',
        tier='automatic_operational_screening' if eligible else 'unqualified',
        image_quality_eligible=eligible,actual_image_review=False,population_accuracy_estimate=False,
        certified_medium_large_ship=False,policy_version=('marine-expanded-nomination-v2'
            if nomination_thresholds is not None else 'marine-operational-mixed-cloud-v1'),
        scoring_evidence_pin=copy.deepcopy(evidence_pin),policy_pin=copy.deepcopy(policy_pin),
        nomination_thresholds=dict(historical_water_fraction=[lower,upper],estimated_cloud_fraction_max=cloud_limit),
        historical_water_fraction=water,estimated_cloud_fraction=cloud,
        estimated_cloud_fraction_is_ground_truth=False,
        hr_uint8_chw_sha256=score['hr_uint8_chw_sha256'],
        original_support_claim_scope='successful intake context; full original replay still required',
        selected_stratified_actual_image_audit_required=True,
        caveats=['Historical shoreline may differ from the acquisition.',
                 'Cloud and marine proxies have no population accuracy estimate.',
                 'The cloud model can miss seams, haze and other image artifacts.',
                 'No ship presence or physical ship size is certified.'])
    return reviewed
