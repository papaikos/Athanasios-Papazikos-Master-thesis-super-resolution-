"""Check attributable visual coverage of the actually selected additions.

This checks one component of a release audit. It cannot qualify an unreviewed
image, prove population accuracy, or replace source/scientific quality gates.
Callers must verify the generation/CRC/SHA pins before supplying review records.
"""
from collections import Counter
import math
import re


def _identity(row):
    hashes = (row.get('candidate_sha256'), row.get('hr_float32_sha256'))
    if any(not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value)
           for value in hashes):
        raise ValueError('Exact candidate and canonical HR SHA-256 required')
    if not row.get('sample_id') or not row.get('source'):
        raise ValueError('Sample identity and source required')
    return (row['source'], *hashes)


def addition_audit(additions, reviews, *, prior_rejected_ids=(), fraction=.05,
                   strata_fields=()):
    """Require per-source positive actual-image checks and preserve all rejects.

    Reviews use verdict pass/reject/hold, exact candidate and canonical float32
    HR hashes, and actual_image_review=True. Prediction-only records do not
    count. A contradictory rejection or unresolved hold is fail-closed.
    """
    if not 0 < fraction <= 1:
        raise ValueError('Visual review fraction must lie in (0,1]')
    identities, strata_by_id = {}, {}
    for row in additions:
        sample_id = row['sample_id']
        if sample_id in identities:
            raise ValueError('Duplicate selected sample identity')
        identities[sample_id] = _identity(row)
        strata_by_id[sample_id] = []
        for field in strata_fields:
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError('Requested stratum requires explicit metadata: ' + field)
            strata_by_id[sample_id].append(f'{row["source"]}:{field}={value}')
    gaps = set()
    positive, held = set(), set()
    rejected = set(prior_rejected_ids)
    for row in reviews:
        if row.get('verdict') not in {'pass', 'reject', 'hold'}:
            raise ValueError('Explicit pass/reject/hold review verdict required')
        identity = _identity(row)
        sample_id = row['sample_id']
        if row.get('actual_image_review') is not True:
            continue
        if row['verdict'] == 'reject':
            rejected.add(sample_id)
        elif row['verdict'] == 'hold':
            held.add(sample_id)
        if sample_id not in identities:
            continue
        if identity != identities[sample_id]:
            gaps.add('review_identity_mismatch:' + sample_id)
            continue
        if row['verdict'] == 'pass':
            if not row.get('reviewer'):
                raise ValueError('An attributable reviewer is required')
            positive.add(sample_id)
    for sample_id in set(identities).intersection(rejected):
        gaps.add('observed_reject_selected:' + sample_id)
    for sample_id in set(identities).intersection(held):
        gaps.add('unresolved_review_selected:' + sample_id)
    positive.difference_update(rejected, held)
    counts = Counter(identity[0] for identity in identities.values())
    reviewed = Counter(identities[sample_id][0] for sample_id in positive)
    per_source = {}
    for source, count in sorted(counts.items()):
        required = math.ceil(count * fraction)
        per_source[source] = dict(added_count=count, required_review_count=required,
                                  reviewed_added_count=reviewed[source])
        if reviewed[source] < required:
            gaps.add('insufficient_selected_reviews:' + source)
    if not identities:
        gaps.add('no_added_samples')
    stratum_counts = Counter(label for labels in strata_by_id.values() for label in labels)
    stratum_reviewed = Counter(label for sample_id in positive
                              for label in strata_by_id[sample_id])
    per_stratum = {}
    for label, count in sorted(stratum_counts.items()):
        required = math.ceil(count * fraction)
        per_stratum[label] = dict(added_count=count, required_review_count=required,
                                  reviewed_added_count=stratum_reviewed[label])
        if stratum_reviewed[label] < required:
            gaps.add('insufficient_stratum_reviews:' + label)
    return dict(
        schema='marine-selected-addition-visual-coverage-v1',
        passed=not gaps, gaps=sorted(gaps), per_source=per_source,
        required_review_fraction=fraction, strata_fields=list(strata_fields),
        per_stratum=per_stratum,
        reviewed_added_ids=sorted(positive), locked_rejected_ids=sorted(rejected),
        unresolved_ids=sorted(held), independent_human_review=False,
        population_accuracy_estimate=False, automatic_per_image_acceptance=False,
        scope='Exact selected-ID/candidate/HR-hash intersection of positive actual '
              'image reviews. Other source, quality, diversity, duplicate and leakage '
              'gates remain separate requirements.',
    )
