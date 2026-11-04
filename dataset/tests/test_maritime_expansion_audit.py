import pytest

from scripts.maritime_expansion_audit import addition_audit


def candidate(number, source='earthview'):
    return dict(sample_id=f'item-{number}', source=source,
                candidate_sha256=f'{number:064x}', hr_float32_sha256='a'*64)


def review(row, verdict='pass'):
    return dict(row, verdict=verdict, actual_image_review=True,
                reviewer='Codex multimodal assistant', independent_human_review=False)


def test_exact_selected_intersection_and_per_source_ceiling():
    rows = [candidate(n) for n in range(21)] + [candidate(21, 'maxar')]
    report = addition_audit(rows, [review(rows[0]), review(candidate(999)), review(rows[21])])
    assert report['per_source']['earthview']['required_review_count'] == 2
    assert report['per_source']['earthview']['reviewed_added_count'] == 1
    assert report['per_source']['maxar']['reviewed_added_count'] == 1
    assert report['passed'] is False
    assert 'insufficient_selected_reviews:earthview' in report['gaps']


def test_old_or_changed_pixels_cannot_count_as_review():
    row = candidate(1)
    old = review(dict(row, hr_float32_sha256='b'*64))
    report = addition_audit([row], [old])
    assert report['passed'] is False
    assert report['per_source']['earthview']['reviewed_added_count'] == 0
    assert 'review_identity_mismatch:item-1' in report['gaps']


def test_exclusion_wins_over_positive_review():
    row = candidate(1)
    report = addition_audit([row], [review(row), review(row, 'reject')])
    assert report['passed'] is False
    assert report['locked_rejected_ids'] == ['item-1']
    assert 'observed_reject_selected:item-1' in report['gaps']


def test_known_prior_rejects_cannot_be_overridden():
    row = candidate(1)
    report = addition_audit([row], [review(row)], prior_rejected_ids=['item-1'])
    assert report['passed'] is False


def test_unresolved_or_prediction_only_review_cannot_clear_quality():
    row = candidate(1)
    prediction = dict(review(row), actual_image_review=False)
    report = addition_audit([row], [prediction, review(row, 'hold')])
    assert report['passed'] is False
    assert 'unresolved_review_selected:item-1' in report['gaps']
    assert report['per_source']['earthview']['reviewed_added_count'] == 0


def test_selected_positive_visual_checks_are_not_human_ground_truth():
    row = candidate(1)
    report = addition_audit([row], [review(row)])
    assert report['passed'] is True
    assert report['independent_human_review'] is False
    assert report['population_accuracy_estimate'] is False
    assert report['automatic_per_image_acceptance'] is False


def test_duplicate_selected_ids_and_missing_hash_fail_closed():
    row = candidate(1)
    with pytest.raises(ValueError, match='Duplicate'):
        addition_audit([row, row], [])
    with pytest.raises(ValueError, match='SHA'):
        addition_audit([dict(row, candidate_sha256='')], [])


def test_zero_reviews_not_sufficient_and_duplicate_reviews_count_once():
    rows = [candidate(n) for n in range(21)]
    report = addition_audit(rows, [review(rows[0]), review(rows[0])])
    assert report['passed'] is False
    assert report['per_source']['earthview']['reviewed_added_count'] == 1
    assert addition_audit([], [])['passed'] is False


def test_reviewed_source_total_does_not_hide_unreviewed_added_stratum():
    rows = [dict(candidate(n), composition='shoreline') for n in range(20)]
    rows.append(dict(candidate(20), composition='vessel'))
    report = addition_audit(rows, [review(rows[0]), review(rows[1])],
                            strata_fields=['composition'])
    assert report['per_source']['earthview']['reviewed_added_count'] == 2
    assert 'insufficient_stratum_reviews:earthview:composition=vessel' in report['gaps']
    assert report['passed'] is False
    report = addition_audit(rows, [review(rows[0]), review(rows[1]), review(rows[20])],
                            strata_fields=['composition'])
    assert report['passed'] is True


def test_requested_stratification_cannot_silently_drop_missing_metadata():
    row = candidate(1)
    with pytest.raises(ValueError, match='stratum'):
        addition_audit([row], [review(row)], strata_fields=['composition'])
