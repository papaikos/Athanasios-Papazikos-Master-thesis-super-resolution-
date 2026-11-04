import hashlib
import pytest

from scripts.maritime_selected_audit_plan import selected_audit_plan


def row(i, source='naip_aerial', water='mixed'):
    return dict(sample_id=f'image-{i:03}', source=source,
        candidate_sha256=hashlib.sha256(f'candidate-{i}'.encode()).hexdigest(),
        hr_float32_sha256=hashlib.sha256(f'HR-{i}'.encode()).hexdigest(),
        water_bucket=water)


def test_deterministic_sample_covers_every_selected_source_and_stratum():
    rows=[row(i) for i in range(80)]+[row(i,'noaa_aerial','water_heavy') for i in range(80,101)]
    plan=selected_audit_plan(rows,strata_fields=['water_bucket'])
    assert plan==selected_audit_plan(list(reversed(rows)),strata_fields=['water_bucket'])
    assert len(plan['required_sample_ids'])==6
    for group in plan['groups'].values():
        assert len(group['required_sample_ids'])==group['required_review_count']
    assert plan['dataset_ready'] is False
    assert plan['population_accuracy_estimate'] is False


def test_full_auxiliary_audit_requires_all_samples():
    rows=[row(i) for i in range(38)]
    plan=selected_audit_plan(rows,fraction=1)
    assert set(plan['required_sample_ids'])=={r['sample_id'] for r in rows}


@pytest.mark.parametrize('change',[lambda r:r.update(candidate_sha256='bad'),
    lambda r:r.pop('water_bucket'),lambda r:r.update(source=''),
    lambda r:r.update(sample_id='')])
def test_missing_exact_identity_or_stratum_fails_closed(change):
    item=row(0);change(item)
    with pytest.raises(ValueError):
        selected_audit_plan([item],strata_fields=['water_bucket'])


def test_duplicate_selected_identity_fails_closed():
    with pytest.raises(ValueError):selected_audit_plan([row(0),row(0)])


def test_zero_or_empty_sampling_cannot_satisfy_quality_gate():
    with pytest.raises(ValueError):selected_audit_plan([])
    with pytest.raises(ValueError):selected_audit_plan([row(0)],fraction=0)


def test_exact_candidate_and_hr_changes_are_visible_in_selection_fingerprint():
    before=row(0);after=row(0);after['candidate_sha256']='a'*64
    assert selected_audit_plan([before])['selection_sha256']!=selected_audit_plan([after])['selection_sha256']


def test_removing_unselected_reject_does_not_shuffle_every_other_audit_image():
    rows=[row(i) for i in range(100)]
    before=selected_audit_plan(rows)
    outsider=next(r['sample_id'] for r in rows if r['sample_id'] not in before['required_sample_ids'])
    after=selected_audit_plan([r for r in rows if r['sample_id']!=outsider])
    assert after['required_sample_ids']==before['required_sample_ids']
    assert after['selection_sha256']!=before['selection_sha256']
