import copy
import pytest


def baseline():
    return dict(schema='maritime-baseline-quantized-hr-census-v1',complete=True,sample_count=1,
        rows=[dict(sample_id='old',hr_uint8_chw_sha256='a'*64)])


def candidate(identity,digest='b'*64,priority=3,tier='automatic_operational_screening'):
    return dict(sample_id=identity,hr_uint8_chw_sha256=digest,
        screening=dict(eligible=True,priority=priority),qualification=dict(tier=tier))


def test_preserved_baseline_and_best_qualified_new_image_win_exact_quantized_duplicates():
    from scripts.maritime_quantized_hr_dedup import deduplicate_quantized_hr
    rows=[candidate('new-baseline-copy','a'*64),candidate('lower-rank'),candidate('ship',priority=1),candidate('other','c'*64)]
    saved=copy.deepcopy(rows)
    kept,rejected=deduplicate_quantized_hr(rows,baseline())
    assert rows==saved
    assert {r['sample_id'] for r in kept}=={'ship','other'}
    assert {r['reason'] for r in rejected}=={'duplicate_preserved_baseline_uint8_hr','duplicate_added_uint8_hr'}


def test_visual_witness_wins_rank_tie_and_order_is_deterministic():
    from scripts.maritime_quantized_hr_dedup import deduplicate_quantized_hr
    rows=[candidate('automatic'),candidate('reviewed',tier='assistant_actual_image_review')]
    a=deduplicate_quantized_hr(rows,baseline());b=deduplicate_quantized_hr(list(reversed(rows)),baseline())
    assert a==b and a[0][0]['sample_id']=='reviewed'


def test_missing_actual_quantized_hash_and_incomplete_baseline_fail_closed():
    from scripts.maritime_quantized_hr_dedup import deduplicate_quantized_hr
    row=candidate('new');row.pop('hr_uint8_chw_sha256')
    with pytest.raises(ValueError):deduplicate_quantized_hr([row],baseline())
    document=baseline();document['complete']=False
    with pytest.raises(ValueError):deduplicate_quantized_hr([candidate('new')],document)


def test_conflicting_hashes_for_same_new_sample_fail_before_order_can_choose_pixels():
    from scripts.maritime_quantized_hr_dedup import deduplicate_quantized_hr
    with pytest.raises(ValueError,match='Conflicting'):
        deduplicate_quantized_hr([candidate('same','b'*64),candidate('same','c'*64)],baseline())


def aerial(identity,day='2020-01-01',crs='EPSG:32610',priority=3):
    row=candidate(identity,priority=priority)
    row['source_kind']='aerial_rgb'
    row['source']=dict(source_type='naip_aerial',acquisition_day=day,output_crs=crs,
                       physical_core_bounds=[0,0,1280,1280],parent_tile_id=identity)
    return row


def test_same_verified_day_core_across_different_parents_retains_best_one():
    from scripts.maritime_quantized_hr_dedup import deduplicate_aerial_day_cells
    rows=[aerial('shore'),aerial('vessel',priority=1),aerial('temporal',day='2021-01-01')]
    kept,rejected=deduplicate_aerial_day_cells(rows)
    assert {r['sample_id'] for r in kept}=={'vessel','temporal'}
    assert rejected[0]['sample_id']=='shore'
    assert rejected[0]['reason']=='same_aerial_calendar_day_core'
    assert deduplicate_aerial_day_cells(list(reversed(rows)))==(kept,rejected)


def test_unknown_dates_and_different_original_crs_are_not_invented_as_same_capture():
    from scripts.maritime_quantized_hr_dedup import deduplicate_aerial_day_cells
    rows=[aerial('unknown-a',day=None),aerial('unknown-b',day=None),
          aerial('known'),aerial('other-original-grid',crs='EPSG:32611')]
    kept,rejected=deduplicate_aerial_day_cells(rows)
    assert len(kept)==4 and not rejected


def test_invalid_calendar_metadata_or_physical_geometry_fails_closed():
    from scripts.maritime_quantized_hr_dedup import deduplicate_aerial_day_cells
    with pytest.raises(ValueError):deduplicate_aerial_day_cells([aerial('bad',day='invented')])
    row=aerial('bad');row['source']['physical_core_bounds']=[0,0,100,100]
    with pytest.raises(ValueError):deduplicate_aerial_day_cells([row])
