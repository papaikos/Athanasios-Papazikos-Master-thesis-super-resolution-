import copy
import pytest

from scripts.maritime_expansion_select import build_selection


def row(identity, x, *, split='train', site=None, capture=None, rank=4):
    return dict(sample_id=identity, source_kind='maxar', source={'immutable': identity},
                acquisition_id=capture or identity, crs='EPSG:32631',
                core_bounds=[x, 0, x+1280, 1280], split=split,
                site_id=site or identity, geographic_group_id=site or identity,
                screening=dict(eligible=True, priority=rank,
                               semantic_evidence='model_predicted'))


def baseline(rows):
    return dict(schema='ultimate-selection-v1', selection=rows,
                splits=dict(assignment={r['geographic_group_id']: r['split'] for r in rows},
                            groups=[dict(geographic_group_id=r['geographic_group_id'],
                                         site_ids=[r['site_id']], split=r['split']) for r in rows]),
                review_policy={'baseline': 'preserve'})


def test_preserves_baseline_and_prefers_medium_large_rank():
    old = baseline([row('old', 0)])
    snapshot = copy.deepcopy(old)
    out = build_selection(old, [row('shore', 10000, rank=4), row('ship', 20000, rank=0)],
                          minimum_bytes=2*835584, target_bytes=2*835584, max_site_fraction=1)
    assert old == snapshot
    assert out['selection'][0] == old['selection'][0]
    assert [r['sample_id'] for r in out['selection']] == ['old', 'ship']
    assert out['summary']['train_rgb_bytes_per_recipe'] == 2*835584
    assert out['status'] == 'REVIEW_PENDING'
    assert not any(out['expansion_gates'].values())


def test_filters_holdout_acquisitions_sites_and_geographic_buffer():
    old = baseline([row('old', 0), row('val', 20000, split='validation', site='protected')])
    candidates = [row('acq', 40000, capture='val'), row('site', 50000, site='protected'),
                  row('near', 24000), row('safe', 60000)]
    out = build_selection(old, candidates, minimum_bytes=835584, target_bytes=20*835584, max_site_fraction=1)
    assert [r['sample_id'] for r in out['selection']] == ['old', 'val', 'safe']
    assert {r['reason'] for r in out['rejections']} == {'protected_acquisition', 'protected_site', 'protected_geography'}
    assert out['splits']['assignment']['protected'] == 'validation'


def test_keeps_baseline_when_duplicate_ids_or_shifted_capture_overlap():
    old = baseline([row('old', 0, capture='capture')])
    candidates = [row('old', 50000), row('overlap', 1000, capture='capture'),
                  row('edge', 1280, capture='capture')]
    out = build_selection(old, candidates, minimum_bytes=835584, target_bytes=20*835584, max_site_fraction=1)
    assert [r['sample_id'] for r in out['selection']] == ['old', 'edge']
    assert out['selection'][0]['core_bounds'] == [0, 0, 1280, 1280]


def test_diversifies_sites_and_reports_real_shortfall_not_tar_bytes():
    old = baseline([row('old', 0)])
    candidates = [row('a'+str(i), 10000+i*1280, site='a', rank=0) for i in range(5)]
    candidates += [row('b', 40000, site='b', rank=4), row('c', 50000, site='c', rank=4)]
    out = build_selection(old, candidates, minimum_bytes=6*835584, target_bytes=6*835584,
                          max_site_fraction=.34)
    assert out['summary']['train_pairs'] == 5
    assert out['summary']['minimum_shortfall_bytes'] == 835584
    assert out['summary']['target_shortfall_bytes'] == 835584
    assert sum(r['site_id']=='a' for r in out['selection']) == 2


def test_rejects_unscreened_and_development_data():
    old = baseline([row('old', 0)])
    unqualified = row('bad', 20000)
    unqualified['screening']['eligible'] = False
    out = build_selection(old, [unqualified, row('dev', 30000, site='dev')],
                          development_sites={'dev'}, minimum_bytes=835584, target_bytes=20*835584, max_site_fraction=1)
    assert len(out['selection']) == 1
    assert {r['reason'] for r in out['rejections']} == {'screening_not_qualified', 'development_site'}


def test_open_water_negatives_cannot_dominate_actual_additions():
    old = baseline([row('old', 0)])
    candidates = [row('shore'+str(i), 10000+i*2000, rank=3) for i in range(18)]
    negatives = [row('water'+str(i), 60000+i*2000, rank=5) for i in range(20)]
    for candidate in negatives:
        candidate['screening']['composition'] = 'open_water_negative'
    out = build_selection(old, candidates+negatives, minimum_bytes=835584,
                          target_bytes=100*835584, max_site_fraction=1)
    added = out['selection'][1:]
    assert len(added) == 20
    assert sum(r['screening'].get('composition') == 'open_water_negative' for r in added) == 2
    assert out['summary']['added_open_water_negative_fraction'] == .1
    assert out['summary']['composition_limited_candidates'] == 18


def test_only_open_water_pool_does_not_fill_quota_or_change_baseline():
    old = baseline([row('old', 0)])
    negative = row('water', 20000, rank=5)
    negative['screening']['composition'] = 'open_water_negative'
    out = build_selection(old, [negative], minimum_bytes=2*835584,
                          target_bytes=2*835584, max_site_fraction=1)
    assert out['selection'] == old['selection']
    assert out['summary']['minimum_shortfall_bytes'] == 835584


def test_protected_geography_index_skips_only_proven_distant_latitudes(monkeypatch):
    import scripts.maritime_expansion_select as selector
    import scripts.maritime_aerial_evaluation_release as evaluation
    calls=[]
    original=selector.near
    def counted(a,b,distance):
        calls.append((a,b));return original(a,b,distance)
    monkeypatch.setattr(selector,'near',counted)
    monkeypatch.setattr(evaluation,'near',counted)
    protected=row('val',20000,split='validation')
    protected['core_bounds']=[20000,6000000,21280,6001280]
    out=build_selection(baseline([row('old',0),protected]),[row('far',60000)],
        minimum_bytes=835584,target_bytes=20*835584,max_site_fraction=1)
    assert [r['sample_id'] for r in out['selection']]==['old','val','far']
    assert calls==[]


def test_open_water_negative_requires_lowest_rank_and_does_not_block_context():
    old = baseline([row('old', 0)])
    negative = row('a-water', 20000, site='coast', rank=0)
    negative['screening']['composition'] = 'open_water_negative'
    with pytest.raises(ValueError, match='negative'):
        build_selection(old, [negative])
    negative['screening']['priority'] = 5
    context = row('z-shore', 40000, site='coast', rank=5)
    out = build_selection(old, [negative, context], minimum_bytes=835584,
                          target_bytes=3*835584, max_site_fraction=1)
    assert [r['sample_id'] for r in out['selection']] == ['old', 'z-shore']

