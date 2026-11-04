import pytest
pytest.importorskip('shapely')
from scripts.maritime_release_select import geographic_groups, select_release


def candidate(cid='a',site='site',left=500000,acquisition='capture'):
    return dict(candidate_id=cid,canonical_core_id=cid,alternate_candidate_ids=[],site_id=site,
        acquisition_id=acquisition,crs='EPSG:32632',core_bounds=[left,0,left+1280,1280],
        country='Greece',core_bytes=7680000,flags=[],mask={'provenance':'original_embedded_alpha_mask'},
        native_preview={'sha256':'a'*64},source_sha256='b'*64,license='CC-BY-4.0')


def decision(cid='a',**changes):
    row=dict(candidate_id=cid,decision='accept',reviewer='reviewer1',native_scale_reviewed=True,
        quality_status='pass',duplicate_status='clear',support_status='verified',
        primary_category='marina_small_craft',vessel_presence='present',small_craft_presence='present',hard_negative=False)
    return dict(row,**changes)


def test_geographic_two_kilometre_buffers_bridge_transitively_and_shared_capture():
    rows=[candidate('a','one',500000,'a'),candidate('b','two',505000,'b'),candidate('c','three',510000,'c')]
    groups=geographic_groups(rows,[],set())
    assert len(groups)==1 and len(groups[0]['site_ids'])==3
    rows=[candidate('a','one',500000,'same'),candidate('b','two',600000,'same')]
    assert len(geographic_groups(rows,[],set()))==1


def test_known_training_footprint_never_becomes_holdout():
    row=candidate()
    prior=dict(site_id='old',crs=row['crs'],bounds=row['core_bounds'])
    result=select_release({'candidates':[row]},[decision()],[],prior_training=[prior])
    assert result['splits']['groups'][0]['split']=='train'
    assert not result['gates']['full_release_passed']


def test_missing_explicit_native_review_cannot_accept():
    result=select_release({'candidates':[candidate()]},[],[])
    assert result['size_audit']['accepted_candidate_count']==0
    assert result['status']=='NO_REVIEWED_SELECTION'
    result=select_release({'candidates':[candidate()]},[decision(quality_status='pending')],[])
    assert result['size_audit']['accepted_candidate_count']==0


def test_unknown_geography_is_rejected_with_measured_gate_failure():
    row=candidate()
    row['core_bounds']=None
    result=select_release({'candidates':[row]},[decision()],[])
    assert result['size_audit']['accepted_candidate_count']==0
    assert not result['gates']['requirements']['geographic_split_integrity']
    assert 'Unresolved geography' in result['rejections'][0]['reason']


def test_useful_coastline_with_unconfirmed_tiny_craft_remains_eligible():
    row=decision(primary_category='coastline',vessel_presence='unconfirmed',small_craft_presence='unconfirmed')
    result=select_release({'candidates':[candidate()]},[row],[])
    assert result['size_audit']['accepted_candidate_count']==1
    assert result['size_audit']['confirmed_vessel_count']==0
    assert result['size_audit']['hard_negative_count']==0


def test_positive_selection_is_provisional_when_size_diversity_cap_or_audit_fail():
    result=select_release({'candidates':[candidate()]},[decision()],[])
    assert result['status']=='PROVISIONAL_SELECTION'
    assert result['gates']['requirements']['site_concentration'] is False
    assert result['gates']['requirements']['semantic_audit'] is False
    assert result['gates']['full_release_passed'] is False
    assert result['size_audit']['target_shortfall_bytes']==60000000000-7680000


def test_audit_must_be_independent_and_aliases_resolve():
    row=candidate()
    row['alternate_candidate_ids']=['alias']
    audit=dict(decision('alias'),decision='agree',independent=True,reviewer='reviewer2')
    result=select_release({'candidates':[row]},[decision('alias')],[audit],development_sites={'site'})
    assert not result['gates']['requirements']['semantic_audit']
    assert result['development_audit_report']['independent_agreements']==1
    assert result['splits']['groups'][0]['split']=='development'
    assert result['selection']==[]
    assert result['development_selection'][0]['canonical_core_id']=='a'
    audit['reviewer']='reviewer1'
    assert not select_release({'candidates':[row]},[decision('alias')],[audit])['gates']['requirements']['semantic_audit']


def test_development_is_excluded_from_final_mother_size_and_quotas(monkeypatch):
    import scripts.maritime_release_select as module
    # Scale core bytes to expose the reported 60 GB vs actual 54 GB fixture
    # without constructing thousands of irrelevant raster footprints.
    monkeypatch.setattr(module,'TARGET_CORES',100)
    monkeypatch.setattr(module,'CORE_BYTES',600000000)
    rows=[candidate(str(i),'site'+str(i),500000+i*10000,'capture'+str(i)) for i in range(100)]
    development={'site'+str(i) for i in range(70,80)}
    groups=geographic_groups(rows,[],development)
    assignment={}
    for group in groups:
        index=int(group['site_ids'][0][4:])
        assignment[group['geographic_group_id']]='train' if index<70 else 'development' if index<80 else 'test' if index<90 else 'validation'
    result=select_release({'candidates':rows},[decision(str(i)) for i in range(100)],[],
        prior_training=[],development_sites=development,frozen_splits=assignment)
    assert result['size_audit']['accepted_candidate_count']==90
    assert result['size_audit']['accepted_unique_core_bytes']==54000000000
    assert result['size_audit']['development_unique_core_bytes']==6000000000
    assert not result['gates']['requirements']['size']
    assert len(result['development_selection'])==10
    assert all(row['inspected_for_development'] for row in result['development_selection'])
    assert set(result['size_audit']['split_core_counts'])=={'train','validation','test'}


def test_repeat_observation_requires_meaningful_change_annotation():
    rows=[candidate('a',acquisition='old'),candidate('b',acquisition='new')]
    rows[1]['flags']=['near_duplicate_review_pending']
    result=select_release({'candidates':rows},[decision('a'),decision('b')],[])
    assert result['size_audit']['accepted_candidate_count']==1
    result=select_release({'candidates':rows},[decision('a'),decision('b',repeat_observation_reason='New vessel arrangement visible')],[])
    assert result['size_audit']['accepted_candidate_count']==2


def test_balanced_categories_cap_presence_and_independent_audit():
    categories=['marina_small_craft']*8+['commercial_fishing_port']*5+['coastline']*5+['open_water']*2
    rows,reviews,audits=[],[],[]
    for index,category in enumerate(categories):
        cid=str(index)
        row=candidate(cid,'site'+str(index//2),500000+(index//2)*10000,'capture'+cid)
        row['country']=['Greece','Namibia','USA'][index%3]
        rows.append(row)
        negative=index==19
        review=decision(cid,primary_category=category,hard_negative=negative,
            vessel_presence='absent' if negative else 'confirmed',
            small_craft_presence='confirmed' if category=='marina_small_craft' else 'absent')
        reviews.append(review)
        audits.append(dict(review,decision='agree',reviewer='independent',independent=True))
    result=select_release({'candidates':rows},reviews,audits)
    for gate in ['category_balance','site_concentration','confirmed_vessels','confirmed_small_craft','hard_negatives','semantic_audit']:
        assert result['gates']['requirements'][gate]
    assert not result['gates']['full_release_passed']


def test_audit_disagreement_needs_explicit_resolution():
    disagree=dict(decision(),decision='disagree',reviewer='independent',independent=True)
    agree=dict(disagree,decision='agree')
    result=select_release({'candidates':[candidate()]},[decision()],[disagree,agree])
    assert not result['gates']['requirements']['semantic_audit']
    agree['resolution_reference']='review-discussion-1'
    assert select_release({'candidates':[candidate()]},[decision()],[disagree,agree])['gates']['requirements']['semantic_audit']


def test_new_bridge_cannot_reassign_frozen_holdout_sites():
    from scripts.maritime_release_select import assign_splits
    groups=[dict(geographic_group_id='new',site_ids=['one','two'],prior_training=False,development=False)]
    frozen=dict(assignment={'old1':'train','old2':'test'},groups=[
        dict(geographic_group_id='old1',site_ids=['one'],split='train'),
        dict(geographic_group_id='old2',site_ids=['two'],split='test')])
    with pytest.raises(ValueError,match='bridge'):
        assign_splits(groups,[],frozen)
