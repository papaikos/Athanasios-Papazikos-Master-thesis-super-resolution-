import pytest
from scripts.maritime_aerial_evaluation import site_allocation,allocate_source


def source(ids):
    return dict(epsg=32631,projected_bounds=[500000,0,503000,3000],
        port_AOI_matches=[dict(port_id=i) for i in ids])


def ids_for(role):
    return [str(i) for i in range(1000) if site_allocation(str(i))==role][:2]


def test_site_split_is_stable_across_flights_and_uses_ten_percent_hash():
    assert site_allocation('16070')==site_allocation('16070')
    assert 60<sum(site_allocation(str(i))=='evaluation' for i in range(1000))<140
    with pytest.raises(ValueError):site_allocation('')


def test_parent_mixed_sites_are_held_and_single_domain_parents_assigned():
    train=ids_for('train');evaluation=ids_for('evaluation')
    assert allocate_source(source(train),[])['role']=='train'
    assert allocate_source(source(evaluation),[])['role']=='evaluation'
    assert allocate_source(source([train[0],evaluation[0]]),[])['role']=='hold'


def test_training_parent_near_evaluation_port_is_held_even_without_port_match():
    evaluation_port=dict(port_id=ids_for('evaluation')[0],longitude=3.,latitude=.01)
    decision=allocate_source(source(ids_for('train')),[evaluation_port])
    assert decision['role']=='hold' and decision['reason']=='evaluation_geography_reservation'


def test_parent_without_stable_port_identity_is_not_training_eligible():
    assert allocate_source(source([]),[])['role']=='hold'
    with pytest.raises(ValueError):allocate_source(source(['']),[])
