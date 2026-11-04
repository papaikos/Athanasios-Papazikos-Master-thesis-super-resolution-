from scripts.maritime_expansion_coast import screen_candidates


def candidate():
    return dict(sample_id='test', source_kind='skysat', source={}, site_id='port',
        acquisition_id='capture', crs='EPSG:32631', core_bounds=[0, 0, 1280, 1280])


class Coast:
    def __init__(self, status):
        self.status = status

    def classify(self, record, margin_m):
        assert record['physical_core_bounds'] == [0, 0, 1280, 1280]
        assert record['source']['epsg'] == 32631
        assert margin_m == 1000
        return dict(status=self.status, marine_area_m2=100, shore_distance_m=0)


def test_marine_prior_preserves_source_and_grants_no_quality_approval():
    original = candidate()
    result = screen_candidates(dict(candidates=[original]), Coast('MARINE_INTERSECTION'))
    assert result['candidates'][0]['source'] == original['source']
    assert 'screening' not in result['candidates'][0]
    assert 'marine_geography_prior' not in original
    assert result['candidates'][0]['marine_geography_prior']['establishes_ship_presence'] is False


def test_inland_and_unknown_are_excluded():
    for status in ('INLAND_FAR_FROM_MARINE_COAST', 'UNKNOWN_INVALID_GEOMETRY', 'UNKNOWN_UNSUPPORTED_GEOGRAPHY'):
        result = screen_candidates(dict(candidates=[candidate()]), Coast(status))
        assert result['candidates'] == []
        assert result['exclusions'][0]['sample_id'] == 'test'


def test_declared_coast_margin_is_allowed_as_uncertain_prior():
    result = screen_candidates(dict(candidates=[candidate()]), Coast('COAST_MARGIN_UNCERTAIN'))
    assert result['statuses'] == {'COAST_MARGIN_UNCERTAIN': 1}
