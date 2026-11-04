import pytest
from scripts.maritime_aerial_exposure_census import normalize_row, build_population, validate_coverage


def row():
    return dict(sample_id='one',site_id='site',acquisition_id='capture',source_kind='maxar',split='test',crs='EPSG:32611',core_bounds=[400000,3700000,401280,3701280],source=dict(source_original_object=dict(sha256='a'*64),physical_context_bounds=[399960,3699960,401320,3701320]))


def test_common_rows_preserve_true_split_parent_and_expand_context():
    out=normalize_row(row(),'common')
    assert out['split']=='test' and out['source']['source_original_object']['sha256']=='a'*64
    assert out['core_bounds']==[399960,3699960,401320,3701320]


def test_missing_geometry_or_identity_fails_closed():
    bad=row();bad.pop('site_id')
    with pytest.raises(ValueError):normalize_row(bad,'common')
    bad=row();bad['core_bounds']=None;bad['source']={}
    with pytest.raises(ValueError):normalize_row(bad,'common')


def test_inventory_uses_entire_original_parent_and_all_ports():
    source=dict(source_id='parent',source_type='naip_aerial',acquisition_day='2022-06-01',epsg=32611,projected_bounds=[400000,3700000,410000,3710000],port_AOI_matches=[dict(port_id='1'),dict(port_id='2')],href='https://original')
    out=normalize_row(source,'aerial_parent');assert out['core_bounds']==source['projected_bounds']
    assert out['source_parent_urls']==['https://original'] and out['site_aliases']==['aerial-port-1','aerial-port-2']


def test_coverage_must_be_explicit_and_never_omit_reviewed_id():
    p=build_population('source_development',[(dict(sha256='b'*64),[normalize_row(row(),'common')])])
    assert p['complete'] is False
    assert validate_coverage(p,[dict(name='review',sample_ids=['one'])])['covered']
    with pytest.raises(ValueError,match='missing'):validate_coverage(p,[dict(name='model',sample_ids=['unknown'])])


def test_parent_coverage_requires_matching_original_url_and_enclosing_footprint():
    from scripts.maritime_aerial_exposure_census import cover_rows
    original=dict(source_id='parent',source_type='naip_aerial',epsg=32611,projected_bounds=[400000,3700000,410000,3710000],port_AOI_matches=[dict(port_id='1')],href='https://original')
    parent=normalize_row(original,'aerial_parent')
    child=row();child['core_bounds']=[401000,3701000,402280,3702280];child['source']['physical_context_bounds']=[400960,3700960,402320,3702320];child['source']['source_href']='https://original'
    assert cover_rows([parent],[normalize_row(child,'common')])['covered_count']==1
    child['source']['source_href']='https://other'
    with pytest.raises(ValueError,match='uncovered'):cover_rows([parent],[normalize_row(child,'common')])
