import copy

import pytest

from scripts.maritime_expansion_inventory import normalize_candidate, normalize_manifest


def earth(epsg=32654, timestamp='2022-10-31T05:31:04.315342+00:00'):
    return dict(schema='earthview-candidate-v1', source_kind='earthview', sample_id='earthview-example',
        acquisition_identity=[timestamp, 'newsat21', epsg, 1., 'L1'],
        source_crs='EPSG:'+str(epsg), physical_core_bounds=[0, 0, 1280, 1280],
        port=dict(id=123, country='JP', size='L'), final_eligible=False,
        requires_semantic_review=True, requires_native_quality_review=True)


def test_earth_capture_identity_survives_crs_and_timestamp_encoding():
    a=normalize_candidate(earth())
    b=normalize_candidate(earth(32653, '2022-10-31T05:31:04.315342Z'))
    assert a['acquisition_id']==b['acquisition_id']
    assert a['crs']!=b['crs']
    assert a['site_id']=='earthview-port-123'
    assert a['vessel_presence']=='unconfirmed'
    assert 'screening' not in a and 'split' not in a


def test_preserves_source_and_wrapped_sky_without_approving_it():
    original=earth(); snapshot=copy.deepcopy(original)
    assert normalize_candidate(original)['source']==snapshot
    assert original==snapshot
    sky=dict(sample_id='sky', source_kind='skysat_raw_window', source={'pinned':True},
             acquisition_id='catalog', site_id='site', crs='EPSG:32631', core_bounds=[0,0,1280,1280])
    normalized=normalize_candidate(sky)
    assert normalized==sky and normalized is not sky and normalized['source'] is not sky['source']


def test_maxar_uses_original_capture_and_published_port():
    source=dict(schema='maxar-mixed-native-v1', sample_id='maxar',
        source=dict(catalog_id='worldview-capture',epsg=32631),
        nearest_port=dict(port_id=42,country_code_as_published='US'),
        physical_core_bounds=[0,0,1280,1280])
    row=normalize_candidate(source)
    assert row['acquisition_id']=='worldview-capture'
    assert row['site_id']=='maxar-port-42' and row['source']==source
    assert row['requires_stratified_native_audit'] is True and 'screening' not in row


def test_invalid_and_duplicate_sources_fail_closed():
    with pytest.raises(ValueError,match='Unsupported'):
        normalize_candidate(dict(sample_id='unknown',source_kind='unknown'))
    with pytest.raises(ValueError,match='Duplicate'):
        normalize_manifest(dict(candidates=[earth(),earth()]))
    bad=earth(); bad['acquisition_identity'][0]='2022-10-31T05:31:04'
    with pytest.raises(ValueError,match='timezone'):
        normalize_candidate(bad)
