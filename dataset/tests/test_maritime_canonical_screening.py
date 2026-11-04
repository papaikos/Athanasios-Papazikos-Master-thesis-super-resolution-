import hashlib
import io
import json

import numpy as np
import pytest

from scripts import maritime_canonical_screening as screening


def payload(context=None, hr=None):
    if context is None:
        context = np.zeros((3, 544, 544), dtype=np.float32)
        context[:, 16:528, 16:528] = np.float32(0.5)
    if hr is None:
        hr = context[:, 16:528, 16:528].copy()
    stream = io.BytesIO()
    np.savez_compressed(stream, hr=hr, hr_context=context)
    return stream.getvalue()


def test_hashes_use_chw_float32_and_frozen_rounding():
    hr, rgb, hashes = screening.canonical_arrays(payload())
    assert rgb.dtype == np.uint8 and rgb.shape == (3, 512, 512)
    assert np.all(rgb == 128)
    assert hashes['hr_float32_sha256'] == hashlib.sha256(hr.tobytes()).hexdigest()
    assert hashes['hr_uint8_chw_sha256'] == hashlib.sha256(rgb.tobytes()).hexdigest()


@pytest.mark.parametrize('defect', ['dtype', 'layout', 'crop', 'nan', 'range'])
def test_invalid_canonical_arrays_fail(defect):
    context = np.zeros((3, 544, 544), dtype=np.float32)
    hr = context[:, 16:528, 16:528].copy()
    if defect == 'dtype': hr = hr.astype(np.float64)
    if defect == 'layout': hr = hr.transpose(1, 2, 0)
    if defect == 'crop': hr[0, 0, 0] = 0.5
    if defect == 'nan': context[0, 0, 0] = np.nan
    if defect == 'range': context[0, 0, 0] = 1.01
    with pytest.raises(ValueError): screening.canonical_arrays(payload(context, hr))


@pytest.mark.parametrize('area', [-1, float('nan'), True, 1280**2 + 1])
def test_invalid_historical_area_fails(area):
    with pytest.raises(ValueError): screening.historical_water_fraction({'marine_screening': {'marine_area_m2': area}, 'physical_core_bounds': [0, 0, 1280, 1280]})


def test_geometry_must_be_physical_1280m():
    source = {'marine_screening': {'marine_area_m2': 1280**2 / 2}, 'physical_core_bounds': [0, 0, 1280, 1280]}
    assert screening.historical_water_fraction(source) == 0.5
    source['physical_core_bounds'][2] = 512
    with pytest.raises(ValueError): screening.historical_water_fraction(source)


def test_constant_image_is_deferred_and_global_normalization_is_exact():
    assert screening.cloud_input(np.zeros((3, 512, 512), np.uint8)) is None
    rgb = np.zeros((3, 512, 512), np.uint8)
    rgb[1] = 128
    rgb[2] = 255
    tensor = screening.cloud_input(rgb)
    assert tensor.shape == (3, 256, 256)
    assert np.all(tensor[0] == 0) and np.all(tensor[2] == 1)
    assert np.all(tensor[1] == np.float32(128/255))


def test_result_is_compact_and_preserves_support_uncertainty():
    pin = {'uri': 'gs://bucket/x', 'generation': '1', 'bytes': 1, 'crc32c': 'AAAAAA==', 'sha256': 'a'*64}
    source = {'sample_id': 'earth-1', 'source_kind': 'earthview', 'marine_screening': {'marine_area_m2': 0}, 'physical_core_bounds': [0, 0, 1280, 1280]}
    result = screening.result_record(source, pin, pin, pin, {'hr_float32_sha256': 'b'*64, 'hr_uint8_chw_sha256': 'c'*64}, 0.02, True)
    assert 'source_candidate' not in result
    assert result['original_assembly_replayed'] is False
    assert result['original_support_verified'] is True
    assert result['final_eligible'] is False
    assert 'declared nodata' in result['original_support_verification_basis']


def test_standalone_source_identity_is_verified_before_reading_hr():
    raw = json.dumps({'sample_id': 'wrong', 'hr': {'x': 1}}).encode()
    ref = {'sample_id': 'expected', 'candidate_pin': {'sha256': hashlib.sha256(raw).hexdigest()}, 'hr_pin': {'x': 1}}
    calls = []
    def fetch(pin): calls.append(pin); return raw
    with pytest.raises(ValueError): screening.prepare_candidate(ref, fetch, validator=lambda source: None)
    assert len(calls) == 1


def test_valid_standalone_source_and_exact_hr_pin_are_preserved():
    hrpin = {'sha256': hashlib.sha256(payload()).hexdigest()}
    source = {'sample_id': 'earth-1', 'hr': hrpin, 'marine_screening': {'marine_area_m2': 1280**2 / 2}, 'physical_core_bounds': [0, 0, 1280, 1280]}
    raw = json.dumps(source).encode()
    ref = {'sample_id': 'earth-1', 'candidate_pin': {'sha256': hashlib.sha256(raw).hexdigest()}, 'hr_pin': hrpin}
    prepared = screening.prepare_candidate(ref, lambda pin: raw if pin is ref['candidate_pin'] else payload(), validator=lambda source: None)
    assert prepared[0] == source and prepared[1] is ref
    assert prepared[3] is None


def test_resume_rejects_wrong_policy_or_hash_layout():
    pin = {'sha256': 'a'*64}
    source = {'sample_id': 'earth-1', 'marine_screening': {'marine_area_m2': 0}, 'physical_core_bounds': [0, 0, 1280, 1280]}
    record = screening.result_record(source, pin, pin, pin, {'hr_float32_sha256': 'b'*64, 'hr_uint8_chw_sha256': 'c'*64}, None, False)
    ref = {'sample_id': 'earth-1', 'candidate_pin': pin, 'hr_pin': pin}
    screening.validate_resumed(record, ref, pin)
    with pytest.raises(ValueError): screening.validate_resumed(record, ref, {'sha256': 'd'*64})
    record['hr_uint8_chw_sha256'] = 'wrong'
    with pytest.raises(ValueError): screening.validate_resumed(record, ref, pin)


def test_policy_fails_closed_without_explicit_earth_scope_and_locks():
    with pytest.raises(ValueError): screening.validate_policy({'source_family': 'naip_aerial'})


@pytest.mark.parametrize('field,value', [('historical_water_fraction', float('nan')), ('estimated_cloud_fraction', 2), ('finite_rgb', False), ('original_support_verified', False), ('nonconstant_rgb', True)])
def test_resume_rejects_invalid_measurements_or_support(field, value):
    pin = {'sha256': 'a'*64}
    source = {'sample_id': 'earth-1', 'marine_screening': {'marine_area_m2': 0}, 'physical_core_bounds': [0, 0, 1280, 1280]}
    record = screening.result_record(source, pin, pin, pin, {'hr_float32_sha256': 'b'*64, 'hr_uint8_chw_sha256': 'c'*64}, None, False)
    record[field] = value
    with pytest.raises(ValueError): screening.validate_resumed(record, {'sample_id': 'earth-1', 'candidate_pin': pin, 'hr_pin': pin}, pin)
