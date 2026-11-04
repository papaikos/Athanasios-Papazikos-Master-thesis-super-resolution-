import hashlib

import numpy as np
import pytest
from scipy.ndimage import convolve, gaussian_filter

from src.maritime_recipes import _mother_area_weights, array_hash, canonical_hr, degrade
from src.skysat_prepare import degrade_rgb


def constant(value=0.5):
    return np.full((3, 544, 544), value, dtype=np.float32)


def test_canonical_constant_and_core_share_exact_bounds():
    context, hr = canonical_hr(np.full((3, 1700, 1700), 128, np.uint8),
                               np.ones((1700, 1700), bool))
    assert context.shape == (3, 544, 544)
    assert hr.shape == (3, 512, 512)
    assert context.dtype == hr.dtype == np.float32
    np.testing.assert_allclose(context, 128 / 255, atol=1e-7)
    np.testing.assert_array_equal(hr, context[:, 16:-16, 16:-16])


def test_canonical_noninteger_weights_use_every_overlap():
    rgb = np.zeros((3, 1700, 1700), np.uint8)
    rgb[:, :, 3] = 255
    context, _ = canonical_hr(rgb, np.ones((1700, 1700), bool))
    # First output covers [0,3.125), next [3.125,6.25).
    np.testing.assert_allclose(context[:, :, 0], 0.125 / 3.125, atol=1e-7)
    np.testing.assert_allclose(context[:, :, 1], 0.875 / 3.125, atol=1e-7)
    assert not context[:, :, 2:].any()


def test_noninteger_area_accumulates_float32_and_preserves_integral():
    weights = _mother_area_weights()
    assert weights.dtype == np.float32
    np.testing.assert_allclose(np.asarray(weights.sum(axis=1)), 1, atol=1e-7)
    image = np.random.default_rng(2).random((1700, 1700), dtype=np.float32)
    first_pass = weights @ image
    second_pass = weights @ first_pass.T
    assert first_pass.dtype == second_pass.dtype == np.float32
    # Audit the conserved integral with float64 summation so NumPy's platform-
    # dependent float32 mean rounding is not mistaken for a resampling failure.
    np.testing.assert_allclose(second_pass.mean(dtype=np.float64),
                               image.mean(dtype=np.float64), atol=1e-7)


def test_canonical_two_dimensional_boundary_overlap():
    rgb = np.zeros((3, 1700, 1700), np.uint8)
    rgb[0, 3, 3] = 255
    context, _ = canonical_hr(rgb, np.ones((1700, 1700), bool))
    expected = np.array([[0.125**2, 0.125 * 0.875],
                         [0.125 * 0.875, 0.875**2]]) / 3.125**2
    np.testing.assert_allclose(context[0, :2, :2], expected, atol=1e-8)
    assert not context[1:].any()


@pytest.mark.parametrize('mutation', ['halo', 'core', 'shape', 'dtype', 'mask_dtype'])
def test_canonical_rejects_incomplete_support_and_wrong_format(mutation):
    rgb = np.zeros((3, 1700, 1700), np.uint8)
    mask = np.ones((1700, 1700), bool)
    if mutation == 'halo':
        mask[0, 0] = False
    elif mutation == 'core':
        mask[850, 850] = False
    elif mutation == 'shape':
        rgb = rgb[:, :-1]
    elif mutation == 'dtype':
        rgb = rgb.astype(np.float32)
    else:
        mask = mask.astype(np.uint8)
    with pytest.raises(ValueError):
        canonical_hr(rgb, mask)


def test_d2_constant_and_shared_masks():
    result = degrade(constant(), 'sample', 'd2')
    np.testing.assert_allclose(result['lr'], 0.5, atol=1e-7)
    assert result['lr'].shape == (3, 128, 128)
    assert result['lr'].dtype == np.float32
    assert result['hr_mask'].shape == (512, 512)
    assert result['lr_mask'].shape == (128, 128)
    assert result['hr_mask'].all() and result['lr_mask'].all()
    assert result['metadata']['recipe'] == 'd2-gaussian-area-context-v1'


def test_d2_alignment_and_real_context_historical_interior():
    rng = np.random.default_rng(8)
    context = rng.random((3, 544, 544), dtype=np.float32)
    result = degrade(context, 'sample', 'd2')['lr']
    expected = gaussian_filter(context, (0, 2, 2), truncate=4, mode='reflect')
    expected = expected.reshape(3, 136, 4, 136, 4).mean((2, 4))[:, 4:-4, 4:-4]
    np.testing.assert_array_equal(result, expected)
    historical, _ = degrade_rgb(context[:, 16:-16, 16:-16], np.ones((512, 512), bool))
    np.testing.assert_array_equal(result[:, 2:-2, 2:-2], historical[:, 2:-2, 2:-2])
    assert np.max(np.abs(result - historical)) > 1e-4
    # Symmetric signal is centred on the LR pixel containing HR pixels 216..219.
    impulse = constant(0)
    impulse[0, 216:220, 216:220] = 1
    lr = degrade(impulse, 'impulse', 'd2')['lr']
    assert np.unravel_index(lr[0].argmax(), lr[0].shape) == (50, 50)
    assert not lr[1:].any()


def test_d1_matches_pinned_torch_coordinate_convention():
    torch = pytest.importorskip('torch')
    context = np.random.default_rng(9).random((3, 544, 544), dtype=np.float32)
    expected = torch.nn.functional.interpolate(torch.from_numpy(context[None]),
        scale_factor=0.25, mode='bicubic', align_corners=False, antialias=True)
    expected = np.clip(expected.numpy()[0, :, 4:-4, 4:-4], 0, 1)
    result = degrade(context, 'sample', 'd1')
    np.testing.assert_array_equal(result['lr'], expected)
    np.testing.assert_allclose(degrade(constant(), 'constant', 'd1')['lr'], 0.5, atol=1e-6)
    impulse = constant(0)
    impulse[0, 216:220, 216:220] = 1
    lr = degrade(impulse, 'impulse', 'd1')['lr']
    assert np.unravel_index(lr[0].argmax(), lr[0].shape) == (50, 50)
    assert not lr[1:].any()


def test_d3_retry_and_order_independence_and_variant_identity():
    context = constant()
    first = degrade(context, 'a', 'd3')
    degrade(context, 'b', 'd3')
    repeated = degrade(context, 'a', 'd3')
    np.testing.assert_array_equal(first['lr'], repeated['lr'])
    assert first['metadata'] == repeated['metadata']
    variant = degrade(context, 'a', 'd3', variant_id='1')
    assert first['metadata']['parameter_seed'] != variant['metadata']['parameter_seed']
    assert first['metadata']['noise_seed'] != variant['metadata']['noise_seed']


def test_d3_reconstructs_recorded_kernel_and_noise():
    context = np.random.default_rng(18).random((3, 544, 544), dtype=np.float32)
    result = degrade(context, 'reconstruct', 'd3')
    meta = result['metadata']
    y, x = np.mgrid[-12:13, -12:13]
    angle = meta['angle_radians']
    rx = np.cos(angle) * x + np.sin(angle) * y
    ry = -np.sin(angle) * x + np.cos(angle) * y
    kernel = np.exp(-0.5 * ((rx / meta['sigma_x_hr_pixels']) ** 2 +
                           (ry / meta['sigma_y_hr_pixels']) ** 2))
    kernel = (kernel / kernel.sum()).astype(np.float32)
    assert array_hash(kernel) == meta['kernel_sha256']
    assert kernel.shape == (25, 25) and (kernel >= 0).all()
    assert kernel.sum() == pytest.approx(1, abs=1e-7)
    expected = convolve(context, kernel[None], mode='reflect', output=np.float32)
    expected = expected.reshape(3, 136, 4, 136, 4).mean((2, 4))[:, 4:-4, 4:-4]
    rng = np.random.Generator(np.random.PCG64(meta['noise_seed']))
    if meta['noise_sigma_normalized']:
        expected = expected + rng.normal(0, meta['noise_sigma_normalized'], expected.shape).astype(np.float32)
    assert meta['clipping_fraction'] == np.count_nonzero((expected < 0) | (expected > 1)) / expected.size
    np.testing.assert_array_equal(result['lr'], np.clip(expected, 0, 1))
    assert 1 <= meta['sigma_x_hr_pixels'] <= 3
    assert 1 <= meta['sigma_y_hr_pixels'] <= 3
    assert 0 <= meta['angle_radians'] < np.pi
    assert 0 <= meta['noise_sigma_normalized'] <= 2 / 255


def test_d3_no_noise_preserves_constant_and_isotropic_parameters():
    # Find a frozen zero-noise realization using the cheap constant input.
    for index in range(30):
        result = degrade(constant(), str(index), 'd3')
        meta = result['metadata']
        if meta['kernel_type'] == 'isotropic':
            assert meta['sigma_x_hr_pixels'] == meta['sigma_y_hr_pixels']
            assert meta['angle_radians'] == 0
        if meta['noise_sigma_normalized'] == 0:
            np.testing.assert_allclose(result['lr'], 0.5, atol=1e-6)
            break
    else:
        pytest.fail('No zero-noise realization among fixed fixture IDs')


def test_d3_clips_noisy_dark_water_once_and_release_seed_changes_streams():
    result = degrade(constant(0), 'clipping', 'd3')
    meta = result['metadata']
    assert meta['noise_sigma_normalized'] > 0
    assert 0.45 < meta['clipping_fraction'] < 0.55
    assert result['lr'].min() == 0 and result['lr'].max() <= 1
    assert np.isfinite(result['lr']).all()
    assert array_hash(result['lr']) == meta['lr_sha256']
    changed = degrade(constant(0), 'clipping', 'd3', release_seed=20261007)
    assert changed['metadata']['parameter_seed'] != meta['parameter_seed']
    assert changed['metadata']['noise_seed'] != meta['noise_seed']


@pytest.mark.parametrize('invalid', ['shape', 'dtype', 'nan', 'range', 'recipe'])
def test_degrade_rejects_bad_context(invalid):
    context = constant()
    recipe = 'd2'
    if invalid == 'shape':
        context = context[:, :-1]
    elif invalid == 'dtype':
        context = context.astype(np.float64)
    elif invalid == 'nan':
        context[0, 0, 0] = np.nan
    elif invalid == 'range':
        context[0, 0, 0] = 1.1
    else:
        recipe = 'bad'
    with pytest.raises(ValueError):
        degrade(context, 'sample', recipe)


def test_array_hash_normalizes_order_and_endianness():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    expected = hashlib.sha256(values.astype('<f4').tobytes(order='C')).hexdigest()
    assert array_hash(values) == expected
    assert array_hash(np.asfortranarray(values)) == expected
    assert array_hash(values.astype('>f4')) == expected
