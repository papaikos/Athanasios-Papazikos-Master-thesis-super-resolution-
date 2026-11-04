"""Versioned synthetic maritime recipes on fully observed, real HR context.

The supplied arrays are corner-aligned: 1700 mother pixels span the same bounds
as 544 HR pixels. Removing 16 HR or four LR pixels gives the same core bounds.
This module does not read imagery, fabricate halos, or change historic rgb-v2.
"""

from functools import lru_cache
import hashlib
import json

import numpy as np
import scipy
from scipy.ndimage import convolve, gaussian_filter
from scipy.sparse import csr_matrix


RECIPE_VERSIONS = {
    'd1': 'd1-bicubic-aa-v1',
    'd2': 'd2-gaussian-area-context-v1',
    'd3': 'd3-variable-psf-noise-v1',
}


def array_hash(array):
    """SHA-256 of decoded C-contiguous little-endian float32 values only."""
    return hashlib.sha256(np.ascontiguousarray(array, dtype='<f4').tobytes()).hexdigest()


@lru_cache(maxsize=1)
def _mother_area_weights():
    """Exact interval overlaps for [0,1700) -> [0,544), stored as float32.

    Boundaries are multiples of 25/8, exactly representable in binary. Each
    row includes every intersecting mother pixel, with weight overlap / 3.125.
    Sparse float32 multiplication keeps both separable accumulations float32.
    """
    scale = 3.125
    rows, columns, values = [], [], []
    for index in range(544):
        left, right = index * scale, (index + 1) * scale
        for pixel in range(int(np.floor(left)), int(np.ceil(right))):
            rows.append(index)
            columns.append(pixel)
            values.append((min(right, pixel + 1) - max(left, pixel)) / scale)
    return csr_matrix((np.asarray(values, np.float32), (rows, columns)),
                      shape=(544, 1700), dtype=np.float32)


def canonical_hr(rgb, mask):
    """Return canonical (544-square context, 512-square core), normalized RGB.

    Reject any invalid original support, including the halo: acceptance is a
    common-sample decision before recipe branching, never nodata averaging.
    Geometry/CRS validation belongs to the source reader supplying this window.
    """
    if rgb.shape != (3, 1700, 1700) or rgb.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB with shape (3,1700,1700)')
    if mask.shape != (1700, 1700) or mask.dtype != np.bool_ or not mask.all():
        raise ValueError('Expected fully valid boolean mother core and halo mask')
    normalized = rgb.astype(np.float32) / np.float32(255)
    weights = _mother_area_weights()
    context = np.empty((3, 544, 544), np.float32)
    for channel in range(3):
        reduced_rows = weights @ normalized[channel]
        context[channel] = (weights @ reduced_rows.T).T
    return context, np.ascontiguousarray(context[:, 16:-16, 16:-16])


def _area4(context):
    side = context.shape[1] // 4
    return context.reshape(3, side, 4, side, 4).mean(axis=(2, 4), dtype=np.float32)[:, 4:-4, 4:-4]


def _seed(release_seed, sample_id, recipe_version, variant_id, purpose):
    # JSON is length-delimited, so IDs containing punctuation cannot collide.
    identity = [release_seed, sample_id, recipe_version, variant_id, purpose]
    payload = json.dumps(identity, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest(), 'little')


def degrade(context, sample_id, recipe='d1', release_seed=20261006, variant_id='0'):
    """Build one deterministic LR realization and its shared support masks.

    ``context`` must be frozen float32 normalized RGB, with full original
    support established by ``canonical_hr``. D1 needs the pinned CPU PyTorch
    runtime; no alternate interpolation implementation is substituted.
    """
    if (context.shape not in ((3, 544, 544), (3, 96, 96)) or context.dtype != np.float32
            or not np.isfinite(context).all() or (context < 0).any() or (context > 1).any()):
        raise ValueError('Expected finite float32 RGB context (3,544,544) or observed chip context (3,96,96) in [0,1]')
    # Both layouts retain 16 actual HR halo pixels per side. The small layout
    # is a separately declared curated supplement; no padding creates context.
    side = context.shape[1]
    hr_side, lr_side = side - 32, side // 4 - 8
    if recipe not in RECIPE_VERSIONS:
        raise ValueError('Unknown degradation recipe')
    if not isinstance(sample_id, str) or not sample_id or not isinstance(variant_id, str):
        raise ValueError('Stable nonempty sample ID and string variant ID required')
    if not isinstance(release_seed, int) or isinstance(release_seed, bool):
        raise ValueError('Integer release seed required')
    version = RECIPE_VERSIONS[recipe]
    meta = dict(recipe=version, sample_id=sample_id, variant_id=variant_id,
                release_seed=release_seed, numpy_version=np.__version__,
                lr_origin='synthetic_hr_only', scale=0.25,
                context_shape=[3, side, side], hr_shape=[3, hr_side, hr_side],
                lr_shape=[3, lr_side, lr_side], crop_lr_pixels=4,
                context_sha256=array_hash(context),
                hr_sha256=array_hash(context[:, 16:-16, 16:-16]),
                normalization='uint8 / 255, float32',
                array_hash_encoding='C-contiguous little-endian float32',
                clipping='once after LR core crop and optional noise')
    if recipe == 'd1':
        import torch

        with torch.no_grad():
            resized = torch.nn.functional.interpolate(
                torch.from_numpy(np.ascontiguousarray(context))[None], scale_factor=0.25,
                mode='bicubic', align_corners=False, antialias=True)
        lr = resized.numpy()[0, :, 4:-4, 4:-4]
        meta.update(implementation='torch.nn.functional.interpolate CPU',
                    implementation_version=torch.__version__, align_corners=False,
                    antialias=True, coordinate_convention='half-pixel centres',
                    operator_order=['bicubic-antialias', 'crop', 'clip'], noise_sigma_normalized=0.0)
    elif recipe == 'd2':
        blurred = gaussian_filter(context, sigma=(0, 2, 2), truncate=4,
                                  mode='reflect', output=np.float32)
        lr = _area4(blurred)
        meta.update(implementation='scipy.ndimage.gaussian_filter + numpy.mean',
                    implementation_version=scipy.__version__, sigma_hr_pixels=2.0,
                    sigma_meters=5.0, truncate=4.0, kernel_radius_hr_pixels=8,
                    boundary='reflect outer context', reduction='exact 4x4 area mean',
                    operator_order=['gaussian', 'area4', 'crop', 'clip'], noise_sigma_normalized=0.0)
    else:
        parameter_seed = _seed(release_seed, sample_id, version, variant_id, 'parameters')
        noise_seed = _seed(release_seed, sample_id, version, variant_id, 'noise')
        rng = np.random.Generator(np.random.PCG64(parameter_seed))
        isotropic = rng.random() < 0.5
        sigma_x = float(rng.uniform(1, 3))
        sigma_y = sigma_x if isotropic else float(rng.uniform(1, 3))
        angle = 0.0 if isotropic else float(rng.uniform(0, np.pi))
        y, x = np.mgrid[-12:13, -12:13]
        rotated_x = np.cos(angle) * x + np.sin(angle) * y
        rotated_y = -np.sin(angle) * x + np.cos(angle) * y
        kernel = np.exp(-0.5 * ((rotated_x / sigma_x) ** 2 + (rotated_y / sigma_y) ** 2))
        kernel = (kernel / kernel.sum()).astype(np.float32)
        blurred = convolve(context, kernel[None], mode='reflect', output=np.float32)
        lr = _area4(blurred)
        noise_sigma = 0.0 if rng.random() < 0.2 else float(rng.uniform(0, 2) / 255)
        if noise_sigma:
            noise_rng = np.random.Generator(np.random.PCG64(noise_seed))
            lr = lr + noise_rng.normal(0, noise_sigma, lr.shape).astype(np.float32)
        meta.update(implementation='scipy.ndimage.convolve + numpy.mean',
                    implementation_version=scipy.__version__,
                    kernel_type='isotropic' if isotropic else 'anisotropic',
                    sigma_x_hr_pixels=sigma_x, sigma_y_hr_pixels=sigma_y,
                    angle_radians=angle, kernel_shape=[25, 25],
                    kernel_sha256=array_hash(kernel), boundary='reflect outer context',
                    reduction='exact 4x4 area mean', noise_sigma_normalized=noise_sigma,
                    noise_distribution='independent zero-mean Gaussian per channel/pixel',
                    parameter_seed=parameter_seed, noise_seed=noise_seed,
                    seed_derivation='SHA256 UTF-8 compact JSON [release_seed,sample_id,recipe,variant_id,purpose], little-endian uint256',
                    rng='numpy.random.PCG64',
                    operator_order=['gaussian-psf', 'area4', 'crop', 'noise', 'clip'])
    meta['clipping_fraction'] = float(np.count_nonzero((lr < 0) | (lr > 1)) / lr.size)
    lr = np.ascontiguousarray(np.clip(lr, 0, 1), dtype=np.float32)
    meta['lr_sha256'] = array_hash(lr)
    return dict(lr=lr, hr_mask=np.ones((hr_side, hr_side), bool),
                lr_mask=np.ones((lr_side, lr_side), bool), metadata=meta)
