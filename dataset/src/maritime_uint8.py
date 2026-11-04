"""Versioned final quantization of existing, fully observed maritime pairs.

Degradation takes place in the original float32 recipe before this module runs.
Quantization is lossy; the original float hashes remain provenance, whereas
storage hashes describe the actual decoded uint8 bytes.
"""
import copy
import hashlib

import numpy as np

from src.maritime_recipes import RECIPE_VERSIONS, array_hash


STORAGE_VERSION = 'maritime-rgb-uint8-v1'
QUANTIZATION = 'floor(clip(x, 0, 1) * 255 + 0.5)'
RGB_SHAPES = {'hr': (3, 512, 512), 'lr': (3, 128, 128)}
RGB_PAYLOAD_BYTES = 835584


def quantize_rgb(normalized):
    """Half-up rounding with float64 arithmetic on supplied finite floats."""
    values = np.asarray(normalized)
    if values.dtype.kind != 'f' or not np.isfinite(values).all():
        raise ValueError('Quantization requires finite floating-point values')
    scaled = np.clip(values.astype(np.float64), 0, 1) * 255.0
    return np.ascontiguousarray(np.floor(scaled + 0.5), dtype=np.uint8)


def uint8_array_hash(array):
    """SHA-256 of decoded C-contiguous uint8 bytes, without NPZ headers."""
    if not isinstance(array, np.ndarray) or array.dtype != np.uint8:
        raise ValueError('Byte hash requires a uint8 array')
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def decoded_rgb_bytes(pair):
    """Count only HR and LR RGB payload; exclude masks and container bytes."""
    for key, shape in RGB_SHAPES.items():
        if not isinstance(pair[key], np.ndarray) or pair[key].shape != shape or pair[key].dtype != np.uint8:
            raise ValueError('Invalid uint8 training array: ' + key)
    return sum(pair[key].nbytes for key in RGB_SHAPES)


def validate_provenance(metadata, recipe, sample_id):
    """Validate the preserved float recipe identity and common physical grid."""
    try:
        provenance = metadata['recipe']
        if (recipe not in RECIPE_VERSIONS or provenance['recipe'] != RECIPE_VERSIONS[recipe]
                or provenance['sample_id'] != sample_id or metadata['sample_id'] != sample_id
                or not isinstance(sample_id, str) or not sample_id
                or metadata['lr_origin'] != 'synthetic_hr_only'):
            raise ValueError('Training recipe/provenance identity mismatch')
        if metadata['hr_sha256'] != provenance['hr_sha256']:
            raise ValueError('Common float HR hash mismatch')
        for key in ('hr_sha256', 'lr_sha256'):
            value = provenance[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Invalid original float array hash')
        hr, lr = metadata['hr_transform'], metadata['lr_transform']
        if (not isinstance(hr, list) or not isinstance(lr, list) or len(hr) != 6 or len(lr) != 6
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in hr + lr)
                or not np.isfinite(hr + lr).all()
                or hr[:2] != [2.5, 0] or hr[3:5] != [0, -2.5]
                or lr[:2] != [10, 0] or lr[3:5] != [0, -10]
                or hr[2] != lr[2] or hr[5] != lr[5]):
            raise ValueError('Training grid geometry mismatch')
    except (KeyError, TypeError) as error:
        raise ValueError('Malformed training provenance') from error


def encode_uint8_pair(hr, lr, metadata, *, hr_mask=None, lr_mask=None):
    """Encode a fully supported original pair and copy its float provenance.

    Omitted masks declare full support, as required by the original release.
    Callers retaining source masks can supply them explicitly for validation.
    """
    try:
        version = metadata['recipe']['recipe']
        recipe = next(key for key, value in RECIPE_VERSIONS.items() if value == version)
        validate_provenance(metadata, recipe, metadata['sample_id'])
    except (KeyError, TypeError, StopIteration) as error:
        raise ValueError('Unknown or malformed original recipe') from error
    pair = {}
    for key, values in [('hr', hr), ('lr', lr)]:
        if (not isinstance(values, np.ndarray) or values.shape != RGB_SHAPES[key] or values.dtype != np.float32
                or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()):
            raise ValueError('Invalid original float training array: ' + key)
        if array_hash(values) != metadata['recipe'][key + '_sha256']:
            raise ValueError('Original float array hash mismatch: ' + key)
        pair[key] = quantize_rgb(values)
    for key, mask, side in [('hr_mask', hr_mask, 512), ('lr_mask', lr_mask, 128)]:
        if mask is None:
            mask = np.ones((side, side), bool)
        if not isinstance(mask, np.ndarray) or mask.shape != (side, side) or mask.dtype != np.bool_ or not mask.all():
            raise ValueError('Invalid full-support training mask: ' + key)
        pair[key] = np.array(mask, copy=True)
    encoded = copy.deepcopy(metadata)
    encoded['storage'] = dict(version=STORAGE_VERSION, dtype='uint8',
        quantization=QUANTIZATION, quantization_arithmetic='float64 on original float32 values',
        quantization_order='after all original float recipe operators and final clipping',
        array_hash_encoding='C-contiguous uint8 bytes',
        rgb_payload_bytes=decoded_rgb_bytes(pair),
        hr_sha256=uint8_array_hash(pair['hr']), lr_sha256=uint8_array_hash(pair['lr']))
    return pair, encoded
