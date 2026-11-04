import copy
import hashlib
import io
import json
import tarfile

import numpy as np
import pytest

from src.maritime_recipes import RECIPE_VERSIONS, array_hash


def inputs(recipe='d2'):
    hr = np.full((3, 512, 512), 0.5, np.float32)
    lr = np.full((3, 128, 128), 0.25, np.float32)
    metadata = dict(sample_id='s', split='train', lr_origin='synthetic_hr_only',
        hr_sha256=array_hash(hr), hr_transform=[2.5, 0, 0, 0, -2.5, 1280],
        lr_transform=[10, 0, 0, 0, -10, 1280],
        recipe=dict(recipe=RECIPE_VERSIONS[recipe], sample_id='s',
                    hr_sha256=array_hash(hr), lr_sha256=array_hash(lr)))
    return hr, lr, metadata


def write_shard(tmp_path, pair, metadata, filename='pairs.tar'):
    stream = io.BytesIO()
    np.savez_compressed(stream, **pair, provenance=np.array(json.dumps(metadata)))
    payload = stream.getvalue()
    path = tmp_path / filename
    with tarfile.open(path, 'w') as archive:
        info = tarfile.TarInfo('train/s.npz')
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return path


def test_quantization_half_up_clips_and_hashes_actual_uint8_bytes():
    from src.maritime_uint8 import quantize_rgb, uint8_array_hash
    values = np.array([-1, 0, 0.5 / 255, 1.5 / 255, 0.5, 1, 2], np.float32)
    encoded = quantize_rgb(values)
    np.testing.assert_array_equal(encoded, [0, 0, 1, 2, 128, 255, 255])
    assert encoded.dtype == np.uint8
    assert uint8_array_hash(encoded) == hashlib.sha256(encoded.tobytes()).hexdigest()
    with pytest.raises(ValueError):
        quantize_rgb(np.array([np.nan], np.float32))


def test_encoder_keeps_float_recipe_provenance_and_counts_only_rgb():
    from src.maritime_uint8 import encode_uint8_pair, decoded_rgb_bytes
    hr, lr, metadata = inputs()
    original = copy.deepcopy(metadata)
    pair, encoded = encode_uint8_pair(hr, lr, metadata)
    assert metadata == original
    assert encoded['recipe'] == original['recipe']
    assert encoded['storage']['rgb_payload_bytes'] == decoded_rgb_bytes(pair) == 835584
    assert encoded['storage']['quantization'] == 'floor(clip(x, 0, 1) * 255 + 0.5)'
    assert pair['hr'].dtype == pair['lr'].dtype == np.uint8
    assert pair['hr'][0, 0, 0] == 128
    metadata['recipe']['lr_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='hash'):
        encode_uint8_pair(hr, lr, metadata)


def test_loader_decodes_uint8_and_normalizes_only_for_training(tmp_path):
    from src.maritime_uint8 import encode_uint8_pair
    from src.maritime_uint8_dataset import MaritimeUint8Dataset
    pair, metadata = encode_uint8_pair(*inputs())
    dataset = MaritimeUint8Dataset([write_shard(tmp_path, pair, metadata)], 'd2')
    stored, decoded = dataset.read_arrays(0)
    assert stored['hr'].dtype == np.uint8
    assert decoded['recipe'] == metadata['recipe']
    torch = pytest.importorskip('torch')
    sample = dataset[0]
    assert sample['hr'].dtype == torch.float32
    assert sample['hr'][0, 0, 0].item() == np.float32(128 / 255)
    assert sample['hr_mask'].dtype == torch.bool


@pytest.mark.parametrize('corruption', ['pixels', 'dtype', 'shape', 'mask', 'transform',
    'nonfinite_transform', 'recipe', 'identity', 'storage', 'payload', 'common_hash'])
def test_loader_rejects_corrupt_storage_and_provenance(tmp_path, corruption):
    from src.maritime_uint8 import encode_uint8_pair
    from src.maritime_uint8_dataset import MaritimeUint8Dataset
    pair, metadata = encode_uint8_pair(*inputs())
    if corruption == 'pixels': pair['hr'][0, 0, 0] = 1
    elif corruption == 'dtype': pair['lr'] = pair['lr'].astype(np.float32)
    elif corruption == 'shape': pair['lr'] = pair['lr'][:, :-1]
    elif corruption == 'mask': pair['lr_mask'][0, 0] = False
    elif corruption == 'transform': metadata['lr_transform'][2] = 1
    elif corruption == 'nonfinite_transform': metadata['hr_transform'][2] = float('nan')
    elif corruption == 'recipe': metadata['recipe']['recipe'] = RECIPE_VERSIONS['d1']
    elif corruption == 'identity': metadata['recipe']['sample_id'] = 'other'
    elif corruption == 'storage': metadata['storage']['version'] = 'unknown'
    elif corruption == 'payload': metadata['storage']['rgb_payload_bytes'] += 1
    elif corruption == 'common_hash': metadata['hr_sha256'] = 'wrong'
    dataset = MaritimeUint8Dataset([write_shard(tmp_path, pair, metadata)], 'd2')
    with pytest.raises(ValueError): dataset.read_arrays(0)


def test_duplicate_ids_rejected_across_shards(tmp_path):
    from src.maritime_uint8 import encode_uint8_pair
    from src.maritime_uint8_dataset import MaritimeUint8Dataset
    pair, metadata = encode_uint8_pair(*inputs())
    a = write_shard(tmp_path, pair, metadata, 'a.tar')
    b = write_shard(tmp_path, pair, metadata, 'b.tar')
    with pytest.raises(ValueError, match='Duplicate'):
        MaritimeUint8Dataset([a, b], 'd2')


@pytest.mark.parametrize('recipe', ['d1', 'd2', 'd3'])
def test_all_original_recipe_identities_round_trip_without_rewriting(tmp_path, recipe):
    from src.maritime_uint8 import encode_uint8_pair
    from src.maritime_uint8_dataset import MaritimeUint8Dataset
    pair, metadata = encode_uint8_pair(*inputs(recipe))
    dataset = MaritimeUint8Dataset([write_shard(tmp_path, pair, metadata)], recipe)
    stored, decoded = dataset.read_arrays(0)
    assert decoded['recipe']['recipe'] == RECIPE_VERSIONS[recipe]
    np.testing.assert_array_equal(stored['lr'], np.full((3, 128, 128), 64, np.uint8))


@pytest.mark.parametrize('key', ['hr', 'lr', 'hr_mask', 'lr_mask'])
def test_encoder_rejects_non_array_input_with_value_error(key):
    from src.maritime_uint8 import encode_uint8_pair
    hr, lr, metadata = inputs()
    arguments = dict(hr=hr, lr=lr, metadata=metadata)
    arguments[key] = []
    with pytest.raises(ValueError):
        encode_uint8_pair(**arguments)
