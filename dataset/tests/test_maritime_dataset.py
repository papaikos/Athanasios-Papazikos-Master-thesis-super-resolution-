import io
import json
import tarfile

import numpy as np
import pytest

from scripts.maritime_pilot import deterministic_npz
from src.maritime_dataset import MaritimeDataset, smoke_batch
from src.maritime_recipes import array_hash, RECIPE_VERSIONS


def shard(tmp_path, recipe='d1', corrupt=False):
    hr = np.zeros((3, 512, 512), np.float32)
    lr = np.zeros((3, 128, 128), np.float32)
    metadata = dict(sample_id='s', mother_id='s', split='development',
        lr_origin='synthetic_hr_only', hr_sha256=array_hash(hr),
        hr_transform=[2.5,0,0,0,-2.5,1280], lr_transform=[10,0,0,0,-10,1280],
        recipe=dict(recipe=RECIPE_VERSIONS[recipe], sample_id='s',
                    hr_sha256=array_hash(hr), lr_sha256='wrong' if corrupt else array_hash(lr)))
    payload = deterministic_npz(dict(hr=hr, lr=lr, hr_mask=np.ones((512,512),bool),
        lr_mask=np.ones((128,128),bool), provenance=np.array(json.dumps(metadata))))
    path = tmp_path / 'pairs.tar'
    with tarfile.open(path, 'w') as archive:
        info = tarfile.TarInfo('development/s.npz')
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return path


@pytest.mark.parametrize('recipe', ['d1','d2','d3'])
def test_actual_torch_loader_batches_each_recipe(tmp_path, recipe):
    pytest.importorskip('torch')
    dataset = MaritimeDataset([shard(tmp_path, recipe)], recipe=recipe)
    result = smoke_batch(dataset)
    assert result['hr_shape'] == [1,3,512,512]
    assert result['lr_shape'] == [1,3,128,128]
    assert result['sample_ids'] == ['s']


def test_decoded_hash_corruption_and_wrong_recipe_rejected(tmp_path):
    dataset = MaritimeDataset([shard(tmp_path, corrupt=True)], recipe='d1')
    with pytest.raises(ValueError, match='hash'):
        dataset.read_arrays(0)
    dataset = MaritimeDataset([shard(tmp_path)], recipe='d2')
    with pytest.raises(ValueError, match='recipe'):
        dataset.read_arrays(0)


def test_duplicate_sample_ids_under_different_tar_directories_rejected(tmp_path):
    path = shard(tmp_path)
    with tarfile.open(path, 'r') as archive:
        payload = archive.extractfile('development/s.npz').read()
    other = tmp_path / 'other.tar'
    with tarfile.open(other,'w') as archive:
        info = tarfile.TarInfo('another-directory/s.npz')
        info.size = len(payload)
        archive.addfile(info,io.BytesIO(payload))
    with pytest.raises(ValueError, match='Duplicate'):
        MaritimeDataset([path,other], recipe='d1')
