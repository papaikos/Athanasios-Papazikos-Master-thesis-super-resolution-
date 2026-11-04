"""Recipe-aware RGB training loader for immutable NPZ tar shards."""
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from src.maritime_recipes import RECIPE_VERSIONS, array_hash


class MaritimeDataset:
    """Map-style torch-compatible dataset; each worker opens its own shard."""

    def __init__(self, shards, recipe):
        if recipe not in RECIPE_VERSIONS:
            raise ValueError('Unknown recipe')
        self.recipe = recipe
        self.members = []
        seen = set()
        for path in shards:
            path = Path(path)
            with tarfile.open(path, 'r') as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith('.npz'):
                        raise ValueError('Unexpected training shard member')
                    identity = Path(member.name).stem
                    if identity in seen:
                        raise ValueError('Duplicate sample in training shards')
                    seen.add(identity)
                    self.members.append((path, member.name))
        if not self.members:
            raise ValueError('Empty training dataset')

    def __len__(self):
        return len(self.members)

    def read_arrays(self, index):
        path, name = self.members[index]
        with tarfile.open(path, 'r') as archive:
            payload = archive.extractfile(name).read()
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            provenance = data['provenance']
            if provenance.shape != () or provenance.dtype.kind != 'U':
                raise ValueError('Provenance must be a scalar Unicode JSON string')
            metadata = json.loads(str(provenance))
            pair = {key: np.array(data[key], copy=True) for key in ('hr','lr','hr_mask','lr_mask')}
        recipe = metadata['recipe']
        if recipe['recipe'] != RECIPE_VERSIONS[self.recipe]:
            raise ValueError('Wrong recipe for this training dataset')
        if (metadata['lr_origin'] != 'synthetic_hr_only'
                or metadata['sample_id'] != recipe['sample_id']
                or Path(name).stem != metadata['sample_id']):
            raise ValueError('Training provenance identity mismatch')
        for key, shape in [('hr',(3,512,512)), ('lr',(3,128,128))]:
            values = pair[key]
            if (values.shape != shape or values.dtype != np.float32
                    or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()):
                raise ValueError('Invalid training array: ' + key)
            if array_hash(values) != recipe[key + '_sha256']:
                raise ValueError('Decoded training array hash mismatch: ' + key)
        if metadata['hr_sha256'] != recipe['hr_sha256']:
            raise ValueError('Common HR hash mismatch')
        for key, shape in [('hr_mask',(512,512)), ('lr_mask',(128,128))]:
            if pair[key].shape != shape or pair[key].dtype != np.bool_ or not pair[key].all():
                raise ValueError('Invalid full-support training mask: ' + key)
        hr, lr = metadata['hr_transform'], metadata['lr_transform']
        if (hr[:2] != [2.5,0] or hr[3:5] != [0,-2.5]
                or lr[:2] != [10,0] or lr[3:5] != [0,-10]
                or hr[2] != lr[2] or hr[5] != lr[5]):
            raise ValueError('Training grid geometry mismatch')
        return pair, metadata

    def __getitem__(self, index):
        import torch
        pair, metadata = self.read_arrays(index)
        return dict(**{key: torch.from_numpy(value) for key,value in pair.items()},
                    sample_id=metadata['sample_id'])


def smoke_batch(dataset, batch_size=2):
    """Exercise real torch collation, including a tensor operation on each batch."""
    import torch
    from torch.utils.data import DataLoader
    count = 0
    first = None
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        if not torch.isfinite(batch['hr'].mean()) or not torch.isfinite(batch['lr'].mean()):
            raise ValueError('Nonfinite training batch')
        count += len(batch['sample_id'])
        if first is None:
            first = dict(hr_shape=list(batch['hr'].shape), lr_shape=list(batch['lr'].shape),
                         sample_ids=batch['sample_id'])
    return dict(**first, verified_samples=count, recipe=dataset.recipe,
                loader='torch.utils.data.DataLoader', torch_version=torch.__version__)
