"""D3x10 shard access, x10 aligned crops and consumed-position sampling.

Pairs are read straight from the shard TARs by byte offset (index written by
ops/audit_data.py after every shard's SHA-256 matched the manifest). Crops and
augmentations are keyed to the consumed sample position, as in the D-series
runs, so DataLoader workers and prefetching cannot change what is trained on;
the trainer's cursor alone decides where a resumed run continues.
"""
import io
import json
from pathlib import Path

import numpy as np
import torch

SCALE, LR_CROP, LR_FULL = 10, 64, 128
HR_CROP, HR_FULL = LR_CROP * SCALE, LR_FULL * SCALE
SEED = 42
AUGMENTATION_SCHEME = 'SeedSequence(seed,consumed_position,314159)-x10-v1'


def load_index(path):
    index = json.loads(Path(path).read_text())
    if index.get('schema') != 'd3x10-shard-index-v1':
        raise ValueError('Unknown shard index schema')
    return index


def read_pair(data_root, row):
    """Exact stored uint8 arrays of one sample (no decoding of the provenance JSON)."""
    with open(Path(data_root) / row['shard'], 'rb') as handle:
        handle.seek(row['offset'])
        payload = handle.read(row['size'])
    if len(payload) != row['size']:
        raise ValueError('Truncated shard member: ' + row['sample_id'])
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        pair = {key: data[key] for key in ('hr', 'lr', 'hr_mask', 'lr_mask')}
    if (pair['hr'].shape != (3, HR_FULL, HR_FULL) or pair['lr'].shape != (3, LR_FULL, LR_FULL)
            or pair['hr'].dtype != np.uint8 or pair['lr'].dtype != np.uint8
            or pair['hr_mask'].shape != (HR_FULL, HR_FULL) or pair['lr_mask'].shape != (LR_FULL, LR_FULL)
            or pair['hr_mask'].dtype != np.bool_ or pair['lr_mask'].dtype != np.bool_):
        raise ValueError('Malformed D3x10 pair: ' + row['sample_id'])
    return pair


def to_float(array):
    # uint8 -> float32 in [0,1], divided by 255 exactly once (D-series loader convention).
    return torch.from_numpy(np.ascontiguousarray(array)).to(torch.float32).div_(255.0)


def crop_choice(position, seed=SEED):
    rng = np.random.default_rng(np.random.SeedSequence([seed, position, 314159]))
    y, x = rng.integers(0, LR_FULL - LR_CROP + 1, size=2)
    return int(y), int(x), int(rng.integers(0, 8))


def aligned_crop(pair, position, seed=SEED):
    """LR crop origin (x,y) -> HR (10x,10y); one of 8 square symmetries applied to all arrays."""
    y, x, geometry = crop_choice(position, seed)
    result = {}
    for key in ('lr', 'hr', 'lr_mask', 'hr_mask'):
        scale = SCALE if key.startswith('hr') else 1
        size = LR_CROP * scale
        value = pair[key][..., y * scale:y * scale + size, x * scale:x * scale + size]
        value = to_float(value) if key in ('lr', 'hr') else torch.from_numpy(np.ascontiguousarray(value))
        value = torch.rot90(value, geometry % 4, (-2, -1))
        if geometry >= 4:
            value = torch.flip(value, (-1,))
        result[key] = value.contiguous()
    return result, {'position': position, 'x': x, 'y': y, 'geometry': geometry}


def permutation(size, epoch, seed=SEED):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(size)


class TrainPositions(torch.utils.data.Dataset):
    """Map consumed position -> (sample in that epoch's permutation, its crop/augmentation)."""

    def __init__(self, data_root, rows, seed=SEED):
        if not rows:
            raise ValueError('Empty TRAIN')
        self.data_root, self.rows, self.seed = str(data_root), rows, seed
        self._epoch, self._perm = None, None

    def __len__(self):
        return 2 ** 62  # positions are unbounded; the sampler decides which are drawn

    def sample_index(self, position):
        epoch, offset = divmod(position, len(self.rows))
        if epoch != self._epoch:
            self._perm, self._epoch = permutation(len(self.rows), epoch, self.seed), epoch
        return int(self._perm[offset])

    def __getitem__(self, position):
        row = self.rows[self.sample_index(position)]
        crop, choice = aligned_crop(read_pair(self.data_root, row), position, self.seed)
        choice['sample_id'] = row['sample_id']
        return dict(crop, position=position, choice=json.dumps(choice))


class PositionBatches(torch.utils.data.Sampler):
    """Consecutive batches of positions from a start cursor; restarting at the cursor reproduces the stream."""

    def __init__(self, start, batch_size=16):
        self.start, self.batch_size = start, batch_size

    def __iter__(self):
        position = self.start
        while True:
            yield list(range(position, position + self.batch_size))
            position += self.batch_size


def train_loader(data_root, rows, cursor, workers, batch_size=16):
    return torch.utils.data.DataLoader(
        TrainPositions(data_root, rows), batch_sampler=PositionBatches(cursor, batch_size),
        num_workers=workers, pin_memory=True, persistent_workers=False,
        prefetch_factor=4 if workers else None)


class FullPairs:
    """Full-frame pairs for evaluation; uint8 cached in RAM, float on access."""

    def __init__(self, data_root, rows, cache=True):
        self.data_root, self.rows = str(data_root), list(rows)
        self.members = [(row['shard'], row['sample_id'] + '.npz') for row in self.rows]  # for visuals.py
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.rows)

    def _raw(self, index):
        if self._cache is not None and index in self._cache:
            return self._cache[index]
        pair = read_pair(self.data_root, self.rows[index])
        if self._cache is not None:
            self._cache[index] = pair
        return pair

    def __getitem__(self, index):
        if not 0 <= index < len(self.rows):
            raise IndexError(index)
        pair = self._raw(index)
        return {'lr': to_float(pair['lr']), 'hr': to_float(pair['hr']),
                'hr_mask': torch.from_numpy(pair['hr_mask'].copy()), 'sample_id': self.rows[index]['sample_id']}

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
