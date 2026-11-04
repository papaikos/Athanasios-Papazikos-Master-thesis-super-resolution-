"""Strict training loader for versioned uint8 maritime NPZ TAR shards."""
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np

from src.maritime_dataset import MaritimeDataset
from src.maritime_uint8 import (QUANTIZATION, STORAGE_VERSION, decoded_rgb_bytes,
                              uint8_array_hash, validate_provenance)


class MaritimeUint8Dataset(MaritimeDataset):
    """Keep decoded RGB uint8 in read_arrays; normalize only for training."""

    def __init__(self, shards, recipe):
        super().__init__(shards, recipe)
        for _, name in self.members:
            path = PurePosixPath(name)
            if path.is_absolute() or '\\' in name or any(part in ('.', '..') for part in name.split('/')):
                raise ValueError('Unsafe training shard member path')

    def read_arrays(self, index):
        path, name = self.members[index]
        try:
            with tarfile.open(path, 'r') as archive:
                payload = archive.extractfile(name).read()
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                provenance = data['provenance']
                if provenance.shape != () or provenance.dtype.kind != 'U':
                    raise ValueError('Provenance must be a scalar Unicode JSON string')
                metadata = json.loads(str(provenance))
                pair = {key: np.array(data[key], copy=True) for key in ('hr', 'lr', 'hr_mask', 'lr_mask')}
            validate_provenance(metadata, self.recipe, Path(name).stem)
            storage = metadata['storage']
            if (storage['version'] != STORAGE_VERSION or storage['dtype'] != 'uint8'
                    or storage['quantization'] != QUANTIZATION
                    or storage['quantization_arithmetic'] != 'float64 on original float32 values'
                    or storage['quantization_order'] != 'after all original float recipe operators and final clipping'
                    or storage['array_hash_encoding'] != 'C-contiguous uint8 bytes'):
                raise ValueError('Wrong uint8 storage convention')
            if storage['rgb_payload_bytes'] != decoded_rgb_bytes(pair):
                raise ValueError('Decoded RGB payload size mismatch')
            for key in ('hr', 'lr'):
                if uint8_array_hash(pair[key]) != storage[key + '_sha256']:
                    raise ValueError('Decoded training byte hash mismatch: ' + key)
            for key, shape in [('hr_mask', (512, 512)), ('lr_mask', (128, 128))]:
                if pair[key].shape != shape or pair[key].dtype != np.bool_ or not pair[key].all():
                    raise ValueError('Invalid full-support training mask: ' + key)
        except (KeyError, TypeError, AttributeError, json.JSONDecodeError) as error:
            raise ValueError('Malformed uint8 training sample') from error
        return pair, metadata

    def __getitem__(self, index):
        import torch
        pair, metadata = self.read_arrays(index)
        tensors = {key: torch.from_numpy(value) for key, value in pair.items()}
        for key in ('hr', 'lr'):
            tensors[key] = tensors[key].to(dtype=torch.float32).div_(255.0)
        return dict(**tensors, sample_id=metadata['sample_id'])
