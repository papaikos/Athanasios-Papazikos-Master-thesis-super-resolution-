"""Torch-compatible quarantined curated SR chips with local image coordinates."""
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from src.maritime_recipes import RECIPE_VERSIONS,array_hash


class CuratedMaritimeDataset:
    """Separate 64/16 training branch; never mixes with geographic holdouts."""
    def __init__(self,shards,recipe):
        if recipe not in RECIPE_VERSIONS:raise ValueError('Unknown recipe')
        self.recipe=recipe;self.members=[];seen=set()
        for value in shards:
            path=Path(value)
            with tarfile.open(path,'r') as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith('.npz'):raise ValueError('Unexpected curated TAR member')
                    identity=Path(member.name).stem
                    if identity in seen:raise ValueError('Duplicate curated sample ID')
                    seen.add(identity);self.members.append((path,member.name))
        if not self.members:raise ValueError('Empty curated dataset')

    def __len__(self):return len(self.members)

    def read_arrays(self,index):
        path,name=self.members[index]
        with tarfile.open(path,'r') as archive:payload=archive.extractfile(name).read()
        with np.load(io.BytesIO(payload),allow_pickle=False) as data:
            provenance=data['provenance']
            if provenance.shape!=() or provenance.dtype.kind!='U':raise ValueError('Scalar Unicode provenance required')
            meta=json.loads(str(provenance))
            arrays={key:np.array(data[key],copy=True) for key in ('hr','lr','hr_mask','lr_mask')}
        recipe=meta['recipe']
        if (meta['split']!='train' or meta['georeferencing'] is not None
            or meta['quarantined_training_supplement'] is not True
            or meta['geographic_holdout_isolation_verified'] is not False):
            raise ValueError('Curated geographic quarantine invariant violated')
        if (recipe['recipe']!=RECIPE_VERSIONS[self.recipe] or meta['lr_origin']!='synthetic_hr_only'
            or recipe['sample_id']!=meta['sample_id'] or Path(name).stem!=meta['sample_id']):
            raise ValueError('Curated recipe/sample identity mismatch')
        if recipe['context_shape']!=[3,96,96] or recipe['hr_shape']!=[3,64,64] or recipe['lr_shape']!=[3,16,16]:
            raise ValueError('Curated physical context/grid mismatch')
        if (meta['native_gsd_m']!=.3 or meta['hr_spacing_m']!=2.5 or meta['lr_spacing_m']!=10
            or meta['coordinate_system']!='source_image_local_axes_unknown_earth_orientation'):
            raise ValueError('Curated resolution or coordinate basis mismatch')
        for key,shape in [('hr',(3,64,64)),('lr',(3,16,16))]:
            array=arrays[key]
            if (array.shape!=shape or array.dtype!=np.float32 or not np.isfinite(array).all()
                or (array<0).any() or (array>1).any() or array_hash(array)!=recipe[key+'_sha256']):
                raise ValueError('Invalid curated training array: '+key)
        if meta['hr_sha256']!=recipe['hr_sha256']:raise ValueError('Shared curated HR hash mismatch')
        for key,shape in [('hr_mask',(64,64)),('lr_mask',(16,16))]:
            if arrays[key].shape!=shape or arrays[key].dtype!=np.bool_ or not arrays[key].all():raise ValueError('Curated full-support mask mismatch')
        return arrays,meta

    def __getitem__(self,index):
        import torch
        arrays,meta=self.read_arrays(index)
        return dict(**{key:torch.from_numpy(value) for key,value in arrays.items()},sample_id=meta['sample_id'])
