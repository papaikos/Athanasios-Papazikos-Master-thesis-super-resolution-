"""Consumed-position sampling; the frozen loader owns payload validation."""
import sys
from pathlib import Path
import numpy as np
from .experiment import KEY, NAME, RECIPE

class SampleStream:
    def __init__(self,size,seed=42):
        if size<1:raise ValueError('Empty TRAIN')
        self.size=size;self.seed=seed;self.cursor=0;self._epoch=None;self._permutation=None
    def indices(self,count):
        values=[]
        for position in range(self.cursor,self.cursor+count):
            epoch,offset=divmod(position,self.size)
            if epoch!=self._epoch:
                self._permutation=np.random.default_rng(np.random.SeedSequence([self.seed,epoch])).permutation(self.size);self._epoch=epoch
            values.append(int(self._permutation[offset]))
        self.cursor+=count
        return values
    def state_dict(self):return {'size':self.size,'seed':self.seed,'cursor':self.cursor,'scheme':'numpy-PCG64-SeedSequence-permutation-v1'}
    def load_state_dict(self,state):
        if state['size']!=self.size or state['seed']!=self.seed or state['cursor']<0:raise ValueError('Sampler identity mismatch')
        self.cursor=state['cursor'];self._epoch=None;self._permutation=None

def aligned_crop(pair,position,seed=42):
    import torch
    rng=np.random.default_rng(np.random.SeedSequence([seed,position,314159]))
    y,x=rng.integers(0,65,size=2);geometry=int(rng.integers(0,8));result={}
    for key in ('lr','hr','lr_mask','hr_mask'):
        scale=4 if key.startswith('hr') else 1
        value=pair[key][...,y*scale:(y+64)*scale,x*scale:(x+64)*scale]
        value=torch.rot90(value,geometry%4,(-2,-1))
        if geometry>=4:value=torch.flip(value,(-1,))
        result[key]=value.contiguous()
    return result,{'position':position,'x':int(x),'y':int(y),'geometry':geometry,'sample_id':pair.get('sample_id')}

def datasets_from_audit(data_root,source_root,audit):
    if audit.get('status')!='PASS':raise ValueError('Data audit must PASS')
    if audit.get('staged_shard_integrity',{}).get('status')!='PASS':raise ValueError('Staged shard integrity must PASS')
    if audit.get('recipe')!=RECIPE:raise ValueError(NAME+' recipe required')
    seen=set()
    for selection,expected_split,scope,prefix in [('train','train','main',NAME+'/train'),('validation','validation','main',NAME+'/validation'),('aerial_validation','validation','aerial',NAME+'/evaluation/aerial-positive63-v1/validation')]:
        for row in audit['membership'].get(selection,[]):
            relative=Path(row['shard'])
            if (row.get('split')!=expected_split or row.get('scope')!=scope
                    or relative.parent.as_posix()!=prefix or relative.suffix!='.tar'
                    or row['member']!=row['sample_id']+'.npz' or row['sample_id'] in seen):
                raise ValueError('Invalid or overlapping '+NAME+' selection: '+selection)
            seen.add(row['sample_id'])
    sys.path.insert(0,str(Path(source_root).resolve()))
    from src.maritime_uint8_dataset import MaritimeUint8Dataset
    result={}
    for split,count in [('train',17952),('validation',115),('aerial_validation',15)]:
        membership=audit['membership'][split]
        if len(membership)!=count:raise ValueError('Incomplete '+split)
        shards=sorted({str(Path(data_root)/r['shard']) for r in membership})
        dataset=MaritimeUint8Dataset(shards,KEY)
        selected={(str(Path(data_root)/r['shard']),r['member']):r['sample_id'] for r in membership}
        dataset.members=[member for member in dataset.members if (str(member[0]),member[1]) in selected]
        if len(dataset)!=count:raise ValueError('Manifest/loader membership mismatch: '+split)
        result[split]=dataset
    return result
