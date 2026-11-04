import io
import json
import tarfile

import numpy as np
import pytest

from src.maritime_recipes import array_hash
from src.maritime_uint8 import quantize_rgb, uint8_array_hash


def archive(path, *, bad_mask=False):
    context=np.full((3,544,544),.25,np.float32)
    hr=np.ascontiguousarray(context[:,16:-16,16:-16])
    row=dict(sample_id='sample',hr_sha256=array_hash(hr),context_sha256=array_hash(context),
        split='train',source_kind='maxar',site_id='port',acquisition_id='capture')
    mask=np.ones((512,512),bool)
    if bad_mask:mask[0,0]=False
    payload=io.BytesIO();np.savez_compressed(payload,context=context,hr=hr,hr_mask=mask,
        provenance=np.asarray(json.dumps(row)))
    with tarfile.open(path,'w') as handle:
        member=tarfile.TarInfo('sample.npz');member.size=len(payload.getvalue())
        handle.addfile(member,io.BytesIO(payload.getvalue()))
    return row,hr


def test_baseline_hashes_actual_quantized_hr_and_preserves_float_identity(tmp_path):
    from scripts.maritime_baseline_hr_census import census_shard
    path=tmp_path/'mother.tar';row,hr=archive(path)
    result=census_shard(path,{'sample':row},['sample'])
    assert result[0]['hr_float32_sha256']==row['hr_sha256']
    assert result[0]['hr_uint8_chw_sha256']==uint8_array_hash(quantize_rgb(hr))
    assert result[0]['split']=='train'


def test_census_fails_closed_on_hr_change_or_unexpected_membership(tmp_path):
    from scripts.maritime_baseline_hr_census import census_shard
    path=tmp_path/'mother.tar';row,_=archive(path)
    changed=dict(row,hr_sha256='a'*64)
    with pytest.raises(ValueError,match='hash'):
        census_shard(path,{'sample':changed},['sample'])
    with pytest.raises(ValueError,match='membership'):
        census_shard(path,{'sample':row},['absent'])


def test_census_cannot_hash_unsupported_hr_as_usable_baseline(tmp_path):
    from scripts.maritime_baseline_hr_census import census_shard
    path=tmp_path/'mother.tar';row,_=archive(path,bad_mask=True)
    with pytest.raises(ValueError,match='support'):
        census_shard(path,{'sample':row},['sample'])
