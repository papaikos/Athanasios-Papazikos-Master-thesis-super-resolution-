import numpy as np
import pytest
from src.maritime_recipes import degrade,array_hash


@pytest.mark.parametrize('recipe',['d1','d2','d3'])
def test_actual_96_context_produces_shared64hr_and16lr(recipe):
    values=np.random.default_rng(42).random((3,96,96),dtype=np.float32)
    result=degrade(values,'qualified-chip',recipe)
    again=degrade(values,'qualified-chip',recipe)
    assert result['lr'].shape==(3,16,16)
    assert result['hr_mask'].shape==(64,64)
    assert result['lr_mask'].shape==(16,16)
    assert result['metadata']['hr_sha256']==array_hash(values[:,16:-16,16:-16])
    np.testing.assert_array_equal(result['lr'],again['lr'])


def test_chip_operators_use_real_observed_context():
    import torch
    values=np.random.default_rng(21).random((3,96,96),dtype=np.float32)
    expected=torch.nn.functional.interpolate(torch.from_numpy(values)[None],scale_factor=.25,
        mode='bicubic',align_corners=False,antialias=True).numpy()[0,:,4:-4,4:-4]
    np.testing.assert_array_equal(degrade(values,'id','d1')['lr'],np.clip(expected,0,1))


def test_arbitrary_context_shape_still_rejected():
    with pytest.raises(ValueError):degrade(np.ones((3,100,100),np.float32),'id')
