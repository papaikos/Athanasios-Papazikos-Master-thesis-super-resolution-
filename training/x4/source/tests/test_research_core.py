import math
import numpy as np
import pytest

def test_schedule_boundaries():
    from research.train import learning_rate
    assert learning_rate(1) == pytest.approx(1e-7)
    assert learning_rate(2000) == pytest.approx(2e-4)
    assert learning_rate(100000) == pytest.approx(1e-6)

def test_metric_identity_and_independent_ssim():
    from research.metrics import score_image
    from scipy.ndimage import gaussian_filter
    rng=np.random.default_rng(3)
    a=rng.random((3,20,21)); b=np.clip(a+0.02,0,1)
    same=score_image(a,a,np.ones((20,21),bool))
    assert same['psnr_infinite'] and same['ssim']==pytest.approx(1)
    result=score_image(b,a,np.ones((20,21),bool))
    means=[]
    for x,y in zip(b,a):
        ux=gaussian_filter(x,1.5,truncate=10/3)[5:-5,5:-5]
        uy=gaussian_filter(y,1.5,truncate=10/3)[5:-5,5:-5]
        xx=gaussian_filter(x*x,1.5,truncate=10/3)[5:-5,5:-5]-ux*ux
        yy=gaussian_filter(y*y,1.5,truncate=10/3)[5:-5,5:-5]-uy*uy
        xy=gaussian_filter(x*y,1.5,truncate=10/3)[5:-5,5:-5]-ux*uy
        means.append(np.mean(((2*ux*uy+.01**2)*(2*xy+.03**2))/((ux*ux+uy*uy+.01**2)*(xx+yy+.03**2))))
    assert result['ssim']==pytest.approx(np.mean(means),abs=1e-6)
    assert result['psnr']==pytest.approx(-10*math.log10(np.mean((a-b)**2)))
    with pytest.raises(ValueError): score_image(a,a,np.zeros((20,21),bool))

def test_sampler_resume_and_aligned_geometry():
    import torch
    from research.data_adapter import SampleStream, aligned_crop
    stream=SampleStream(5,42); first=stream.indices(16); state=stream.state_dict()
    assert sorted(first[:5])==list(range(5))
    restored=SampleStream(5,42);restored.load_state_dict(state)
    assert stream.indices(20)==restored.indices(20)
    grid=torch.arange(128*128).reshape(128,128)
    pair={'lr':grid.repeat(3,1,1).float(),'hr':grid.repeat_interleave(4,0).repeat_interleave(4,1).repeat(3,1,1).float(),'lr_mask':torch.ones(128,128,dtype=torch.bool),'hr_mask':torch.ones(512,512,dtype=torch.bool)}
    for position in range(16):
        crop,choice=aligned_crop(pair,position,42)
        assert torch.equal(crop['lr'],crop['hr'][:,::4,::4])
        assert crop['lr'].shape==(3,64,64)

def test_checkpoint_corruption_rejected(tmp_path):
    from research.checkpointing import atomic_checkpoint, load_checkpoint
    path,receipt=atomic_checkpoint({'schema':1,'completed_updates':7},tmp_path,7)
    assert load_checkpoint(path,receipt)['completed_updates']==7
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError):load_checkpoint(path,receipt)
