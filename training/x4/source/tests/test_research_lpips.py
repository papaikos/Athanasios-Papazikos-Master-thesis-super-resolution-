import copy
import random
import numpy as np
import pytest
import torch

class FakeLPIPS(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.rand(1));self.inputs=None
    def forward(self,x,y,normalize=False):
        assert not normalize and not torch.is_grad_enabled() and not self.training
        self.inputs=(x.clone(),y.clone())
        return (x-y).square().mean().reshape(1,1,1,1)

def test_lpips_construction_and_scoring_are_rng_pure_and_frozen():
    from research.lpips_metric import LPIPSMetric
    from research.evaluate import rng_state
    before=rng_state();created=[]
    def factory(**kwargs):
        random.random();np.random.random();value=FakeLPIPS();created.append((value,kwargs));return value
    metric=LPIPSMetric('cpu',factory=factory)
    after=rng_state()
    assert before['python']==after['python'] and np.array_equal(before['numpy'][1],after['numpy'][1]) and torch.equal(before['torch'],after['torch'])
    assert all(not p.requires_grad for p in created[0][0].parameters())
    assert created[0][1]['net']=='alex' and created[0][1]['version']=='0.1' and created[0][1]['spatial'] is False
    prediction=torch.full((3,16,16),1.2);target=torch.full_like(prediction,.25);mask=torch.ones(16,16,dtype=torch.bool)
    assert metric.score(prediction,target,mask)==pytest.approx(2.25)
    x,y=created[0][0].inputs
    assert x.dtype==y.dtype==torch.float32 and x.shape==(1,3,16,16)
    assert torch.equal(x,torch.ones_like(x)) and torch.equal(y,torch.full_like(y,-.5))

def test_lpips_rejects_partial_support_and_nonfinite():
    from research.lpips_metric import LPIPSMetric
    metric=LPIPSMetric('cpu',factory=lambda **_:FakeLPIPS());a=torch.zeros(3,16,16);mask=torch.ones(16,16,dtype=torch.bool);mask[0,0]=False
    with pytest.raises(ValueError,match='full support'):metric.score(a,a,mask)
    mask[:]=True;a[0,0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):metric.score(a,a,mask)

def test_evaluation_records_per_image_and_mean_lpips(tmp_path):
    import csv
    from research.lpips_metric import LPIPSMetric
    from research.evaluate import evaluate
    model=torch.nn.Identity();model.train();metric=LPIPSMetric('cpu',factory=lambda **_:FakeLPIPS())
    pairs=[{'sample_id':str(i),'lr':torch.full((3,16,16),value),'hr':torch.zeros(3,16,16),'hr_mask':torch.ones(16,16,dtype=torch.bool)} for i,value in enumerate((.25,.5))]
    rng=torch.get_rng_state().clone();result=evaluate(model,pairs,'cpu',tmp_path/'rows.csv',2,lpips_metric=metric)
    assert result['mean_per_image_lpips']==pytest.approx(.625)
    rows=list(csv.DictReader((tmp_path/'rows.csv').open()))
    assert [float(row['lpips']) for row in rows]==pytest.approx([.25,1.])
    assert torch.equal(rng,torch.get_rng_state()) and model.training

def test_initial_lpips_reference_refresh_only_at_true_step_zero():
    from research.train import needs_lpips_reference_refresh
    assert needs_lpips_reference_refresh(0,{'bicubic':{'count':115}})
    complete={'bicubic':{'mean_per_image_lpips':.5},'initial_validation':{'mean_per_image_lpips':.6}}
    assert not needs_lpips_reference_refresh(0,complete)
    with pytest.raises(ValueError,match='step zero'):needs_lpips_reference_refresh(7,{})

def test_production_constructor_verifies_both_pins_before_instantiation(tmp_path,monkeypatch):
    import hashlib
    import sys
    from types import SimpleNamespace
    import research.lpips_metric as metric_module
    package=tmp_path/'package';calibration=package/'weights/v0.1/alex.pth';calibration.parent.mkdir(parents=True);calibration.write_bytes(b'calibration-fixture')
    cache=tmp_path/'hub';alex=cache/'checkpoints/alexnet-owt-7be5be79.pth';alex.parent.mkdir(parents=True);alex.write_bytes(b'alexnet-fixture')
    created=[]
    def factory(**kwargs):created.append(kwargs);return FakeLPIPS()
    monkeypatch.setitem(sys.modules,'lpips',SimpleNamespace(__file__=str(package/'__init__.py'),LPIPS=factory))
    monkeypatch.setattr(metric_module,'version',lambda _: '0.1.4')
    monkeypatch.setattr(torch.hub,'get_dir',lambda:str(cache))
    monkeypatch.setattr(metric_module,'CALIBRATION_SHA256',hashlib.sha256(calibration.read_bytes()).hexdigest())
    monkeypatch.setattr(metric_module,'ALEXNET_SHA256',hashlib.sha256(alex.read_bytes()).hexdigest())
    metric=metric_module.LPIPSMetric('cpu')
    assert created[0]['model_path']==str(calibration)
    assert metric.provenance['alexnet_file']['bytes']==len(b'alexnet-fixture')
    alex.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='cache pin'):metric_module.LPIPSMetric('cpu')
    assert len(created)==1
    calibration.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='calibration pin'):metric_module.LPIPSMetric('cpu')
    assert len(created)==1
