"""Independent scientific gate checks; no production or dataset payload access."""
import copy
from pathlib import Path
import shutil
import torch


def test_preflight_resets_scratch_and_replays_real_next_batch(tmp_path,monkeypatch):
    import research.train as training
    from research.evaluate import rng_state
    from research.checkpointing import sha256

    class TinyX4(torch.nn.Module):
        def __init__(self):
            super().__init__();self.bias=torch.nn.Parameter(torch.zeros(1,3,1,1))
        def forward(self,x):
            return torch.nn.functional.interpolate(x,scale_factor=4,mode='nearest')+self.bias

    class TrainingOnly:
        def __len__(self):return 3
        def __getitem__(self,index):
            assert 0<=index<3
            return {'lr':torch.full((3,128,128),.2+index*.01),
                    'hr':torch.full((3,512,512),.6+index*.01),
                    'lr_mask':torch.ones(128,128,dtype=torch.bool),
                    'hr_mask':torch.ones(512,512,dtype=torch.bool),
                    'sample_id':f'train-{index}'}

    class Store:
        prefix='test';bucket=None
        def __init__(self):self.bucket=self;self.paths={};self.remote_name=None
        def upload_verified(self,path,name):
            self.paths[self.prefix+'/'+name]=Path(path)
            return {'generation':'1','sha256':sha256(path),'bytes':Path(path).stat().st_size}
        def download_verified(self,name,target,receipt):
            source=self.paths[self.prefix+'/'+name]
            assert sha256(source)==receipt['sha256'] and source.stat().st_size==receipt['bytes']
            shutil.copyfile(source,target)
        def blob(self,name):self.remote_name=name;return self
        def download_to_filename(self,target,**kwargs):shutil.copyfile(self.paths[self.remote_name],target)

    torch.set_num_threads(1);torch.manual_seed(9)
    model=TinyX4();initial=copy.deepcopy(model.state_dict());before=rng_state()
    original=training.optimizer_update
    monkeypatch.setattr(training,'optimizer_update',lambda *args,**kwargs:original(*args,**dict(kwargs,bf16=False)))
    store=Store()
    chosen=training.preflight(model,TrainingOnly(),'cpu',tmp_path,initial,{'fixture':True},store)
    assert chosen==4
    assert all(torch.equal(value,initial[key]) for key,value in model.state_dict().items())
    assert torch.equal(before['torch'],rng_state()['torch'])
    import json
    proof=json.loads((tmp_path/'preflight/results.json').read_text())
    assert proof['status']=='PASS' and proof['overfit_after']<proof['overfit_before']
    assert proof['resume_max_absolute_error']==0
    assert [row['position'] for row in proof['resume_next_choices']]==list(range(16,32))
    assert all(row['sample_id'].startswith('train-') for row in proof['resume_next_choices'])
    assert 'test/preflight/results.json' in store.paths
    assert 'test/preflight/resume-verification.json' in store.paths


def test_loss_is_fp32_and_averages_all_valid_rgb_values():
    from research.train import masked_l1
    prediction=torch.zeros(2,3,12,12,dtype=torch.bfloat16,requires_grad=True)
    target=torch.full((2,3,12,12),.5,dtype=torch.float32)
    loss=masked_l1(prediction,target,torch.ones(2,12,12,dtype=torch.bool))
    assert loss.dtype==torch.float32 and float(loss.detach())==.5
    loss.backward()
    expected=-1/(2*3*12*12)
    assert torch.allclose(prediction.grad.float(),torch.full_like(prediction.grad.float(),expected),atol=1e-5)
