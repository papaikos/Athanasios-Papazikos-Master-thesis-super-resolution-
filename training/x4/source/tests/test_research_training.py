from pathlib import Path
import random
import numpy as np
import pytest
import torch

def test_accumulation_matches_batch_and_resume_next_update(tmp_path):
    from research.train import optimizer_update,pack_state,restore_state
    from research.data_adapter import SampleStream
    from research.checkpointing import atomic_checkpoint,load_checkpoint
    torch.manual_seed(4)
    a=torch.nn.Conv2d(3,3,1);b=torch.nn.Conv2d(3,3,1);b.load_state_dict(a.state_dict())
    pairs=[{'lr':torch.rand(3,16,16),'hr':torch.rand(3,16,16),'hr_mask':torch.ones(16,16,dtype=torch.bool)} for _ in range(16)]
    oa=torch.optim.Adam(a.parameters());ob=torch.optim.Adam(b.parameters())
    optimizer_update(a,oa,pairs,4,'cpu',1,bf16=False)
    optimizer_update(b,ob,pairs,16,'cpu',1,bf16=False)
    for x,y in zip(a.parameters(),b.parameters()):assert torch.allclose(x,y,atol=1e-7)
    stream=SampleStream(19);stream.indices(16)
    state=pack_state(a,oa,stream,1,{},{});p,r=atomic_checkpoint(state,tmp_path,1)
    optimizer_update(a,oa,pairs,4,'cpu',2,bf16=False)
    next_indices=stream.indices(16)
    restore_state(load_checkpoint(p,r),b,ob,stream,{})
    assert stream.indices(16)==next_indices
    optimizer_update(b,ob,pairs,4,'cpu',2,bf16=False)
    for x,y in zip(a.parameters(),b.parameters()):assert torch.equal(x,y)

def test_evaluation_restores_rng_and_mode(tmp_path):
    from research.evaluate import evaluate
    model=torch.nn.Sequential(torch.nn.Conv2d(3,3,1),torch.nn.Dropout(.5));model.train()
    pairs=[{'lr':torch.rand(3,16,16),'hr':torch.rand(3,16,16),'hr_mask':torch.ones(16,16,dtype=torch.bool),'sample_id':'one'}]
    before=torch.get_rng_state().clone();weights={k:v.clone() for k,v in model.state_dict().items()}
    result=evaluate(model,pairs,'cpu',tmp_path/'v.csv',1)
    assert result['count']==1 and model.training and torch.equal(before,torch.get_rng_state())
    assert all(torch.equal(weights[k],v) for k,v in model.state_dict().items())

def test_config_rejects_scientific_drift():
    import copy
    from research.model_factory import CONFIG,validate_config
    c=copy.deepcopy(CONFIG);c['seed']=43
    with pytest.raises(ValueError):validate_config(c)
    c=copy.deepcopy(CONFIG);c['seed']=42.0
    with pytest.raises(ValueError):validate_config(c)
    c=copy.deepcopy(CONFIG);c['model']['ape']=0
    with pytest.raises(ValueError):validate_config(c)

def test_resume_reconciles_validation_sidecar_without_future_state():
    from research.train import reconcile_events
    state={'completed_updates':500,'cadence':{'validation_step':0},'best':None}
    saved={'completed_updates':500,'cadence':{'validation_step':500},'best':{'step':500}}
    assert reconcile_events(state,saved)['best']['step']==500
    with pytest.raises(ValueError):reconcile_events(state,dict(saved,completed_updates=501))

def test_next_batch_uses_consumed_positions_after_resume():
    from research.train import next_batch
    from research.data_adapter import SampleStream
    dataset=[{'lr':torch.full((3,128,128),float(i)),'hr':torch.full((3,512,512),float(i)),'lr_mask':torch.ones(128,128,dtype=torch.bool),'hr_mask':torch.ones(512,512,dtype=torch.bool),'sample_id':str(i)} for i in range(3)]
    stream=SampleStream(3);next_batch(dataset,stream);state=stream.state_dict()
    pairs,choices=next_batch(dataset,stream)
    replay=SampleStream(3);replay.load_state_dict(state);pairs2,choices2=next_batch(dataset,replay)
    assert choices==choices2 and choices[0]['position']==16
    assert all(torch.equal(a['lr'],b['lr']) for a,b in zip(pairs,pairs2))

def test_best_selection_excludes_initialization_and_keeps_earliest_ties():
    from research.train import is_improvement
    assert not is_improvement(0,{'mean_per_image_rgb_psnr':40,'psnr_infinite':False},None)
    assert is_improvement(1,{'mean_per_image_rgb_psnr':40,'psnr_infinite':False},None)
    assert not is_improvement(2,{'mean_per_image_rgb_psnr':40,'psnr_infinite':False},{'score':40,'psnr_infinite':False})
    assert not is_improvement(2,{'mean_per_image_rgb_psnr':None,'psnr_infinite':True},{'score':None,'psnr_infinite':True})

def test_nonfinite_update_does_not_step_optimizer():
    from research.train import optimizer_update
    model=torch.nn.Conv2d(3,3,1);before={k:v.clone() for k,v in model.state_dict().items()};optimizer=torch.optim.Adam(model.parameters())
    pairs=[{'lr':torch.full((3,16,16),float('nan')),'hr':torch.zeros(3,16,16),'hr_mask':torch.ones(16,16,dtype=torch.bool)}]*16
    with pytest.raises(ValueError):optimizer_update(model,optimizer,pairs,4,'cpu',1,bf16=False)
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items()) and not optimizer.state

def test_weight_hash_is_stable_across_serialization_and_detects_mutation(tmp_path):
    from research.train import weight_hash
    model=torch.nn.Conv2d(3,3,1);before=weight_hash(model.state_dict())
    torch.save(model.state_dict(),tmp_path/'weights.pt')
    assert weight_hash(torch.load(tmp_path/'weights.pt',weights_only=True))==before
    with torch.no_grad():model.weight[0,0,0,0]+=1
    assert weight_hash(model.state_dict())!=before

def test_initial_reference_is_persisted_before_next_checkpoint(tmp_path):
    import json
    from research.train import save_initial_reference
    from research.checkpointing import atomic_json
    cadence={'latest_validation':{'step':0,'count':115,'mean_per_image_rgb_psnr':12.5}}
    def persist():atomic_json(tmp_path/'events.json',{'completed_updates':0,'cadence':cadence,'best':None})
    save_initial_reference(cadence,persist)
    saved=json.loads((tmp_path/'events.json').read_text())
    assert saved['cadence']['initial_validation']=={'step':0,'count':115,'mean_per_image_rgb_psnr':12.5}
    cadence['latest_validation']['step']=500
    assert cadence['initial_validation']['step']==0

def test_full_validation_record_retains_step_time_and_scope(tmp_path):
    import json
    from research.train import record_validation
    aggregate={'step':0,'utc':'2026-10-09T00:00:00+00:00','count':115,'mean_per_image_rgb_psnr':12.5,'psnr_infinite':False}
    record_validation(tmp_path,aggregate)
    row=json.loads((tmp_path/'logs/validation.jsonl').read_text())
    assert row==dict(aggregate,scope='full_main_validation')

def test_frozen_run_metadata_is_uploaded_with_readback_contract(tmp_path):
    from research.train import publish_frozen_metadata
    class Store:
        def __init__(self):self.verified=[]
        def upload_verified(self,path,relative):
            self.verified.append((relative,path.read_text()))
            return {'sha256':'verified','bytes':path.stat().st_size}
    expected=['provenance/initial-weights.json','provenance/step-one-reset.json','provenance/environment.json','provenance/model.json','config/resolved.json','config/resolved.yaml']
    for relative in expected:
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(relative)
    store=Store();publish_frozen_metadata(tmp_path,store)
    assert dict(store.verified)=={relative:relative for relative in expected}

def test_trainer_utc_controls_parse_with_python310_iso_parser(monkeypatch):
    from datetime import datetime,timezone,timedelta
    import infra.lifecycle as lifecycle
    import research.train as training
    class Python310Datetime(datetime):
        @classmethod
        def fromisoformat(cls,value):
            if value.endswith('Z'):raise ValueError('Python3.10 rejects trailing Z')
            return super().fromisoformat(value)
    monkeypatch.setattr(lifecycle,'datetime',Python310Datetime)
    control={'t0_utc':'2026-10-09T09:23:57Z','soft_cutoff_utc':'2026-10-14T03:23:57Z','cleanup_cutoff_utc':'2026-10-14T07:23:57Z'}
    start=training.utc(control['t0_utc']);soft=training.utc(control['soft_cutoff_utc']);cleanup=training.utc(control['cleanup_cutoff_utc'])
    assert start==datetime(2026,10,9,9,23,57,tzinfo=timezone.utc)
    assert soft-start==timedelta(hours=114) and cleanup-start==timedelta(hours=118)
    assert training.utc('2026-10-09T12:23:57+03:00')==start
    with pytest.raises(ValueError):training.utc('2026-10-09T09:23:57')
