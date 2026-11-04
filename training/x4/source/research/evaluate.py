"""Serial full-frame evaluation with RNG, state and training-mode purity."""
import csv
import random
from datetime import datetime,timezone
import numpy as np
import torch
from .metrics import score_image,aggregate

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])

def evaluate(model,dataset,device,output,expected_count,*,bicubic=False,deadline=None,lpips_metric=None):
    if len(dataset)!=expected_count:raise ValueError('Full evaluation membership required')
    state=rng_state();modes={m:m.training for m in model.modules()};before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};rows=[]
    try:
        model.eval()
        with torch.inference_mode():
            for pair in dataset:
                if deadline and datetime.now(timezone.utc)>=deadline:raise TimeoutError('Bounded evaluation deadline reached')
                x=pair['lr'].unsqueeze(0).to(device)
                prediction=torch.nn.functional.interpolate(x,size=pair['hr'].shape[-2:],mode='bicubic',align_corners=False,antialias=False) if bicubic else model(x)
                if prediction.shape[1:]!=pair['hr'].shape:raise ValueError('Incorrect x4 inference shape')
                row=score_image(prediction[0].cpu().numpy(),pair['hr'].numpy(),pair['hr_mask'].numpy())
                if lpips_metric is not None:row['lpips']=lpips_metric.score(prediction[0],pair['hr'],pair['hr_mask'])
                row['sample_id']=pair['sample_id'];rows.append(row)
        if any(not torch.equal(before[k],v.detach().cpu()) for k,v in model.state_dict().items()):raise ValueError('Evaluation mutated model state')
    finally:
        for module,training in modes.items():module.training=training
        restore_rng(state)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=aggregate(rows)
    if lpips_metric is not None:result['mean_per_image_lpips']=float(np.mean([row['lpips'] for row in rows],dtype=np.float64))
    return result
