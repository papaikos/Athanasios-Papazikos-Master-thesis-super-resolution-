"""Small CPU supervised x4 RGB format smoke; does not train a thesis model."""
import argparse
import importlib
import json
from pathlib import Path
import re

from scripts.maritime_uint8_download import FOLDERS,build_jobs,build_auxiliary_jobs
from src.maritime_uint8_dataset import MaritimeUint8Dataset


def local_shards(root,recipe='d1',split='train',*,evaluation=None):
    if recipe not in ('d1','d2','d3'): raise ValueError('A child recipe d1/d2/d3 is required')
    root=Path(root).expanduser().resolve()
    if evaluation is not None:
        if not isinstance(evaluation,str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*',evaluation):
            raise ValueError('Safe auxiliary evaluation name required')
        if split=='train':raise ValueError('Auxiliary evaluation cannot be used as the training split')
        document=json.loads((root/FOLDERS[recipe]/'evaluation'/evaluation/'manifest.json').read_text(encoding='utf-8-sig'))
        audit=json.loads((root/'control'/'evaluation'/evaluation/'quality-audit.json').read_text(encoding='utf-8-sig'))
        jobs=build_auxiliary_jobs({recipe:document},audit,name=evaluation,recipes=[recipe],splits=[split])
    else:
        document=json.loads((root/FOLDERS[recipe]/'manifest.json').read_text(encoding='utf-8-sig'))
        jobs=build_jobs({recipe:document},recipes=[recipe],splits=[split])
    paths=[root/relative for _,relative in jobs]
    if not paths or any(not path.is_file() for path in paths): raise ValueError('Selected split shards are missing locally')
    return paths


def masked_supervised_loss(prediction,target,mask,kind='l1'):
    """Mean supervised RGB error over valid HR pixels and all three channels."""
    import torch
    if (prediction.shape!=target.shape or prediction.ndim!=4 or prediction.shape[1]!=3
            or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()
            or mask.dtype!=torch.bool):
        raise ValueError('Finite matching NCHW RGB and boolean support required')
    if mask.ndim==3: mask=mask.unsqueeze(1)
    if mask.shape!=(target.shape[0],1,target.shape[2],target.shape[3]) or not mask.any():
        raise ValueError('Valid nonempty HR support mask required')
    if kind=='l1': error=(prediction.float()-target.float()).abs()
    elif kind=='mse': error=(prediction.float()-target.float()).square()
    else: raise ValueError('Supported supervised losses are l1 and mse')
    weights=mask.expand_as(error)
    return torch.where(weights,error,0).sum()/weights.sum()


def engineering_model():
    import torch
    return torch.nn.Sequential(torch.nn.Conv2d(3,8,3,padding=1),torch.nn.ReLU(),
        torch.nn.Conv2d(8,3*4*4,3,padding=1),torch.nn.PixelShuffle(4))


def training_smoke(root,*,recipe='d1',split='train',steps=2,batch_size=1,loss='l1',
                   learning_rate=0.0001,seed=42,model_factory=None,model_kwargs=None,evaluation=None):
    import torch
    from torch.utils.data import DataLoader
    if not 1<=steps<=20 or not 1<=batch_size<=4 or not 0<learning_rate<=1:
        raise ValueError('Smoke steps 1..20, batch 1..4 and positive finite learning rate required')
    torch.set_num_threads(1);torch.manual_seed(seed)
    dataset=MaritimeUint8Dataset(local_shards(root,recipe,split,evaluation=evaluation),recipe)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0,generator=generator)
    if model_factory:
        module,symbol=model_factory.split(':',1)
        model=getattr(importlib.import_module(module),symbol)(**(model_kwargs or {})).cpu()
        model_name=model_factory
    else:
        model=engineering_model();model_name='small engineering Conv2d/PixelShuffle x4'
    model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    before=[parameter.detach().clone() for parameter in model.parameters()]
    iterator=iter(loader);losses=[];sample_ids=[]
    for _ in range(steps):
        try: batch=next(iterator)
        except StopIteration: iterator=iter(loader);batch=next(iterator)
        if batch['lr'].shape[1:]!=(3,128,128) or batch['hr'].shape[1:]!=(3,512,512):
            raise ValueError('Expected LR128 and HR512 RGB x4 pairs')
        prediction=model(batch['lr'])
        optimizer.zero_grad(set_to_none=True)
        value=masked_supervised_loss(prediction,batch['hr'],batch['hr_mask'],loss)
        value.backward()
        if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
            raise ValueError('Nonfinite model gradients')
        optimizer.step();losses.append(float(value.detach()));sample_ids.extend(batch['sample_id'])
    updated=any(not torch.equal(old,new.detach()) for old,new in zip(before,model.parameters()))
    if not updated: raise ValueError('Optimizer did not update model parameters')
    return dict(scope='engineering format smoke; not a trained thesis model',recipe=recipe,split=split,
        auxiliary_evaluation=evaluation,
        dataset_samples=len(dataset),steps=steps,loss=loss,loss_values=losses,
        hr_shape=list(batch['hr'].shape),lr_shape=list(batch['lr'].shape),
        stored_dtype='uint8',tensor_dtype='float32',normalization='uint8 /255 on load',
        optimizer='Adam',optimizer_updated_parameters=updated,model=model_name,
        sampled_ids=sample_ids,torch_version=torch.__version__,device='cpu',
        scientific_performance_claim=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/maritime-uint8-training-example.json')
    parser.add_argument('--root',help='Override example dataset root')
    parser.add_argument('--recipe',choices=['d1','d2','d3'])
    parser.add_argument('--split',choices=['train','validation','test'])
    parser.add_argument('--evaluation',help='Separate auxiliary evaluation name; validation/test only')
    parser.add_argument('--steps',type=int)
    parser.add_argument('--output',help='Optional engineering smoke report JSON')
    args=parser.parse_args(argv)
    config=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    report=training_smoke(args.root or config['dataset_root'],recipe=args.recipe or config['recipe'],
        split=args.split or config.get('split','train'),steps=args.steps if args.steps is not None else config.get('steps',2),
        batch_size=config.get('batch_size',1),loss=config.get('loss','l1'),
        learning_rate=config.get('learning_rate',0.0001),seed=config.get('seed',42),
        evaluation=args.evaluation or config.get('auxiliary_evaluation'),
        model_factory=config.get('model_factory'),model_kwargs=config.get('model_kwargs',{}))
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__': main()
