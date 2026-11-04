"""Frozen D1 inference on the exact preserved D3 demonstration inputs."""
import csv, hashlib, html, importlib.util, json, os, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(os.environ.get('RUN_ARCHIVE', Path(__file__).resolve().parents[1]))  # archive folder of one run, e.g. 'D1 run'
D3=Path(os.environ.get('D3_ARCHIVE', ROOT.parent / 'D3 run'))
PROJECT=ROOT/'training-run/project'
CHECKOUT=D3/'reproducibility/checkout'
sys.path.insert(0,str(CHECKOUT))
from research.model_factory import make_model
from research.checkpointing import load_checkpoint, sha256
from research.metrics import score_image, aggregate
from research.lpips_metric import LPIPSMetric
GROUPS=['sentinel2-additional-ports','sentinel2-comparisons','skysat-comparisons']

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def checked_copy(src,dst,expected=None):
    digest=sha256(src)
    if expected and digest!=expected: raise ValueError(f'Source hash mismatch: {src}')
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    if sha256(dst)!=digest: raise ValueError(f'Copy mismatch: {dst}')
    return {'sha256':digest,'bytes':dst.stat().st_size,'original_path':str(src)}

def rgb(a):
    return Image.fromarray(np.rint(np.clip(a.transpose(1,2,0),0,1)*255).astype(np.uint8))

def panel(path,sample,arrays,labels,crop=None,sky=False):
    side=256 if sky else 512
    rows=(len(arrays)+2)//3
    canvas=Image.new('RGB',(side*3,rows*(side+54)+74),'white')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16 if sky else 19)
    for i,(a,label) in enumerate(zip(arrays,labels)):
        pic=rgb(a) if a.ndim==3 else Image.fromarray(np.rint(np.clip(a/.25,0,1)*255).astype(np.uint8)).convert('RGB')
        if crop: pic=pic.crop(crop)
        x,y=i%3*side,i//3*(side+54)
        draw.text((x+5,y+8),label,fill='black',font=font)
        draw.text((x+5,y+30),'Display only; nearest-neighbor enlargement',fill='#555555',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',11 if sky else 15))
        canvas.paste(pic.resize((side,side),Image.Resampling.NEAREST),(x,y+54))
    draw.text((5,rows*(side+54)+8),sample+(' | identical D3 detail crop' if crop else ''),fill='black',font=font)
    draw.text((5,rows*(side+54)+33),'Synthetic x4 reference; error scale 0..0.25.' if sky else 'No independent HR ground truth; nominal 2.5m grid is not verified resolving power.',fill='black',font=font)
    canvas.save(path)

def prepare_models():
    verification=json.loads((ROOT/'backup-verification.json').read_text())
    if verification['status']!='PASS': raise ValueError('Verified complete backup required')
    folder=PROJECT/'results/checkpoints'
    bestpointer=json.loads((folder/'best.json').read_text())
    pointers={'best':bestpointer['checkpoint'],'final':json.loads((folder/'latest.json').read_text())}
    identities={}; reports={}
    model,info=make_model(CHECKOUT/'vendor/network_swinir.py')
    expected_hash='9e143898679ebeebc5d2fc94ad1b89c38aa4a4d43da4e0fcba0f93e476994913'
    if sha256(CHECKOUT/'vendor/network_swinir.py')!=expected_hash: raise ValueError('Architecture source changed')
    for label,receipt in pointers.items():
        p=folder/receipt['filename']; state=load_checkpoint(p,receipt)
        step=84000 if label=='best' else 100000
        checks={'receipt_step':receipt['step']==step,'state_step':state['completed_updates']==step,
          'target':state['target_updates']==100000,'sampler_cursor':state['sampler']['cursor']==step*16,
          'crop_exposure':state['crop_exposure']==step*16,
          'batch':(state['microbatch'],state['accumulation_steps'],state['effective_batch'])==(4,4,16),
          'schedule':state['scheduler']=={'completed_updates':step,'original_target':100000},
          'model_source':state['identity']['model_source_sha256']==expected_hash,
          'finite_model':all(torch.isfinite(v).all().item() for v in state['model'].values()),
          'optimizer_steps':all(int(v['step'])==step for v in state['optimizer']['state'].values())}
        model.load_state_dict(state['model'],strict=True)
        if not all(checks.values()): raise ValueError(checks)
        dest=ROOT/'models'/f'{label}-step-{step}-weights.pth'
        torch.save(state['model'],dest)
        reloaded=torch.load(dest,map_location='cpu',weights_only=True)
        if not all(torch.equal(v,reloaded[k]) for k,v in state['model'].items()): raise ValueError('Weight export mismatch')
        identities[label]={'step':step,'checkpoint_path':str(p),'checkpoint_sha256':receipt['sha256'],
          'weights_path':str(dest),'weights_sha256':sha256(dest),'scientific_identity':state['identity']}
        reports[label]={'status':'PASS','checks':checks,'receipt':receipt}
    write(ROOT/'models/model-identities.json',identities)
    write(ROOT/'models/full-state-verification.json',reports)
    return identities,info

def main():
    if torch.__version__!='2.8.0+cpu' or np.__version__!='2.2.6': raise ValueError('Original D3 inference runtime required')
    torch.set_num_threads(6);torch.manual_seed(42);np.random.seed(42);torch.use_deterministic_algorithms(True)
    identities,info=prepare_models()
    d3identities=json.loads((D3/'models/model-identities.json').read_text())
    for ident in d3identities.values():
        if sha256(Path(ident['weights_path']))!=ident['weights_sha256']: raise ValueError('D3 weights changed')
    model,_=make_model(CHECKOUT/'vendor/network_swinir.py');model.eval().requires_grad_(False)
    lpips=LPIPSMetric('cpu')
    rows=[]; allrecords=[]; cards=[]
    for name in GROUPS:
        group=ROOT/name; original=D3/name
        samples=sorted(p for p in (original/'results').iterdir() if p.is_dir())
        if len(samples)!=10: raise ValueError('Exactly ten preserved images per group required')
        sky=name=='skysat-comparisons'; records=[]
        for src in samples:
            sample=src.name; dest=group/'results'/sample; dest.mkdir(parents=True,exist_ok=True)
            provenance=json.loads((src/'provenance.json').read_text(encoding='utf-8'))
            inventory=provenance['outputs']; copies={}
            lrname='LR-x4-fp32.npy' if sky else 'LR-input-fp32.npy'
            lrpng='LR-x4.png' if sky else 'LR-input.png'
            names=[lrname,lrpng]+(['HR-reference.png'] if sky else [])
            for filename in names:
                copies[filename]=checked_copy(src/filename,dest/filename,inventory[filename]['sha256'])
            lr=np.load(dest/lrname,allow_pickle=False)
            pixels=np.asarray(Image.open(dest/lrpng)).transpose(2,0,1).astype(np.float32)/255
            if lr.dtype!=np.float32 or not np.array_equal(lr,pixels) or not np.isfinite(lr).all(): raise ValueError('Saved LR is not exact D3 input')
            lr_t=torch.from_numpy(lr)
            hr=np.asarray(Image.open(dest/'HR-reference.png')).transpose(2,0,1).astype(np.float32)/255 if sky else None
            shape=hr.shape if sky else (3,512,512)
            predictions={}; timings={}
            for label in ['final','best']:
                for filename in [label+'-raw-fp32.npy',label+'.png']:
                    copies['D3-'+filename]=checked_copy(src/filename,dest/('D3-'+filename),inventory[filename]['sha256'])
                d3=np.load(dest/('D3-'+label+'-raw-fp32.npy'),allow_pickle=False)
                if d3.dtype!=np.float32 or d3.shape!=shape or not np.isfinite(d3).all():raise ValueError('Invalid preserved D3 prediction')
                predictions['D3-'+label]=d3
                weights=torch.load(identities[label]['weights_path'],map_location='cpu',weights_only=True)
                model.load_state_dict(weights,strict=True);started=time.perf_counter()
                with torch.inference_mode():
                    prediction=model(lr_t[None])[0].float().cpu();repeat=model(lr_t[None])[0].float().cpu()
                if tuple(prediction.shape)!=shape or not torch.isfinite(prediction).all() or not torch.equal(prediction,repeat):raise ValueError('Invalid or nonrepeatable D1 prediction')
                if any(not torch.equal(weights[k],v) for k,v in model.state_dict().items()):raise ValueError('Inference mutated model')
                timings[label]=time.perf_counter()-started;predictions['D1-'+label]=prediction.numpy()
                np.save(dest/(label+'-raw-fp32.npy'),prediction.numpy());rgb(prediction.numpy()).save(dest/(label+'.png'))
                print(json.dumps({'group':name,'sample':sample,'model':'D1-'+label,'seconds':timings[label],'repeat_equal':True}),flush=True)
            detail=provenance['detail_crop_xyxy'] if sky else provenance['detail_crop_output_xyxy']
            enlarged=torch.nn.functional.interpolate(lr_t[None],size=shape[-2:],mode='bicubic' if sky else 'nearest',**({'align_corners':False,'antialias':False} if sky else {}))[0].numpy()
            for mode in ['D1','best','final']:
                if mode=='D1':
                    left,right=predictions['D1-final'],predictions['D1-best'];labels=['D1 final (100000)','D1 best (84000)'];base='comparison'
                else:
                    left,right=predictions['D1-'+mode],predictions['D3-'+mode]
                    labels=[f'D1 {mode} ({identities[mode]["step"]})',f'D3 {mode} ({d3identities[mode]["step"]})'];base='D1-vs-D3-'+mode
                arrays=[hr if sky else enlarged,left,right];titles=['HR source reference' if sky else 'Original Sentinel-2 input (10m)',*labels]
                if sky:arrays.extend([enlarged,np.abs(np.clip(left,0,1)-hr).mean(0),np.abs(np.clip(right,0,1)-hr).mean(0)]);titles.extend(['LR bicubic display x4','Mean absolute error: left','Mean absolute error: right'])
                panel(dest/(base+'.png'),sample,arrays,titles,sky=sky)
                panel(dest/('detail.png' if mode=='D1' else base+'-detail.png'),sample,arrays,titles,crop=detail,sky=sky)
            if sky:
                for label,pred in predictions.items():
                    mask=torch.ones(hr.shape[-2:],dtype=torch.bool)
                    scores=score_image(pred,hr,mask.numpy());scores['lpips']=lpips.score(torch.from_numpy(pred),torch.from_numpy(hr),mask)
                    scores.update(sample_id=sample,model=label,step=(identities if label.startswith('D1') else d3identities)[label.split('-')[1]]['step'])
                    rows.append(scores)
                    np.save(dest/(label+'-absolute-error.npy'),np.abs(np.clip(pred,0,1)-hr).mean(0))
                    Image.fromarray(np.rint(np.clip(np.abs(np.clip(pred,0,1)-hr).mean(0)/.25,0,1)*255).astype(np.uint8)).save(dest/(label+'-error-0-to-0.25.png'))
            if sky:
                for label in ['best','final']:
                    checked_copy(dest/('D1-'+label+'-absolute-error.npy'),dest/(label+'-absolute-error.npy'))
                    checked_copy(dest/('D1-'+label+'-error-0-to-0.25.png'),dest/(label+'-error-0-to-0.25.png'))
            record={'group':name,'id':sample,'original_D3_provenance':provenance,'original_provenance_sha256':sha256(src/'provenance.json'),
                'copied_inputs_and_D3_outputs':copies,'exact_D3_input_reused':True,'D1_repeat_equal':True,'model_state_unchanged':True,
                'input_shape':list(lr.shape),'output_shape':list(shape),'seconds_two_forwards':timings,'detail_crop_xyxy':detail,
                'outputs':{p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in dest.iterdir() if p.is_file() and p.name!='provenance.json'}}
            write(dest/'provenance.json',record);records.append(record);allrecords.append(record)
            cards.append(f'<h2>{html.escape(name+" / "+sample)}</h2><a href="../{name}/results/{sample}/D1-vs-D3-best.png"><img style="max-width:100%" src="../{name}/results/{sample}/D1-vs-D3-best.png"></a><p><a href="../{name}/results/{sample}/comparison.png">D1 final/best</a> | <a href="../{name}/results/{sample}/D1-vs-D3-final.png">Final100k comparison</a> | <a href="../{name}/results/{sample}/D1-vs-D3-best-detail.png">Same detail crop</a> | <a href="../{name}/results/{sample}/provenance.json">Provenance</a></p>')
        write(group/'inference.json',{'samples':records,'count':10,'ground_truth':'Synthetic reference only' if sky else 'Unavailable; no HR quality metrics'})
        (group/'README.txt').write_text('D1 inference on exact preserved D3 inputs. D1 best=84000; D3 best=82000; finals=100000. Original D3 predictions copied byte-for-byte, not rerun. Same full-frame CPU FP32 runtime and source architecture; repeated D1 inference equals exactly; weights remain unchanged. '+('SkySat metrics are a synthetic x4 demonstration, not independent HR satellite validation.' if sky else 'Sentinel-2 has no independent HR ground truth. Nominal 2.5m output grid does not establish physical 2.5m resolving power.')+'\n',encoding='utf-8')
    with (ROOT/'skysat-comparisons/metrics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={label:dict(aggregate([r for r in rows if r['model']==label]),mean_per_image_lpips=float(np.mean([r['lpips'] for r in rows if r['model']==label]))) for label in ['D1-best','D1-final','D3-best','D3-final']}
    for ident in identities.values():
        if sha256(Path(ident['weights_path']))!=ident['weights_sha256'] or sha256(Path(ident['checkpoint_path']))!=ident['checkpoint_sha256']:raise ValueError('D1 weights or full checkpoint changed')
    references={p.name:{'sha256':sha256(p),'bytes':p.stat().st_size,'role':'Original D3 inference script; reference only, not executed unchanged'} for p in (ROOT/'tools').glob('D3-original-reference-*.py')}
    protocol={'reference_scripts':references,'adapted_script_sha256':sha256(Path(__file__)),'status':'PASS','created_utc':datetime.now(timezone.utc).isoformat(),'count':len(allrecords),'model_forwards':120,
      'inference':'CPU FP32, full-frame, no tiling, eval/inference_mode; seed42; six threads; deterministic; two identical forwards per D1 model and image; immutable weights',
      'inputs':'Exact existing D3 LR float32 arrays reused without regenerating degradation; PNG agreement and recorded SHA256 checked',
      'runtime':{'torch':torch.__version__,'numpy':np.__version__,'threads':torch.get_num_threads()},'D1_models':identities,'D3_models':d3identities,
      'architecture':info,'lpips':lpips.provenance,'SkySat_summary':summary,'samples':allrecords}
    if len(allrecords)!=30:raise ValueError('Not all 30 scenes complete')
    write(ROOT/'reports/comparison-verification.json',protocol)
    write(ROOT/'skysat-comparisons/evaluation.json',{'scope':'external synthetic demonstration; not sealed test','metrics':'Identical original D3 score_image and LPIPSMetric, full RGB support; arithmetic per-image means','summary':summary,'lpips':lpips.provenance})
    (ROOT/'reports/comparison-index.html').write_text('<!doctype html><meta charset="utf-8"><title>D1 versus D3</title><body style="font-family:Arial;max-width:1550px;margin:2em auto"><h1>D1 versus D3: identical 30 comparison inputs</h1><p>D1 best84000 versus D3 best82000. Matching final100k comparisons linked below. Sentinel images have no independent HR ground truth; SkySat metrics use preserved synthetic references.</p><pre>'+html.escape(json.dumps(summary,indent=2))+'</pre>'+''.join(cards)+'</body>',encoding='utf-8')
    print(json.dumps({'status':'PASS','count':30,'summary':summary}),flush=True)
if __name__=='__main__':main()
