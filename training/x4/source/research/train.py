"""Classical SwinIR fit with verified GCS or filesystem/Slurm persistence."""
import argparse
import copy
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import time
import threading
import numpy as np
import torch
from infra.lifecycle import utc
from .checkpointing import atomic_json,atomic_checkpoint,load_checkpoint,GCSStore,sha256
from .data_adapter import SampleStream,aligned_crop,datasets_from_audit
from .evaluate import evaluate,rng_state,restore_rng
from .experiment import NAME
from .model_factory import CONFIG,validate_config,make_model
from .reporting import event,utcnow

def make_store(backend,durable_root,control):
    if backend=='filesystem':
        from .hpc_store import FilesystemStore
        if not durable_root:raise ValueError('--durable-root required for filesystem storage')
        return FilesystemStore(durable_root)
    return GCSStore(control['output_uri'])


def preflight_microbatches(backend):
    return (4,) if backend=='filesystem' else (4,2,1)


def validate_runtime_microbatch(backend,microbatch):
    if backend=='filesystem' and microbatch!=4:
        raise ValueError(NAME+' comparison requires microbatch4/accumulation4; smaller microbatch prohibited')


def job_runtime_expired(started,max_seconds,*,now=None):
    return max_seconds is not None and (time.monotonic() if now is None else now)-started>=max_seconds


def sync_progress(root,store):
    for directory in ('logs','metrics','provenance','config','preflight','tensorboard'):
        for path in sorted((Path(root)/directory).rglob('*')):
            if path.is_file() and path.suffix!='.tmp':
                store.sync_file(path,path.relative_to(root).as_posix())


def pause_run(root,store,checkpoint,status):
    """Publish an update boundary without evaluation/export or completion claims."""
    checkpoint()
    sync_progress(root,store)
    value=status()
    value.update(phase='PAUSED',failure_reason=value.get('failure'),training_complete=False,cleanup_ready=False,export_outcome='DEFERRED_UNTIL_TARGET')
    atomic_json(Path(root)/'status/guest-status.json',value)
    store.pointer('status/guest-status.json',value)
    return value


def defer_target_finalization(root,store,checkpoint,status):
    """Persist the final training boundary for evaluation in a fresh allocation."""
    checkpoint()
    sync_progress(root,store)
    value=status()
    value.update(phase='AWAITING_FINALIZATION',outcome='TARGET_REACHED',training_complete=True,cleanup_ready=False,export_outcome='DEFERRED_UNTIL_FINALIZATION_JOB')
    atomic_json(Path(root)/'status/guest-status.json',value)
    store.pointer('status/guest-status.json',value)
    return value


def pending_target_validation(backend,deferred,step,cadence):
    return backend=='filesystem' and not deferred and step==100000 and cadence.get('validation_step')!=100000


def recovery_storage_control(control,storage_path,scientific_path,root):
    if not storage_path:return control
    from ops.bootstrap_guest import validate_recovery_controls
    operational=json.loads(Path(storage_path).read_text())
    parent=operational['recovery_parent_control_pin']
    if sha256(scientific_path)!=parent['sha256']:raise ValueError('Scientific recovery control byte pin differs')
    validate_recovery_controls(control,operational,parent)
    if not (Path(root)/'checkpoints/latest.json').is_file():raise ValueError('Recovery checkpoint is required; scratch fallback prohibited')
    return operational


def verify_recovery_best(state,root):
    best=state.get('best')
    if best:
        receipt=best['checkpoint']
        filename=receipt['filename']
        if Path(filename).name!=filename or '/' in filename or '\\' in filename:raise ValueError('Unsafe recovery best checkpoint filename')
        if best['step']>state['completed_updates'] or receipt['step']>state['completed_updates']:raise ValueError('Recovery best checkpoint is ahead')
        load_checkpoint(Path(root)/'checkpoints'/filename,receipt)

def weight_hash(state):
    digest=hashlib.sha256()
    for key,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        digest.update(json.dumps([key,str(value.dtype),list(value.shape)],separators=(',',':')).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def reconcile_events(state,saved):
    if saved['completed_updates']>state['completed_updates']:raise ValueError('Event state is ahead of resumable checkpoint')
    result=dict(state)
    if saved['completed_updates']==state['completed_updates']:result.update(cadence=saved['cadence'],best=saved['best'])
    return result

def next_batch(dataset,stream):
    position=stream.cursor;indices=stream.indices(16);pairs=[];choices=[]
    for offset,index in enumerate(indices):
        pair,choice=aligned_crop(dataset[index],position+offset);pairs.append(pair);choices.append(choice)
    return pairs,choices

def is_improvement(step,value,best):
    if step<1:return False
    if best is None:return True
    if best['psnr_infinite']:return False
    return value['psnr_infinite'] or value['mean_per_image_rgb_psnr']>best['score']

def save_initial_reference(cadence,persist):
    cadence['initial_validation']=copy.deepcopy(cadence['latest_validation'])
    persist()

def record_validation(root,value):
    event(root,'validation',dict(value,scope='full_main_validation'))

def publish_frozen_metadata(root,store):
    for relative in ('provenance/initial-weights.json','provenance/step-one-reset.json','provenance/environment.json','provenance/model.json','config/resolved.json','config/resolved.yaml'):
        store.upload_verified(Path(root)/relative,relative)

def needs_lpips_reference_refresh(step,cadence):
    complete=all('mean_per_image_lpips' in cadence.get(key,{}) for key in ('bicubic','initial_validation'))
    if not complete and step!=0:raise ValueError('Missing LPIPS scratch references can only be recovered at step zero')
    return not complete

def learning_rate(update):
    if not 1<=update<=100000:raise ValueError('Update outside original schedule')
    return .0002*update/2000 if update<=2000 else .000001+.5*(.0002-.000001)*(1+math.cos(math.pi*(update-2000)/98000))

def masked_l1(prediction,target,mask):
    if prediction.shape!=target.shape or prediction.ndim!=4 or prediction.shape[1]!=3 or mask.dtype!=torch.bool:raise ValueError('Matching RGB and boolean support required')
    if mask.shape!=(target.shape[0],*target.shape[-2:]) or not mask.flatten(1).any(1).all():raise ValueError('Empty or malformed support')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():raise ValueError('Nonfinite loss input')
    error=(prediction.float()-target.float()).abs();weights=mask[:,None].expand_as(error)
    return torch.where(weights,error,0).sum()/weights.sum()

def optimizer_update(model,optimizer,pairs,microbatch,device,update,*,bf16=True):
    if len(pairs)!=16 or 16%microbatch:raise ValueError('Effective batch must be 16')
    model.train();optimizer.zero_grad(set_to_none=True);total=0.
    for group in optimizer.param_groups:group['lr']=learning_rate(update)
    for start in range(0,16,microbatch):
        part=pairs[start:start+microbatch]
        x=torch.stack([p['lr'] for p in part]).to(device);y=torch.stack([p['hr'] for p in part]).to(device);mask=torch.stack([p['hr_mask'] for p in part]).to(device)
        with torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16,enabled=bf16):prediction=model(x)
        loss=masked_l1(prediction,y,mask);(loss*(len(part)/16)).backward();total+=float(loss.detach())*len(part)/16
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):raise ValueError('Nonfinite or missing gradient')
    norm=float(torch.sqrt(sum(g.float().square().sum() for g in gradients)))
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite weights')
    return {'raw_l1':total,'gradient_norm':norm,'lr':learning_rate(update),'microbatches':16//microbatch}

def pack_state(model,optimizer,stream,step,cadence,identity,best=None,microbatch=4):
    return {'schema':1,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':{'completed_updates':step,'original_target':100000},'scaler':None,'completed_updates':step,'target_updates':100000,'sampler':stream.state_dict(),'rng':rng_state(),'cadence':copy.deepcopy(cadence),'identity':identity,'best':best,'microbatch':microbatch,'accumulation_steps':16//microbatch,'effective_batch':16,'crop_exposure':step*16,'augmentation_scheme':'SeedSequence(seed,consumed_position,314159)-v1'}

def restore_state(state,model,optimizer,stream,identity):
    if state['schema']!=1 or state['identity']!=identity or state['target_updates']!=100000 or state['sampler']['cursor']!=state['completed_updates']*16:raise ValueError('Resume identity/accounting mismatch')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);stream.load_state_dict(state['sampler']);restore_rng(state['rng'])

def optimizer_for(model):return torch.optim.Adam(model.parameters(),lr=.0002,betas=(.9,.99),eps=1e-8,weight_decay=0)

def preflight(model,dataset,device,run_dir,initial,identity,store,*,storage_backend="gcs"):
    """Actual model shape, TRAIN overfit and save/reload-next-update gate."""
    root=Path(run_dir);initial_rng=rng_state();result={};chosen=None
    fixed=[aligned_crop(dataset[i%len(dataset)],i)[0] for i in range(16)]
    try:
        with torch.inference_mode():
            model.eval()
            assert model(fixed[0]['lr'][None].to(device)).shape==(1,3,256,256)
            assert model(dataset[0]['lr'][None].to(device)).shape==(1,3,512,512)
        for microbatch in preflight_microbatches(storage_backend):
            model.load_state_dict(initial);restore_rng(initial_rng);optimizer=optimizer_for(model)
            try:optimizer_update(model,optimizer,fixed,microbatch,device,2000);chosen=microbatch;break
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        if chosen is None:raise RuntimeError('Actual model cannot fit required microbatch4/accumulation4' if storage_backend=='filesystem' else 'Actual model cannot fit effective batch16')
        model.load_state_dict(initial);optimizer=optimizer_for(model)
        def deterministic_loss():
            model.eval()
            with torch.inference_mode():return float(masked_l1(model(fixed[0]['lr'][None].to(device)),fixed[0]['hr'][None].to(device),fixed[0]['hr_mask'][None].to(device)))
        before=deterministic_loss()
        for _ in range(8):optimizer_update(model,optimizer,fixed,chosen,device,2000)
        after=deterministic_loss()
        if not after<before:raise ValueError('Bounded TRAIN overfit failed to lower deterministic L1')
        model.load_state_dict(initial);restore_rng(initial_rng);optimizer=optimizer_for(model);stream=SampleStream(len(dataset))
        first_pairs,_=next_batch(dataset,stream);optimizer_update(model,optimizer,first_pairs,chosen,device,1)
        path,receipt=atomic_checkpoint(pack_state(model,optimizer,stream,1,{},identity,microbatch=chosen),root/'preflight',1)
        remote=store.upload_verified(path,'preflight/'+path.name)
        # Resume directly from downloaded durable bytes, rather than trusting the local file.
        download=root/'preflight/remote-resume.pt'
        store.download_verified('preflight/'+path.name,download,remote)
        second_pairs,next_choices=next_batch(dataset,stream);optimizer_update(model,optimizer,second_pairs,chosen,device,2);expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        state=load_checkpoint(download,receipt);restore_state(state,model,optimizer,stream,identity)
        replay_pairs,replay_choices=next_batch(dataset,stream)
        if replay_choices!=next_choices or any(not torch.equal(a['lr'],b['lr']) or not torch.equal(a['hr'],b['hr']) for a,b in zip(second_pairs,replay_pairs)):raise ValueError('Resume next sample/crop/augmentation differs')
        optimizer_update(model,optimizer,replay_pairs,chosen,device,2)
        max_error=max(float((v.detach().cpu()-expected[k]).abs().max()) for k,v in model.state_dict().items())
        if max_error>1e-6:raise ValueError('Resume next update differs above 1e-6')
        result={'status':'PASS','real_model':True,'overfit_before':before,'overfit_after':after,'resume_max_absolute_error':max_error,'resume_tolerance':1e-6,'resume_next_choices':next_choices,'durable_checkpoint':remote,'microbatch':chosen,'accumulation':16//chosen,'grad_scaler':False}
    finally:model.load_state_dict(initial);restore_rng(initial_rng)
    atomic_json(root/'preflight/results.json',result);atomic_json(root/'preflight/resume-verification.json',result)
    store.upload_verified(root/'preflight/results.json','preflight/results.json')
    store.upload_verified(root/'preflight/resume-verification.json','preflight/resume-verification.json')
    return chosen

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data-root','run-dir','control','source-root','model-source','audit'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--storage-backend',choices=('gcs','filesystem'),default='gcs')
    parser.add_argument('--durable-root')
    parser.add_argument('--job-max-runtime-seconds',type=float)
    parser.add_argument('--checkpoint-retention',choices=('all','best-latest'),default='all')
    parser.add_argument('--defer-finalization',action='store_true')
    parser.add_argument('--config');parser.add_argument('--storage-control');parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args(argv)
    job_started=time.monotonic()
    if args.job_max_runtime_seconds is not None and (not math.isfinite(args.job_max_runtime_seconds) or args.job_max_runtime_seconds<=0):parser.error('--job-max-runtime-seconds must be positive and finite')
    root=Path(args.run_dir);root.mkdir(parents=True,exist_ok=True);control=json.loads(Path(args.control).read_text());audit=json.loads(Path(args.audit).read_text())
    storage_control=recovery_storage_control(control,args.storage_control,args.control,root)
    store=make_store(args.storage_backend,args.durable_root,storage_control)
    if args.storage_backend=='filesystem':
        store.restore_run(root)
        for diagnostic in store.recovery_diagnostics:event(root,'events',diagnostic)
    config=CONFIG
    if args.config:
        import yaml
        config=validate_config(yaml.safe_load(Path(args.config).read_text()))
    atomic_json(root/'config/resolved.json',config)
    import yaml
    (root/'config/resolved.yaml').write_text(yaml.safe_dump(config,sort_keys=True),encoding='utf-8')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 hardware required for substantive run')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    datasets=datasets_from_audit(args.data_root,args.source_root,audit);model,model_info=make_model(args.model_source);device='cuda';model=model.to(device)
    identity={'config_sha256':sha256(root/'config/resolved.json'),'audit_sha256':sha256(args.audit),'model_source_sha256':sha256(args.model_source),'control_sha256':sha256(args.control),'torch':torch.__version__}
    atomic_json(root/'provenance/model.json',model_info);atomic_json(root/'provenance/environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'tf32':False,'deterministic_algorithms':True,'grad_scaler':False})
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};initial_weight_hash=weight_hash(initial);initial_rng=rng_state();stream=SampleStream(len(datasets['train']));optimizer=optimizer_for(model)
    atomic_json(root/'provenance/initial-weights.json',{'initial_weight_sha256':initial_weight_hash,'hash_scheme':'sorted-key-dtype-shape-contiguous-tensor-bytes-v1','before_preflight':True})
    from torch.utils.tensorboard import SummaryWriter
    writer=SummaryWriter(str(root/'tensorboard'))
    cadence={};best=None;step=0;stop={'requested':False};microbatch=4;finalizing=None;telemetry={};metric_log_utc=None;observed_successful_updates=0
    status_lock=threading.RLock();heartbeat_stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.update(requested=True));signal.signal(signal.SIGINT,lambda *_:stop.update(requested=True))
    def status(phase,**extra):
        value=dict(run_id=control['run_id'],phase='FINALIZING' if finalizing else phase,operation=phase,cleanup_ready=False,failure_reason=None,finalization_deadline_utc=finalizing,heartbeat_utc=utcnow(),completed_updates=step,target_updates=100000,crop_exposure=step*16,last_remote_checkpoint=cadence.get('remote'),last_verified_checkpoint=cadence.get('remote'),best=best,best_validation=best,full_validation=cadence.get('latest_validation'),lr=learning_rate(max(1,step)),cadence=cadence,telemetry=telemetry,telemetry_utc=telemetry.get('utc'),metric_log_utc=metric_log_utc,**extra)
        with status_lock:
            atomic_json(root/'status/guest-status.json',value)
            store.pointer('status/guest-status.json',value)
        return value
    def heartbeat():
        while not heartbeat_stop.wait(60):
            try:
                with status_lock:
                    path=root/'status/guest-status.json'
                    if path.exists():
                        value=json.loads(path.read_text());value['heartbeat_utc']=utcnow();atomic_json(path,value);store.pointer('status/guest-status.json',value)
            except Exception as error:event(root,'events',{'heartbeat_upload_error':str(error)})
    heartbeat_thread=threading.Thread(target=heartbeat,daemon=True);heartbeat_thread.start()
    def save_events():
        value={'completed_updates':step,'cadence':cadence,'best':best}
        atomic_json(root/'checkpoints/events.json',value);store.pointer('checkpoints/events.json',value)
    def checkpoint():
        cadence['checkpoint_utc']=utcnow()
        path,receipt=atomic_checkpoint(pack_state(model,optimizer,stream,step,cadence,identity,best,microbatch),root/'checkpoints',step)
        status('UPLOADING');remote=store.upload_verified(path,'checkpoints/'+path.name);remote.update(step=step,verified_utc=utcnow());cadence['remote']=remote;cadence['checkpoint_utc']=utcnow()
        pointer=dict(receipt,**remote);pointer['milestone']=step in (5000,10000,25000,50000,75000,100000)
        store.pointer('checkpoints/latest.json',pointer);atomic_json(root/'checkpoints/latest.json',pointer);save_events()
        if args.storage_backend=='filesystem' and args.checkpoint_retention=='best-latest':store.prune_checkpoints(root)
        return pointer
    def validation():
        nonlocal best
        pointer=checkpoint();status('VALIDATING')
        value=evaluate(model,datasets['validation'],device,root/f'metrics/main-validation-step-{step}-lpips-v1.csv',115,lpips_metric=lpips_metric)
        value.update(step=step,utc=utcnow())
        value['metric_schema']='rgb-v1-plus-lpips-alex-v0.1-v1'
        for key,metric in value.items():
            if isinstance(metric,(int,float)) and not isinstance(metric,bool):writer.add_scalar('full_main_validation/'+key,metric,step)
        record_validation(root,value);cadence['validation_utc']=utcnow();cadence['validation_step']=step;cadence['latest_validation']=value
        score=value['mean_per_image_rgb_psnr']
        if is_improvement(step,value,best):
            best=dict(step=step,score=score,psnr_infinite=value['psnr_infinite'],checkpoint=pointer);atomic_json(root/'checkpoints/best.json',best);store.pointer('checkpoints/best.json',best)
        diagnostic=[datasets['train'][i] for i in np.random.default_rng(42).choice(len(datasets['train']),32,replace=False)]
        atomic_json(root/'config/fixed-train-diagnostic-ids.json',[v['sample_id'] for v in diagnostic])
        diagnostic_metrics=evaluate(model,diagnostic,device,root/f'metrics/fixed-train-step-{step}-lpips-v1.csv',32,lpips_metric=lpips_metric)
        event(root,'train-diagnostic',dict(diagnostic_metrics,step=step,scope='fixed_32_train_full_frame',metric_schema='rgb-v1-plus-lpips-alex-v0.1-v1'))
        save_events()
        if args.storage_backend=='filesystem':
            sync_progress(root,store)
            if args.checkpoint_retention=='best-latest':store.prune_checkpoints(root)
        from .reporting import refresh_curves
        refresh_curves(root)
        if step in (0,1000) or step%10000==0:
            from .visuals import write_panels
            write_panels(model,datasets['validation'],device,root,step)
    def elapsed(key):return (datetime.now(timezone.utc)-utc(cadence.get(key,control['t0_utc']))).total_seconds()
    failure=None;outcome='FAILED';lpips_metric=None
    try:
        from .lpips_metric import LPIPSMetric
        status('METRIC_INITIALIZATION')
        lpips_metric=LPIPSMetric(device)
        atomic_json(root/'provenance/lpips-metric-v1.json',lpips_metric.provenance)
        store.upload_verified(root/'provenance/lpips-metric-v1.json','provenance/lpips-metric-v1.json')
        pointer=root/'checkpoints/latest.json'
        if pointer.exists():
            receipt=json.loads(pointer.read_text());state=load_checkpoint(root/'checkpoints'/receipt['filename'],receipt)
            if (root/'checkpoints/events.json').exists():state=reconcile_events(state,json.loads((root/'checkpoints/events.json').read_text()))
            if args.storage_control or args.storage_backend=='filesystem':verify_recovery_best(state,root)
            restore_state(state,model,optimizer,stream,identity);step=state['completed_updates'];cadence=state['cadence'];best=state['best'];microbatch=state['microbatch']
            validate_runtime_microbatch(args.storage_backend,microbatch)
            if args.storage_backend=='filesystem' and step==0:
                preflight_result=root/'preflight/results.json'
                passed=preflight_result.is_file() and json.loads(preflight_result.read_text()).get('status')=='PASS'
                if args.preflight_only or not passed:
                    status('PREFLIGHT');microbatch=preflight(model,datasets['train'],device,root,initial,identity,store,storage_backend=args.storage_backend)
                    restore_state(state,model,optimizer,stream,identity)
                    if args.preflight_only:checkpoint();sync_progress(root,store);return
        else:
            path,receipt=atomic_checkpoint(pack_state(model,optimizer,stream,0,{},identity),root/'checkpoints',0);store.upload_verified(path,'checkpoints/'+path.name)
            status('PREFLIGHT');microbatch=preflight(model,datasets['train'],device,root,initial,identity,store,storage_backend=args.storage_backend)
            model.load_state_dict(initial);restore_rng(initial_rng);optimizer=optimizer_for(model);stream=SampleStream(len(datasets['train']))
            reset_hash=weight_hash(model.state_dict())
            if reset_hash!=initial_weight_hash:raise ValueError('Scratch initialization reset hash differs')
            atomic_json(root/'provenance/step-one-reset.json',{'initial_weight_sha256':initial_weight_hash,'step_one_weight_sha256':reset_hash,'optimizer_state_empty':not bool(optimizer.state),'sampler_cursor':stream.cursor,'rng_restored':True})
            publish_frozen_metadata(root,store)
            if args.preflight_only:
                if args.storage_backend=='filesystem':checkpoint();sync_progress(root,store)
                return
        if args.storage_backend=='filesystem' and step<100000 and (stop['requested'] or job_runtime_expired(job_started,args.job_max_runtime_seconds)):
            outcome='INTERRUPTED' if stop['requested'] else 'TIME_LIMITED'
            return
        if needs_lpips_reference_refresh(step,cadence):
            if weight_hash(model.state_dict())!=initial_weight_hash:raise ValueError('Step-zero LPIPS recovery requires exact scratch weights')
            status('REFERENCE_RECOMPUTATION')
            event(root,'events',{'event':'LPIPS_V1_REFERENCE_RECOMPUTATION','step':step,'reason':'missing diagnostic metric references; previous metric history preserved','initial_weight_sha256':initial_weight_hash})
            baseline=evaluate(model,datasets['validation'],device,root/'metrics/bicubic-main-validation-lpips-v1.csv',115,bicubic=True,lpips_metric=lpips_metric)
            cadence['bicubic']=dict(baseline,step=0,utc=utcnow(),metric_schema='rgb-v1-plus-lpips-alex-v0.1-v1')
            validation();save_initial_reference(cadence,save_events)
            atomic_json(root/'provenance/lpips-step-zero-references-v1.json',{'step':step,'initial_weight_sha256':initial_weight_hash,'bicubic':cadence['bicubic'],'scratch':cadence['initial_validation'],'recorded_utc':utcnow(),'historical_metric_files_preserved':True})
            store.upload_verified(root/'provenance/lpips-step-zero-references-v1.json','provenance/lpips-step-zero-references-v1.json')
        losses=[];interval_started=time.monotonic();last_heartbeat=0.;last_sync=0.
        while step<100000 and not stop['requested'] and not job_runtime_expired(job_started,args.job_max_runtime_seconds) and (args.storage_backend=='filesystem' or datetime.now(timezone.utc)<utc(control['soft_cutoff_utc'])):
            # Backup recovery at a completed boundary; no new updates with a stale durable checkpoint.
            if elapsed('checkpoint_utc')>1800:checkpoint()
            start=time.monotonic();position=stream.cursor;pairs,choices=next_batch(datasets['train'],stream)
            try:record=optimizer_update(model,optimizer,pairs,microbatch,device,step+1)
            except BaseException:stream.cursor=position;raise
            step+=1;losses.append(record);event(root,'samples',dict(step=step,choices=choices))
            observed_successful_updates=step
            now=time.monotonic()
            if step%50==0 or now-interval_started>=60:
                event(root,'train',dict(step=step,interval_updates=len(losses),raw_l1=float(np.mean([v['raw_l1'] for v in losses])),gradient_norm=float(np.mean([v['gradient_norm'] for v in losses])),lr=record['lr'],updates_per_second=len(losses)/(now-interval_started),crop_exposure=step*16,microbatch=microbatch,accumulation=16//microbatch));losses=[];interval_started=now
                metric_log_utc=utcnow()
                for key,metric in record.items():writer.add_scalar('train/'+key,metric,step)
            if now-last_heartbeat>=60:
                status('TRAINING',latest_train=record,update_seconds=now-start);last_heartbeat=now
                import psutil
                telemetry={'utc':utcnow(),'step':step,'gpu_allocated_bytes':torch.cuda.memory_allocated(),'gpu_reserved_bytes':torch.cuda.memory_reserved(),'gpu_peak_bytes':torch.cuda.max_memory_allocated(),'disk_free_bytes':shutil.disk_usage(root).free,'host_ram_available_bytes':psutil.virtual_memory().available,'host_ram_used_bytes':psutil.virtual_memory().used,'update_seconds':now-start}
                try:telemetry['nvidia_smi']=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=5).stdout.strip()
                except Exception as error:telemetry['telemetry_gap']=str(error)
                event(root,'telemetry',telemetry)
            if step==100000 and args.storage_backend=='filesystem' and args.defer_finalization:break
            if step<100000 and (stop['requested'] or job_runtime_expired(job_started,args.job_max_runtime_seconds)):break
            if step in (500,1000) or step%2000==0 or elapsed('validation_utc')>=3600:validation()
            elif step%500==0 or elapsed('checkpoint_utc')>=900:checkpoint()
            if now-last_sync>=300:
                if args.storage_backend=='filesystem':sync_progress(root,store)
                else:
                    for path in (root/'logs').glob('*.jsonl'):store.sync_file(path,'logs/'+path.name)
                # Status and the heartbeat already publish under status_lock.
                # Republishing here would race their generation-guarded uploads.
                last_sync=now
        if pending_target_validation(args.storage_backend,args.defer_finalization,step,cadence):validation()
        outcome='TARGET_REACHED' if step==100000 else ('INTERRUPTED' if stop['requested'] else 'TIME_LIMITED')
    except BaseException as error:
        failure=f'{type(error).__name__}: {error}';event(root,'events',{'failure':failure})
        # A failed step may have mutated optimizer/master parameters. Restore a complete
        # known checkpoint before any final save; never publish uncertain weights.
        pointer=root/'checkpoints/latest.json'
        if pointer.exists():
            observed=step;receipt=json.loads(pointer.read_text());state=load_checkpoint(root/'checkpoints'/receipt['filename'],receipt)
            if (root/'checkpoints/events.json').exists():
                saved=json.loads((root/'checkpoints/events.json').read_text())
                if saved['completed_updates']<=state['completed_updates']:state=reconcile_events(state,saved)
            restore_state(state,model,optimizer,stream,identity);step=state['completed_updates'];cadence=state['cadence'];best=state['best']
            event(root,'events',{'recovered_checkpoint_step':step,'observed_successful_updates':observed,'potentially_lost_local_updates':observed-step})
        else:
            model.load_state_dict(initial);restore_rng(initial_rng);optimizer=optimizer_for(model);stream=SampleStream(len(datasets['train']));step=0
    finally:
        heartbeat_stop.set();heartbeat_thread.join(timeout=10)
        writer.flush();writer.close()
        if args.preflight_only and failure is None:
            status('READY',preflight_only=True);store.pointer('status/guest-status.json',json.loads((root/'status/guest-status.json').read_text()));return
        if args.storage_backend=='filesystem' and step<100000:
            pause_run(root,store,checkpoint,lambda:status('PAUSED',outcome=outcome,failure=failure))
            if failure:raise RuntimeError(failure)
            return
        if args.storage_backend=='filesystem' and args.defer_finalization and step==100000 and failure is None:
            defer_target_finalization(root,store,checkpoint,lambda:status('AWAITING_FINALIZATION',outcome='TARGET_REACHED'))
            return
        entry=datetime.now(timezone.utc);grace=entry+timedelta(hours=2) if args.storage_backend=='filesystem' else min(entry+timedelta(hours=2),utc(control['cleanup_cutoff_utc']))
        finalizing=grace.isoformat()
        observed_successful_updates=max(observed_successful_updates,step)
        final_status=status('FINALIZING',outcome=outcome);final_status.update(failure_reason=failure,finalization_deadline_utc=grace.isoformat(),best=best,evaluation_gaps=[],identity=identity,initial_weight_sha256=initial_weight_hash,observed_successful_updates=observed_successful_updates,resumable_updates=step,potentially_lost_local_updates=observed_successful_updates-step);atomic_json(root/'status/guest-status.json',final_status)
        try:
            checkpoint()
            if best and datetime.now(timezone.utc)<grace:
                receipt=best['checkpoint'];state=load_checkpoint(root/'checkpoints'/receipt['filename'],receipt);model.load_state_dict(state['model'])
                final_status['final_main']=evaluate(model,datasets['validation'],device,root/'metrics/final-main-validation-lpips-v1.csv',115,deadline=grace,lpips_metric=lpips_metric)
                final_status['final_aerial']=evaluate(model,datasets['aerial_validation'],device,root/'metrics/final-aerial-validation-lpips-v1.csv',15,deadline=grace,lpips_metric=lpips_metric)
                from .visuals import write_panels
                write_panels(model,datasets['validation'],device,root,'final')
            else:final_status['evaluation_gaps'].append('Selected-checkpoint final evaluation unavailable')
            from .reporting import refresh_curves
            refresh_curves(root)
            from .finalize import export
            export(root,store,final_status);final_status['cleanup_ready']=True;final_status['export_outcome']='VERIFIED'
        except BaseException as error:final_status['export_outcome']='FINAL_EXPORT_INCOMPLETE';final_status['export_error']=str(error)
        atomic_json(root/'status/guest-status.json',final_status);store.pointer('status/guest-status.json',final_status)
    if failure:raise RuntimeError(failure)

if __name__=='__main__':main()
