"""SwinIR x10 on D3x10 with verified filesystem persistence and Slurm-chained pauses.

Adapted from the D2 Aristotle trainer. Scientific recipe unchanged (Adam 2e-4,
2k warm-up + cosine to 1e-6, batch 16, 64x64 LR crops, bf16, masked L1, 100k
updates); engineering changes: one 16-sample pass instead of 4x4 accumulation
(same batch-mean gradient), parallel position-keyed data workers, single-sync
finiteness checks, metric scoring in 8 CPU processes during validation,
checkpoints every 1,000 updates or 15 minutes, best+latest retention. --benchmark measures throughput options without touching the run.
"""
import argparse
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import threading
import time

import numpy as np
import torch

from .checkpointing import atomic_json, atomic_checkpoint, load_checkpoint, sha256
from .data import AUGMENTATION_SCHEME, FullPairs, load_index, train_loader
from .evaluate import evaluate, rng_state, restore_rng
from .hpc_store import FilesystemStore
from .model_factory import CONFIG, validate_config, make_model
from .reporting import event, utcnow, refresh_curves

TARGET, BATCH, LR_SHAPE, HR_SHAPE = 100000, 16, (3, 64, 64), (3, 640, 640)
CHECKPOINT_EVERY, CHECKPOINT_MINUTES = 1000, 15
SCORE_WORKERS = 8  # CPU processes scoring validation images while the GPU infers the next


def utc(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


def learning_rate(update):
    if not 1 <= update <= TARGET:
        raise ValueError('Update outside original schedule')
    return .0002 * update / 2000 if update <= 2000 else .000001 + .5 * (.0002 - .000001) * (1 + math.cos(math.pi * (update - 2000) / 98000))


def masked_l1(prediction, target, mask):
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 3 or mask.dtype != torch.bool:
        raise ValueError('Matching RGB and boolean support required')
    if mask.shape != (target.shape[0], *target.shape[-2:]) or not mask.flatten(1).any(1).all():
        raise ValueError('Empty or malformed support')
    error = (prediction.float() - target.float()).abs()
    weights = mask[:, None].expand_as(error)
    return torch.where(weights, error, 0).sum() / weights.sum()


def configure_backends(deterministic, tf32):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def optimizer_update(model, optimizer, batch, device, update):
    x = batch['lr'].to(device, non_blocking=True)
    y = batch['hr'].to(device, non_blocking=True)
    mask = batch['hr_mask'].to(device, non_blocking=True)
    if x.shape != (BATCH, *LR_SHAPE) or y.shape != (BATCH, *HR_SHAPE):
        raise ValueError(f'Unexpected batch shapes {tuple(x.shape)} {tuple(y.shape)}')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lr = learning_rate(update)
    for group in optimizer.param_groups:
        group['lr'] = lr
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        prediction = model(x)
    loss = masked_l1(prediction, y, mask)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients:
        raise ValueError('Missing gradients')
    norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients)))
    loss_value, norm_value = float(loss.detach()), float(norm)  # one sync for both
    if not (math.isfinite(loss_value) and math.isfinite(norm_value)):
        raise ValueError('Nonfinite loss or gradient')
    optimizer.step()
    return {'raw_l1': loss_value, 'gradient_norm': norm_value, 'lr': lr, 'microbatches': 1}


def weights_finite(model):
    return bool(torch.isfinite(torch.stack(torch._foreach_norm(list(model.parameters())))).all())


def weight_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def optimizer_for(model):
    return torch.optim.Adam(model.parameters(), lr=.0002, betas=(.9, .99), eps=1e-8, weight_decay=0)


def pack_state(model, optimizer, cursor, size, step, cadence, identity, best):
    return {'schema': 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': {'completed_updates': step, 'original_target': TARGET}, 'scaler': None,
            'completed_updates': step, 'target_updates': TARGET,
            'sampler': {'cursor': cursor, 'size': size, 'seed': 42, 'scheme': 'numpy-PCG64-SeedSequence-permutation-v1'},
            'rng': rng_state(), 'cadence': copy.deepcopy(cadence), 'identity': identity, 'best': best,
            'microbatch': BATCH, 'accumulation_steps': 1, 'effective_batch': BATCH, 'crop_exposure': step * BATCH,
            'augmentation_scheme': AUGMENTATION_SCHEME}


def restore_state(state, model, optimizer, size, identity):
    sampler = state['sampler']
    if (state['schema'] != 1 or state['identity'] != identity or state['target_updates'] != TARGET
            or sampler['cursor'] != state['completed_updates'] * BATCH or sampler['size'] != size):
        raise ValueError('Resume identity/accounting mismatch')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    return sampler['cursor']


def reconcile_events(state, saved):
    if saved['completed_updates'] > state['completed_updates']:
        raise ValueError('Event state is ahead of resumable checkpoint')
    result = dict(state)
    if saved['completed_updates'] == state['completed_updates']:
        result.update(cadence=saved['cadence'], best=saved['best'])
    return result


def is_improvement(step, value, best):
    if step < 1:
        return False
    if best is None:
        return True
    if best['psnr_infinite']:
        return False
    return value['psnr_infinite'] or value['mean_per_image_rgb_psnr'] > best['score']


def sync_progress(root, store):
    for directory in ('logs', 'metrics', 'provenance', 'config', 'preflight', 'tensorboard', 'figures'):
        for path in sorted((Path(root) / directory).rglob('*')):
            if path.is_file() and path.suffix != '.tmp':
                store.sync_file(path, path.relative_to(root).as_posix())


def gpu_sample():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except Exception as error:
        return 'unavailable: ' + str(error)


def fixed_batch(train_rows, data_root):
    from .data import TrainPositions
    ds = TrainPositions(data_root, train_rows)
    items = [ds[p] for p in range(BATCH)]
    return {'lr': torch.stack([i['lr'] for i in items]), 'hr': torch.stack([i['hr'] for i in items]),
            'hr_mask': torch.stack([i['hr_mask'] for i in items]), 'position': torch.arange(BATCH)}


def preflight(model, optimizer_factory, train_rows, validation, data_root, device, root, initial, identity, store, workers):
    """Actual model shapes, bounded TRAIN overfit and a durable save/reload next-update test."""
    root = Path(root)
    initial_rng = rng_state()
    result = {}
    try:
        batch = fixed_batch(train_rows, data_root)
        with torch.inference_mode():
            model.eval()
            if model(batch['lr'][:1].to(device)).shape != (1, *HR_SHAPE):
                raise ValueError('Crop inference shape is not 640x640')
            if model(validation[0]['lr'][None].to(device)).shape != (1, 3, 1280, 1280):
                raise ValueError('Full-frame inference shape is not 1280x1280')
        optimizer = optimizer_factory(model)

        def deterministic_loss():
            model.eval()
            with torch.inference_mode():
                return float(masked_l1(model(batch['lr'][:1].to(device)), batch['hr'][:1].to(device), batch['hr_mask'][:1].to(device)))
        before = deterministic_loss()
        for _ in range(8):
            optimizer_update(model, optimizer, batch, device, 2000)
        after = deterministic_loss()
        if not after < before:
            raise ValueError('Bounded TRAIN overfit failed to lower deterministic L1')
        torch.cuda.reset_peak_memory_stats()
        model.load_state_dict(initial)
        restore_rng(initial_rng)
        optimizer = optimizer_factory(model)
        loader = iter(train_loader(data_root, train_rows, 0, workers))
        first = next(loader)
        optimizer_update(model, optimizer, first, device, 1)
        path, receipt = atomic_checkpoint(pack_state(model, optimizer, BATCH, len(train_rows), 1, {}, identity, None), root / 'preflight', 1)
        remote = store.upload_verified(path, 'preflight/' + path.name)
        download = root / 'preflight/remote-resume.pt'
        store.download_verified('preflight/' + path.name, download, remote)
        second = next(loader)
        optimizer_update(model, optimizer, second, device, 2)
        expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        del loader
        state = load_checkpoint(download, receipt)
        cursor = restore_state(state, model, optimizer, len(train_rows), identity)
        replay = next(iter(train_loader(data_root, train_rows, cursor, workers)))
        if replay['choice'] != second['choice'] or not torch.equal(replay['lr'], second['lr']) or not torch.equal(replay['hr'], second['hr']):
            raise ValueError('Resume next sample/crop/augmentation differs')
        optimizer_update(model, optimizer, replay, device, 2)
        max_error = max(float((v.detach().cpu() - expected[k]).abs().max()) for k, v in model.state_dict().items())
        if max_error > 1e-6:
            raise ValueError(f'Resume next update differs by {max_error} > 1e-6')
        result = {'status': 'PASS', 'real_model': True, 'overfit_before': before, 'overfit_after': after,
                  'resume_max_absolute_error': max_error, 'resume_tolerance': 1e-6,
                  'resume_next_choices': [json.loads(c) for c in second['choice']], 'durable_checkpoint': remote,
                  'microbatch': BATCH, 'accumulation': 1, 'grad_scaler': False,
                  'peak_gpu_bytes_batch16': torch.cuda.max_memory_allocated()}
    finally:
        model.load_state_dict(initial)
        restore_rng(initial_rng)
    atomic_json(root / 'preflight/results.json', result)
    store.upload_verified(root / 'preflight/results.json', 'preflight/results.json')
    return result


def benchmark(args, model, train_rows, validation, device, root):
    """Throughput options measured once before the run; nothing is saved to the run."""
    out, initial = {}, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    reads = []
    for workers in sorted({0, args.workers // 2, args.workers}):
        loader = iter(train_loader(args.data_root, train_rows, 10 ** 6, workers))
        next(loader)
        start = time.monotonic()
        for _ in range(20):
            next(loader)
        reads.append({'workers': workers, 'samples_per_second': 20 * BATCH / (time.monotonic() - start)})
        del loader
    out['loader'] = reads
    for deterministic in (True, False):
        configure_backends(deterministic, False)
        model.load_state_dict(initial)
        optimizer = optimizer_for(model)
        loader = iter(train_loader(args.data_root, train_rows, 0, args.workers))
        for update in range(1, 11):
            optimizer_update(model, optimizer, next(loader), device, update)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        waits, utils, start = 0., [], time.monotonic()
        for update in range(11, 71):
            t = time.monotonic()
            batch = next(loader)
            waits += time.monotonic() - t
            optimizer_update(model, optimizer, batch, device, update)
            if update % 15 == 0:
                utils.append(gpu_sample())
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        out['deterministic' if deterministic else 'nondeterministic'] = {
            'updates_per_second': 60 / elapsed, 'data_wait_fraction': waits / elapsed,
            'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'nvidia_smi_samples': utils}
        del loader
    model.load_state_dict(initial)
    from .lpips_metric import LPIPSMetric
    sub = FullPairs(args.data_root, validation.rows[:10])
    start = time.monotonic()
    evaluate(model, sub, device, Path(root) / 'benchmark-validation.csv', 10, lpips_metric=LPIPSMetric(device))
    out['validation_seconds_per_image'] = (time.monotonic() - start) / 10
    out['environment'] = {'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()}
    atomic_json(Path(args.benchmark), out)
    print(json.dumps(out, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data-root', 'index', 'run-dir', 'durable-root', 'control', 'model-source', 'audit'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--job-max-runtime-seconds', type=float)
    parser.add_argument('--defer-finalization', action='store_true')
    parser.add_argument('--benchmark', help='write a throughput benchmark JSON here and exit')
    args = parser.parse_args(argv)
    job_started = time.monotonic()
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    control = json.loads(Path(args.control).read_text())
    audit = json.loads(Path(args.audit).read_text())
    if audit.get('status') != 'PASS':
        raise ValueError('Data audit must PASS')
    import yaml
    config = validate_config(yaml.safe_load(Path(args.config).read_text()))
    training = config['training']
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 GPU required')
    configure_backends(training['deterministic_algorithms'], training['tf32'])
    random.seed(42), np.random.seed(42), torch.manual_seed(42), torch.cuda.manual_seed_all(42)
    index = load_index(args.index)
    if sha256(args.index) != audit['index_sha256']:
        raise ValueError('Shard index differs from the audited one')
    train_rows = [r for r in index['rows'] if r['split'] == 'train']
    validation = FullPairs(args.data_root, [r for r in index['rows'] if r['split'] == 'validation'])
    if len(train_rows) != config['data']['train'] or len(validation) != config['data']['validation']:
        raise ValueError('Data counts differ from configuration')
    model, model_info = make_model(args.model_source)
    device = 'cuda'
    model = model.to(device)
    if args.benchmark:
        return benchmark(args, model, train_rows, validation, device, root)

    store = FilesystemStore(args.durable_root)
    store.restore_run(root)
    for diagnostic in store.recovery_diagnostics:
        event(root, 'events', diagnostic)
    atomic_json(root / 'config/resolved.json', config)
    (root / 'config/resolved.yaml').write_text(yaml.safe_dump(config, sort_keys=True), encoding='utf-8')
    identity = {'config_sha256': sha256(root / 'config/resolved.json'), 'audit_sha256': sha256(args.audit),
                'index_sha256': sha256(args.index), 'model_source_sha256': sha256(args.model_source),
                'control_sha256': sha256(args.control), 'torch': torch.__version__}
    atomic_json(root / 'provenance/model.json', model_info)
    atomic_json(root / 'provenance/environment.json', {
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(),
        'tf32': training['tf32'], 'deterministic_algorithms': training['deterministic_algorithms'],
        'grad_scaler': False, 'workers': args.workers, 'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    initial_weight_hash = weight_hash(initial)
    initial_rng = rng_state()
    optimizer = optimizer_for(model)
    atomic_json(root / 'provenance/initial-weights.json', {'initial_weight_sha256': initial_weight_hash,
                'hash_scheme': 'sorted-key-dtype-shape-contiguous-tensor-bytes-v1', 'before_preflight': True})
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(root / 'tensorboard'))
    cadence, best, step, cursor = {}, None, 0, 0
    stop = {'requested': False}
    telemetry, metric_log_utc, finalizing = {}, None, None
    status_lock, heartbeat_stop = threading.RLock(), threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(requested=True))
    expired = lambda: args.job_max_runtime_seconds is not None and time.monotonic() - job_started >= args.job_max_runtime_seconds

    def status(phase, **extra):
        value = dict(run_id=control['run_id'], phase='FINALIZING' if finalizing else phase, operation=phase,
                     failure_reason=None, heartbeat_utc=utcnow(), completed_updates=step, target_updates=TARGET,
                     crop_exposure=step * BATCH, last_verified_checkpoint=cadence.get('remote'), best=best,
                     full_validation=cadence.get('latest_validation'), lr=learning_rate(max(1, step)),
                     cadence=cadence, telemetry=telemetry, metric_log_utc=metric_log_utc,
                     slurm_job_id=os.environ.get('SLURM_JOB_ID'), **extra)
        with status_lock:
            atomic_json(root / 'status/guest-status.json', value)
            store.pointer('status/guest-status.json', value)
        return value

    def heartbeat():
        while not heartbeat_stop.wait(60):
            try:
                with status_lock:
                    path = root / 'status/guest-status.json'
                    if path.exists():
                        value = json.loads(path.read_text())
                        value['heartbeat_utc'] = utcnow()
                        atomic_json(path, value)
                        store.pointer('status/guest-status.json', value)
            except Exception as error:
                event(root, 'events', {'heartbeat_error': str(error)})
    threading.Thread(target=heartbeat, daemon=True).start()

    def save_events():
        value = {'completed_updates': step, 'cadence': cadence, 'best': best}
        atomic_json(root / 'checkpoints/events.json', value)
        store.pointer('checkpoints/events.json', value)

    def checkpoint():
        cadence['checkpoint_utc'] = utcnow()
        path, receipt = atomic_checkpoint(pack_state(model, optimizer, cursor, len(train_rows), step, cadence, identity, best), root / 'checkpoints', step)
        remote = store.upload_verified(path, 'checkpoints/' + path.name)
        remote.update(step=step, verified_utc=utcnow())
        cadence['remote'] = remote
        pointer = dict(receipt, **remote)
        store.pointer('checkpoints/latest.json', pointer)
        atomic_json(root / 'checkpoints/latest.json', pointer)
        save_events()
        store.prune_checkpoints(root)
        return pointer

    diagnostic_rows = [train_rows[i] for i in np.random.default_rng(42).choice(len(train_rows), 32, replace=False)]
    diagnostic = FullPairs(args.data_root, diagnostic_rows)
    atomic_json(root / 'config/fixed-train-diagnostic-ids.json', [r['sample_id'] for r in diagnostic_rows])

    def validation_event():
        nonlocal best
        pointer = checkpoint()
        status('VALIDATING')
        started = time.monotonic()
        value = evaluate(model, validation, device, root / f'metrics/main-validation-step-{step}.csv', len(validation), lpips_metric=lpips_metric, score_workers=SCORE_WORKERS)
        value.update(step=step, utc=utcnow(), seconds=time.monotonic() - started, metric_schema='rgb-v1-plus-lpips-alex-v0.1-v1')
        for key, metric in value.items():
            if isinstance(metric, (int, float)) and not isinstance(metric, bool):
                writer.add_scalar('full_main_validation/' + key, metric, step)
        event(root, 'validation', dict(value, scope='full_main_validation'))
        cadence.update(validation_utc=utcnow(), validation_step=step, latest_validation=value)
        if is_improvement(step, value, best):
            best = dict(step=step, score=value['mean_per_image_rgb_psnr'], psnr_infinite=value['psnr_infinite'], checkpoint=pointer)
            atomic_json(root / 'checkpoints/best.json', best)
            store.pointer('checkpoints/best.json', best)
        diag = evaluate(model, diagnostic, device, root / f'metrics/fixed-train-step-{step}.csv', 32, lpips_metric=lpips_metric, score_workers=SCORE_WORKERS)
        event(root, 'train-diagnostic', dict(diag, step=step, scope='fixed_32_train_full_frame', metric_schema='rgb-v1-plus-lpips-alex-v0.1-v1'))
        save_events()
        store.prune_checkpoints(root)
        refresh_curves(root)
        if step in (0, 1000) or step % 10000 == 0:
            from .visuals import write_panels
            write_panels(model, validation, device, root, step)
        sync_progress(root, store)

    def elapsed(key):
        return (datetime.now(timezone.utc) - utc(cadence.get(key, control['t0_utc']))).total_seconds()

    failure, outcome, lpips_metric, observed = None, 'FAILED', None, 0
    try:
        from .lpips_metric import LPIPSMetric
        status('METRIC_INITIALIZATION')
        lpips_metric = LPIPSMetric(device)
        atomic_json(root / 'provenance/lpips-metric-v1.json', lpips_metric.provenance)
        pointer = root / 'checkpoints/latest.json'
        if pointer.exists():
            receipt = json.loads(pointer.read_text())
            state = load_checkpoint(root / 'checkpoints' / receipt['filename'], receipt)
            if (root / 'checkpoints/events.json').exists():
                state = reconcile_events(state, json.loads((root / 'checkpoints/events.json').read_text()))
            cursor = restore_state(state, model, optimizer, len(train_rows), identity)
            step, cadence, best = state['completed_updates'], state['cadence'], state['best']
            event(root, 'events', {'event': 'RESUMED', 'step': step, 'cursor': cursor, 'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
        else:
            path, _ = atomic_checkpoint(pack_state(model, optimizer, 0, len(train_rows), 0, {}, identity, None), root / 'preflight', 0)
            status('PREFLIGHT')
            event(root, 'events', {'event': 'PREFLIGHT_START'})
            preflight(model, optimizer_for, train_rows, validation, args.data_root, device, root, initial, identity, store, args.workers)
            model.load_state_dict(initial)
            restore_rng(initial_rng)
            optimizer = optimizer_for(model)
            reset_hash = weight_hash(model.state_dict())
            if reset_hash != initial_weight_hash:
                raise ValueError('Scratch initialization reset hash differs')
            atomic_json(root / 'provenance/step-one-reset.json', {'initial_weight_sha256': initial_weight_hash,
                        'step_one_weight_sha256': reset_hash, 'optimizer_state_empty': not bool(optimizer.state),
                        'sampler_cursor': 0, 'rng_restored': True})
            status('REFERENCE_VALIDATION')
            baseline = evaluate(model, validation, device, root / 'metrics/bicubic-main-validation.csv', len(validation), bicubic=True, lpips_metric=lpips_metric, score_workers=SCORE_WORKERS)
            cadence['bicubic'] = dict(baseline, step=0, utc=utcnow(), metric_schema='rgb-v1-plus-lpips-alex-v0.1-v1')
            event(root, 'validation', dict(cadence['bicubic'], scope='bicubic_reference'))
            validation_event()
            cadence['initial_validation'] = copy.deepcopy(cadence['latest_validation'])
            save_events()
        if step < TARGET and (stop['requested'] or expired()):
            outcome = 'INTERRUPTED' if stop['requested'] else 'TIME_LIMITED'
            return
        loader = iter(train_loader(args.data_root, train_rows, cursor, args.workers))
        losses, waits, interval_started, last_beat, last_sync = [], 0., time.monotonic(), 0., time.monotonic()
        while step < TARGET and not stop['requested'] and not expired():
            if elapsed('checkpoint_utc') > 1800:
                checkpoint()
            wait_start = time.monotonic()
            batch = next(loader)
            waits += time.monotonic() - wait_start
            if batch['position'].tolist() != list(range(cursor, cursor + BATCH)):
                raise ValueError('Loader positions differ from the authoritative cursor')
            start = time.monotonic()
            record = optimizer_update(model, optimizer, batch, device, step + 1)
            step, cursor, observed = step + 1, cursor + BATCH, step + 1
            losses.append(record)
            event(root, 'samples', {'step': step, 'choices': [json.loads(c) for c in batch['choice']]})
            now = time.monotonic()
            if step % 50 == 0 or now - interval_started >= 60:
                if not weights_finite(model):
                    raise ValueError('Nonfinite weights')
                interval = now - interval_started
                event(root, 'train', dict(step=step, interval_updates=len(losses),
                      raw_l1=float(np.mean([v['raw_l1'] for v in losses])),
                      gradient_norm=float(np.mean([v['gradient_norm'] for v in losses])), lr=record['lr'],
                      updates_per_second=len(losses) / interval, data_wait_fraction=waits / interval,
                      crop_exposure=step * BATCH, microbatch=BATCH, accumulation=1))
                for key in ('raw_l1', 'gradient_norm', 'lr'):
                    writer.add_scalar('train/' + key, record[key], step)
                writer.add_scalar('train/updates_per_second', len(losses) / interval, step)
                losses, waits, interval_started, metric_log_utc = [], 0., now, utcnow()
            if now - last_beat >= 60:
                import psutil
                telemetry = {'utc': utcnow(), 'step': step, 'gpu_allocated_bytes': torch.cuda.memory_allocated(),
                             'gpu_peak_bytes': torch.cuda.max_memory_allocated(), 'nvidia_smi': gpu_sample(),
                             'disk_free_bytes': shutil.disk_usage(root).free,
                             'host_ram_available_bytes': psutil.virtual_memory().available, 'update_seconds': now - start}
                event(root, 'telemetry', telemetry)
                status('TRAINING', latest_train=record)
                last_beat = now
            if step == TARGET and args.defer_finalization:
                break
            if step < TARGET and (stop['requested'] or expired()):
                break
            if step in (500, 1000) or step % 2000 == 0 or elapsed('validation_utc') >= 3600:
                validation_event()
            elif step % CHECKPOINT_EVERY == 0 or elapsed('checkpoint_utc') >= CHECKPOINT_MINUTES * 60:
                checkpoint()
            if now - last_sync >= 300:
                sync_progress(root, store)
                last_sync = now
        if not args.defer_finalization and step == TARGET and cadence.get('validation_step') != TARGET:
            validation_event()
        outcome = 'TARGET_REACHED' if step == TARGET else ('INTERRUPTED' if stop['requested'] else 'TIME_LIMITED')
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        event(root, 'events', {'failure': failure})
        pointer = root / 'checkpoints/latest.json'
        if pointer.exists():
            receipt = json.loads(pointer.read_text())
            state = load_checkpoint(root / 'checkpoints' / receipt['filename'], receipt)
            if (root / 'checkpoints/events.json').exists():
                saved = json.loads((root / 'checkpoints/events.json').read_text())
                if saved['completed_updates'] <= state['completed_updates']:
                    state = reconcile_events(state, saved)
            cursor = restore_state(state, model, optimizer, len(train_rows), identity)
            step, cadence, best = state['completed_updates'], state['cadence'], state['best']
            event(root, 'events', {'recovered_checkpoint_step': step, 'observed_successful_updates': observed,
                                   'potentially_lost_local_updates': max(0, observed - step)})
    finally:
        heartbeat_stop.set()
        writer.flush()
        writer.close()
        if (root / 'checkpoints/latest.json').exists() or step > 0:
            if step < TARGET or (args.defer_finalization and failure is None):
                checkpoint()
                sync_progress(root, store)
                deferred = step == TARGET
                value = status('AWAITING_FINALIZATION' if deferred else 'PAUSED',
                               outcome='TARGET_REACHED' if deferred else outcome, failure=failure, training_complete=deferred,
                               export_outcome='DEFERRED_UNTIL_FINALIZATION_JOB' if deferred else 'DEFERRED_UNTIL_TARGET')
                if failure:
                    raise RuntimeError(failure)
                return value
        if failure:
            raise RuntimeError(failure)
        finalizing = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        final = status('FINALIZING', outcome=outcome)
        final.update(failure_reason=failure, finalization_deadline_utc=finalizing, best=best,
                     evaluation_gaps=['No aerial validation set exists for D3x10; test split sealed by design'],
                     identity=identity, initial_weight_sha256=initial_weight_hash,
                     observed_successful_updates=max(observed, step), resumable_updates=step)
        try:
            checkpoint()
            receipt = best['checkpoint']
            model.load_state_dict(load_checkpoint(root / 'checkpoints' / receipt['filename'], receipt)['model'])
            final['final_main'] = evaluate(model, validation, device, root / 'metrics/final-main-validation.csv', len(validation),
                                           deadline=utc(finalizing), lpips_metric=lpips_metric, score_workers=SCORE_WORKERS)
            from .visuals import write_panels
            write_panels(model, validation, device, root, 'final')
            refresh_curves(root)
            from .finalize import export
            export(root, store, final)
            final['export_outcome'] = 'VERIFIED'
        except BaseException as error:
            final['export_outcome'] = 'FINAL_EXPORT_INCOMPLETE'
            final['export_error'] = f'{type(error).__name__}: {error}'
        atomic_json(root / 'status/guest-status.json', final)
        store.pointer('status/guest-status.json', final)


if __name__ == '__main__':
    main()
