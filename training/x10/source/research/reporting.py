"""Strict JSON event records, learning curves and the final scientific report."""
from datetime import datetime, timezone
import json
from pathlib import Path


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def event(root, name, value):
    path = Path(root) / 'logs' / f'{name}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'utc': utcnow(), **value}, allow_nan=False) + '\n')


def scientific_report(root, status):
    path = Path(root) / 'final/report.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    cadence = status.get('cadence', {})
    lines = [
        '# D3x10 SwinIR x10 (1 m / 10 m), 100k-update run', '',
        f"Outcome: {status['outcome']}. Successful updates: {status['completed_updates']}/100000; training seed 42; "
        'random initialization (no pretraining).', '',
        'Model: classical SwinIR (embed 180, 6x6 RSTB, window 8), x10 head = SwinIR x2 stage + x5 pixel-shuffle stage. '
        'Recipe identical to the D1-D3 x4 runs (Adam 2e-4, betas 0.9/0.99, 2k warm-up + cosine to 1e-6, batch 16, '
        '64x64 LR crops -> 640x640 HR, bf16 autocast, masked L1).', '',
        'Data: D3x10 TRAIN 16,835 (after rules_v5 rejection and 88 manual restores) and validation 115 (all Maxar). '
        'The test split is sealed and was not used.', '',
        'Metrics: clipped, unquantized RGB; per-image float64 PSNR averaged over images; valid 11x11 Gaussian SSIM; '
        'LPIPS AlexNet v0.1 as a diagnostic only. Checkpoint selection: highest mean validation PSNR after step 1.', '',
        f"Selected checkpoint: `{json.dumps(status.get('best'))}`", '',
        f"Observed successful updates: {status.get('observed_successful_updates', status['completed_updates'])}; "
        f"resumable updates: {status.get('resumable_updates', status['completed_updates'])}.", '',
        f"Scratch initialization weight SHA-256: `{status.get('initial_weight_sha256')}`", '']
    for label, value in [('Bicubic reference', cadence.get('bicubic')),
                         ('Scratch initialization', cadence.get('initial_validation')),
                         ('Selected checkpoint, main validation', status.get('final_main')),
                         ('Configuration/data/source identity', status.get('identity'))]:
        lines += [f'{label}: `{json.dumps(value, allow_nan=False)}`', '']
    selected = (status.get('final_main') or {}).get('mean_per_image_rgb_psnr')
    baseline = (cadence.get('bicubic') or {}).get('mean_per_image_rgb_psnr')
    if selected is not None and baseline is not None:
        lines += [f'Selected checkpoint minus bicubic RGB PSNR: {selected - baseline:.6f} dB '
                  '(reported regardless of sign).', '']
    lines += ['Limitations: one seed; synthetic LR (D3 degradation at x10); validation is Maxar only; a bounded run '
              'does not establish convergence or architecture superiority. Evaluation gaps: '
              f"{json.dumps(status.get('evaluation_gaps', []))}.", '']
    path.write_text('\n'.join(lines), encoding='utf-8')
    return path


def refresh_curves(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)

    def rows(name):
        path = root / 'logs' / f'{name}.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    train, validation = rows('train'), rows('validation')
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    panels = [(train, 'raw_l1', 'TRAIN interval raw L1'), (validation, 'mean_per_image_rgb_psnr', 'Validation RGB PSNR (dB)'),
              (validation, 'ssim', 'Validation SSIM'), (validation, 'mean_per_image_lpips', 'Validation LPIPS (lower better)')]
    for ax, (data, key, label) in zip(axes, panels):
        data = [r for r in data if r.get(key) is not None and r.get('step', 0) >= 1]
        ax.plot([r['step'] for r in data], [r[key] for r in data])
        ax.set_xlabel('successful optimizer update')
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    (root / 'figures').mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(root / 'figures/learning-curves.png', dpi=150)
    plt.close(fig)
