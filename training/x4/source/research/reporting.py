"""Strict JSON event records and private readable scientific report."""
from datetime import datetime,timezone
import json
from pathlib import Path
from .experiment import NAME

def utcnow():return datetime.now(timezone.utc).isoformat()

def event(root,name,value):
    path=Path(root)/'logs'/f'{name}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    record={'utc':utcnow(),**value}
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()

def scientific_report(root,status):
    path=Path(root)/'final/report.md';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(f"# {NAME} SwinIR confirmation\n\nOutcome: {status['outcome']}. Successful updates: {status['completed_updates']}/100000; seed 42.\n\nSelected full-main-validation checkpoint: {json.dumps(status.get('best'))}.\n\nMetrics use clipped unquantized RGB, float64 per-image PSNR and valid 11x11 Gaussian SSIM. Main validation is 115 Maxar images; aerial validation is diagnostic. Tests remain sealed.\n\nOne seed and bounded training do not establish convergence, architecture superiority, cross-degradation superiority or seed variability. Final evaluation gaps: {json.dumps(status.get('evaluation_gaps',[]))}.\n",encoding='utf-8')
    cadence=status.get('cadence',{})
    with path.open('a',encoding='utf-8') as f:
        f.write(f"\nObserved successful optimizer updates: {status.get('observed_successful_updates',status['completed_updates'])}; saved resumable updates: {status.get('resumable_updates',status['completed_updates'])}; potentially lost local interval: {status.get('potentially_lost_local_updates',0)}.\n")
        f.write(f"\nCanonical scratch weight SHA256: `{status.get('initial_weight_sha256')}`. Training scope: {NAME} TRAIN17952, recipe seed20261006; training seed42.\n")
        for label,value in [('Bicubic reference',cadence.get('bicubic')),('Scratch initialization',cadence.get('initial_validation')),('Selected-checkpoint main validation',status.get('final_main')),('Final aerial diagnostic',status.get('final_aerial')),('Configuration/data/source identity',status.get('identity'))]:
            f.write(f'\n{label}: `{json.dumps(value,allow_nan=False)}`\n')
        selected=status.get('final_main',{}).get('mean_per_image_rgb_psnr');baseline=cadence.get('bicubic',{}).get('mean_per_image_rgb_psnr')
        if selected is not None and baseline is not None:f.write(f'\nSelected checkpoint minus bicubic RGB PSNR: {selected-baseline:.6f} dB. This comparison is reported regardless of its sign.\n')
    return path

def refresh_curves(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name,key,ax,label in [('train','raw_l1',axes[0],'TRAIN interval raw L1'),('validation','mean_per_image_rgb_psnr',axes[1],'Full main validation RGB PSNR')]:
        path=root/'logs'/f'{name}.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        rows=[r for r in rows if r.get(key) is not None]
        ax.plot([r['step'] for r in rows],[r[key] for r in rows]);ax.set_xlabel('successful optimizer update');ax.set_ylabel(label);ax.grid(alpha=.2)
    (root/'figures').mkdir(exist_ok=True);fig.tight_layout();fig.savefig(root/'figures/learning-curves.png',dpi=150);plt.close(fig)
