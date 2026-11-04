"""Read-only progress summary of the D3x10 x10 run (python3 ops/inspect_run.py on the login node)."""
import json
import os
from pathlib import Path
import subprocess

# Run folder on the cluster (inputs/, results/, slurm/); defaults to the folder above ops/.
PROJECT = Path(os.environ.get('PROJECT_DIR', Path(__file__).resolve().parents[1]))
RESULTS = PROJECT / 'results'


def tail_jsonl(path, n=1):
    if not path.exists():
        return []
    lines = path.read_text(encoding='utf-8').splitlines()
    return [json.loads(line) for line in lines[-n:] if line.strip()]


def main():
    status = json.loads((RESULTS / 'status/guest-status.json').read_text()) if (RESULTS / 'status/guest-status.json').exists() else {}
    train = tail_jsonl(RESULTS / 'logs/train.jsonl', 20)
    validation = [v for v in tail_jsonl(RESULTS / 'logs/validation.jsonl', 200) if v.get('scope') == 'full_main_validation']
    bicubic = next((v for v in tail_jsonl(RESULTS / 'logs/validation.jsonl', 10 ** 6) if v.get('scope') == 'bicubic_reference'), None)
    telemetry = tail_jsonl(RESULTS / 'logs/telemetry.jsonl', 5)
    queue = subprocess.run(['squeue', '-u', os.environ.get('USER', ''), '-o', '%i %j %T %M %l %R'], capture_output=True, text=True).stdout
    summary = {
        'phase': status.get('phase'), 'operation': status.get('operation'), 'heartbeat_utc': status.get('heartbeat_utc'),
        'completed_updates': status.get('completed_updates'), 'slurm_job': status.get('slurm_job_id'),
        'failure': status.get('failure') or status.get('failure_reason'),
        'recent_updates_per_second': round(sum(t['updates_per_second'] for t in train) / len(train), 3) if train else None,
        'recent_data_wait_fraction': round(sum(t.get('data_wait_fraction', 0) for t in train) / len(train), 3) if train else None,
        'recent_raw_l1': train[-1]['raw_l1'] if train else None,
        'gpu_util_mem_used_total_temp_power': [t.get('nvidia_smi') for t in telemetry],
        'bicubic_psnr': bicubic and bicubic.get('mean_per_image_rgb_psnr'),
        'latest_validation': validation and {k: validation[-1].get(k) for k in ('step', 'mean_per_image_rgb_psnr', 'ssim', 'mean_per_image_lpips', 'seconds')},
        'best': status.get('best') and {k: status['best'].get(k) for k in ('step', 'score')},
        'last_verified_checkpoint_step': (status.get('last_verified_checkpoint') or {}).get('step'),
        'chain': json.loads((PROJECT / 'chain-state.json').read_text()) if (PROJECT / 'chain-state.json').exists() else None,
        'queue': queue.strip().splitlines(),
    }
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
