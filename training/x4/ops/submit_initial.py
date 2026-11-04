"""Idempotent initial submission after complete data staging."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
from after_job import reconcile_submission
from research.checkpointing import atomic_json

# Run folder on the cluster (inputs/, results/, slurm/); defaults to the folder above ops/.
PROJECT = Path(os.environ.get('PROJECT_DIR', Path(__file__).resolve().parents[1]))


def main():
    with (PROJECT / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (PROJECT / 'STOP').exists():
            raise ValueError('STOP file is present')
        ready = json.loads((PROJECT / 'inputs/data-ready.json').read_text())
        if ready.get('status') != 'TRANSFER_COMPLETE':
            raise ValueError('Data transfer is not complete')
        receipt = PROJECT / 'initial-job.json'
        if receipt.exists():
            print(receipt.read_text())
            return
        token = 'd2-initial-20261014'
        intent = PROJECT / 'initial-job-intent.json'
        if intent.exists():
            if json.loads(intent.read_text())['scheduler_comment'] != token:
                raise ValueError('Initial submission identity differs')
            job = reconcile_submission(token)
        else:
            atomic_json(intent, {'scheduler_comment': token})
            result = subprocess.run(['sbatch', '--parsable', '--comment=' + token, str(PROJECT / 'ops/train.sbatch')], capture_output=True, text=True, check=True, cwd=PROJECT)
            job = result.stdout.strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('Unexpected scheduler response')
        value = {'job_id': job, 'state': 'SUBMITTED', 'completed_updates': 0, 'target_updates': 100000, 'gpu': 'A100 40GB'}
        atomic_json(receipt, value)
        atomic_json(PROJECT / 'active-job.json', value)
        print(json.dumps(value))


if __name__ == '__main__':
    main()
