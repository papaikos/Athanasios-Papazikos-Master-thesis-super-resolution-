"""Continue only a verified, successful pause; complete only a verified export (adapted from D2)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

# Run folder on the cluster (inputs/, results/, slurm/); defaults to the folder above ops/.
PROJECT = Path(os.environ.get('PROJECT_DIR', Path(__file__).resolve().parents[1]))
RESULTS = PROJECT / 'results'


def read_status():
    path = RESULTS / 'status/guest-status.json'
    return json.loads(path.read_text()) if path.exists() else {'completed_updates': 0}


def decision(status, start_step):
    step = status['completed_updates']
    if not 0 <= start_step <= step <= 100000:
        raise ValueError('Invalid update accounting')
    if status.get('failure') or status.get('failure_reason'):
        raise ValueError('Trainer failure requires inspection')
    if step == 100000:
        if status.get('outcome') != 'TARGET_REACHED':
            raise ValueError('Target boundary lacks successful outcome')
        if status.get('phase') == 'AWAITING_FINALIZATION' and status.get('export_outcome') == 'DEFERRED_UNTIL_FINALIZATION_JOB':
            return 'finalize'
        return 'complete' if status.get('export_outcome') == 'VERIFIED' else 'retry_export'
    if status.get('phase') != 'PAUSED' or status.get('outcome') not in ('TIME_LIMITED', 'INTERRUPTED'):
        raise ValueError('No verified pause to continue')
    if step <= start_step:
        raise ValueError('No training progress; refusing an endless job chain')
    return 'continue'


def verify_complete():
    from research.hpc_store import FilesystemStore
    store = FilesystemStore(RESULTS)
    # A verified marker identifies the exact immutable manifest for this attempt.
    marker = store._read_json_verified('final/FINALIZED.json')
    if marker.get('finalized') is not True:
        raise ValueError('Finalized marker required')
    relative = Path(marker['manifest_object'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe final manifest path')
    path = RESULTS / relative
    if path.stat().st_size != marker['manifest_bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != marker['manifest_sha256']:
        raise ValueError('Final manifest identity mismatch')
    manifest = json.loads(path.read_text())
    if marker['attempt_id'] != manifest['attempt_id']:
        raise ValueError('Final attempt identity mismatch')
    for artifact in manifest['required_artifacts']:
        relative = Path(artifact['object'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe final artifact path')
        path = RESULTS / relative
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                checksum.update(chunk)
        if path.stat().st_size != artifact['size'] or checksum.hexdigest() != artifact['sha256']:
            raise ValueError('Final artifact mismatch: ' + artifact['object'])


def reconcile_submission(token):
    queries = [
        ['squeue', '--user=' + os.environ.get('USER', ''), '--noheader', '--format=%i|%k'],
        ['sacct', '--allocations', '--user=' + os.environ.get('USER', ''), '--starttime=2026-10-21',
         '--noheader', '--parsable2', '--format=JobIDRaw,Comment'],
    ]
    for command in queries:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        matches = set()
        for line in result.stdout.splitlines():
            fields = line.strip().split('|')
            if len(fields) >= 2 and fields[1].strip() == token and fields[0].strip().isdigit():
                matches.add(fields[0].strip())
        if len(matches) > 1:
            raise ValueError('Multiple successors recorded; inspection required')
        if matches:
            return matches.pop()
    raise ValueError('Previous submission outcome unknown; refusing duplicate submission')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read-step', action='store_true')
    parser.add_argument('--start-step', type=int)
    parser.add_argument('--parent-job')
    args = parser.parse_args()
    status = read_status()
    if args.read_step:
        print(status['completed_updates'])
        return
    action = decision(status, args.start_step)
    if action == 'complete':
        verify_complete()
        print('TARGET_REACHED_100000_AND_EXPORT_VERIFIED')
        return
    if (PROJECT / 'STOP').exists():
        print('STOP file present; verified state retained; no next job')
        return
    state_path = PROJECT / 'chain-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'jobs': 0, 'export_retries': 0}
    if state['jobs'] >= 60:
        raise ValueError('Job-chain safety bound reached')
    if action == 'retry_export':
        if state['export_retries'] >= 3:
            raise ValueError('Final export retry bound reached')
        state['export_retries'] += 1
    parent = args.parent_job
    if not parent or not parent.isdigit():
        raise ValueError('Validated parent Slurm ID required')
    receipt = PROJECT / 'slurm' / ('next-from-' + parent + '.json')
    if receipt.exists():
        print(receipt.read_text())
        return
    from research.checkpointing import atomic_json
    token = 'd3x10-successor-of-' + parent
    intent_path = PROJECT / 'slurm' / ('intent-from-' + parent + '.json')
    if intent_path.exists():
        intent = json.loads(intent_path.read_text())
        if intent['scheduler_comment'] != token:
            raise ValueError('Submission identity mismatch')
        job = reconcile_submission(token)
    else:
        # Commit intent before the external call. An interrupted/unknown outcome
        # must be reconciled or stopped, never silently submitted a second time.
        atomic_json(intent_path, {'parent_job': parent, 'scheduler_comment': token})
        result = subprocess.run(['sbatch', '--parsable', '--comment=' + token, '--dependency=afterany:' + parent, str(PROJECT / 'ops/train.sbatch')], capture_output=True, text=True, check=True)
        job = result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise ValueError('Unexpected Slurm submission response')
    atomic_json(receipt, {'parent_job': parent, 'next_job': job, 'completed_updates': status['completed_updates'], 'action': action})
    state.update(jobs=state['jobs'] + 1, latest_job=job, last_completed_updates=status['completed_updates'])
    atomic_json(state_path, state)
    atomic_json(PROJECT / 'active-job.json', {'job_id': job, 'state': 'SUBMITTED', 'completed_updates': status['completed_updates']})
    print(json.dumps({'next_job': job, 'completed_updates': status['completed_updates'], 'action': action}))


if __name__ == '__main__':
    main()
