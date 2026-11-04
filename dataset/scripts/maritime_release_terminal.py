"""GCE SDK supervisor receipts: Docker exit evidence is separate from artifacts.

Called by maritime_release_supervise.sh under its per-phase flock. Publication
uses immutable GCS writes and generation-locked byte readback. No READY writes.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

from scripts.maritime_cloud_inventory import create_only, json_bytes, parse_gs, require_cloud_vm

PHASES = ('mother', 'd1', 'd2', 'd3', 'finalize')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(json_bytes(value)); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def begin(scratch, phase, selection, prefix, image_id):
    """Resume unpublished attempts; permit at most three distinct launches."""
    if phase not in PHASES or not re.fullmatch(r'maritime@sha256:[0-9a-f]{64}', image_id):
        raise ValueError('Known phase and pinned maritime runtime image identity required')
    scratch = Path(scratch); path = scratch/(phase+'-supervisor.json')
    from hashlib import sha256
    configuration = dict(selection_sha256=sha256(Path(selection).read_bytes()).hexdigest(),
                         output_prefix=prefix.rstrip('/'), container_image_id=image_id,
                         provisional=True, pilot=True)
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous:
        if previous['configuration'] != configuration:
            raise ValueError('Supervisor resume configuration mismatch')
        if previous.get('published') is not True:
            return previous
        if previous.get('exit_code') == 0:
            return previous
    attempt = (previous['attempt_number'] if previous else 0)+1
    if attempt > 3: raise ValueError('Three-attempt limit reached')
    identity = uuid.uuid4().hex
    record = dict(phase=phase, configuration=configuration, attempt_number=attempt,
                  job_attempt_id=identity, supervisor_id='sdk-'+identity,
                  container_name='maritime-'+phase+'-'+identity, state='PENDING',
                  worker_process_exited=False, logs_closed=False, published=False,
                  log=str(scratch/(phase+'-'+identity+'.log')))
    atomic_json(path, record)
    return record


def pin_read(client, ref):
    if not isinstance(ref, dict) or not {'uri','generation','bytes','crc32c','sha256'} <= ref.keys():
        raise ValueError('Complete object pin required')
    bucket, name = parse_gs(ref['uri'])
    blob = client.bucket(bucket).blob(name, generation=int(ref['generation']))
    blob.reload()
    if str(blob.generation) != str(ref['generation']) or int(blob.size) != ref['bytes'] or blob.crc32c != ref['crc32c']:
        raise ValueError('Pinned evidence metadata mismatch')
    payload = blob.download_as_bytes(if_generation_match=int(ref['generation']), checksum='crc32c')
    from hashlib import sha256
    if len(payload) != ref['bytes'] or sha256(payload).hexdigest() != ref['sha256']:
        raise ValueError('Pinned evidence byte readback mismatch')
    return payload


def put(client, prefix, suffix, payload):
    bucket, root = parse_gs(prefix.rstrip('/'))
    name = root+'/'+suffix
    return dict(uri='gs://'+bucket+'/'+name,
                **create_only(client.bucket(bucket).blob(name), payload))


@contextmanager
def evidence_lock(path):
    # Production supervisor runs on Linux GCE; tests can exercise append logic
    # without claiming a Windows cross-process locking guarantee.
    import fcntl
    with open(str(path)+'.lock', 'a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def append_receipt(path, ref):
    path = Path(path)
    with evidence_lock(path):
        evidence = json.loads(path.read_text()) if path.exists() else {'receipts': []}
        if not isinstance(evidence.get('receipts'), list): raise ValueError('Invalid receipts file')
        pin_key = (ref['uri'], str(ref['generation']))
        matches = [r for r in evidence['receipts'] if (r['uri'],str(r['generation'])) == pin_key]
        if matches and matches != [ref]: raise ValueError('Conflicting terminal receipt pin')
        if not matches: evidence['receipts'].append(ref)
        atomic_json(path, evidence)


def publish(client, scratch, phase, prefix, inspect_path, exit_code, instance_id):
    """Only called after docker wait and synchronous docker logs have closed."""
    scratch = Path(scratch); state_path = scratch/(phase+'-supervisor.json')
    state = json.loads(state_path.read_text())
    if state.get('published'):
        if phase != 'finalize':
            pin_read(client, state['terminal_receipt'])
            append_receipt(scratch/'terminal-evidence.json', state['terminal_receipt'])
        return state
    inspect = json.loads(Path(inspect_path).read_text())
    if not isinstance(inspect, list) or len(inspect) != 1: raise ValueError('One Docker inspection required')
    docker = inspect[0]; runtime = docker['State']
    image_id = state['configuration']['container_image_id']
    if (docker.get('Name') != '/'+state['container_name']
        or docker.get('Image') != image_id.split('@',1)[1]
        or runtime.get('Running') is not False or runtime.get('Status') != 'exited'
        or type(exit_code) is not int or type(runtime.get('ExitCode')) is not int
        or runtime['ExitCode'] != exit_code or not runtime.get('FinishedAt')
        or not str(instance_id).isdigit()):
        raise ValueError('Docker terminal/image/instance evidence mismatch')
    log = Path(state['log'])
    if not log.is_file(): raise ValueError('Closed Docker log missing')
    terminal = dict(producer='sdk_supervisor', supervisor_id=state['supervisor_id'], phase=phase,
                    job_attempt_id=state['job_attempt_id'], instance_id=str(instance_id),
                    container_image_id=image_id, state='EXITED', exit_code=exit_code,
                    worker_process_exited=True, logs_closed=True)
    if phase != 'finalize':
        checkpoint = json.loads((scratch/(phase+'-attempt.json')).read_text())
        for key in ('phase','job_attempt_id','instance_id','container_image_id'):
            if checkpoint.get(key) != terminal[key]: raise ValueError('Worker checkpoint identity mismatch')
        terminal['config_sha256'] = checkpoint['config_sha256']
        terminal['start_record'] = checkpoint['start_record']
        start = json.loads(pin_read(client, terminal['start_record']))
        for key in ('phase','job_attempt_id','instance_id','container_image_id','config_sha256'):
            if start.get(key) != terminal[key]: raise ValueError('Pinned worker start mismatch')
    suffix = 'control/supervisor/'+phase+'/'+state['job_attempt_id']
    terminal['closed_log'] = put(client, prefix, suffix+'.log', log.read_bytes())
    terminal['docker_inspection'] = put(client, prefix, suffix+'-docker.json', json_bytes(inspect))
    receipt = put(client, prefix, suffix+'-exited.json', json_bytes(terminal))
    if phase != 'finalize': append_receipt(scratch/'terminal-evidence.json', receipt)
    state.update(state='EXITED', exit_code=exit_code, worker_process_exited=True,
                 logs_closed=True, published=True, terminal_receipt=receipt)
    atomic_json(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['begin','publish'])
    parser.add_argument('--phase', choices=PHASES, required=True)
    parser.add_argument('--scratch', required=True)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--selection')
    parser.add_argument('--container-image-id')
    parser.add_argument('--inspect')
    parser.add_argument('--exit-code', type=int)
    args = parser.parse_args()
    instance = require_cloud_vm()
    if args.action == 'begin':
        state = begin(args.scratch,args.phase,args.selection,args.output_prefix,args.container_image_id)
        print('COMPLETE' if state.get('published') and state.get('exit_code') == 0 else state['job_attempt_id'])
    else:
        from google.cloud import storage
        state = publish(storage.Client(),args.scratch,args.phase,args.output_prefix,args.inspect,args.exit_code,instance)
        print(json.dumps(state,sort_keys=True))


if __name__ == '__main__': main()
