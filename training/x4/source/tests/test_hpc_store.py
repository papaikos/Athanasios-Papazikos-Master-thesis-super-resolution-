"""Real durable filesystem behavior; no GPU, cloud or optimizer execution."""
import json
import pytest


def store_at(path):
    from research.hpc_store import FilesystemStore
    return FilesystemStore(path)


def test_verified_publish_and_empty_scratch_resume(tmp_path):
    store = store_at(tmp_path / 'durable')
    source = tmp_path / 'checkpoint.pt'
    source.write_bytes(b'checkpoint bytes')
    receipt = store.upload_verified(source, 'checkpoints/step-1.pt')
    assert receipt['bytes'] == 16
    store.pointer('checkpoints/latest.json', {'filename': 'step-1.pt', **receipt})
    store.pointer('checkpoints/best.json', {'checkpoint': {'filename': 'step-1.pt', **receipt}})
    store.pointer('checkpoints/events.json', {'completed_updates': 1})
    scratch = tmp_path / 'new-scratch'
    store.restore_run(scratch)
    assert (scratch / 'checkpoints/step-1.pt').read_bytes() == b'checkpoint bytes'
    assert json.loads((scratch / 'checkpoints/latest.json').read_text())['filename'] == 'step-1.pt'
    assert json.loads((scratch / 'checkpoints/best.json').read_text())['checkpoint']['filename'] == 'step-1.pt'
    assert json.loads((scratch / 'checkpoints/events.json').read_text()) == {'completed_updates': 1}


def test_immutable_publish_rejects_changed_bytes_without_overwrite(tmp_path):
    store = store_at(tmp_path / 'durable')
    source = tmp_path / 'source'; source.write_bytes(b'first')
    store.upload_verified(source, 'checkpoints/a.pt')
    source.write_bytes(b'second')
    with pytest.raises(ValueError, match='immutable'):
        store.upload_verified(source, 'checkpoints/a.pt')
    assert (store.root / 'checkpoints/a.pt').read_bytes() == b'first'


def test_download_rejects_corrupt_object_and_preserves_destination(tmp_path):
    store = store_at(tmp_path / 'durable')
    source = tmp_path / 'source'; source.write_bytes(b'correct')
    receipt = store.upload_verified(source, 'checkpoints/a.pt')
    (store.root / 'checkpoints/a.pt').write_bytes(b'corrupt')
    destination = tmp_path / 'destination'; destination.write_bytes(b'previous')
    with pytest.raises(ValueError, match='integrity'):
        store.download_verified('checkpoints/a.pt', destination, receipt)
    assert destination.read_bytes() == b'previous'
    with pytest.raises(ValueError, match='integrity'):
        store.restore_run(tmp_path / 'scratch')


@pytest.mark.parametrize('relative', ['../escape', '/absolute', 'C:/escape', r'..\escape', '.receipts/forged.json', 'a/../../escape'])
def test_rejects_escaping_and_reserved_object_paths(tmp_path, relative):
    store = store_at(tmp_path / 'durable')
    source = tmp_path / 'source'; source.write_bytes(b'payload')
    with pytest.raises(ValueError, match='path'):
        store.upload_verified(source, relative)
    with pytest.raises(ValueError, match='path'):
        store.pointer(relative, {})


def test_mutable_pointer_and_log_are_replaced_and_readback_verified(tmp_path):
    store = store_at(tmp_path / 'durable')
    store.pointer('checkpoints/events.json', {'completed_updates': 1})
    store.pointer('checkpoints/events.json', {'completed_updates': 2})
    source = tmp_path / 'log'; source.write_bytes(b'first\n')
    store.sync_file(source, 'logs/events.jsonl')
    source.write_bytes(b'first\nsecond\n')
    store.sync_file(source, 'logs/events.jsonl')
    store.download_verified('checkpoints/events.json', tmp_path / 'scratch/checkpoints/events.json')
    store.download_verified('logs/events.jsonl', tmp_path / 'scratch/logs/events.jsonl')
    assert json.loads((tmp_path / 'scratch/checkpoints/events.json').read_text()) == {'completed_updates': 2}
    assert (tmp_path / 'scratch/logs/events.jsonl').read_bytes() == b'first\nsecond\n'


def test_missing_receipt_fails_closed_for_existing_durable_object(tmp_path):
    store = store_at(tmp_path / 'durable')
    source = tmp_path / 'source'; source.write_bytes(b'checkpoint')
    store.upload_verified(source, 'checkpoints/a.pt')
    (store.root / '.receipts/checkpoints/a.pt.json').unlink()
    with pytest.raises(ValueError, match='receipt'):
        store.restore_run(tmp_path / 'scratch')

@pytest.mark.parametrize('filename', ['../outside.pt', '/absolute.pt', r'..\outside.pt'])
def test_resume_rejects_unsafe_checkpoint_pointer(tmp_path, filename):
    store = store_at(tmp_path / 'durable')
    store.pointer('checkpoints/latest.json', {'filename': filename, 'sha256': '0' * 64, 'bytes': 1})
    with pytest.raises(ValueError, match='checkpoint'):
        store.restore_run(tmp_path / 'scratch')


def test_partial_durable_run_cannot_fall_back_to_stale_scratch(tmp_path):
    store = store_at(tmp_path / 'durable')
    store.pointer('checkpoints/events.json', {'completed_updates': 2})
    scratch = tmp_path / 'scratch/checkpoints'; scratch.mkdir(parents=True)
    (scratch / 'latest.json').write_text('{}')
    with pytest.raises(ValueError, match='Partial'):
        store.restore_run(scratch.parent)


def checkpoint_fixture(store, scratch, step):
    import hashlib
    payload = ('step ' + str(step)).encode()
    digest = hashlib.sha256(payload).hexdigest()
    filename = f'step-{step:09d}-{digest}.pt'
    path = scratch / 'checkpoints' / filename; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
    return {'filename': filename, 'step': step, **store.upload_verified(path, 'checkpoints/' + filename)}


def test_retention_keeps_verified_best_latest_and_metrics(tmp_path):
    store = store_at(tmp_path / 'durable'); scratch = tmp_path / 'scratch'
    old = checkpoint_fixture(store, scratch, 5000)
    best = checkpoint_fixture(store, scratch, 7000)
    latest = checkpoint_fixture(store, scratch, 8000)
    store.pointer('checkpoints/latest.json', latest)
    selected = {'step': 7000, 'checkpoint': best}
    store.pointer('checkpoints/best.json', selected)
    store.pointer('checkpoints/events.json', {'completed_updates': 8000, 'best': selected})
    store.pointer('metrics/main.json', {'psnr': 12})
    store.prune_checkpoints(scratch)
    assert not (store.root / 'checkpoints' / old['filename']).exists()
    assert not (scratch / 'checkpoints' / old['filename']).exists()
    assert {p.name for p in (store.root / 'checkpoints').glob('*.pt')} == {best['filename'], latest['filename']}
    assert (store.root / 'metrics/main.json').is_file()
    store.restore_run(tmp_path / 'resumed')


def test_retention_refuses_uncommitted_events_boundary(tmp_path):
    store = store_at(tmp_path / 'durable'); scratch = tmp_path / 'scratch'
    old = checkpoint_fixture(store, scratch, 1)
    latest = checkpoint_fixture(store, scratch, 2)
    store.pointer('checkpoints/latest.json', latest)
    store.pointer('checkpoints/events.json', {'completed_updates': 1, 'best': None})
    with pytest.raises(ValueError, match='boundary'):
        store.prune_checkpoints(scratch)
    assert (store.root / 'checkpoints' / old['filename']).is_file()
    assert (scratch / 'checkpoints' / old['filename']).is_file()


def test_retention_refuses_corrupt_pointer_before_any_deletion(tmp_path):
    store = store_at(tmp_path / 'durable'); scratch = tmp_path / 'scratch'
    old = checkpoint_fixture(store, scratch, 1)
    latest = checkpoint_fixture(store, scratch, 2)
    store.pointer('checkpoints/latest.json', latest)
    store.pointer('checkpoints/events.json', {'completed_updates': 2, 'best': None})
    (store.root / 'checkpoints/latest.json').write_text('{}')
    with pytest.raises(ValueError, match='integrity'):
        store.prune_checkpoints(scratch)
    assert (store.root / 'checkpoints' / old['filename']).is_file()

def test_best_latest_removes_preflight_tensor_bytes_only_after_pass_and_reset(tmp_path):
    store = store_at(tmp_path / 'durable'); scratch = tmp_path / 'scratch'
    latest = checkpoint_fixture(store, scratch, 0)
    preflight = scratch / 'preflight'; preflight.mkdir()
    tensor = preflight / latest['filename']; tensor.write_bytes(b'preflight optimizer')
    store.upload_verified(tensor, 'preflight/' + tensor.name)
    (preflight / 'remote-resume.pt').write_bytes(b'preflight optimizer')
    store.pointer('checkpoints/latest.json', latest)
    store.pointer('checkpoints/events.json', {'completed_updates': 0, 'best': None})
    store.prune_checkpoints(scratch)
    assert tensor.is_file()
    store.pointer('preflight/results.json', {'status': 'PASS', 'resume_max_absolute_error': 0})
    store.prune_checkpoints(scratch)
    assert not tensor.exists()
    assert not (preflight / 'remote-resume.pt').exists()
    assert not (store.root / 'preflight' / tensor.name).exists()
    assert (store.root / 'preflight/results.json').is_file()
    assert (store.root / 'checkpoints' / latest['filename']).is_file()

@pytest.mark.parametrize('relative', ['.', './'])
def test_rejects_root_as_object_path(tmp_path, relative):
    store = store_at(tmp_path / 'durable')
    with pytest.raises(ValueError, match='path'):
        store.pointer(relative, {})


def test_receipt_only_durable_state_cannot_be_treated_as_fresh(tmp_path):
    store = store_at(tmp_path / 'durable')
    receipt = store.root / '.receipts/checkpoints/latest.json.json'
    receipt.parent.mkdir(parents=True); receipt.write_text('{}')
    with pytest.raises(ValueError, match='Partial'):
        store.restore_run(tmp_path / 'scratch')

def test_resume_ignores_and_diagnoses_store_temporary_after_process_death(tmp_path):
    import subprocess
    import sys
    store = store_at(tmp_path / 'durable'); scratch = tmp_path / 'scratch'
    latest = checkpoint_fixture(store, scratch, 7)
    store.pointer('checkpoints/latest.json', latest)
    source = tmp_path / 'next-log'; source.write_bytes(b'uncommitted new log')
    code = "import os,sys; from research.hpc_store import FilesystemStore; os.fsync=lambda descriptor:os._exit(17); FilesystemStore(sys.argv[1]).sync_file(sys.argv[2], 'logs/events.jsonl')"
    result = subprocess.run([sys.executable, '-c', code, str(store.root), str(source)], check=False)
    assert result.returncode == 17
    assert store.restore_run(tmp_path / 'resumed') is True
    assert (tmp_path / 'resumed/checkpoints' / latest['filename']).read_bytes() == b'step 7'
    assert not (tmp_path / 'resumed/logs/events.jsonl').exists()
    assert len(store.recovery_diagnostics) == 1
    assert store.recovery_diagnostics[0]['event'] == 'IGNORED_UNCOMMITTED_TEMPORARY'
    assert len(list((store.root / 'logs').glob('*.tmp'))) == 1


def test_unknown_temporary_is_not_silently_ignored(tmp_path):
    store = store_at(tmp_path / 'durable'); scratch = tmp_path / 'scratch'
    latest = checkpoint_fixture(store, scratch, 7)
    store.pointer('checkpoints/latest.json', latest)
    unknown = store.root / 'logs/interrupted.tmp'; unknown.parent.mkdir(); unknown.write_bytes(b'unknown bytes')
    with pytest.raises(ValueError, match='receipt'):
        store.restore_run(tmp_path / 'resumed')


def test_reserved_temporary_names_cannot_be_published_as_objects(tmp_path):
    store = store_at(tmp_path / 'durable')
    with pytest.raises(ValueError, match='path'):
        store.pointer('logs/.hpc-publish-abcd1234.tmp', {})
