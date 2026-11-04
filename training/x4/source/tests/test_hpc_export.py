"""Filesystem final exports can resume changed logs without mutable collisions."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest
from research.checkpointing import sha256
from research.finalize import export
from research.hpc_store import FilesystemStore


def status():
    return {'run_id': 'd2-export-test', 'outcome': 'TARGET_REACHED', 'completed_updates': 100000,
            'finalization_deadline_utc': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}


def test_interrupted_filesystem_export_retries_with_changed_logs(tmp_path):
    run = tmp_path / 'run'; (run / 'logs').mkdir(parents=True)
    log = run / 'logs/train.jsonl'; log.write_bytes(b'{"step":100000}\n')
    class InterruptedStore(FilesystemStore):
        interrupted = False
        def upload_verified(self, path, relative):
            receipt = super().upload_verified(path, relative)
            if relative.endswith('logs/train.jsonl') and not self.interrupted:
                self.interrupted = True
                raise RuntimeError('simulated crash after log snapshot commit')
            return receipt
    store = InterruptedStore(tmp_path / 'durable')
    with pytest.raises(RuntimeError, match='simulated crash'):
        export(run, store, status())
    assert not (store.root / 'final/FINALIZED.json').exists()
    log.write_bytes(b'{"step":100000}\n{"event":"recovery"}\n')
    manifest = export(run, store, status())
    snapshots = [entry for entry in manifest['required_artifacts'] if entry['object'].endswith('logs/train.jsonl')]
    assert len(snapshots) == 1
    assert (store.root / snapshots[0]['object']).read_bytes() == b'{"step":100000}\n{"event":"recovery"}\n'
    for entry in manifest['required_artifacts']:
        assert sha256(store.root / entry['object']) == entry['sha256']
        assert (store.root / entry['object']).stat().st_size == entry['size']
    marker = json.loads((store.root / 'final/FINALIZED.json').read_text())
    assert marker['finalized'] is True
    assert marker['attempt_id'] == manifest['attempt_id']
    assert sha256(store.root / marker['manifest_object']) == marker['manifest_sha256']
    assert (store.root / marker['manifest_object']).stat().st_size == marker['manifest_bytes']
    assert json.loads((store.root / marker['manifest_object']).read_text()) == manifest


def test_failed_retry_invalidates_previous_completion_marker(tmp_path):
    run = tmp_path / 'run'; (run / 'logs').mkdir(parents=True)
    (run / 'logs/train.jsonl').write_bytes(b'original\n')
    store = FilesystemStore(tmp_path / 'durable')
    export(run, store, status())
    expired = status(); expired['finalization_deadline_utc'] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with pytest.raises(TimeoutError):
        export(run, store, expired)
    assert not (store.root / 'final/FINALIZED.json').exists()
    assert not (run / 'final/FINALIZED.json').exists()


def test_snapshot_readback_corruption_prevents_final_marker(tmp_path, monkeypatch):
    import os
    run = tmp_path / 'run'; (run / 'logs').mkdir(parents=True)
    (run / 'logs/train.jsonl').write_bytes(b'original\n')
    store = FilesystemStore(tmp_path / 'durable')
    original_link = os.link
    def corrupt_after_link(source, target, **kwargs):
        result = original_link(source, target, **kwargs)
        if Path(target).as_posix().endswith('/logs/train.jsonl'):
            Path(target).write_bytes(b'corrupt bytes')
        return result
    monkeypatch.setattr(os, 'link', corrupt_after_link)
    with pytest.raises(ValueError, match='integrity'):
        export(run, store, status())
    assert not (store.root / 'final/FINALIZED.json').exists()
