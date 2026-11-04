import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import pytest

class LocalStore:
    """Filesystem transport fixture exercising actual immutable export bytes."""
    def __init__(self,root):self.root=root
    def upload_verified(self,path,relative):
        from research.checkpointing import sha256
        target=self.root/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.read_bytes()!=Path(path).read_bytes():raise ValueError('immutable collision')
        target.write_bytes(Path(path).read_bytes())
        return {'sha256':sha256(target),'bytes':target.stat().st_size}

def test_final_export_freezes_logs_and_excludes_mutable_status(tmp_path):
    from research.finalize import export
    run=tmp_path/'run';run.mkdir();(run/'logs').mkdir();(run/'status').mkdir()
    (run/'logs/train.jsonl').write_text('{"step":1}\n');(run/'status/guest-status.json').write_text('{}')
    store=LocalStore(tmp_path/'remote')
    status={'run_id':'test','outcome':'TIME_LIMITED','completed_updates':1,'finalization_deadline_utc':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()}
    result=export(run,store,status)
    names={r['object'] for r in result['required_artifacts']}
    assert 'final/artifacts/logs/train.jsonl' in names
    assert not any(n.startswith('status/') for n in names)
    (run/'logs/train.jsonl').write_text('changed')
    assert (store.root/'final/artifacts/logs/train.jsonl').read_text()=='{"step":1}\n'
    assert (store.root/'final/FINALIZED.json').exists()

def test_expired_export_does_not_publish_finalized(tmp_path):
    from research.finalize import export
    root=tmp_path/'run';root.mkdir();store=LocalStore(tmp_path/'remote')
    status={'run_id':'test','outcome':'FAILED','completed_updates':0,'finalization_deadline_utc':(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()}
    with pytest.raises(TimeoutError):export(root,store,status)
    assert not (store.root/'final/FINALIZED.json').exists()
