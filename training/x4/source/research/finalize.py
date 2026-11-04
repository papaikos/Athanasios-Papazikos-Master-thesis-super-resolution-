"""Bounded export; FINALIZED exists only after required artifact readback."""
from pathlib import Path
import shutil
from datetime import datetime,timezone
from .checkpointing import atomic_json,sha256
from .hpc_store import FilesystemStore
import uuid
from .reporting import scientific_report,utcnow

def export(run_dir,store,status):
    root=Path(run_dir);filesystem=isinstance(store,FilesystemStore)
    if filesystem:
        store.invalidate_finalized()
        (root/'final/FINALIZED.json').unlink(missing_ok=True)
    scientific_report(root,status)
    artifacts=[]
    paths=list(sorted(root.rglob('*')))
    for path in paths:
        if datetime.now(timezone.utc)>=datetime.fromisoformat(status['finalization_deadline_utc']):raise TimeoutError('Finalization deadline reached')
        if not path.is_file() or path.suffix=='.tmp' or path.name in ('FINALIZED.json','artifacts-manifest.json'):continue
        relative=path.relative_to(root).as_posix()
        if relative.startswith(('control/','status/','progress/','final/artifacts/','final/manifests/')):continue
        if not (relative.startswith('checkpoints/') and path.suffix=='.pt'):
            snapshot=root/'final/artifacts'/relative;snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,snapshot)
            path=snapshot;relative=path.relative_to(root).as_posix()
            if filesystem:relative='final/artifacts/'+sha256(snapshot)+'/'+snapshot.relative_to(root/'final/artifacts').as_posix()
        receipt=store.upload_verified(path,relative)
        artifacts.append({'object':relative,'size':receipt['bytes'],'sha256':receipt['sha256']})
    manifest={'run_id':status['run_id'],'required_artifacts':artifacts,'verified_utc':utcnow()}
    if filesystem:
        manifest['attempt_id']=uuid.uuid4().hex
        return _commit_filesystem_manifest(root,store,status,manifest)
    atomic_json(root/'final/artifacts-manifest.json',manifest);store.upload_verified(root/'final/artifacts-manifest.json','final/artifacts-manifest.json')
    atomic_json(root/'final/FINALIZED.json',manifest);store.upload_verified(root/'final/FINALIZED.json','final/FINALIZED.json')
    return manifest


def _commit_filesystem_manifest(root,store,status,manifest):
    """Completion refers to exact immutable manifest bytes from this attempt."""
    def within_deadline():
        if datetime.now(timezone.utc)>=datetime.fromisoformat(status['finalization_deadline_utc']):raise TimeoutError('Finalization deadline reached')
    within_deadline()
    path=root/'final/artifacts-manifest.json';atomic_json(path,manifest)
    digest=sha256(path);relative='final/manifests/'+digest+'.json'
    receipt=store.upload_verified(path,relative)
    store.sync_file(path,'final/artifacts-manifest.json')
    within_deadline()
    marker=dict(manifest,finalized=True,manifest_object=relative,manifest_sha256=receipt['sha256'],manifest_bytes=receipt['bytes'])
    marker_path=root/'final/FINALIZED.json';atomic_json(marker_path,marker)
    store.sync_file(marker_path,'final/FINALIZED.json')
    return manifest
