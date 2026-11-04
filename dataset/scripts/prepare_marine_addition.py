"""Bounded local preparation and immutable extension of the frozen RGB dataset.

No acquisition, cloud compute, training, or deletion of cloud data is performed.
Run without arguments; a supervisor enforces a persisted two-hour deadline.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from src.cloud_config import GCS_BUCKET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GCLOUD = 'C:/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd'
BUCKET = GCS_BUCKET
REPORT = ROOT / 'reports/marine-addition-20260927/pair-preparation'
INPUT = REPORT.parent
SCRATCH = Path(os.environ.get('LOCALAPPDATA', os.path.expanduser('~/AppData/Local'))) / 'Temp/marine-pair-preparation'
FROZEN = ROOT / 'reports/satellite-training-20260926/training-dataset.json'
PREFIX = f'gs://{BUCKET}/skysat/prepared/marine-addition-v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def storage(*args):
    result = subprocess.run([GCLOUD, 'storage', *map(str, args)], check=True,
                            capture_output=True, text=True, encoding='utf-8')
    return result.stdout


def checksum(value):
    return hashlib.sha256(json.dumps({k: v for k, v in value.items() if k != 'sha256'},
                                    sort_keys=True).encode()).hexdigest()


def merge_dataset(old, additions):
    if checksum(old) != old['sha256']:
        raise ValueError('Frozen dataset checksum mismatch')
    # Empty padded TIFFs remain in the preparation audit, not the training set.
    for source in additions:
        if source['pairs'] == 0 and (source['accepted_source_bytes'] != 0 or source['shards']):
            raise ValueError('Zero-pair receipt has nonzero coverage or shards')
    additions = [source for source in additions if source['pairs'] > 0]
    hashes = {s['source_sha256'] for s in old['sources'] if s['pairs'] > 0}
    objects = {s['source_object']['uri'] for s in old['sources']}
    heldout_groups = {s['group_id'] for s in old['sources'] if s['split'] != 'train'}
    for source in additions:
        if source['split'] != 'train' or source['group_id'] in heldout_groups:
            raise ValueError('New source conflicts with frozen heldouts')
        if source['source_sha256'] in hashes or source['source_object']['uri'] in objects:
            raise ValueError('Duplicate native source hash or object')
        hashes.add(source['source_sha256'])
        objects.add(source['source_object']['uri'])
    result = {**old, 'sources': old['sources'] + additions}
    result['totals'] = {
        split: {'pairs': sum(s['pairs'] for s in result['sources'] if s['split'] == split),
                'accepted_source_bytes': sum(s['accepted_source_bytes'] for s in result['sources']
                                             if s['split'] == split)}
        for split in ('train', 'test', 'validation')}
    result['description'] = 'Frozen original dataset plus verified coastal-source RGB pairs; unchanged heldouts.'
    result['parent_sha256'] = old['sha256']
    result['marine_content_review'] = {
        'status': 'pending per-cell review', 'verified_maritime_accepted_bytes': 0,
        'note': 'Coastal source scenes also contain inland cells; not every pair is maritime.'}
    result['sha256'] = checksum(result)
    assert [s for s in result['sources'] if s['split'] != 'train'] == [
        s for s in old['sources'] if s['split'] != 'train']
    return result


def validate_pair_arrays(hr, lr, mask, info, row, receipt):
    import numpy as np
    from rasterio.crs import CRS
    if (info.get('asset_id') != row['asset_id'] or info.get('split') != 'train'
            or any(info.get(key) != receipt[key] for key in (
                'source_id', 'group_id', 'split', 'source_object', 'manifest_sha256', 'recipe'))
            or info.get('group_id') != row['group_id'] or info.get('recipe') != 'rgb-v2'):
        raise ValueError('Pair provenance mismatch')
    if (hr.shape != (3, 512, 512) or lr.shape != (3, 128, 128)
            or hr.dtype != np.float32 or lr.dtype != np.float32
            or mask.shape != (128, 128) or mask.dtype != np.bool_
            or not np.isfinite(hr).all() or not np.isfinite(lr).all()
            or hr.min() < 0 or hr.max() > 1 or lr.min() < 0 or lr.max() > 1):
        raise ValueError('Invalid RGB pair arrays')
    expected_mask = np.zeros((128, 128), bool)
    expected_mask[2:-2, 2:-2] = True
    if not np.array_equal(mask, expected_mask):
        raise ValueError('Invalid synthetic LR support mask')
    cell = info.get('cell')
    if (not isinstance(cell, list) or len(cell) != 2 or any(type(c) is not int for c in cell)
            or type(info.get('native_covered_bytes')) is not int
            or info['native_covered_bytes'] <= 0 or info['native_covered_bytes'] % 3):
        raise ValueError('Invalid native coverage provenance')
    if (info.get('transform') != [2.5, 0, cell[0] * 1280, 0, -2.5, (cell[1] + 1) * 1280]
            or info.get('lr_transform') != [10, 0, cell[0] * 1280, 0, -10, (cell[1] + 1) * 1280]):
        raise ValueError('Pair grid differs from aligned 2.5m/10m cell')
    grid = row['grid']
    crs = CRS.from_user_input(info.get('crs'))
    if (crs != CRS.from_user_input(grid.get('crsCode') or grid['crsWkt'])
            or not crs.is_projected or crs.linear_units != 'metre'):
        raise ValueError('Pair CRS differs from native metric CRS')
    return (info['asset_id'], info['crs'], *cell)


def verify_pairs(receipts, rows, seen_cells):
    import numpy as np
    from src.skysat_prepare import degrade_rgb, digest
    for receipt in receipts:
        count = covered = 0
        row = rows[receipt['source_id']]
        for shard in receipt['shards']:
            metadata = json.loads(storage('objects', 'describe', shard['uri'], '--format=json'))
            generation = str(metadata['generation'])
            if not generation.isdigit():
                raise ValueError('Invalid shard generation')
            path = SCRATCH / 'verify-shard.tar'
            storage('cp', shard['uri'] + '#' + generation, path)
            if path.stat().st_size != int(metadata['size']) or digest(path) != shard['sha256']:
                raise ValueError('Shard identity mismatch')
            shard.update(generation=generation, bytes=int(metadata['size']))
            with tarfile.open(path) as archive:
                members = archive.getmembers()
                if not members or len({m.name for m in members}) != len(members):
                    raise ValueError('Empty shard or duplicate pair members')
                for index, member in enumerate(members):
                    if not member.isfile() or not member.name.endswith('.npz'):
                        raise ValueError('Unexpected shard member')
                    with np.load(io.BytesIO(archive.extractfile(member).read()), allow_pickle=False) as pair:
                        info = json.loads(str(pair['metadata']))
                        hr, lr, mask = pair['hr'], pair['lr'], pair['mask']
                        cell = validate_pair_arrays(hr, lr, mask, info, row, receipt)
                        if cell in seen_cells:
                            raise ValueError('Duplicate observation cell')
                        seen_cells.add(cell)
                        if index == 0:
                            expected_lr, expected_mask = degrade_rgb(hr, np.ones((512, 512), bool))
                            if not np.allclose(lr, expected_lr, atol=1e-7, rtol=1e-6) or not np.array_equal(mask, expected_mask):
                                raise ValueError('Synthetic LR degradation mismatch')
                        count += 1
                        covered += int(info['native_covered_bytes'])
            path.unlink()
        if count != receipt['pairs'] or covered != receipt['accepted_source_bytes']:
            raise ValueError('Receipt pair/native-byte accounting mismatch')


def worker():
    from src.skysat_prepare import prepare, validate_manifest, completed_objects, publish
    old = read(FROZEN)
    old_sources = {s['source_id'] for s in old['sources']}
    old_assets = set()
    for path in (ROOT / 'reports').glob('skysat-acquisition-manifest-*.json'):
        old_assets.update(s['asset_id'] for s in read(path)['sources'] if s['source_id'] in old_sources)
    if not old_assets:
        raise ValueError('Cannot prove old asset exclusion')
    additions, seen_cells, seen_assets = [], set(), set()
    args = SimpleNamespace(gcloud=GCLOUD)
    for batch in (1, 2, 3):
        manifest_path = INPUT / f'manifest-batch{batch}.json'
        state_path = INPUT / ('task-state.json' if batch == 1 else f'task-state-batch{batch}.json')
        manifest, state = read(manifest_path), read(state_path)
        validate_manifest(manifest)
        if manifest['bucket'] != BUCKET:
            raise ValueError('Unexpected raw bucket')
        rows = {r['source_id']: r for r in manifest['sources']}
        for row in rows.values():
            if row['asset_id'] in old_assets or row['asset_id'] in seen_assets:
                raise ValueError('Asset overlaps frozen data or another addition')
            if row['split'] != 'train' or state['tasks'][row['source_id']]['state'] != 'COMPLETED':
                raise ValueError('Only completed TRAIN sources may be prepared')
            seen_assets.add(row['asset_id'])
        inventory = REPORT / f'objects-batch{batch}.json'
        if not inventory.exists():
            save(inventory, json.loads(storage('ls', f"gs://{BUCKET}/{manifest['prefix']}/**", '--json')))
        items = completed_objects(manifest, state, read(inventory))
        if {r['source_id'] for r, _ in items} != set(rows):
            raise ValueError('Completed source missing from pinned object inventory')
        args = SimpleNamespace(gcloud=GCLOUD, manifest=str(manifest_path), task_state=str(state_path),
                               objects=str(inventory), scratch=str(SCRATCH), max_sources=len(rows),
                               max_pairs_per_source=1000000, output_prefix=f'{PREFIX}/batch{batch}/full',
                               progress_file=str(SCRATCH / f'progress-batch{batch}.json'),
                               summary_file=str(REPORT / f'summary-batch{batch}.json'))
        receipts = prepare(args)
        verify_pairs(receipts, rows, seen_cells)
        save(REPORT / f'verified-batch{batch}.json', receipts)
        additions.extend(receipts)
    result = merge_dataset(old, additions)
    output = REPORT / 'training-dataset-marine-v1.json'
    save(output, result)
    publish(args, output, f'{PREFIX}/training-dataset-marine-v1.json')
    save(REPORT / 'completion.json', {
        'stage': 'complete', 'completed_at': datetime.now(timezone.utc).isoformat(),
        'manifest': f'{PREFIX}/training-dataset-marine-v1.json', 'sha256': result['sha256'],
        'added_train_pairs': sum(s['pairs'] for s in additions),
        'added_accepted_source_bytes': sum(s['accepted_source_bytes'] for s in additions),
        'verified_maritime_accepted_bytes': 0, 'totals': result['totals'], 'training_started': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    options = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if options.worker:
        worker()
        return
    import msvcrt
    with (SCRATCH / 'controller.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        launch_path = REPORT / 'launch.json'
        if not launch_path.exists():
            save(launch_path, {'deadline': (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
                               'cloud_compute': False, 'training': False})
        seconds = (datetime.fromisoformat(read(launch_path)['deadline']) - datetime.now(timezone.utc)).total_seconds()
        if seconds <= 0:
            raise RuntimeError('Recorded two-hour preparation deadline expired; receipts preserved')
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'], cwd=ROOT)
        save(REPORT / 'process.json', {'controller_pid': os.getpid(), 'worker_pid': process.pid})
        try:
            code = process.wait(timeout=seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], check=False)
            process.wait()
            save(REPORT / 'stop.json', {'reason': 'deadline_or_interrupt', 'receipts_preserved': True})
            raise
        if code:
            save(REPORT / 'stop.json', {'reason': 'worker_failed', 'exit_code': code, 'receipts_preserved': True})
            raise SystemExit(code)


if __name__ == '__main__':
    main()
