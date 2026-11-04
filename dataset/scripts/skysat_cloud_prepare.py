"""Cloud-only proof followed by two disjoint, bounded preparation workers."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace

import numpy as np
from scipy.ndimage import gaussian_filter

from src.skysat_prepare import completed_objects, prepare, digest, validate_manifest, validate_receipt
from src.cloud_config import GCS_BUCKET


PARENT_HELDOUT_PREFIX = ('gs://'+GCS_BUCKET+'/'
                         'skysat/prepared/pairs-v1/full')


def object_prefix(output_prefix, obj):
    identity = hashlib.sha256((obj['uri'] + '#' + obj['generation']).encode()).hexdigest()[:24]
    return output_prefix + '/' + identity


def load_heldout_reference(inputs, new_manifest):
    """Validate frozen parent receipts without counting them as new TRAIN data."""
    reference = json.loads((inputs / 'heldout-reference.json').read_text())
    parent = json.loads((inputs / 'heldout-manifest.json').read_text())
    validate_manifest(parent)
    if (reference.get('schema') != 1
            or reference.get('parent_output_prefix') != PARENT_HELDOUT_PREFIX
            or reference.get('manifest_sha256') != parent['sha256']):
        raise ValueError('Heldout parent reference/checksum differs from pinned contract')
    receipts_path = inputs / 'heldout-receipts.json'
    if digest(receipts_path) != reference.get('receipts_sha256'):
        raise ValueError('Heldout receipts checksum mismatch')
    receipts = json.loads(receipts_path.read_text())
    if isinstance(receipts, dict):
        receipts = receipts['objects']
    if not isinstance(receipts, list) or not receipts:
        raise ValueError('Missing heldout receipts')
    rows = {row['source_id']: row for row in parent['sources']}
    if len(rows) != len(parent['sources']):
        raise ValueError('Duplicate parent source identity')
    group_splits = {}
    for row in parent['sources']:
        if group_splits.setdefault(row['group_id'], row['split']) != row['split']:
            raise ValueError('Parent geographic split conflict')
    heldout_rows = [row for row in parent['sources'] if row['split'] in ('validation', 'test')]
    for key in ('source_id', 'asset_id', 'group_id'):
        if {row[key] for row in heldout_rows} & {row[key] for row in new_manifest['sources']}:
            raise ValueError('New TRAIN sources overlap parent heldout ' + key)
    counts, seen = {'validation': 0, 'test': 0}, set()
    for receipt in receipts:
        row = rows.get(receipt.get('source_id'))
        if row is None or row['split'] not in counts:
            raise ValueError('Receipt is not a named parent heldout source')
        obj = receipt['source_object']
        prefix = f"gs://{parent['bucket']}/{parent['prefix']}/{row['source_id']}/"
        if (not obj.get('uri', '').startswith(prefix) or '#' in obj['uri']
                or not obj['uri'].lower().endswith(('.tif', '.tiff'))
                or not isinstance(obj.get('generation'), str) or not obj['generation'].isdigit()
                or not isinstance(obj.get('bytes'), int) or obj['bytes'] <= 0
                or receipt.get('raw_bytes') != obj['bytes']):
            raise ValueError('Invalid parent source object identity')
        if obj['uri'] in seen:
            raise ValueError('Duplicate heldout source object')
        seen.add(obj['uri'])
        validate_receipt(receipt, row, obj, dict(recipe='rgb-v2',
                         manifest_sha256=parent['sha256'], max_pairs_per_object=1000000))
        expected = object_prefix(PARENT_HELDOUT_PREFIX, obj)
        for index, shard in enumerate(receipt['shards']):
            checksum = shard.get('sha256', '')
            if (shard['uri'] != expected + f'/pairs-{index:05d}.tar'
                    or len(checksum) != 64
                    or any(c not in '0123456789abcdef' for c in checksum)):
                raise ValueError('Heldout shard identity or hash differs from parent contract')
        counts[row['split']] += receipt['pairs']
    if not all(counts.values()):
        raise ValueError('Both validation and test require positive parent pairs')
    return receipts, reference


def write_status(path, **values):
    values['updated_at'] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(values, indent=2))
    temporary.replace(path)


def validate_pair(hr, lr, mask, metadata):
    if hr.shape != (3, 512, 512) or lr.shape != (3, 128, 128) or mask.shape != (128, 128):
        raise ValueError('Wrong RGB pair dimensions')
    if not np.isfinite(hr).all() or not np.isfinite(lr).all():
        raise ValueError('Nonfinite pair')
    if hr.min() < 0 or hr.max() > 1 or float(hr.std()) < 1e-5:
        raise ValueError('Invalid normalization or constant proof image')
    expected = gaussian_filter(hr, (0, 2, 2), truncate=4, mode='reflect')
    expected = expected.reshape(3, 128, 4, 128, 4).mean(axis=(2, 4))
    error = float(np.max(np.abs(expected - lr)))
    if error > 2e-6:
        raise ValueError('LR does not match declared same-image degradation')
    if mask.dtype != np.bool_ or not mask[2:-2, 2:-2].all() or mask[:2].any() or mask[-2:].any() or mask[:, :2].any() or mask[:, -2:].any():
        raise ValueError('Gaussian support validity mask differs')
    x, y = metadata['cell']
    if metadata['transform'] != [2.5, 0.0, x*1280, 0.0, -2.5, (y+1)*1280]:
        raise ValueError('HR grid is not aligned to source cell')
    return dict(max_degradation_error=error, hr_std=float(hr.std()), valid_lr_pixels=int(mask.sum()))


def verify_receipts(receipts, scratch):
    evidence = []
    for receipt in receipts:
        pair_count = 0
        names = set()
        for shard in receipt['shards']:
            with tempfile.TemporaryDirectory(dir=scratch) as directory:
                archive_path = Path(directory) / 'pairs.tar'
                subprocess.run(['gcloud', 'storage', 'cp', shard['uri'], str(archive_path)], check=True)
                if digest(archive_path) != shard['sha256']:
                    raise ValueError('Pair tar readback SHA256 mismatch')
                with tarfile.open(archive_path) as archive:
                    for member in archive.getmembers():
                        if (not member.isfile() or not member.name.endswith('.npz')
                                or '/' in member.name or '\\' in member.name
                                or member.name in names or not 0 < member.size <= 16000000):
                            raise ValueError('Unexpected pair archive member')
                        names.add(member.name)
                        with np.load(io.BytesIO(archive.extractfile(member).read()), allow_pickle=False) as pair:
                            meta = json.loads(str(pair['metadata']))
                            if any(meta.get(key) != receipt[key] for key in (
                                'source_id', 'group_id', 'split', 'manifest_sha256', 'source_object', 'recipe'
                            )):
                                raise ValueError('Pair provenance disagrees with receipt')
                            evidence.append(dict(source_id=receipt['source_id'], split=receipt['split'],
                                                 **validate_pair(pair['hr'], pair['lr'], pair['mask'], meta)))
                            pair_count += 1
        if pair_count != receipt['pairs']:
            raise ValueError('Verified pair count differs from receipt')
    if not evidence:
        raise ValueError('No accepted proof pairs')
    return evidence


def run(args):
    root, scratch, status = Path(args.root), Path(args.scratch), Path(args.status_file)
    scratch.mkdir(parents=True, exist_ok=True)
    inputs = root / 'inputs'
    manifest = json.loads((inputs / 'manifest.json').read_text())
    validate_manifest(manifest)
    state = json.loads((inputs / 'state.json').read_text())
    objects = json.loads((inputs / 'objects.json').read_text())
    items = completed_objects(manifest, state, objects)
    if len(items) != len(objects) or not items:
        raise ValueError('Frozen listing contains incomplete or unexpected objects')
    object_map = {o['url'].split('#')[0]: o for o in objects}

    def prepare_subset(selected, label, max_pairs, prefix):
        listing = scratch / (label + '-objects.json')
        listing.write_text(json.dumps([object_map[obj['uri']] for _, obj in selected]))
        return prepare(SimpleNamespace(manifest=str(inputs / 'manifest.json'),
            task_state=str(inputs / 'state.json'), objects=str(listing), gcloud='gcloud',
            output_prefix=prefix, scratch=str(scratch), max_sources=len(manifest['sources']),
            max_pairs_per_source=max_pairs, summary_file=str(scratch / (label + '-summary.json')),
            progress_file=str(scratch / (label + '-progress.json'))))

    proof = []
    incremental = {row['split'] for row in manifest['sources']} == {'train'}
    heldout_reference = None
    heldout_evidence = []
    if incremental:
        write_status(status, stage='verifying_parent_heldout', total_objects=len(items))
        heldout_receipts, heldout_reference = load_heldout_reference(inputs, manifest)
        heldout_evidence = verify_receipts(heldout_receipts, scratch)
    write_status(status, stage='proof', total_objects=len(items))
    for split in (('train',) if incremental else ('train', 'validation', 'test')):
        candidates = sorted((item for item in items if item[0]['split'] == split),
                            key=lambda item: -item[1]['bytes'])
        found = False
        for index, item in enumerate(candidates[:3]):
            receipts = prepare_subset([item], f'proof-{split}-{index}', 8, args.output_prefix + '/proof')
            if sum(r['pairs'] for r in receipts) > 0:
                proof.extend(verify_receipts(receipts, scratch))
                found = True
                break
        if not found:
            raise ValueError('No valid proof pairs for ' + split + '; full preparation blocked')
    write_status(status, stage='full_preparation', proof=proof, total_objects=len(items), workers=2)
    results = []
    completed_workers = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(prepare_subset, items[i::2], f'worker-{i}', 1000000,
                               args.output_prefix + '/full') for i in range(2)]
        for future in as_completed(futures):
            results.extend(future.result())
            completed_workers += 1
            write_status(status, stage='full_preparation', proof=proof,
                         completed_workers=completed_workers, completed_objects=len(results), total_objects=len(items))
    totals = {}
    hashes = set()
    for receipt in results:
        identity = receipt.get('source_sha256')
        if not identity:
            raise ValueError('Missing native content hash')
        if identity in hashes:
            raise ValueError('Duplicate native TIFF content; needs deduplication before acceptance')
        hashes.add(identity)
        split = receipt['split']
        total = totals.setdefault(split, dict(pairs=0, accepted_source_bytes=0, valid_native_rgb_bytes=0))
        for key in total:
            total[key] += receipt[key]
    write_status(status, stage='preparation_complete', proof=proof, totals=totals,
                 completed_objects=len(results), total_objects=len(items), optimizer_consumed_bytes=0,
                 heldout_reference=heldout_reference, parent_heldout_verified_pairs=len(heldout_evidence),
                 training_authorized=False,
                 candidate_train_100gb=totals.get('train', {}).get('accepted_source_bytes', 0) >= 100000000000,
                 aggregation_scope='current batch only; cross-batch deduplication and optimizer consumption pending')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'scratch', 'output-prefix', 'status-file'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        path = Path(args.status_file)
        previous = json.loads(path.read_text()) if path.exists() else {}
        previous.pop('updated_at', None)
        previous.update(stage='failed', error=str(error))
        write_status(path, **previous)
        raise


if __name__ == '__main__':
    main()
