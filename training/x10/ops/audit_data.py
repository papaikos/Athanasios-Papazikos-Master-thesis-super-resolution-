"""One-time D3x10 data audit on AUTH; writes inputs/data-audit.json and inputs/data-index.json.

Checks the manifest and samples.csv against their pinned SHA-256, every TRAIN and
validation shard against the manifest, member lists, counts, split disjointness,
the absence of test/rejected pairs, and records site/group overlap between TRAIN
and validation (reported, never used to re-split).
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--control', required=True)
    parser.add_argument('--audit-out', required=True)
    parser.add_argument('--index-out', required=True)
    args = parser.parse_args()
    root, control = Path(args.data_root), json.loads(Path(args.control).read_text())
    pins = control['data_pins']
    if sha256(root / 'manifest.json') != pins['manifest_sha256']:
        raise ValueError('manifest.json differs from the pinned build')
    if sha256(root / 'samples.csv') != pins['samples_csv_sha256']:
        raise ValueError('samples.csv differs from the pinned build')
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('status') != 'FILTERED' or manifest.get('recipe') != 'd3x10-variable-psf-noise-v1':
        raise ValueError('Unexpected manifest status/recipe')
    shards = [s for s in manifest['shards'] if s['split'] in ('train', 'validation')]
    on_disk = sorted(p.relative_to(root).as_posix() for p in root.rglob('*.tar'))
    if on_disk != sorted(s['file'] for s in shards):
        raise ValueError('Shard files on disk differ from the TRAIN/validation manifest entries')
    rows, seen = [], set()
    for number, shard in enumerate(shards, 1):
        path = root / shard['file']
        if path.stat().st_size != shard['bytes'] or sha256(path) != shard['sha256']:
            raise ValueError('Shard checksum mismatch: ' + shard['file'])
        with tarfile.open(path) as tar:
            members = tar.getmembers()
        if [m.name for m in members] != [i + '.npz' for i in shard['sample_ids']]:
            raise ValueError('Shard members differ from manifest: ' + shard['file'])
        for m in members:
            sid = m.name[:-4]
            if sid in seen:
                raise ValueError('Duplicate sample: ' + sid)
            seen.add(sid)
            rows.append({'sample_id': sid, 'split': shard['split'], 'shard': shard['file'],
                         'offset': m.offset_data, 'size': m.size})
        if number % 25 == 0:
            print(f'{number}/{len(shards)} shards verified', flush=True)
    counts = {split: sum(r['split'] == split for r in rows) for split in ('train', 'validation')}
    if counts != {'train': 16835, 'validation': 115}:
        raise ValueError(f'Unexpected counts {counts}')
    with open(root / 'samples.csv', newline='', encoding='utf-8') as handle:
        table = {r['sample_id']: r for r in csv.DictReader(handle)}
    in_shards = {r['sample_id']: r['split'] for r in rows}
    expected = {k: v['split'] for k, v in table.items() if v['split'] in ('train', 'validation')}
    if in_shards != expected:
        raise ValueError('samples.csv TRAIN/validation membership differs from shards')
    overlap = {}
    for key in ('site_id', 'geographic_group_id', 'acquisition_id'):
        train = {table[s][key] for s, sp in in_shards.items() if sp == 'train' and table[s][key]}
        val = {table[s][key] for s, sp in in_shards.items() if sp == 'validation' and table[s][key]}
        shared = sorted(train & val)
        overlap[key] = {'shared_values': len(shared),
                        'validation_samples_affected': sum(1 for s, sp in in_shards.items()
                                                           if sp == 'validation' and table[s][key] in shared)}
    index = {'schema': 'd3x10-shard-index-v1', 'manifest_sha256': pins['manifest_sha256'],
             'rows': sorted(rows, key=lambda r: (r['split'], r['shard'], r['offset']))}
    Path(args.index_out).write_text(json.dumps(index), encoding='utf-8')
    audit = {'status': 'PASS', 'utc': datetime.now(timezone.utc).isoformat(), 'counts': counts,
             'shards_verified': len(shards), 'test_on_aristotle': False,
             'manifest_sha256': pins['manifest_sha256'], 'samples_csv_sha256': pins['samples_csv_sha256'],
             'index_sha256': sha256(args.index_out), 'train_validation_overlap': overlap,
             'note': 'Overlap is recorded, not acted on; validation is 100% Maxar, TRAIN is mixed-source.',
             'python': sys.version.split()[0]}
    Path(args.audit_out).write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
