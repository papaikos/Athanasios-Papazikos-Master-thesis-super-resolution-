"""Reinstate rejected D3x10 pairs that the user cleared by manual review.

Each listed NPZ in rejected/<split>/ is checked against rejected_samples.csv
(SHA-256) and put back into its original shard at its original position (the
build's shard plan order), written with the same deterministic TAR settings.
Only after the rewritten shard is verified and swapped in is the copy in
rejected/ removed, so every pair is always in exactly one place. Then
samples.csv (adds a `restored_by_manual_review` column), rejected_samples.csv
and manifest.json are updated. Only this dataset (D3x10) is touched.

  python restore_d3x10.py --data Maritime_D3x10_1m10m --list rejected/<list>.txt --full-run runs/<UTC>_full
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import tarfile

import pandas as pd

from reject_d3x10 import REJECTED, ROOT, sha256_bytes, sha256_file, utc, write_tar


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True)
    parser.add_argument('--list', required=True)
    parser.add_argument('--full-run', required=True)
    args = parser.parse_args()
    data = (ROOT / args.data).resolve()
    if not data.is_relative_to(ROOT):
        parser.error('only the D3x10 dataset inside D3x10 build may be changed')
    ids = [PureWindowsPath(line.strip()).stem for line in open(ROOT / args.list, encoding='utf-8') if line.strip()]
    if len(ids) != len(set(ids)):
        parser.error('duplicate ids in list')
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%SZ') + '_restore')
    (run / 'before').mkdir(parents=True)
    shutil.copy2(__file__, run / 'script_snapshot.py')
    shutil.copy2(ROOT / args.list, run / Path(args.list).name)
    for name, src in (('manifest.json', data / 'manifest.json'), ('samples.csv', data / 'samples.csv'),
                      ('rejected_samples.csv', REJECTED / 'rejected_samples.csv')):
        shutil.copy2(src, run / 'before' / name)

    def log(line):
        print(line, flush=True)
        with open(run / 'progress.log', 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')

    rej = pd.read_csv(REJECTED / 'rejected_samples.csv', low_memory=False)
    restore = rej[rej.sample_id.isin(ids)]
    if len(restore) != len(ids):
        raise ValueError('some listed ids are not in rejected_samples.csv')
    order = {s['sample_id']: (job['name'] + '.tar', i)
             for job in json.loads((ROOT / args.full_run / 'shard_plan.json').read_text())
             for i, s in enumerate(job['samples'])}
    manifest = json.loads((data / 'manifest.json').read_text())
    entries = {Path(s['file']).name: s for s in manifest['shards']}
    log(f'{utc()} restore {len(restore)} samples into {restore.x10_shard.nunique()} shards')

    changes = []
    for shard, group in restore.groupby('x10_shard'):
        entry = entries[shard]
        path = data / entry['file']
        before_sha = sha256_file(path)
        with tarfile.open(path) as tar:
            members = {m.name: tar.extractfile(m).read() for m in tar.getmembers()}
        added = []
        for r in group.itertuples():
            src = ROOT / r.rejected_file
            payload = src.read_bytes()
            if sha256_bytes(payload) != r.rejected_npz_sha256:
                raise ValueError('rejected file changed since rejection: ' + r.sample_id)
            if order[r.sample_id][0] != shard:
                raise ValueError('shard plan disagrees for ' + r.sample_id)
            members[r.sample_id + '.npz'] = payload
            added.append(dict(sample_id=r.sample_id, npz_sha256=r.rejected_npz_sha256, file=str(src)))
        ordered = sorted(members.items(), key=lambda kv: order[kv[0][:-4]][1])
        partial = Path(str(path) + '.restore.partial')
        write_tar(partial, ordered)
        with tarfile.open(partial) as tar:
            if [(m.name, tar.extractfile(m).read()) for m in tar.getmembers()] != ordered:
                partial.unlink()
                raise ValueError('rewritten shard check failed: ' + shard)
        after_sha = sha256_file(partial)
        os.replace(partial, path)
        for a in added:
            Path(a['file']).unlink()
        entry.update(sha256=after_sha, bytes=path.stat().st_size, samples=len(ordered),
                     sample_ids=[name[:-4] for name, _ in ordered])
        changes.append(dict(shard=entry['file'], before_sha256=before_sha, sha256=after_sha,
                            member_order=[name for name, _ in ordered], added=added))
    (run / 'shard_changes.json').write_text(json.dumps(changes, indent=1), encoding='utf-8')
    log(f'{utc()} shards rewritten: {len(changes)}')

    samples = pd.read_csv(data / 'samples.csv', low_memory=False)
    if 'restored_by_manual_review' not in samples:
        samples['restored_by_manual_review'] = False
    back = restore.drop(columns=['reject_reasons', 'rejected_file', 'rejected_npz_sha256']).copy()
    back['restored_by_manual_review'] = True
    samples = pd.concat([samples, back[samples.columns]], ignore_index=True)
    samples = samples.sort_values(['split', 'sample_id'], kind='stable').reset_index(drop=True)
    samples.to_csv(data / 'samples.csv', index=False)
    rej[~rej.sample_id.isin(ids)].to_csv(REJECTED / 'rejected_samples.csv', index=False)
    restored_rows = restore.copy()
    restored_rows.to_csv(run / 'restored_samples.csv', index=False)

    ids_in_shards = [i for s in manifest['shards'] for i in s['sample_ids']]
    if sorted(ids_in_shards) != sorted(samples.sample_id) or len(ids_in_shards) != len(set(ids_in_shards)):
        raise ValueError('manifest and samples.csv disagree after restore')
    manifest['counts'] = samples.split.value_counts().reindex(['train', 'validation', 'test']).astype(int).to_dict()
    manifest['filter'].setdefault('manual_restores', []).append(dict(
        utc=utc(), run=run.name, restored=len(restore), by='user manual review of rejected images',
        list_file=f'rejected/{Path(args.list).name}', restored_reasons=restore.reject_reasons.value_counts().to_dict()))
    manifest['filter']['rejected_after_restores'] = int((~rej.sample_id.isin(ids)).sum())
    manifest['samples_csv'].update(sha256=sha256_file(data / 'samples.csv'))
    (data / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    shutil.copy2(data / 'manifest.json', run / 'manifest.json')
    summary = dict(finished_utc=utc(), restored=len(restore), counts=manifest['counts'],
                   still_rejected=manifest['filter']['rejected_after_restores'],
                   shards_changed=[c['shard'] for c in changes],
                   restored_by_reason=restore.reject_reasons.value_counts().to_dict(),
                   restored_by_source=restore.source_kind.value_counts().to_dict())
    (run / 'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    log(f"{utc()} done: counts {manifest['counts']}, still rejected {summary['still_rejected']}")


if __name__ == '__main__':
    main()
