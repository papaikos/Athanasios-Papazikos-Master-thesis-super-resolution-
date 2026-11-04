"""Move rejected D3x10 pairs out of the shards into D3x10 build/rejected.

Rule (user decision 2026-10-20): reject when Jev nodata > 0.70 or seams > 0.60
or color_artifacts > 0.75 or P(clouds more than half) = p3 + p4 >= 0.50.
Applied to every split.

Each rejected sample's NPZ is written byte-for-byte to rejected/<split>/. Each
affected shard is rewritten with the remaining members in their original order,
using the same deterministic TAR settings as build_d3x10.py, verified, and then
swapped in. Reruns are safe: a shard that no longer holds rejected members is
left alone. Only this dataset (D3x10) is touched.

  python reject_d3x10.py --data Maritime_D3x10_1m10m [--workers 6]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile

import pandas as pd

ROOT = Path(__file__).resolve().parent
REJECTED = ROOT / 'rejected'
POLICY = dict(name='rules_v5_d3x10', decided='2026-10-20 by the user, from the score-band review',
              rules=dict(nodata_gt=0.70, seams_gt=0.60, color_artifacts_gt=0.75, clouds_more_than_half_ge=0.50),
              clouds_definition='clouds_p3 + clouds_p4 (Jev cloud level 3 or 4)', splits='all')


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def reasons(row):
    out = []
    if row.nodata > 0.70: out.append('nodata')
    if row.seams > 0.60: out.append('seams')
    if row.color_artifacts > 0.75: out.append('color_artifacts')
    if row.clouds_p3 + row.clouds_p4 >= 0.50: out.append('mostly_clouds')
    return out


def write_tar(path, members):
    """Same TAR conventions as build_d3x10.build_shard."""
    with open(path, 'wb') as fh, tarfile.open(fileobj=fh, mode='w', format=tarfile.PAX_FORMAT) as tar:
        for name, payload in members:
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(payload), 0o644, 0
            tar.addfile(info, io.BytesIO(payload))


def filter_shard(job):
    path, reject = Path(job['path']), set(job['reject'])
    before_sha = sha256_file(path)
    with tarfile.open(path) as tar:
        members = [(m.name, tar.extractfile(m).read()) for m in tar.getmembers()]
    present = {name[:-4] for name, _ in members} & reject
    if not present:
        return dict(shard=job['file'], changed=False, sha256=before_sha, bytes=path.stat().st_size,
                    sample_ids=[name[:-4] for name, _ in members], rejected=[])
    rejected = []
    for name, payload in members:
        if name[:-4] in present:
            target = REJECTED / job['split'] / name
            tmp = Path(str(target) + '.tmp')
            tmp.write_bytes(payload)
            os.replace(tmp, target)
            if sha256_bytes(target.read_bytes()) != sha256_bytes(payload):
                raise ValueError('rejected copy differs: ' + name)
            rejected.append(dict(sample_id=name[:-4], npz_sha256=sha256_bytes(payload), npz_bytes=len(payload)))
    kept = [(name, payload) for name, payload in members if name[:-4] not in present]
    partial = Path(str(path) + '.filtered.partial')
    write_tar(partial, kept)
    with tarfile.open(partial) as tar:
        check = [(m.name, tar.extractfile(m).read()) for m in tar.getmembers()]
    if check != kept:
        partial.unlink()
        raise ValueError('rewritten shard differs from the kept members: ' + job['file'])
    after_sha = sha256_file(partial)
    os.replace(partial, path)
    return dict(shard=job['file'], changed=True, before_sha256=before_sha, sha256=after_sha,
                bytes=path.stat().st_size, sample_ids=[name[:-4] for name, _ in kept], rejected=rejected)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    data = (ROOT / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data)
    if not data.is_relative_to(ROOT):
        parser.error('only the D3x10 dataset inside D3x10 build may be changed')
    run = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%SZ') + '_reject')
    (run / 'before').mkdir(parents=True)
    shutil.copy2(__file__, run / 'script_snapshot.py')
    log = lambda line: (print(line, flush=True), open(run / 'progress.log', 'a', encoding='utf-8').write(line + '\n'))

    manifest = json.loads((data / 'manifest.json').read_text())
    table = pd.read_csv(data / 'samples.csv', low_memory=False)
    shutil.copy2(data / 'manifest.json', run / 'before' / 'manifest.json')
    shutil.copy2(data / 'samples.csv', run / 'before' / 'samples.csv')
    table['reject_reasons'] = [';'.join(reasons(r)) for r in table.itertuples()]
    reject = table[table.reject_reasons != '']
    write_json = lambda p, v: Path(p).write_text(json.dumps(v, indent=1), encoding='utf-8')
    write_json(run / 'policy.json', dict(**POLICY, created_utc=utc(), rejected=len(reject),
                                         manifest_before_sha256=sha256_file(data / 'manifest.json'),
                                         samples_before_sha256=sha256_file(data / 'samples.csv')))
    log(f'{utc()} reject {len(reject)} of {len(table)} samples: {reject.split.value_counts().to_dict()}')
    for split in ('train', 'validation', 'test'):
        (REJECTED / split).mkdir(parents=True, exist_ok=True)

    by_shard = reject.groupby('x10_shard').sample_id.apply(list).to_dict()
    jobs = [dict(path=str(data / s['file']), file=s['file'], split=s['split'],
                 reject=by_shard.get(Path(s['file']).name, [])) for s in manifest['shards']]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(filter_shard, jobs))
    write_json(run / 'shard_changes.json', results)
    moved = [r for res in results for r in res['rejected']]
    log(f"{utc()} shards rewritten: {sum(r['changed'] for r in results)}; samples moved this run: {len(moved)}")

    # Rejected-sample table: every Jev/build column + reasons + where it came from + exact file hash.
    files = {p.stem: p for split in ('train', 'validation', 'test') for p in (REJECTED / split).glob('*.npz')}
    if set(files) != set(reject.sample_id):
        raise ValueError('rejected folder does not hold exactly the rejected samples')
    rej = reject.copy()
    rej['rejected_file'] = [str(files[s].relative_to(ROOT)) for s in rej.sample_id]
    rej['rejected_npz_sha256'] = [sha256_file(files[s]) for s in rej.sample_id]
    rej.to_csv(REJECTED / 'rejected_samples.csv', index=False)
    write_json(REJECTED / 'policy.json', json.loads((run / 'policy.json').read_text()))

    kept = table[table.reject_reasons == ''].drop(columns='reject_reasons')
    kept.to_csv(data / 'samples.csv', index=False)
    for res, entry in zip(results, manifest['shards']):
        entry.update(sha256=res['sha256'], bytes=res['bytes'], samples=len(res['sample_ids']),
                     sample_ids=res['sample_ids'])
    if sum(s['samples'] for s in manifest['shards']) != len(kept):
        raise ValueError('manifest and samples.csv disagree after filtering')
    manifest.update(status='FILTERED', counts=kept.split.value_counts().reindex(['train', 'validation', 'test']).astype(int).to_dict(),
                    filter=dict(**POLICY, applied_utc=utc(), reject_run=run.name, rejected=len(reject),
                                rejected_by_split=reject.split.value_counts().to_dict(),
                                rejected_folder='D3x10 build/rejected (byte-identical NPZs + rejected_samples.csv)',
                                manifest_before_sha256=json.loads((run / 'policy.json').read_text())['manifest_before_sha256']))
    manifest['samples_csv'].update(sha256=sha256_file(data / 'samples.csv'),
                                   note=manifest['samples_csv']['note'] + '; rejected samples listed in rejected/rejected_samples.csv')
    write_json(data / 'manifest.json', manifest)
    shutil.copy2(data / 'manifest.json', run / 'manifest.json')
    shutil.copy2(data / 'samples.csv', run / 'samples.csv')
    summary = dict(finished_utc=utc(), kept=manifest['counts'], rejected=len(reject),
                   rejected_by_reason=pd.Series([x for r in reject.reject_reasons for x in r.split(';')]).value_counts().to_dict(),
                   rejected_by_source=reject.source_kind.value_counts().to_dict(),
                   shards_changed=[r['shard'] for r in results if r['changed']],
                   total_bytes_after=sum(s['bytes'] for s in manifest['shards']))
    write_json(run / 'summary.json', summary)
    log(f"{utc()} done: kept {manifest['counts']}, rejected {len(reject)} -> {REJECTED}")


if __name__ == '__main__':
    main()
