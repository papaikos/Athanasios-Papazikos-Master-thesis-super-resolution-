import numpy as np
import pytest
import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

from scripts.skysat_cloud_prepare import validate_pair, load_heldout_reference, verify_receipts
from src.skysat_prepare import degrade_rgb


def sample():
    y, x = np.mgrid[:512, :512]
    hr = np.stack([x / 512, y / 512, (x+y) / 1024]).astype(np.float32)
    lr, mask = degrade_rgb(hr, np.ones((512, 512), dtype=bool))
    return hr, lr, mask, {'cell': [2, 3], 'transform': [2.5, 0, 2560, 0, -2.5, 5120]}


def test_pair_proof_accepts_aligned_gradient():
    assert validate_pair(*sample())['max_degradation_error'] == 0


def test_pair_proof_rejects_changed_lr_and_misaligned_grid():
    hr, lr, mask, meta = sample()
    lr[0, 50, 50] += 0.01
    with pytest.raises(ValueError, match='degradation'):
        validate_pair(hr, lr, mask, meta)
    hr, lr, mask, meta = sample()
    meta['transform'][2] += 2.5
    with pytest.raises(ValueError, match='aligned'):
        validate_pair(hr, lr, mask, meta)


def test_pair_proof_rejects_invalid_mask():
    hr, lr, mask, meta = sample()
    mask[0, 0] = True
    with pytest.raises(ValueError, match='validity mask'):
        validate_pair(hr, lr, mask, meta)


PARENT = 'gs://my-maritime-rgb-bucket/skysat/prepared/pairs-v1/full'


def heldout_inputs(tmp_path):
    manifest = dict(bucket='b', prefix='raw', sources=[
        dict(source_id='v', asset_id='asset-v', group_id='gv', split='validation'),
        dict(source_id='t', asset_id='asset-t', group_id='gt', split='test')])
    manifest['sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    receipts = []
    for row in manifest['sources']:
        obj = dict(uri=f"gs://b/raw/{row['source_id']}/rgb.tif", generation='123', bytes=25)
        identity = hashlib.sha256((obj['uri'] + '#123').encode()).hexdigest()[:24]
        receipts.append(dict(**{k: row[k] for k in ('source_id', 'group_id', 'split')},
            recipe='rgb-v2', manifest_sha256=manifest['sha256'], max_pairs_per_object=1000000,
            source_object=obj, source_sha256='a' * 64, raw_bytes=25,
            valid_native_rgb_bytes=100, accepted_source_bytes=90, pairs=1,
            shards=[dict(uri=PARENT + '/' + identity + '/pairs-00000.tar', sha256='b' * 64)]))
    (tmp_path / 'heldout-manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'heldout-receipts.json').write_text(json.dumps(receipts))
    reference = dict(schema=1, parent_output_prefix=PARENT, manifest_sha256=manifest['sha256'],
        receipts_sha256=hashlib.sha256((tmp_path / 'heldout-receipts.json').read_bytes()).hexdigest())
    (tmp_path / 'heldout-reference.json').write_text(json.dumps(reference))
    return manifest, receipts, reference


def test_heldout_reference_accepts_frozen_two_splits(tmp_path):
    _, receipts, _ = heldout_inputs(tmp_path)
    assert load_heldout_reference(tmp_path, {'sources': []})[0] == receipts


@pytest.mark.parametrize('change', ['missing_test', 'group', 'generation', 'prefix', 'checksum'])
def test_heldout_reference_rejects_invalid_contract(tmp_path, change):
    _, receipts, reference = heldout_inputs(tmp_path)
    if change == 'missing_test':
        receipts.pop()
    elif change == 'group':
        receipts[0]['group_id'] = 'wrong'
    elif change == 'generation':
        receipts[0]['source_object']['generation'] = '124'
    elif change == 'prefix':
        reference['parent_output_prefix'] = 'gs://other/untrusted'
    else:
        reference['manifest_sha256'] = 'wrong'
    (tmp_path / 'heldout-receipts.json').write_text(json.dumps(receipts))
    reference['receipts_sha256'] = hashlib.sha256((tmp_path / 'heldout-receipts.json').read_bytes()).hexdigest()
    (tmp_path / 'heldout-reference.json').write_text(json.dumps(reference))
    with pytest.raises(ValueError):
        load_heldout_reference(tmp_path, {'sources': []})


def test_heldout_receipt_file_checksum_and_split_leakage(tmp_path):
    heldout_inputs(tmp_path)
    with pytest.raises(ValueError, match='overlap'):
        load_heldout_reference(tmp_path, {'sources': [dict(source_id='new', asset_id='new', group_id='gv')]})
    with (tmp_path / 'heldout-receipts.json').open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError, match='checksum'):
        load_heldout_reference(tmp_path, {'sources': []})


def make_pair_tar(path, receipt, bad_meta=False):
    hr, lr, mask, meta = sample()
    meta.update({k: receipt[k] for k in ('source_id', 'group_id', 'split', 'manifest_sha256', 'source_object', 'recipe')})
    if bad_meta:
        meta['group_id'] = 'wrong'
    buf = io.BytesIO()
    np.savez_compressed(buf, hr=hr, lr=lr, mask=mask, metadata=json.dumps(meta))
    with tarfile.open(path, 'w') as archive:
        member = tarfile.TarInfo('000000.npz')
        member.size = len(buf.getvalue())
        archive.addfile(member, io.BytesIO(buf.getvalue()))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('bad', ['none', 'hash', 'metadata'])
def test_heldout_readback_verifies_hash_and_full_provenance(tmp_path, monkeypatch, bad):
    _, receipts, _ = heldout_inputs(tmp_path)
    receipt = receipts[0]
    archive = tmp_path / 'source.tar'
    receipt['shards'][0]['sha256'] = make_pair_tar(archive, receipt, bad_meta=bad == 'metadata')
    if bad == 'hash':
        receipt['shards'][0]['sha256'] = '0' * 64
    def copy(command, check):
        Path(command[-1]).write_bytes(archive.read_bytes())
    monkeypatch.setattr('scripts.skysat_cloud_prepare.subprocess.run', copy)
    if bad == 'none':
        assert len(verify_receipts([receipt], tmp_path)) == 1
    else:
        with pytest.raises(ValueError):
            verify_receipts([receipt], tmp_path)


@pytest.mark.parametrize('incremental', [True, False])
def test_runner_incremental_keeps_parent_pairs_out_of_train_totals(tmp_path, monkeypatch, incremental):
    import scripts.skysat_cloud_prepare as runner
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    if incremental:
        heldout_inputs(inputs)
    splits = ['train'] if incremental else ['train', 'validation', 'test']
    rows = [dict(source_id='new-' + split, asset_id='asset-' + split,
                 group_id='new-group-' + split, split=split) for split in splits]
    manifest = dict(bucket='new', prefix='raw', sources=rows)
    manifest['sha256'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    objects = [dict(url=f"gs://new/raw/{row['source_id']}/rgb.tif#1",
                    metadata=dict(size='25', generation='1')) for row in rows]
    state = dict(manifest_sha256=manifest['sha256'],
                 tasks={row['source_id']: dict(state='COMPLETED') for row in rows})
    for name, value in [('manifest', manifest), ('state', state), ('objects', objects)]:
        (inputs / (name + '.json')).write_text(json.dumps(value))
    preparations = []
    def fake_prepare(args):
        selected = json.loads(Path(args.objects).read_text())
        preparations.append((args.max_pairs_per_source, len(selected)))
        return [dict(source_id=obj['url'].split('/')[4], split=obj['url'].split('/')[4][4:],
                     source_sha256=hashlib.sha256(obj['url'].encode()).hexdigest(),
                     pairs=5, accepted_source_bytes=30, valid_native_rgb_bytes=60)
                for obj in selected]
    monkeypatch.setattr(runner, 'prepare', fake_prepare)
    monkeypatch.setattr(runner, 'verify_receipts',
                        lambda receipts, scratch: [dict(split=r['split']) for r in receipts])
    status = tmp_path / 'status.json'
    runner.run(SimpleNamespace(root=str(tmp_path), scratch=str(tmp_path / 'scratch'),
                               status_file=str(status), output_prefix='gs://new/prepared'))
    final = json.loads(status.read_text())
    assert final['stage'] == 'preparation_complete'
    assert final['totals']['train']['pairs'] == 5
    assert final['totals']['train']['accepted_source_bytes'] == 30
    assert final['training_authorized'] is False
    assert final['candidate_train_100gb'] is False
    assert len([p for p in preparations if p[0] == 8]) == len(splits)
    assert len([p for p in preparations if p[0] == 1000000]) == 2
    if incremental:
        assert set(final['totals']) == {'train'}
        assert final['parent_heldout_verified_pairs'] == 2
    else:
        assert set(final['totals']) == {'train', 'validation', 'test'}
