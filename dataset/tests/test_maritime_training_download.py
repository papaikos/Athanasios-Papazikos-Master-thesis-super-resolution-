import copy
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from scripts import maritime_training_download as download


def object_pin(payload=b'synthetic shard'):
    return dict(uri='gs://synthetic/shard', generation='7', bytes=len(payload),
                crc32c='test-crc', sha256=download.digest_bytes(payload))


def manifest(dataset='d1'):
    return dict(dataset=dataset, training_format_verified=True, sample_count=3,
                samples=[dict(sample_id=name, split=split) for name, split in
                         [('a', 'train'), ('b', 'validation'), ('c', 'test')]],
                shards=[dict(split=split, sample_ids=[name], object=object_pin())
                        for name, split in [('a', 'train'), ('b', 'validation'), ('c', 'test')]])


def test_validation_selects_only_frozen_validation_members():
    selected = download.selected_shards(manifest(), 'd1', 'validation')
    assert selected[0]['sample_ids'] == ['b']
    broken = manifest()
    broken['shards'][1]['sample_ids'] = ['a']
    with pytest.raises(ValueError, match='membership'):
        download.selected_shards(broken, 'd1', 'validation')


def test_cli_accepts_actual_validation_split(monkeypatch):
    calls = []
    monkeypatch.setattr(download, 'run', calls.append)
    monkeypatch.setattr(sys, 'argv', ['download', '--index-uri', 'gs://synthetic/index',
                                    '--dataset', 'd2', '--split', 'validation', '--output', 'output'])
    download.main()
    assert calls[0].split == 'validation'


@pytest.fixture
def run_setup(monkeypatch, tmp_path):
    client = object()
    storage_module = ModuleType('google.cloud')
    storage_module.storage = SimpleNamespace(Client=lambda: client)
    monkeypatch.setitem(sys.modules, 'google.cloud', storage_module)
    monkeypatch.setattr(download, 'require_cloud_vm', lambda: '123')
    index = dict(datasets={name: object_pin() for name in ['mother', 'd1', 'd2', 'd3']},
                 sample_count=3)
    monkeypatch.setattr(download, 'read_index', lambda *args, **kwargs: (index, object_pin()))
    args = SimpleNamespace(index_pin=None, index_uri='gs://synthetic/index', dataset='d1',
                           split='validation', output=str(tmp_path), max_bytes=20_000_000_000)
    return args


@pytest.mark.parametrize('dataset', ['mother', 'd1', 'd2', 'd3'])
def test_run_mother_manifest_only_and_child_validation_shards(run_setup, monkeypatch, dataset):
    args = run_setup
    args.dataset = dataset
    document = manifest(dataset)
    monkeypatch.setattr(download, 'verify_output', lambda *args: download.json_bytes(document))
    calls = []
    def fetch(client, pin, target):
        calls.append(pin)
        return target / 'verified.tar'
    monkeypatch.setattr(download, 'download_shard', fetch)
    result = download.run(args)
    assert result['split'] == 'validation' and result['samples'] == 1
    assert json.loads(Path(result['manifest_path']).read_text()) == document
    if dataset == 'mother':
        assert calls == [] and result['shards'] == [] and result['tar_bytes'] == 0
    else:
        assert len(calls) == 1 and result['shards'] == [str(Path(args.output) / 'verified.tar')]
        assert result['tar_bytes'] == object_pin()['bytes']


def test_byte_cap_rejects_before_any_shard_download(run_setup, monkeypatch):
    run_setup.max_bytes = 1
    monkeypatch.setattr(download, 'verify_output', lambda *args: download.json_bytes(manifest()))
    def forbidden(*args):
        pytest.fail('Byte cap should reject before downloading')
    monkeypatch.setattr(download, 'download_shard', forbidden)
    with pytest.raises(ValueError, match='max-bytes'):
        download.run(run_setup)


def test_workstation_guard_precedes_sdk_and_download(monkeypatch):
    def forbidden():
        raise RuntimeError('GCE required')
    monkeypatch.setattr(download, 'require_cloud_vm', forbidden)
    with pytest.raises(RuntimeError, match='GCE required'):
        download.run(SimpleNamespace())


class FakeBlob:
    def __init__(self, payload):
        self.payload = payload
        self.generation = 7
        self.size = len(payload)
        self.crc32c = 'test-crc'
        self.downloads = 0

    def reload(self):
        pass

    def download_to_filename(self, path, **kwargs):
        assert kwargs == dict(if_generation_match=7, checksum='crc32c')
        self.downloads += 1
        Path(path).write_bytes(self.payload)


def fake_client(blob):
    def get_blob(name, generation):
        assert name == 'shard' and isinstance(generation, int)
        return blob
    def get_bucket(name):
        assert name == 'synthetic'
        return SimpleNamespace(blob=get_blob)
    return SimpleNamespace(bucket=get_bucket)


def test_download_resume_and_corrupt_cache_rejection(tmp_path):
    payload = b'synthetic shard'
    blob = FakeBlob(payload)
    client = fake_client(blob)
    path = download.download_shard(client, object_pin(payload), tmp_path)
    assert path.read_bytes() == payload
    assert download.download_shard(client, object_pin(payload), tmp_path) == path
    assert blob.downloads == 1
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='corrupt'):
        download.download_shard(client, object_pin(payload), tmp_path)


@pytest.mark.parametrize('field', ['sha256', 'crc32c', 'generation', 'bytes'])
def test_bad_immutable_pin_rejected_without_partial_files(tmp_path, field):
    payload = b'synthetic shard'
    blob = FakeBlob(payload)
    pin = object_pin(payload)
    pin[field] = {'sha256': '0' * 64, 'crc32c': 'wrong', 'generation': '8', 'bytes': 1}[field]
    with pytest.raises(ValueError):
        download.download_shard(fake_client(blob), pin, tmp_path)
    assert not list(tmp_path.iterdir())


def test_incomplete_receipt_cannot_authorize_training_index(monkeypatch):
    pin = object_pin()
    index = dict(schema='maritime-super-resolution-training-index-v1',
                 state='FOUR_TRAINING_FORMATS_VERIFIED',
                 datasets={name: pin for name in ['mother', 'd1', 'd2', 'd3']},
                 release_receipt=pin, sample_count=3, unique_useful_native_rgb_bytes=60_000_000_000)
    receipt = dict(state='COMPLETE', training_format_verified=True, manifests=index['datasets'],
                   sample_count=3, useful_native_rgb_bytes=60_000_000_000)
    objects = iter([download.json_bytes(index), download.json_bytes(receipt)])
    monkeypatch.setattr(download, 'verify_output', lambda *args: next(objects))
    assert download.read_index(object(), pin=pin)[0] == index
    incomplete = copy.deepcopy(receipt)
    incomplete['state'] = 'RUNNING'
    objects = iter([download.json_bytes(index), download.json_bytes(incomplete)])
    with pytest.raises(ValueError, match='receipt disagree'):
        download.read_index(object(), pin=pin)
