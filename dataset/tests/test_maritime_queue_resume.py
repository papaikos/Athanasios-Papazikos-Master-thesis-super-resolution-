import copy

import pytest

from scripts.maritime_queue_resume import validate_checkpoint, publish_immutable


def checkpoint():
    return dict(next_batch=3, batch_count=6, config={'batches': {'sha256': 'frozen'}},
                launches=[dict(batch=i+1, container_id='id'+str(i)) for i in range(3)],
                active={'1': 'id1', '2': 'id2'}, closed=[{'uri': 'closure1'}])


def test_resumes_own_latest_checkpoint_after_migration_boundary():
    state = checkpoint()
    assert validate_checkpoint(state, 6, expected_config=state['config'], migration_boundary=1) is state


@pytest.mark.parametrize('mutation', ['missing_launch', 'bad_container', 'wrong_batch', 'bad_closed_count', 'wrong_config', 'past_schedule'])
def test_inconsistent_checkpoint_fails_before_launch(mutation):
    state = copy.deepcopy(checkpoint())
    if mutation == 'missing_launch': state['launches'].pop()
    if mutation == 'bad_container': state['active']['2'] = 'unowned'
    if mutation == 'wrong_batch': state['launches'][1]['batch'] = 1
    if mutation == 'bad_closed_count': state['closed'] = []
    if mutation == 'wrong_config': state['config'] = {}
    if mutation == 'past_schedule': state['next_batch'] = 7
    with pytest.raises(ValueError):
        validate_checkpoint(state, 6, expected_config={'batches': {'sha256': 'frozen'}}, migration_boundary=1)


class Blob:
    generation = '12'
    name = 'closure.json'
    def __init__(self, existing): self.existing = existing; self.uploads = 0
    def exists(self): return self.existing is not None
    def reload(self): pass
    def upload_from_string(self, raw, **kwargs):
        assert kwargs == dict(if_generation_match=0, checksum='crc32c')
        self.uploads += 1
        self.existing = raw
    def download_as_bytes(self, **kwargs):
        assert kwargs == dict(if_generation_match=12, checksum='crc32c')
        return self.existing


def test_identical_closure_reused_without_write():
    blob = Blob(b'same')
    publish_immutable(blob, b'same')
    assert blob.uploads == 0


def test_conflicting_closure_never_overwritten():
    blob = Blob(b'old')
    with pytest.raises(ValueError, match='differs'):
        publish_immutable(blob, b'new')
    assert blob.existing == b'old' and blob.uploads == 0


def test_new_closure_create_only_readback():
    blob = Blob(None)
    publish_immutable(blob, b'new')
    assert blob.uploads == 1 and blob.existing == b'new'
