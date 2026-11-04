"""Exercise the real trainer publication paths without a GPU or cloud access."""
import ast
import inspect
import json
import threading
from types import SimpleNamespace

from google.api_core.exceptions import PreconditionFailed

from research.checkpointing import GCSStore, atomic_json
from research import train


def test_periodic_log_sync_cannot_conflict_with_inflight_heartbeat(tmp_path):
    # Execute the production nested heartbeat and periodic-sync block. Extracting
    # these operational paths avoids replacing the scientific trainer with GPU
    # mocks, while ensuring the formerly unprotected publication is exercised.
    main = ast.parse(inspect.getsource(train.main)).body[0]
    heartbeat = next(node for node in main.body
                     if isinstance(node, ast.FunctionDef) and node.name == 'heartbeat')
    loop = next(node for node in ast.walk(main) if isinstance(node, ast.While)
                and any(isinstance(child, ast.Name) and child.id == 'last_sync'
                        for child in ast.walk(node)))
    periodic = next(node for node in loop.body if isinstance(node, ast.If)
                    and any(isinstance(child, ast.Name) and child.id == 'last_sync'
                            for child in ast.walk(node.test)))
    heartbeat_observed = threading.Event()
    release_heartbeat = threading.Event()
    heartbeat_uploaded = threading.Event()
    remote = {'generation': 7, 'data': b'{}'}
    errors = []

    class Blob:
        def __init__(self, name):
            self.name = name

        def reload(self):
            self.generation = remote['generation']
            if threading.current_thread().name == 'trainer-heartbeat':
                heartbeat_observed.set()
            else:
                release_heartbeat.set()

        def upload_from_string(self, raw, **kwargs):
            if threading.current_thread().name == 'trainer-heartbeat':
                assert release_heartbeat.wait(5), 'Periodic sync did not finish'
            else:
                assert heartbeat_uploaded.wait(5), 'Heartbeat did not publish'
            if kwargs['if_generation_match'] != remote['generation']:
                raise PreconditionFailed('Concurrent status generation changed')
            remote.update(generation=remote['generation'] + 1, data=raw)
            heartbeat_uploaded.set()

    class Bucket:
        def blob(self, name):
            assert name == 'run/status/guest-status.json'
            return Blob(name)

    store = GCSStore('gs://offline/run', SimpleNamespace(bucket=lambda _: Bucket()))

    def sync_file(path, relative):
        assert relative == 'logs/train.jsonl'
        assert path.read_text() == 'training-log\n'
        assert heartbeat_observed.wait(5), 'Heartbeat did not read generation'

    store.sync_file = sync_file
    atomic_json(tmp_path / 'status/guest-status.json', {'completed_updates': 19264})
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'logs/train.jsonl').write_text('training-log\n')
    waits = iter((False, True))
    namespace = dict(args=SimpleNamespace(storage_backend='gcs'), root=tmp_path, store=store, json=json, atomic_json=atomic_json,
                     status_lock=threading.RLock(), utcnow=lambda: 'fresh-heartbeat',
                     heartbeat_stop=SimpleNamespace(wait=lambda _: next(waits)),
                     event=lambda *args: errors.append(args), now=300, last_sync=0)
    exec(compile(ast.Module(body=[heartbeat], type_ignores=[]), train.__file__, 'exec'), namespace)
    thread = threading.Thread(target=namespace['heartbeat'], name='trainer-heartbeat')
    thread.start()
    try:
        assert heartbeat_observed.wait(5)
        try:
            exec(compile(ast.Module(body=[periodic], type_ignores=[]), train.__file__, 'exec'), namespace)
        except Exception as error:
            errors.append(error)
    finally:
        release_heartbeat.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == [], f'Training publication failed: {errors}'
    assert json.loads(remote['data']) == {'completed_updates': 19264, 'heartbeat_utc': 'fresh-heartbeat'}
    assert namespace['last_sync'] == 300
