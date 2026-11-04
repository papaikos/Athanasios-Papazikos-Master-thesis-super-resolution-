"""Operational controls run without model construction or optimizer updates."""
import pytest
from research import train
from research.hpc_store import FilesystemStore


def test_pause_persists_boundary_and_logs_without_final_metrics(tmp_path):
    scratch = tmp_path / 'scratch'
    (scratch / 'logs').mkdir(parents=True)
    (scratch / 'logs/train.jsonl').write_text('{"step":7}\n')
    store = FilesystemStore(tmp_path / 'durable')
    def checkpoint():
        source = scratch / 'checkpoint.pt'; source.write_bytes(b'boundary 7')
        receipt = store.upload_verified(source, 'checkpoints/step-7.pt')
        store.pointer('checkpoints/latest.json', {'filename': 'step-7.pt', **receipt})
    value = train.pause_run(scratch, store, checkpoint, lambda: {'completed_updates': 7, 'target_updates': 100000, 'outcome': 'INTERRUPTED'})
    assert value['phase'] == 'PAUSED'
    assert value['cleanup_ready'] is False
    assert value['training_complete'] is False
    assert 'final_main' not in value and 'final_aerial' not in value
    assert (store.root / 'logs/train.jsonl').read_text() == '{"step":7}\n'
    assert not (store.root / 'final/FINALIZED.json').exists()
    store.restore_run(tmp_path / 'new-scratch')
    assert (tmp_path / 'new-scratch/checkpoints/step-7.pt').read_bytes() == b'boundary 7'


def test_job_runtime_budget_is_relative_to_this_invocation():
    assert train.job_runtime_expired(100, 60, now=159) is False
    assert train.job_runtime_expired(100, 60, now=160) is True
    assert train.job_runtime_expired(1000, 60, now=1001) is False
    assert train.job_runtime_expired(100, None, now=100000) is False


def test_filesystem_selection_never_constructs_gcs_client(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('Google client constructed for filesystem backend')
    monkeypatch.setattr(train, 'GCSStore', unexpected)
    store = train.make_store('filesystem', tmp_path / 'durable', {'output_uri': 'gs://unused'})
    assert isinstance(store, FilesystemStore)


def test_filesystem_preflight_and_resume_reject_microbatch_drift():
    assert train.preflight_microbatches('filesystem') == (4,)
    assert train.preflight_microbatches('gcs') == (4, 2, 1)
    with pytest.raises(ValueError, match='microbatch'):
        train.validate_runtime_microbatch('filesystem', 2)
    train.validate_runtime_microbatch('filesystem', 4)

def test_pause_branch_in_main_exits_before_finalization(tmp_path):
    import ast
    import inspect
    from types import SimpleNamespace
    main = ast.parse(inspect.getsource(train.main)).body[0]
    final = next(node.finalbody for node in main.body if isinstance(node, ast.Try) and node.finalbody)
    pause = next(node for node in final if isinstance(node, ast.If) and any(isinstance(child, ast.Name) and child.id == 'step' for child in ast.walk(node.test)))
    branch = ast.FunctionDef(name='finish', args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[pause, ast.Raise(exc=ast.Call(func=ast.Name(id='AssertionError', ctx=ast.Load()), args=[ast.Constant(value='Reached expensive finalization')], keywords=[]))], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[branch], type_ignores=[]))
    calls = []
    namespace = dict(args=SimpleNamespace(storage_backend='filesystem'), step=42, root=tmp_path, store=None, checkpoint=None, failure=None, outcome='INTERRUPTED', status=lambda *args, **kwargs: {}, pause_run=lambda *args: calls.append('paused'))
    exec(compile(module, train.__file__, 'exec'), namespace)
    namespace['finish']()
    assert calls == ['paused']


@pytest.mark.parametrize('budget', ['0', '-1', 'nan', 'inf'])
def test_invalid_operational_budget_fails_before_reading_inputs(budget):
    arguments = []
    for field in ('data-root', 'run-dir', 'control', 'source-root', 'model-source', 'audit'):
        arguments += ['--' + field, 'unused-missing-path']
    with pytest.raises(SystemExit) as error:
        train.main(arguments + ['--job-max-runtime-seconds', budget])
    assert error.value.code == 2

def test_target_deferral_persists_last_and_marks_awaiting_without_final_metrics(tmp_path):
    import json
    scratch = tmp_path / 'scratch'; (scratch / 'logs').mkdir(parents=True)
    (scratch / 'logs/train.jsonl').write_text('{"step":100000}\n')
    store = FilesystemStore(tmp_path / 'durable')
    calls = []
    def checkpoint():
        calls.append('checkpoint')
        source = scratch / 'checkpoint.pt'; source.write_bytes(b'last 100000')
        receipt = store.upload_verified(source, 'checkpoints/last.pt')
        store.pointer('checkpoints/latest.json', {'filename': 'last.pt', 'step': 100000, **receipt})
    def current_status():
        calls.append('status')
        return {'completed_updates': 100000, 'target_updates': 100000, 'outcome': 'TARGET_REACHED'}
    value = train.defer_target_finalization(scratch, store, checkpoint, current_status)
    assert calls == ['checkpoint', 'status']
    assert value['phase'] == 'AWAITING_FINALIZATION'
    assert value['outcome'] == 'TARGET_REACHED' and value['training_complete'] is True
    assert value['cleanup_ready'] is False
    assert value['export_outcome'] == 'DEFERRED_UNTIL_FINALIZATION_JOB'
    assert 'final_main' not in value and 'final_aerial' not in value
    assert not (store.root / 'final/FINALIZED.json').exists()
    store.restore_run(tmp_path / 'resumed')
    assert (tmp_path / 'resumed/checkpoints/last.pt').read_bytes() == b'last 100000'
    assert json.loads((tmp_path / 'resumed/status/guest-status.json').read_text()) == value


def test_target_validation_is_carried_to_finalization_job_once():
    assert train.pending_target_validation('filesystem', True, 100000, {'validation_step': 98000}) is False
    assert train.pending_target_validation('filesystem', False, 100000, {'validation_step': 98000}) is True
    assert train.pending_target_validation('filesystem', False, 100000, {'validation_step': 100000}) is False
    assert train.pending_target_validation('filesystem', False, 99999, {'validation_step': 98000}) is False
    assert train.pending_target_validation('gcs', False, 100000, {'validation_step': 98000}) is False


@pytest.mark.parametrize(('backend', 'flag', 'expected'), [('filesystem', True, 'deferred'), ('filesystem', False, 'finalize'), ('gcs', True, 'finalize')])
def test_target_deferral_branch_allows_next_job_finalization(backend, flag, expected):
    import ast
    import inspect
    from types import SimpleNamespace
    main = ast.parse(inspect.getsource(train.main)).body[0]
    final = next(node.finalbody for node in main.body if isinstance(node, ast.Try) and node.finalbody)
    defer = next(node for node in final if isinstance(node, ast.If) and any(isinstance(child, ast.Name) and child.id == 'defer_target_finalization' for child in ast.walk(node)))
    wrapper = ast.FunctionDef(name='finish', args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), body=[defer, ast.Return(value=ast.Constant(value='finalize'))], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    namespace = dict(args=SimpleNamespace(storage_backend=backend, defer_finalization=flag), step=100000, root=None, store=None, checkpoint=None, failure=None, status=lambda *args, **kwargs: {})
    namespace['defer_target_finalization'] = lambda *args: namespace.update(deferred=True)
    exec(compile(module, train.__file__, 'exec'), namespace)
    result = namespace['finish']()
    assert ('deferred' if namespace.get('deferred') else result) == expected

@pytest.mark.parametrize(('backend', 'flag', 'expected'), [('filesystem', True, []), ('filesystem', False, ['validation']), ('gcs', True, ['validation'])])
def test_training_target_gate_defers_evaluation_to_fresh_job(backend, flag, expected):
    import ast
    import inspect
    from types import SimpleNamespace
    main = ast.parse(inspect.getsource(train.main)).body[0]
    loop = next(node for node in ast.walk(main) if isinstance(node, ast.While) and any(isinstance(child, ast.Name) and child.id == 'optimizer_update' for child in ast.walk(node)))
    gate = next(node for node in loop.body if isinstance(node, ast.If) and any(isinstance(child, ast.Attribute) and child.attr == 'defer_finalization' for child in ast.walk(node.test)))
    record = ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='calls', ctx=ast.Load()), attr='append', ctx=ast.Load()), args=[ast.Constant(value='validation')], keywords=[]))
    module = ast.fix_missing_locations(ast.Module(body=[ast.While(test=ast.Constant(value=True), body=[gate, record, ast.Break()], orelse=[])], type_ignores=[]))
    calls = []
    exec(compile(module, train.__file__, 'exec'), dict(args=SimpleNamespace(storage_backend=backend, defer_finalization=flag), step=100000, calls=calls))
    assert calls == expected
