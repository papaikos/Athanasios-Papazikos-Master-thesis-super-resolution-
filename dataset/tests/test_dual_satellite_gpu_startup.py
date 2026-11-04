from pathlib import Path
import subprocess
import sys
import pytest

SCRIPT = Path('scripts/dual_satellite_gpu_startup.sh')
BASE = 'gs://my-maritime-rgb-bucket/'

def validate(*args):
    code = SCRIPT.read_text().split("<<'VALIDATE'\n",1)[1].split('\nVALIDATE',1)[0]
    return subprocess.run([sys.executable, '-', *map(str,args)], input=code, text=True, capture_output=True)

def test_valid_bounded_training_metadata():
    result = validate(BASE+'training/code.tar.gz#123', 'a'*64, BASE+'training/run-01', 30000, 2000)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '27820'

@pytest.mark.parametrize('change', [
    {0: BASE+'training/code.tar.gz'}, {0: BASE+'training/../code#123'},
    {1:'x'*64}, {2:BASE+'skysat/raw-batch1'}, {3:59601}, {3:2001},
])
def test_reject_unsafe_metadata(change):
    args=[BASE+'training/code.tar.gz#123','a'*64,BASE+'training/run-01',30000,2000]
    for key,value in change.items(): args[key]=value
    assert validate(*args).returncode != 0


def test_checkpoint_sync_failure_is_not_masked_by_log_upload(tmp_path):
    import shutil
    import shlex
    bash = shutil.which("bash")
    if sys.platform == "win32":
        bash = "C:/Program Files/Git/bin/bash.exe"
    if not bash:
        pytest.skip("Bash unavailable")
    root = tmp_path / "worker"
    (root / "repo/runs").mkdir(parents=True)
    (root / "output-uri").write_text(BASE + "training/test")
    (root / "startup.log").write_text("log")
    function = "sync_results() {" + SCRIPT.read_text().split("sync_results() {", 1)[1].split("\nworker() {", 1)[0]
    command = "ROOT=" + shlex.quote(root.as_posix()) + "\n"
    command += "timeout() { if [[ \"$*\" == *\"storage rsync\"* ]]; then return 17; fi; return 0; }\n"
    command += function + "\nif sync_results; then exit 0; else exit $?; fi\n"
    result = subprocess.run([bash, "-c", command], capture_output=True, text=True)
    assert result.returncode == 17, result.stderr


def test_exact_sixteen_hour_deadline_is_accepted():
    result = validate(BASE+'training/code.tar.gz#123', 'a'*64, BASE+'training/run-01', 59600, 2000)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '57420'


def test_sixteen_hours_plus_one_second_is_rejected():
    result = validate(BASE+'training/code.tar.gz#123', 'a'*64, BASE+'training/run-01', 59601, 2000)
    assert result.returncode != 0


def test_sequential_trainers_share_staging_once():
    script = SCRIPT.read_text()
    assert script.count('python -m src.satellite_training_data ') == 1
    assert script.index('python -m src.train ') < script.index('python -m src.swinir_train ')
    assert '--resume' not in script
    assert '--download --channels 3 --pair-coverage' in script
    assert 'cap=25200' in script
    assert 'remaining >= 600' in script


@pytest.mark.parametrize('status,step,rc,expected', [
    ('step_cap', 2000, 0, True), ('completed', 2000, 0, True),
    ('time_or_signal_stop', 1000, 0, False), ('step_cap', 2000, 1, False),
    ('completed', 1999, 0, False),
])
def test_completion_requires_full_success(tmp_path, status, step, rc, expected):
    import json
    output = tmp_path / 'runs' / 'sen2sr'
    output.mkdir(parents=True)
    (output / 'result.json').write_text(json.dumps({'status': status, 'step': step}))
    code = SCRIPT.read_text().split("<<'RECEIPT'\n", 1)[1].split('\nRECEIPT', 1)[0]
    result = subprocess.run([sys.executable, '-', 'sen2sr', str(rc)], input=code,
                            text=True, capture_output=True, cwd=tmp_path)
    assert (result.returncode == 0) == expected, result.stderr
    assert json.loads((output / 'launch-result.json').read_text())['complete'] == expected
