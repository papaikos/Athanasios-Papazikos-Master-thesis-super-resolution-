from pathlib import Path
import subprocess
import sys
import pytest

SCRIPT = Path('scripts/satellite_gpu_startup.sh')
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
    {1:'x'*64}, {2:BASE+'skysat/raw-batch1'}, {3:88401}, {3:2001},
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
    (root / "repo/runs/training").mkdir(parents=True)
    (root / "output-uri").write_text(BASE + "training/test")
    (root / "startup.log").write_text("log")
    function = "sync_results() {" + SCRIPT.read_text().split("sync_results() {", 1)[1].split("\nworker() {", 1)[0]
    command = "ROOT=" + shlex.quote(root.as_posix()) + "\n"
    command += "timeout() { if [[ \"$*\" == *\"storage rsync\"* ]]; then return 17; fi; return 0; }\n"
    command += function + "\nif sync_results; then exit 0; else exit $?; fi\n"
    result = subprocess.run([bash, "-c", command], capture_output=True, text=True)
    assert result.returncode == 17, result.stderr


def test_exact_twenty_four_hour_deadline_is_accepted():
    result = validate(BASE+'training/code.tar.gz#123', 'a'*64, BASE+'training/run-01', 88400, 2000)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '86220'


def test_twenty_four_hours_plus_one_second_is_rejected():
    result = validate(BASE+'training/code.tar.gz#123', 'a'*64, BASE+'training/run-01', 88401, 2000)
    assert result.returncode != 0
