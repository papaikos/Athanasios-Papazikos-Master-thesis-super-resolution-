import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BUCKET = "gs://my-rgb-sr-pilot-bucket/"


class MarineLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.credit = Path(self.temp.name) / "credit.json"
        now = datetime.now(timezone.utc)
        self.credit.write_text(json.dumps(dict(
            available_credit_eur=187.75, reserved_eur=47, committed_eur=0,
            unbilled_eur=15, other_anticipated_eur=20, whole_vm_upper_eur_per_hour=1,
            snapshot_at=now.isoformat(), credit_expires_at=(now + timedelta(days=30)).isoformat(),
            stage_cap_seconds=43200, user_time_cap_seconds=43200,
        )), encoding="utf-8")
        self.args = ["--credit-request", str(self.credit), "--name", "marine-cpu-test",
                     "--service-account", "123456789-compute@developer.gserviceaccount.com",
                     "--code-uri", BUCKET + "code/marine.tar.gz#123",
                     "--code-sha256", "a" * 64,
                     "--source-manifest-uri", BUCKET + "marine100/audit.sources.json#456",
                     "--prepared-prefix", BUCKET + "marine100/prepared/job1",
                     "--status-prefix", BUCKET + "marine100/cpu/job1"]

    def invoke(self, extra=(), gcloud_result=None):
        script = (ROOT / "scripts/launch_marine_cpu.sh").read_text(encoding="utf-8")
        source = script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", ["-", str(ROOT), *self.args, *extra]), \
                patch.object(sys, "path", sys.path[:]), \
                patch("subprocess.run", return_value=gcloud_result or subprocess.CompletedProcess([], 0, "{}", "")) as run, \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(compile(source, "launch_marine_cpu.sh", "exec"), {"__name__": "__main__"})
                code = 0
            except SystemExit as error:
                code = error.code
        return code, stdout.getvalue(), stderr.getvalue(), run

    def test_dryrun_cpu_deadline_metadata(self):
        code, stdout, stderr, run = self.invoke()
        self.assertEqual(code, 0, stderr)
        run.assert_not_called()
        result = json.loads(stdout)
        command = result["command"]
        for flag in ("--machine-type=e2-standard-4", "--zone=europe-west4-b",
                     "--boot-disk-size=60GB", "--boot-disk-auto-delete",
                     "--instance-termination-action=DELETE", "--no-restart-on-failure",
                     "--maintenance-policy=MIGRATE"):
            self.assertIn(flag, command)
        self.assertFalse(any("accelerator" in arg or "max-run-duration" in arg for arg in command))
        self.assertNotIn("--maintenance-policy=TERMINATE", command)
        now = datetime.fromisoformat(result["generated_at"])
        expiry = datetime.fromisoformat(result["termination_time"])
        self.assertGreater((expiry - now).total_seconds(), 43199)
        self.assertLessEqual((expiry - now).total_seconds(), 43200)
        self.assertIn("--termination-time=" + result["termination_time"], command)
        metadata = next(arg for arg in command if arg.startswith("--metadata="))
        self.assertIn("rgb-deadline-epoch=" + str(int(expiry.timestamp())), metadata)
        self.assertIn("rgb-source-manifest-uri=" + BUCKET + "marine100/audit.sources.json#456", metadata)

    def test_execute_only_explicit_and_no_retry(self):
        code, stdout, stderr, run = self.invoke(["--execute"])
        self.assertEqual(code, 0, stderr)
        run.assert_called_once()
        result = json.loads(stdout)
        self.assertFalse(result["dry_run"])
        self.assertEqual(run.call_args.args[0], result["command"])
        self.assertEqual(result["command"][:4], ["gcloud", "compute", "instances", "create"])
        code, _, _, run = self.invoke(["--execute"], subprocess.CompletedProcess([], 1, "", "already exists"))
        self.assertEqual(code, 1)
        run.assert_called_once()

    def test_stale_credit_and_zero_budget_fail_before_execution(self):
        data = json.loads(self.credit.read_text())
        for change in ({"snapshot_at": "2000-01-01T00:00:00Z"}, {"committed_eur": 190}):
            self.credit.write_text(json.dumps({**data, **change}))
            code, _, _, run = self.invoke(["--execute"])
            self.assertNotEqual(code, 0)
            run.assert_not_called()

    def test_policy_floors_and_runtime_cap(self):
        data = json.loads(self.credit.read_text())
        data.update(unbilled_eur=0, other_anticipated_eur=0, whole_vm_upper_eur_per_hour=.1,
                    reserved_eur=46.94, stage_cap_seconds=90000, user_time_cap_seconds=90000)
        self.credit.write_text(json.dumps(data))
        code, stdout, stderr, _ = self.invoke()
        self.assertEqual(code, 0, stderr)
        request = json.loads(stdout)["effective_credit_request"]
        self.assertEqual(request["reserved_eur"], 47)
        self.assertEqual(request["unbilled_eur"], 15)
        self.assertEqual(request["other_anticipated_eur"], 20)
        self.assertEqual(request["whole_vm_upper_eur_per_hour"], 1)
        self.assertEqual(json.loads(stdout)["runtime_seconds"], 43200)

    def test_invalid_or_unpinned_inputs(self):
        for extra in (["--name", "bad;injected"], ["--source-manifest-uri", BUCKET + "marine100/a.json"],
                      ["--code-sha256", "bad"], ["--code-uri", "https://example.com/code"],
                      ["--service-account", "default"], ["--project", "unowned-project"],
                      ["--status-prefix", BUCKET + "marine100/prepared/job1/sub"],
                      ["--runtime-seconds", "43201"], ["--runtime-seconds", "599"],
                      ["--accelerator=count=1"]):
            with self.subTest(extra=extra):
                code, _, _, run = self.invoke(extra)
                self.assertNotEqual(code, 0)
                run.assert_not_called()

    def test_real_bash_dryrun_and_startup_metadata_contract(self):
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        bash = str(git_bash) if git_bash.exists() else shutil.which("bash")
        if not bash:
            self.skipTest("Bash unavailable")
        env = dict(os.environ, PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1")
        process = subprocess.run([bash, str(ROOT / "scripts/launch_marine_cpu.sh"), *self.args],
                                 capture_output=True, text=True, encoding="utf-8", env=env, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)
        output = json.loads(process.stdout)
        self.assertTrue(output["dry_run"])
        deadline = int(datetime.fromisoformat(output["termination_time"]).timestamp())
        process = subprocess.run([
            bash, str(ROOT / "scripts/marine_cpu_startup.sh"), "--validate",
            BUCKET + "code/marine.tar.gz#123", "a" * 64,
            BUCKET + "marine100/audit.sources.json#456",
            BUCKET + "marine100/prepared/job1", BUCKET + "marine100/cpu/job1",
            str(deadline), str(int(datetime.now(timezone.utc).timestamp())),
        ], capture_output=True, text=True, encoding="utf-8", env=env, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == "__main__":
    unittest.main()
