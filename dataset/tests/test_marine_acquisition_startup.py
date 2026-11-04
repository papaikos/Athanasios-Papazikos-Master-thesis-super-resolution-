"""Run acquisition bootstrap input/archive gates without cloud calls."""

import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/marine_acquisition_startup.sh"
BUCKET = "gs://my-rgb-sr-pilot-bucket"


def embedded(name):
    return SCRIPT.read_text().split("<<'" + name + "'\n", 1)[1].split("\n" + name, 1)[0]


def archive(path, entries):
    with tarfile.open(path, "w:gz") as output:
        for name, body in entries.items():
            member = tarfile.TarInfo(name)
            payload = body.encode()
            member.size = len(payload)
            output.addfile(member, io.BytesIO(payload))


class AcquisitionStartupTests(unittest.TestCase):
    def metadata(self, **changes):
        values = dict(
            code=BUCKET + "/code/run.tar.gz",
            code_sha="a" * 64,
            state=BUCKET + "/marine100/handoff/state.tar.gz#123",
            state_sha="b" * 64,
            status=BUCKET + "/marine100/acquisition-runs/stage02",
            deadline="43200",
            now="0",
        )
        values.update(changes)
        return subprocess.run(
            [sys.executable, "-c", embedded("ACQUISITION_METADATA"), *values.values()],
            capture_output=True,
            text=True,
        )

    def test_deadline_hard_limit_and_cleanup_margin(self):
        result = self.metadata()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(int(result.stdout) + 10 + 60 + 5, 43200)
        for deadline in ("599", "43201", "-2", "nan"):
            self.assertNotEqual(self.metadata(deadline=deadline).returncode, 0)

    def test_bucket_archive_hash_and_separate_status(self):
        for changes in (
            {"code_sha": "bad"},
            {"state_sha": ""},
            {"code": "gs://other/run.tar.gz"},
            {"state": BUCKET + "/state.json"},
            {"status": BUCKET + "/marine100/control/v1"},
        ):
            self.assertNotEqual(self.metadata(**changes).returncode, 0)

    def run_extract(self, root, extra=None, task_digest=None):
        digest = "c" * 64
        entries = {
            "inventory.json": json.dumps({"sha256": digest}),
            "tasks.json": json.dumps(
                {
                    "manifest_sha256": task_digest or digest,
                    "tasks": {
                        "source:native": {"task_id": "preserve-me", "state": "RUNNING"}
                    },
                }
            ),
            "snapshots/" + digest + ".json": json.dumps({"sha256": digest}),
        }
        entries.update(extra or {})
        archive(root / "state.tar.gz", entries)
        archive(root / "code.tar.gz", {"src/__init__.py": ""})
        return subprocess.run(
            [sys.executable, "-c", embedded("ACQUISITION_ARCHIVES"), str(root)],
            capture_output=True,
            text=True,
        )

    def test_restores_original_task_ids_and_no_old_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_extract(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            tasks = json.loads((root / "state/tasks.json").read_text())
            self.assertEqual(tasks["tasks"]["source:native"]["task_id"], "preserve-me")
            self.assertFalse((root / "state/loop-state.json").exists())
            # A restart must not roll local current task state back to its bundle.
            self.assertNotEqual(self.run_extract(root).returncode, 0)

    def test_rejects_old_control_traversal_and_unmatched_inventory(self):
        for extra, digest in (
            ({"loop-state.json": "{}"}, None),
            ({"../escaped.json": "{}"}, None),
            ({}, "d" * 64),
            ({}, "../outside"),
        ):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                result = self.run_extract(root, extra, digest)
                self.assertNotEqual(result.returncode, 0, result.stderr)
                self.assertFalse((root / "escaped.json").exists())

    def test_loop_deadline_precedes_timeout(self):
        import time

        deadline = int(time.time()) + 3600
        result = subprocess.run(
            [sys.executable, "-c", embedded("LOOP_DEADLINE"), str(deadline)],
            capture_output=True,
            text=True,
        )
        from datetime import datetime

        self.assertEqual(
            datetime.fromisoformat(result.stdout.strip()).timestamp(), deadline - 180
        )

    def test_bash_syntax_and_only_targeted_retry_flag(self):
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        bash = str(git_bash) if git_bash.exists() else shutil.which("bash")
        if not bash:
            self.skipTest("Bash unavailable")
        result = subprocess.run(
            [bash, "-n", str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        source = SCRIPT.read_text()
        self.assertIn("earthengine-api==1.7.43", source)
        self.assertIn("--retry-known-reference-size-failures", source)
        self.assertNotIn("scheduler_tick", source)


if __name__ == "__main__":
    unittest.main()
