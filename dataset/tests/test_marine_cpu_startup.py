"""Execute metadata gates without cloud access; syntax-check the startup script."""

from pathlib import Path
import shutil
import subprocess
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/marine_cpu_startup.sh"
BUCKET = "gs://my-rgb-sr-pilot-bucket"


class MarineCPUStartupTests(unittest.TestCase):
    def validate(self, **changes):
        fields = dict(
            code=BUCKET + "/code/archive.tar.gz",
            checksum="a" * 64,
            manifest=BUCKET + "/marine100/audit.sources.json#123",
            prepared=BUCKET + "/marine100/prepared/v1",
            status=BUCKET + "/marine100/cpu/v1",
            deadline="43200",
            now="0",
        )
        fields.update(changes)
        validator = (
            SCRIPT.read_text()
            .split("<<'CPU_METADATA'\n", 1)[1]
            .split("\nCPU_METADATA", 1)[0]
        )
        return subprocess.run(
            [sys.executable, "-c", validator, *fields.values()],
            capture_output=True,
            text=True,
        )

    def test_absolute_deadline_and_shutdown_margin(self):
        result = self.validate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(int(result.stdout) + 10 + 60 + 5, 43200)
        for value in ("43201", "599", "-1", "NaN"):
            self.assertNotEqual(self.validate(deadline=value).returncode, 0)

    def test_generation_pin_code_hash_and_bucket_required(self):
        for changes in (
            {"manifest": BUCKET + "/marine100/audit.json"},
            {"checksum": "not-a-hash"},
            {"code": "gs://other/code.tgz"},
            {"manifest": BUCKET + "/sar/audit.json#1"},
        ):
            self.assertNotEqual(self.validate(**changes).returncode, 0)

    def test_progress_cannot_mutate_immutable_output_prefix(self):
        for status in (
            BUCKET + "/marine100/prepared/v1",
            BUCKET + "/marine100/prepared/v1/state",
            BUCKET + "/marine100/prepared",
        ):
            self.assertNotEqual(self.validate(status=status).returncode, 0)

    def test_bash_syntax(self):
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        bash = str(git_bash) if git_bash.exists() else shutil.which("bash")
        if not bash:
            self.skipTest("Bash unavailable")
        result = subprocess.run(
            [bash, "-n", str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
