"""Static safety contract; these checks do not claim a successful cloud run."""

from pathlib import Path
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/t4_startup.sh"
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.exists() else shutil.which("bash")


class StartupContractTests(unittest.TestCase):
    @unittest.skipUnless(BASH, "Bash required for GPU validation")
    def test_gpu_metadata_requires_exactly_one_selected_gpu(self):
        for expected, actual, accepted in [
            ("t4", "Tesla T4", True), ("t4", "NVIDIA Tesla T4", True),
            ("l4", "NVIDIA L4", True), ("l4", "Tesla T4", False),
            ("t4", "NVIDIA L4", False), ("l4", "NVIDIA L4\nNVIDIA L4", False),
            ("l4", "", False), ("a100", "NVIDIA A100", False),
        ]:
            with self.subTest(expected=expected, actual=actual):
                result = subprocess.run([BASH, str(SCRIPT), "--validate-gpu", expected, actual], capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, accepted, result.stderr)

    @unittest.skipUnless(BASH, "Bash required for config validation")
    def test_exactly_one_dataset_source(self):
        for pairs, raw, expected in [
            ("gs://b/p", "", 0),
            ("", "gs://b/r", 0),
            ("", "", 2),
            ("gs://b/p", "gs://b/r", 2),
        ]:
            with self.subTest(pairs=pairs, raw=raw):
                result = subprocess.run(
                    [BASH, str(SCRIPT), "--validate-inputs", pairs, raw],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, expected, result.stderr)

    @unittest.skipUnless(BASH, "Bash required for runtime validation")
    def test_full_runtime_margin_and_hard_limits(self):
        # Actual startup validator must reject expired, malformed, underfunded and unbounded runs.
        validator = (
            SCRIPT.read_text()
            .split("<<'RUNTIME_VALIDATOR'\n", 1)[1]
            .split("\nRUNTIME_VALIDATOR", 1)[0]
        )
        for training, deadline, now, accepted in [
            ("32400", "36000", "0", True),
            ("3600", "6000", "300", True),
            ("3600", "6000", "361", False),
            ("32401", "40000", "0", False),
            ("3600", "43201", "0", False),
            ("3600", "6000", "6000", False),
            ("nan", "6000", "0", False),
        ]:
            with self.subTest(training=training, deadline=deadline, now=now):
                result = subprocess.run(
                    [sys.executable, "-c", validator, "full", training, deadline, now],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode == 0, accepted, result.stderr)
                if accepted:
                    worker = int(result.stdout)
                    self.assertLess(worker + 20 + 60 + 5, int(deadline) - int(now))

    def test_full_run_never_uses_smoke_bypass_or_pilot_resume(self):
        source = SCRIPT.read_text()
        full = source.split("# Recheck after setup;", 1)[1].split("  else", 1)[0]
        self.assertIn("--steps 2000", full)
        self.assertIn("--accumulation 8", full)
        self.assertNotIn("--smoke", full)
        self.assertNotIn("--amp", full)
        self.assertNotIn("--resume", full)
        self.assertIn("Full training requires normal validation", source)

    @unittest.skipUnless(BASH, "Bash required for prepared source validation")
    def test_prepared_prefix_is_exclusive(self):
        for pairs, raw, prepared, accepted in [
            ("", "", "gs://bucket/prepared", True),
            ("gs://bucket/pairs", "", "gs://bucket/prepared", False),
            ("", "gs://bucket/raw", "gs://bucket/prepared", False),
        ]:
            result = subprocess.run(
                [BASH, str(SCRIPT), "--validate-inputs", pairs, raw, prepared],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode == 0, accepted, result.stderr)
            if accepted:
                self.assertEqual(result.stdout.strip(), "prepared")

    def test_raw_manifest_rejects_incomplete_or_out_of_scope_exports(self):
        source = SCRIPT.read_text(encoding="utf-8")
        validator = source.split("<<'RAW_VALIDATOR'\n", 1)[1].split(
            "\nRAW_VALIDATOR", 1
        )[0]
        bucket = "my-rgb-sr-pilot-bucket"
        valid = {
            "status": "export_complete",
            "bucket": bucket,
            "prefix": "pilot/raw",
            "records": [{"id": "harbour"}],
            "tasks": [
                {
                    "site_id": "harbour",
                    "name": name,
                    "state": "COMPLETED",
                    "gcs_uri": f"gs://{bucket}/pilot/raw/harbour_{name}.tif",
                }
                for name in ("naip_hr", "naip_lr", "s2_lr")
            ],
        }
        cases = [("valid", valid, True)]
        for name, modify in [
            ("pending", lambda m: m["tasks"][0].update(state="RUNNING")),
            ("wrong_bucket", lambda m: m.update(bucket="unrelated-sar-bucket")),
            (
                "wrong_uri",
                lambda m: m["tasks"][0].update(gcs_uri="gs://other/data.tif"),
            ),
            ("traversal", lambda m: m["tasks"][0].update(site_id="../harbour")),
            ("missing", lambda m: m["tasks"].pop()),
            ("duplicate", lambda m: m["tasks"].append(m["tasks"][0])),
            ("oversized", lambda m: m.update(tasks=m["tasks"] * 7)),
        ]:
            invalid = copy.deepcopy(valid)
            modify(invalid)
            cases.append((name, invalid, False))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            for name, manifest, accepted in cases:
                with self.subTest(case=name):
                    path.write_text(json.dumps(manifest), encoding="utf-8")
                    result = subprocess.run(
                        [sys.executable, "-c", validator, str(path)],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode == 0, accepted, result.stderr)
                    if accepted:
                        self.assertEqual(len(result.stdout.splitlines()), 3)
                    else:
                        self.assertEqual(result.stdout, "")

    def test_preparation_and_pair_gate_precede_gpu_install(self):
        source = SCRIPT.read_text(encoding="utf-8")
        gpu_install = source.index("bash scripts/setup_gpu.sh")
        self.assertLess(source.index("python -m src.export_ee prepare"), gpu_install)
        self.assertLess(source.index("<<'PAIR_GATE'"), gpu_install)
        self.assertLess(
            source.index('"$CHECKPOINT_URI/data/pairs.tar.gz"'), gpu_install
        )

    def test_checkpoint_gate_precedes_setup_and_training(self):
        source = SCRIPT.read_text(encoding="utf-8")
        gate = source.index('cmp "$ROOT/write-probe.txt" "$ROOT/read-probe.txt"')
        self.assertLess(gate, source.index("bash scripts/setup_gpu.sh"))
        self.assertLess(gate, source.index("python -m src.train"))
        self.assertIn('kill -TERM "$worker_pid"', source)
        self.assertIn("trap finish EXIT", source)
        self.assertIn("while sleep 60", source)

    def test_bounded_full_model_smoke_from_cloud_archives(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("--kill-after=20s 3200s", source)
        self.assertIn("--kill-after=5s 60s", source)
        self.assertLess(3200 + 20 + 60 + 5, 3300)
        self.assertIn(
            "--smoke --download --max-runtime-seconds 1800 --accumulation 1", source
        )
        self.assertIn('gcloud storage cp "$pairs_uri"', source)
        self.assertIn('gcloud storage cp "$code_uri"', source)
        self.assertIn("uv==0.8.22", source)
        self.assertIn("python install 3.11.13", source)
        self.assertIn('validate_gpu "$expected_gpu" "$gpu"', source)
        self.assertIn("common-cu129-ubuntu-2204-nvidia-580-v20260909", source)
        self.assertNotIn("gcloud compute instances create", source)

    @unittest.skipUnless(BASH, "Bash required for syntax validation")
    def test_bash_syntax_and_unknown_option(self):
        checked = subprocess.run(
            [BASH, "-n", str(SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(checked.returncode, 0, checked.stderr)
        rejected = subprocess.run(
            [BASH, str(SCRIPT), "--unknown"], capture_output=True, text=True
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("Unsupported startup argument", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
