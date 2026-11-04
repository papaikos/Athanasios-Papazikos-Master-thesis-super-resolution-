import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone


ROOT = Path(__file__).resolve().parents[1]
GIT_BASH = Path(r"C:\Program Files\Git\bin\bash.exe")
BASH = str(GIT_BASH) if GIT_BASH.exists() else shutil.which("bash")


@unittest.skipUnless(BASH, "Bash required for launcher contract tests")
class LaunchContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.credit = Path(self.temp.name) / "credit.json"
        now = datetime.now(timezone.utc)
        self.request = dict(
            available_credit_eur=200,
            reserved_eur=50,
            committed_eur=0,
            unbilled_eur=0,
            other_anticipated_eur=0,
            whole_vm_upper_eur_per_hour=10,
            snapshot_at=now.isoformat(),
            credit_expires_at=(now + timedelta(days=1)).isoformat(),
            stage_cap_seconds=7200,
            user_time_cap_seconds=7200,
        )

    def run_builder(self, extra=(), prepared=True, **changes):
        self.request.update(changes)
        self.credit.write_text(json.dumps(self.request), encoding="utf-8")
        args = [
            "--credit-request",
            str(self.credit),
            "--project",
            "pilot-project",
            "--zone",
            "europe-west4-a",
            "--name",
            "rgbn-pilot",
            "--service-account",
            "pilot-runner@pilot-project.iam.gserviceaccount.com",
            "--image-project",
            "pilot-project",
            "--image",
            "prepared-t4-image-v1",
            "--checkpoint-uri",
            "gs://pilot-checkpoints/rgbn/run-001",
        ]
        if "full" in extra and prepared:
            index = Path(self.temp.name) / "pairs.json"
            prefix = "gs://my-rgb-sr-pilot-bucket/prepared/test"
            index.write_text(
                json.dumps(
                    [
                        {
                            "site_id": name,
                            "split": split,
                            "path": name + ".npz",
                            "sha256": "a" * 64,
                            "bytes": 10,
                            "uri": prefix + "/" + name + ".npz",
                        }
                        for name, split in [("train", "train"), ("val", "validation")]
                    ]
                )
            )
            index.with_name("source-usage.json").write_text(
                json.dumps(
                    {
                        "pairs_index_sha256": hashlib.sha256(
                            index.read_bytes()
                        ).hexdigest(),
                        "source_manifest_sha256": "b" * 64,
                        "bytes": 100_000_000_000,
                        "sources": [
                            {
                                "uri": "gs://source/native.tif",
                                "generation": "123",
                                "bytes": 100_000_000_000,
                            }
                        ],
                    }
                )
            )
            args += ["--prepared-prefix", prefix, "--prepared-index", str(index)]
        env = dict(os.environ, PYTHON=sys.executable, PYTHONDONTWRITEBYTECODE="1")
        return subprocess.run(
            [BASH, str(ROOT / "scripts/launch_t4_pilot.sh"), *args, *extra],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            timeout=20,
        )

    def test_bounded_dry_run_contract(self):
        process = self.run_builder()
        self.assertEqual(process.returncode, 0, process.stderr)
        output = json.loads(process.stdout)
        self.assertTrue(output["dry_run"])
        command = output["command"]
        self.assertEqual(
            command[:5], ["gcloud", "compute", "instances", "create", "rgbn-pilot"]
        )
        for flag in (
            "--machine-type=n1-standard-4",
            "--provisioning-model=SPOT",
            "--accelerator=count=1,type=nvidia-tesla-t4",
            "--instance-termination-action=DELETE",
            "--maintenance-policy=TERMINATE",
            "--boot-disk-auto-delete",
            "--boot-disk-size=80GB",
            "--no-restart-on-failure",
            "--scopes=https://www.googleapis.com/auth/devstorage.read_write",
        ):
            self.assertIn(flag, command)
        self.assertNotIn("cloud-platform", process.stdout)
        self.assertTrue(output["preflight_commands"])
        self.assert_deadline(output, 3600)

    def assert_deadline(self, output, seconds):
        generated = datetime.fromisoformat(output["generated_at"])
        deadline = datetime.fromisoformat(output["termination_time"])
        self.assertEqual((deadline - generated).total_seconds(), seconds)
        self.assertIn(
            "--termination-time=" + output["termination_time"], output["command"]
        )
        self.assertFalse(
            any(flag.startswith("--max-run-duration") for flag in output["command"])
        )

    def test_full_run_credit_bounded_with_setup_reserve(self):
        result = self.run_builder(
            ["--run-mode", "full"],
            stage_cap_seconds=43200,
            user_time_cap_seconds=43200,
            whole_vm_upper_eur_per_hour=1,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assert_deadline(output, 36000)
        self.assertEqual(output["training_seconds"], 32400)
        metadata = next(x for x in output["command"] if x.startswith("--metadata="))
        self.assertIn("rgb-run-mode=full", metadata)
        self.assertIn("rgb-training-seconds=32400", metadata)
        self.assertGreaterEqual(
            output["pilot_seconds"] - output["training_seconds"], 2400
        )
        result = self.run_builder(["--run-mode", "full"], user_time_cap_seconds=9000)
        output = json.loads(result.stdout)
        self.assert_deadline(output, 9000)
        self.assertEqual(output["training_seconds"], 6600)

    def test_full_requires_prepared_source_manifest(self):
        result = self.run_builder(["--run-mode", "full"], prepared=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --prepared-prefix", result.stderr)

    def test_l4_uses_integrated_gpu_and_supported_disk_with_same_guards(self):
        result = self.run_builder(["--gpu", "l4", "--run-mode", "full"])
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        command = output["command"]
        self.assertIn("--machine-type=g2-standard-4", command)
        self.assertIn("--boot-disk-type=pd-balanced", command)
        self.assertIn("--provisioning-model=SPOT", command)
        self.assertIn("--instance-termination-action=DELETE", command)
        self.assertFalse(any(x.startswith("--accelerator=") for x in command))
        metadata = next(x for x in command if x.startswith("--metadata="))
        self.assertIn("rgb-expected-gpu=l4", metadata)
        self.assertIn("rgb-expected-image=projects/pilot-project/global/images/prepared-t4-image-v1", metadata)
        self.assertNotEqual(self.run_builder(["--gpu", "l4", "--run-mode", "full"], prepared=False).returncode, 0)
        self.assertNotEqual(self.run_builder(["--gpu", "l4"], snapshot_at="2000-01-01T00:00:00Z").returncode, 0)

    def test_l4_rejects_dlvm_boot_image_and_unknown_gpu(self):
        result = self.run_builder(["--gpu", "l4", "--image-project", "deeplearning-platform-release"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not support Deep Learning VM", result.stderr)
        self.assertNotEqual(self.run_builder(["--gpu", "a100"]).returncode, 0)

    def test_explicit_full_disk_size_is_bounded(self):
        result = self.run_builder(["--run-mode", "full", "--boot-disk-gb", "120"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--boot-disk-size=120GB", json.loads(result.stdout)["command"])
        for mode, size in [("full", "79"), ("full", "201"), ("smoke", "120")]:
            self.assertNotEqual(
                self.run_builder(
                    ["--run-mode", mode, "--boot-disk-gb", size]
                ).returncode,
                0,
            )

    def test_full_run_rejects_insufficient_and_excessive_runtime(self):
        result = self.run_builder(["--run-mode", "full"], user_time_cap_seconds=5999)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        for mode, cap in [("full", 43201), ("smoke", 7200)]:
            result = self.run_builder(
                ["--run-mode", mode, "--runtime-seconds", str(cap)]
            )
            self.assertNotEqual(result.returncode, 0)

    def test_recomputes_smaller_runtime(self):
        process = self.run_builder(user_time_cap_seconds=123)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assert_deadline(json.loads(process.stdout), 123)

    def test_existing_compute_default_service_account(self):
        email = "123456789012-compute@developer.gserviceaccount.com"
        process = self.run_builder(["--service-account", email])
        self.assertEqual(process.returncode, 0, process.stderr)
        output = json.loads(process.stdout)
        self.assertIn("--service-account=" + email, output["command"])
        self.assertFalse(
            any(
                "add-iam-policy-binding" in command
                for command in output["preflight_commands"]
            )
        )

    def test_stale_and_exhausted_credit_denied(self):
        for changes in (
            {"snapshot_at": "2000-01-01T00:00:00Z"},
            {"committed_eur": 200},
        ):
            with self.subTest(changes=changes):
                process = self.run_builder(**changes)
                self.assertNotEqual(process.returncode, 0)
                self.assertNotIn("gcloud compute instances create", process.stdout)

    def test_injection_bad_zone_and_service_account_rejected(self):
        for flag, value in (
            ("--name", "bad;touch injected"),
            ("--project", "--quiet"),
            ("--zone", "us-central1-a"),
            ("--service-account", "default"),
            ("--service-account", "abc-compute@developer.gserviceaccount.com"),
            ("--checkpoint-uri", "https://other/bucket"),
            ("--image", "image,extra=bad"),
        ):
            with self.subTest(flag=flag):
                self.assertNotEqual(self.run_builder([flag, value]).returncode, 0)

    def test_no_launch_or_arbitrary_flag_passthrough(self):
        for flag in (
            "--launch",
            "--execute",
            "--scopes=cloud-platform",
            "--boot-disk-size=100",
        ):
            self.assertNotEqual(self.run_builder([flag]).returncode, 0)

    def test_guard_result_cannot_replace_credit_request(self):
        self.request = {
            "allowed_seconds": 3600,
            "spend_allowance_eur": 10,
            "reasons": [],
        }
        self.assertNotEqual(self.run_builder().returncode, 0)

    def test_runtime_below_cloud_minimum_denied(self):
        self.assertNotEqual(self.run_builder(user_time_cap_seconds=29).returncode, 0)


if __name__ == "__main__":
    unittest.main()
