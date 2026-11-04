import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src import diagnose_t4 as diagnostic


class DiagnosticTests(unittest.TestCase):
    def test_checkpoint_snapshot_is_independent_of_later_source_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "live.pt"
            source.write_bytes(b"original")
            frozen = Path(tmp) / "snapshot.pt"
            digest = diagnostic.snapshot_checkpoint(source, frozen)
            source.write_bytes(b"later")
            self.assertEqual(frozen.read_bytes(), b"original")
            self.assertEqual(diagnostic.file_hash(frozen), digest)

    def test_precision_observer_tracks_actual_fftn_and_restores_functions(self):
        class ScanFixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.selective_scan = lambda x: x

            def forward(self, x):
                return self.selective_scan(x)

        net = torch.nn.Module()
        net.sr_model = ScanFixture()
        original_fft = torch.fft.fftn
        with diagnostic.observe_precision(net) as evidence:
            x = net.sr_model(torch.ones(1, 4, 8, 8))
            torch.fft.ifft2(torch.fft.fftn(x, dim=(-2, -1)))
        self.assertEqual(evidence["fft_dtypes"], ["torch.float32"])
        self.assertIs(torch.fft.fftn, original_fft)

    def test_amp_comparison_rejects_material_gradient_drift(self):
        policy = json.loads(diagnostic.DEFAULT_POLICY.read_text())
        reference = {
            "output": torch.ones(1, 4, 8, 8),
            "gradients": {"head": torch.ones(5)},
        }
        close = {
            "output": reference["output"] + 0.001,
            "gradients": {"head": torch.ones(5) * 1.01},
        }
        self.assertTrue(diagnostic.compare_amp(reference, close, policy)["passed"])
        close["gradients"]["head"] *= 2
        self.assertFalse(diagnostic.compare_amp(reference, close, policy)["passed"])

    def test_device_gate_rejects_hidden_extra_or_wrong_gpu(self):
        diagnostic.validate_device(1, "Tesla T4", (7, 5), ["Tesla T4"])
        for values in [
            (2, "Tesla T4", (7, 5), ["Tesla T4"]),
            (1, "Tesla T4", (8, 0), ["Tesla T4"]),
            (1, "Tesla T4", (7, 5), ["Tesla T4", "Tesla T4"]),
            (1, "A100", (8, 0), ["A100"]),
        ]:
            with self.assertRaises(ValueError):
                diagnostic.validate_device(*values)

    def test_numeric_comparison_fails_nonfinite_shape_and_difference(self):
        reference = torch.ones(2, 3)
        self.assertTrue(
            diagnostic.compare_tensor(reference, reference, 0.001, 0.001)["passed"]
        )
        self.assertFalse(
            diagnostic.compare_tensor(reference, reference + 0.1, 0.001, 0.001)[
                "passed"
            ]
        )
        self.assertFalse(
            diagnostic.compare_tensor(
                reference, reference * float("nan"), 0.001, 0.001
            )["passed"]
        )
        self.assertFalse(
            diagnostic.compare_tensor(reference, torch.ones(6), 0.001, 0.001)["passed"]
        )

    def test_resume_tree_checks_counters_rng_sampler_and_float_states(self):
        state = {
            "optimizer": {"step": torch.tensor(101.0), "moment": torch.tensor([0.1])},
            "rng": {
                "torch": torch.tensor([1, 2], dtype=torch.uint8),
                "numpy": np.array([3]),
            },
            "sampled_pairs": ["one.npz"],
            "scaler": {"scale": 128.0},
        }
        self.assertTrue(diagnostic.compare_tree(state, state, 1e-5, 1e-7)["passed"])
        altered = dict(state, sampled_pairs=["two.npz"])
        self.assertFalse(diagnostic.compare_tree(state, altered, 1e-5, 1e-7)["passed"])
        altered = dict(
            state,
            optimizer={"step": torch.tensor(102.0), "moment": torch.tensor([0.1])},
        )
        self.assertFalse(diagnostic.compare_tree(state, altered, 1e-5, 1e-7)["passed"])
        altered = dict(
            state,
            rng={
                "torch": torch.tensor([1, 3], dtype=torch.uint8),
                "numpy": np.array([3]),
            },
        )
        self.assertFalse(diagnostic.compare_tree(state, altered, 1e-5, 1e-7)["passed"])

    def test_frozen_parameter_hash_detects_mutation_only_in_frozen_scope(self):
        net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        net[0].requires_grad_(False)
        initial = diagnostic.frozen_hash(net)
        with torch.no_grad():
            net[1].weight.add_(1)
        self.assertEqual(diagnostic.frozen_hash(net), initial)
        with torch.no_grad():
            net[0].weight.add_(1)
        self.assertNotEqual(diagnostic.frozen_hash(net), initial)

    def test_timeout_is_inconclusive_and_processes_are_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = diagnostic.run_process(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                0.05,
                Path(tmp) / "timeout.log",
            )
            self.assertEqual(result["status"], "inconclusive")
            ids = []
            for index in range(2):
                log = Path(tmp) / f"process-{index}.log"
                result = diagnostic.run_process(
                    [sys.executable, "-c", "import os; print(os.getpid())"], 10, log
                )
                self.assertEqual(result["status"], "completed")
                ids.append(int(log.read_text()))
            self.assertNotEqual(*ids)

    def test_policy_has_pinned_reference_and_registered_tolerances(self):
        policy = json.loads(diagnostic.DEFAULT_POLICY.read_text())
        self.assertEqual(policy["reference"]["version"], "2.2.4")
        self.assertEqual(len(policy["reference"]["interface_sha256"]), 64)
        self.assertGreater(policy["amp"]["output"]["atol"], 0)
        self.assertIn("rationale", policy)


if __name__ == "__main__":
    unittest.main()
