import random
import json
import contextlib
import io
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np
import torch
from src import train
from src.data import degrade
from src.data import save_pair
from src.model import select_stage
from test_model import TinyComposite


class TrainTests(unittest.TestCase):
    def test_full_training_requires_source_accounting_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "pairs.json"
            index.write_text("[]")
            with (
                patch.object(train, "read_pairs", return_value=[Path("pair.npz")]),
                patch.object(train, "load_full_model") as model,
                self.assertRaises(FileNotFoundError),
            ):
                train.main(
                    [
                        "--pairs",
                        str(index),
                        "--output",
                        str(Path(tmp) / "run"),
                        "--max-runtime-seconds",
                        "10",
                    ]
                )
            model.assert_not_called()

    def test_training_only_requires_explicit_bounded_smoke(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train.main(
                [
                    "--pairs",
                    "absent.json",
                    "--output",
                    "unused",
                    "--max-runtime-seconds",
                    "10",
                    "--training-only-smoke",
                ]
            )

    def test_normal_smoke_still_requires_validation_pairs(self):
        def pairs(index, split):
            if split == "validation":
                raise ValueError("No verified validation pairs")
            return [Path("train.npz")]

        with (
            patch.object(train, "read_pairs", side_effect=pairs),
            self.assertRaisesRegex(ValueError, "validation"),
        ):
            train.main(
                [
                    "--pairs",
                    "absent.json",
                    "--output",
                    "unused",
                    "--max-runtime-seconds",
                    "10",
                    "--smoke",
                ]
            )

    def test_training_only_records_missing_validation_and_keeps_update_cap(self):
        net = TinyComposite()
        net.hard_constraint.register_buffer("low_pass_mask", torch.ones(1))
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            root = Path(tmp)
            index = root / "pairs.json"
            index.write_text("[]")
            pairs = stack.enter_context(
                patch.object(train, "read_pairs", return_value=[Path("train.npz")])
            )
            stack.enter_context(
                patch.object(train, "load_full_model", return_value=net)
            )
            stack.enter_context(
                patch.object(
                    train.torch.cuda, "get_device_name", return_value="Tesla T4"
                )
            )
            stack.enter_context(patch.object(train.torch.cuda, "device_count", return_value=1))
            stack.enter_context(
                patch.object(
                    train.torch.cuda, "get_device_capability", return_value=(7, 5)
                )
            )
            stack.enter_context(
                patch.object(train.torch.cuda, "reset_peak_memory_stats")
            )
            stack.enter_context(
                patch.object(
                    train.importlib.metadata, "version", return_value="fixture"
                )
            )
            stack.enter_context(
                patch.object(
                    train.importlib.metadata, "distribution", return_value=MagicMock()
                )
            )
            stack.enter_context(patch.object(train.signal, "signal"))
            stack.enter_context(
                patch.object(train.time, "monotonic", side_effect=[0, 2, 2])
            )
            checkpoint = stack.enter_context(patch.object(train, "save_checkpoint"))
            evaluate = stack.enter_context(patch.object(train, "evaluate"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            train.main(
                [
                    "--pairs",
                    str(index),
                    "--output",
                    str(root / "run"),
                    "--max-runtime-seconds",
                    "1",
                    "--smoke",
                    "--training-only-smoke",
                ]
            )
            pairs.assert_called_once_with(index, "train")
            evaluate.assert_not_called()
            metadata = checkpoint.call_args.kwargs["metadata"]
            self.assertEqual((metadata["a_steps"], metadata["steps"]), (1, 100))
            self.assertTrue(metadata["engineering_only"])
            self.assertTrue(metadata["no_validation"])
            self.assertFalse((root / "run" / "released-baseline.json").exists())
            status = json.loads((root / "run" / "engineering-status.json").read_text())
            self.assertEqual(status["quality_acceptance"], "not_evaluated")

    def test_unbounded_runtime_is_rejected_before_loading_data(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train.main(
                [
                    "--pairs",
                    "absent.json",
                    "--output",
                    "unused",
                    "--max-runtime-seconds",
                    "inf",
                ]
            )

    def test_pair_index_rejects_split_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "sample.npz")
            digest = save_pair(
                path,
                np.full((4, 512, 512), 0.2, np.float32),
                np.ones((512, 512), bool),
                {
                    "site_id": "a",
                    "split": "validation",
                    "naip_id": "n",
                    "s2_id": "s",
                    "calibration_train_sites": ["b"],
                },
            )
            index = Path(tmp, "pairs.json")
            index.write_text(
                json.dumps(
                    [
                        {
                            "path": "sample.npz",
                            "site_id": "a",
                            "split": "train",
                            "sha256": digest,
                        }
                    ]
                )
            )
            with self.assertRaisesRegex(ValueError, "provenance"):
                train.read_pairs(index, "train")
            index.write_text(
                json.dumps(
                    [
                        {
                            "path": "sample.npz",
                            "site_id": "a",
                            "split": "validation",
                            "sha256": digest,
                        }
                    ]
                )
            )
            self.assertEqual(train.read_pairs(index, "validation"), [path])
            tensors = train.load_pair(path, "cpu")
            self.assertEqual(tensors[2].shape, (1, 1, 512, 512))
            with np.load(path, allow_pickle=False) as archive:
                content = {key: archive[key] for key in archive.files}
            stale = json.loads(str(content["provenance"]))
            stale.pop("gaussian_mode")
            content["provenance"] = json.dumps(stale)
            np.savez(path, **content)
            with self.assertRaisesRegex(ValueError, "degradation"):
                train.load_pair(path, "cpu")

    def test_optimizer_no_decay_and_finite_gradient_checks(self):
        net = TinyComposite()
        select_stage(net, "A")
        opt = train.optimizer_for(net, "A")
        train.schedule(opt, 0, 1000)
        self.assertLess(opt.param_groups[0]["lr"], 5e-5)
        for group in opt.param_groups:
            for param in group["params"]:
                if param.ndim == 1:
                    self.assertEqual(group["weight_decay"], 0)
        for param in net.parameters():
            if param.requires_grad:
                param.grad = torch.ones_like(param)
        self.assertTrue(train.gradient_evidence(net, "A")["nonzero_gradient"])
        net.sr_model.conv_last.weight.grad[0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            train.gradient_evidence(net, "A")

    def test_torch_degradation_matches_pair_generator(self):
        hr = np.random.default_rng(17).uniform(0, 0.5, (4, 512, 512)).astype("float32")
        expected, mask = degrade(hr, np.ones((512, 512), bool))
        actual = train.degrade_tensor(torch.from_numpy(hr)[None])[0].numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-7)

    def test_losses_ignore_invalid_pixels_and_dark_spectra(self):
        target = torch.full((1, 4, 32, 32), 0.2)
        pred = target.clone().requires_grad_()
        valid = torch.ones(1, 1, 32, 32, dtype=torch.bool)
        valid[:, :, 12, 12] = False
        target[:, :, 12, 12] = 20
        lr = train.degrade_tensor(pred.detach())
        parts = train.reconstruction_loss(
            pred, target, lr, valid, torch.ones(1, 1, 8, 8, dtype=torch.bool)
        )
        self.assertAlmostEqual(float(parts["charbonnier"].detach()), 0.001, places=6)
        self.assertAlmostEqual(float(parts["gradient"].detach()), 0, places=6)
        parts["total"].backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(float(pred.grad[:, :, 12, 12].abs().sum()), 0)
        dark = train.reconstruction_loss(
            pred * 0,
            pred.detach() * 0,
            lr * 0,
            valid,
            torch.ones(1, 1, 8, 8, dtype=torch.bool),
        )
        self.assertEqual(float(dark["spectral"].detach()), 0)

    def test_checkpoint_restores_optimizer_rng_stage_and_step(self):
        net = TinyComposite()
        opt = torch.optim.AdamW(net.parameters())
        scaler = torch.amp.GradScaler("cpu")
        scaler.scale(net.sr_model.conv_last(torch.ones(1, 4)).sum()).backward()
        scaler.step(opt)
        scaler.update()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "checkpoint.pt")
            train.save_checkpoint(
                path,
                net,
                opt,
                scaler,
                step=7,
                stage="B",
                metadata={"lock": "test"},
                monitor={"best": 0.2, "stale": 2},
            )
            expected = (random.random(), np.random.random(), torch.rand(1))
            with torch.no_grad():
                net.sr_model.conv_last.weight.zero_()
            restored_scaler = torch.amp.GradScaler("cpu", init_scale=32)
            state = train.load_checkpoint(
                path, net, opt, restored_scaler, {"lock": "test"}
            )
            self.assertEqual(restored_scaler.state_dict(), scaler.state_dict())
            self.assertEqual((state["step"], state["stage"]), (7, "B"))
            self.assertEqual(state["monitor"], {"best": 0.2, "stale": 2})
            self.assertEqual(random.random(), expected[0])
            self.assertEqual(np.random.random(), expected[1])
            torch.testing.assert_close(torch.rand(1), expected[2])
            self.assertGreater(
                float(net.sr_model.conv_last.weight.detach().abs().sum()), 0
            )
            self.assertEqual(float(opt.state[net.sr_model.conv_last.weight]["step"]), 1)
            with self.assertRaises(ValueError):
                train.load_checkpoint(path, net, opt, None, {"lock": "other"})


if __name__ == "__main__":
    unittest.main()
