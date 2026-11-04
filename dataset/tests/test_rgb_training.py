import json
import tempfile
import unittest
from pathlib import Path
import torch
import numpy as np
from torch import nn
from src import model, train


class RGBTrainingTests(unittest.TestCase):
    def test_rgb_backbone_adaptation_preserves_body_and_fourier(self):
        net = nn.Module()
        net.sr_model = nn.Module()
        net.sr_model.conv_first = nn.Conv2d(4, 6, 3, padding=1)
        net.sr_model.conv_last = nn.Conv2d(6, 4, 3, padding=1)
        net.sr_model.body = nn.Linear(6, 6)
        net.hard_constraint = nn.Module()
        net.hard_constraint.low_pass_mask = torch.rand(512, 512)
        body = net.sr_model.body
        mask = net.hard_constraint.low_pass_mask.clone()
        first = net.sr_model.conv_first.weight.clone()
        last = net.sr_model.conv_last.weight.clone()
        model.adapt_model_rgb(net, "a" * 64)
        self.assertIs(net.sr_model.body, body)
        torch.testing.assert_close(net.sr_model.conv_first.weight, first[:, :3])
        torch.testing.assert_close(net.sr_model.conv_last.weight, last[:3])
        torch.testing.assert_close(net.hard_constraint.low_pass_mask, mask)
        x = torch.rand(1, 3, 128, 128)
        out = net.sr_model.conv_last(
            torch.nn.functional.interpolate(net.sr_model.conv_first(x), scale_factor=4)
        )
        model.validate_pair(x, out, channels=3)
        with self.assertRaises(ValueError):
            model.validate_pair(x, out)
        opt = torch.optim.AdamW(net.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rgb.pt"
            meta = {"channels": 3, "adaptation": net.rgb_adaptation}
            train.save_checkpoint(
                path, net, opt, None, step=1, stage="A", metadata=meta
            )
            train.load_checkpoint(path, net, opt, None, meta)
            with self.assertRaises(ValueError):
                train.load_checkpoint(path, net, opt, None, {"channels": 4})

    def test_rgb_preflight_restores_weights_after_real_backward_and_checkpoint(self):
        class Group(nn.Module):
            def __init__(self):
                super().__init__()
                self.x_proj_weight = nn.Parameter(torch.ones(1))

            def forward(self, x):
                return x * self.x_proj_weight

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Group() for _ in range(6)])
                for name in model.HEAD:
                    setattr(self, name, nn.Conv2d(3, 3, 1))

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                for name in model.HEAD:
                    x = getattr(self, name)(x)
                return torch.nn.functional.interpolate(x, scale_factor=4)

        class Constraint(nn.Module):
            def __init__(self):
                super().__init__()
                self.low_pass_mask = torch.full((512, 512), 0.5)

            def forward(self, lr, sr):
                up = torch.nn.functional.interpolate(lr, size=sr.shape[-2:])
                return torch.fft.ifft2(
                    torch.fft.fft2(up.float()) * self.low_pass_mask
                    + torch.fft.fft2(sr.float()) * (1 - self.low_pass_mask)
                ).real

        class Composite(nn.Module):
            def __init__(self):
                super().__init__()
                self.sr_model = Backbone()
                self.hard_constraint = Constraint()

            def forward(self, lr):
                return self.hard_constraint(lr, self.sr_model(lr))

        net = Composite()
        initial = {k: v.clone() for k, v in net.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            pair = Path(tmp) / "pair.npz"
            hr = torch.rand(1, 3, 512, 512)
            lr = train.degrade_tensor(hr)
            np.savez(
                pair,
                hr=hr[0].numpy(),
                lr=lr[0].numpy(),
                hr_mask=np.ones((512, 512), bool),
                lr_mask=np.ones((128, 128), bool),
                provenance=json.dumps(
                    dict(
                        lr_origin="synthetic_hr_only",
                        gaussian_sigma_hr=2,
                        gaussian_truncate=4,
                        gaussian_mode="reflect",
                    )
                ),
            )
            report = train.rgb_preflight(net, pair, tmp, "cpu", float("inf"))
            self.assertTrue(report["released_initialization_restored"])
            self.assertTrue(report["stages"][1]["nonzero_scan_gradient"])
            self.assertEqual(report["stages"][1]["output_shape"], [1, 3, 512, 512])
            self.assertEqual(report["optimizer_coverage_draws"], 0)
            for key, value in net.state_dict().items():
                torch.testing.assert_close(value, initial[key], rtol=0, atol=0)

    def test_staged_satellite_pair_loads_and_counts_verified_native_cells(self):
        from test_satellite_training_data import fixture
        from src.satellite_training_data import stage_dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, fetch = fixture(root)
            stage_dataset(manifest, root / "staged", fetch)
            index = root / "staged" / "index.json"
            paths = train.read_pairs(index, "train")
            lr, hr, hm, lm = train.load_pair(paths[0], "cpu", channels=3)
            self.assertEqual(tuple(lr.shape), (1, 3, 128, 128))
            self.assertEqual(tuple(hr.shape), (1, 3, 512, 512))
            self.assertTrue(hm.all())
            self.assertFalse(lm[0, 0, 0, 0])
            coverage = train.PairCoverage(index, paths)
            self.assertEqual(
                coverage.status(1)["accepted_consumed_train_bytes"], 7680000
            )

    def test_patience_waits_for_complete_pair_coverage(self):
        self.assertTrue(train.keep_training(3, {"all_pairs_consumed": False}))
        self.assertFalse(train.keep_training(3, {"all_pairs_consumed": True}))
        self.assertFalse(train.keep_training(3, None))
        self.assertTrue(train.keep_training(2, None))

    def test_pair_coverage_counts_only_successful_draws_and_unique_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "index.json"
            rows = [
                dict(
                    path=f"{i}.npz",
                    split="train",
                    source_id="s",
                    cell=[i, 0],
                    native_covered_bytes=10 + i,
                )
                for i in range(11)
            ]
            p.write_text(json.dumps(rows))
            paths = [p.parent / r["path"] for r in rows]
            coverage = train.PairCoverage(p, paths)
            self.assertEqual(len({coverage.choose(i) for i in range(11)}), 11)
            self.assertEqual(coverage.status(0)["accepted_consumed_train_bytes"], 0)
            self.assertEqual(coverage.status(11)["accepted_consumed_train_bytes"], 165)
            self.assertEqual(coverage.status(19)["accepted_consumed_train_bytes"], 165)
            self.assertEqual(coverage.status(19)["optimizer_draws"], 19)
            self.assertTrue(coverage.status(11)["all_pairs_consumed"])
            self.assertEqual(
                coverage.choose(17), train.PairCoverage(p, paths).choose(17)
            )
            rows[-1]["cell"] = rows[0]["cell"]
            p.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                train.PairCoverage(p, paths)


if __name__ == "__main__":
    unittest.main()
