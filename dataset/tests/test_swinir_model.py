import tempfile
import unittest
from pathlib import Path
from src.swinir_model import verify_asset


class AssetIntegrity(unittest.TestCase):
    def test_tampered_asset_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "weights.pth"
            p.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                verify_asset(p, "0" * 64)

    def test_valid_asset(self):
        import hashlib

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "weights.pth"
            p.write_bytes(b"ok")
            self.assertEqual(verify_asset(p, hashlib.sha256(b"ok").hexdigest()), p)


if __name__ == "__main__":
    unittest.main()


class OfficialArchitecture(unittest.TestCase):
    def test_strict_pretrained_full_model_forward_backward(self):
        try:
            import torch
            import timm  # noqa: F401
        except ImportError:
            self.skipTest("Optional local torch/timm runtime unavailable")
        from src.swinir_model import load_swinir

        torch.set_num_threads(4)
        model, _ = load_swinir("assets/swinir", "configs/swinir-lock.json")
        self.assertGreater(sum(p.numel() for p in model.parameters()), 11000000)
        model.train()
        output = model(torch.rand(1, 3, 16, 16))
        self.assertEqual(tuple(output.shape), (1, 3, 64, 64))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )


class CheckpointDurability(unittest.TestCase):
    def test_atomic_roundtrip_and_temporary_cleanup(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional local torch runtime unavailable")
        from src.swinir_train import checkpoint_roundtrip
        import json

        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_roundtrip(
                root, model, optimizer, scaler, 0, float("inf"), {"test": True}
            )
            self.assertTrue(
                json.loads((root / "checkpoint-roundtrip.json").read_text())["passed"]
            )
            self.assertFalse((root / ".checkpoint-roundtrip.tmp").exists())
