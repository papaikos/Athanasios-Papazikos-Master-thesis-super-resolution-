import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from src import model


class TinyComposite(nn.Module):
    """Structural fixture only; never a replacement for full SEN2SR."""

    def __init__(self):
        super().__init__()
        self.sr_model = nn.Module()
        self.sr_model.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5)) for _ in range(6)]
        )
        for name in (
            "conv_first",
            "conv_after_body",
            "conv_before_upsample",
            "upsample",
            "conv_last",
            "norm",
        ):
            setattr(self.sr_model, name, nn.Linear(4, 4))
        self.hard_constraint = nn.Linear(4, 4)


class ModelTests(unittest.TestCase):
    def test_stages_and_frozen_eval_survive_train(self):
        net = TinyComposite()
        model.select_stage(net, "A")
        model.training_mode(net, "A")
        self.assertFalse(net.sr_model.layers[0].training)
        self.assertTrue(net.sr_model.norm.training)
        self.assertTrue(net.sr_model.conv_after_body.weight.requires_grad)
        self.assertFalse(net.sr_model.layers[5][0].weight.requires_grad)
        model.select_stage(net, "B")
        model.training_mode(net, "B")
        self.assertTrue(net.sr_model.layers[4][0].weight.requires_grad)
        self.assertTrue(net.sr_model.layers[5].training)
        self.assertFalse(net.sr_model.layers[3].training)
        self.assertFalse(net.hard_constraint.training)
        self.assertFalse(net.hard_constraint.weight.requires_grad)

    def test_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "x").write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                model.verify_assets(tmp, {"assets": {"x": {"sha256": "0" * 64}}})

    def test_shape_contract(self):
        model.validate_pair(torch.zeros(1, 4, 128, 128), torch.zeros(1, 4, 512, 512))
        with self.assertRaises(ValueError):
            model.validate_pair(
                torch.zeros(1, 3, 128, 128), torch.zeros(1, 3, 512, 512)
            )

    def test_full_loader_rejects_cpu_substitution(self):
        with self.assertRaisesRegex(RuntimeError, "no CPU/Lite substitute"):
            model.load_full_model("unused", device="cpu")


if __name__ == "__main__":
    unittest.main()
