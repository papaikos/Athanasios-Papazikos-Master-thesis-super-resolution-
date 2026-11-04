import unittest

import torch
from torch.nn import functional as F

from src.rgb_adaptation import adapt_rgbn_state_dict


def source_state():
    return {
        "conv_first.weight": torch.arange(72.0).reshape(2, 4, 3, 3),
        "conv_first.bias": torch.tensor([1.0, 2.0]),
        "conv_last.weight": torch.arange(72.0).reshape(4, 2, 3, 3),
        "conv_last.bias": torch.tensor([3.0, 4.0, 5.0, 6.0]),
        "body.weight": torch.tensor([7.0, 8.0]),
    }


class RGBAdaptationTests(unittest.TestCase):
    def test_rgb_convolutions_match_original_with_zero_nir(self):
        source = source_state()
        rgb, metadata = adapt_rgbn_state_dict(source, source_sha256="a" * 64)
        image = torch.arange(75.0).reshape(1, 3, 5, 5)
        rgbn = torch.cat((image, torch.zeros(1, 1, 5, 5)), dim=1)
        original = F.conv2d(rgbn, source["conv_first.weight"], source["conv_first.bias"])
        adapted = F.conv2d(image, rgb["conv_first.weight"], rgb["conv_first.bias"])
        torch.testing.assert_close(adapted, original, rtol=0, atol=0)
        expected = F.conv2d(original, source["conv_last.weight"], source["conv_last.bias"])
        actual = F.conv2d(adapted, rgb["conv_last.weight"], rgb["conv_last.bias"])
        torch.testing.assert_close(actual, expected[:, :3], rtol=0, atol=0)
        self.assertEqual(metadata["source_sha256"], "a" * 64)
        self.assertEqual(metadata["source_channels"], ["R", "G", "B", "NIR"])
        self.assertEqual(metadata["target_channels"], ["R", "G", "B"])

    def test_every_tensor_is_independent_and_other_values_unchanged(self):
        source = source_state()
        rgb, _ = adapt_rgbn_state_dict(source, source_sha256="b" * 64)
        self.assertEqual(set(rgb), set(source))
        for key in ("body.weight", "conv_first.bias"):
            torch.testing.assert_close(rgb[key], source[key], rtol=0, atol=0)
        for key in rgb:
            before = source[key].clone()
            rgb[key].zero_()
            torch.testing.assert_close(source[key], before, rtol=0, atol=0)

    def test_rejects_missing_and_wrong_shapes(self):
        for key in ("conv_first.weight", "conv_last.weight", "conv_last.bias"):
            state = source_state()
            del state[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                adapt_rgbn_state_dict(state, source_sha256="c" * 64)
        for key, shape in (
            ("conv_first.weight", (2, 3, 3, 3)),
            ("conv_first.weight", (2, 4, 3)),
            ("conv_first.weight", (0, 4, 3, 3)),
            ("conv_last.weight", (3, 2, 3, 3)),
            ("conv_last.weight", (4, 2, 3)),
            ("conv_last.bias", (3,)),
            ("conv_last.bias", (4, 1)),
        ):
            state = source_state()
            state[key] = torch.zeros(shape)
            with self.subTest(key=key, shape=shape), self.assertRaises(ValueError):
                adapt_rgbn_state_dict(state, source_sha256="c" * 64)

    def test_rejects_non_tensor_and_invalid_provenance(self):
        state = source_state()
        state["body.weight"] = "invalid"
        with self.assertRaises(ValueError):
            adapt_rgbn_state_dict(state, source_sha256="d" * 64)
        for digest in (None, "", "g" * 64, "a" * 63):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                adapt_rgbn_state_dict(source_state(), source_sha256=digest)
