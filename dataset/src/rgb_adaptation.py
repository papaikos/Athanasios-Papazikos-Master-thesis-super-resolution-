"""Explicit RGB initialization from a verified RGBN backbone state dictionary.

The caller verifies the original checkpoint bytes and supplies their SHA-256.
This utility neither downloads a model nor configures its architecture or trainer.
"""

import re

import torch


def adapt_rgbn_state_dict(state_dict, *, source_sha256):
    """Drop only NIR input/output weights; return independent tensors and provenance.

    Keys must be the unprefixed backbone keys. RGB is assumed to occupy the first
    three channels in the verified source architecture. No NIR is synthesized.
    The returned hash records caller-provided provenance, not verification here.
    """
    if not isinstance(source_sha256, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", source_sha256
    ):
        raise ValueError("A caller-verified source SHA-256 is required")
    required = ("conv_first.weight", "conv_last.weight", "conv_last.bias")
    if any(key not in state_dict for key in required):
        raise ValueError("Missing RGBN backbone convolution keys")
    if any(not isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValueError("Every state dictionary value must be a tensor")
    first, last, bias = (state_dict[key] for key in required)
    if (
        first.ndim != 4
        or last.ndim != 4
        or any(size <= 0 for size in first.shape)
        or any(size <= 0 for size in last.shape)
        or first.shape[1] != 4
        or last.shape[0] != 4
        or bias.shape != (4,)
    ):
        raise ValueError("Expected nonempty 4-channel RGBN Conv2d weights and bias")
    result = {key: value.detach().clone() for key, value in state_dict.items()}
    result["conv_first.weight"] = first[:, :3].detach().clone()
    result["conv_last.weight"] = last[:3].detach().clone()
    result["conv_last.bias"] = bias[:3].detach().clone()
    metadata = {
        "adaptation": "rgbn_to_rgb_channel_slice_v1",
        "source_sha256": source_sha256,
        "source_channels": ["R", "G", "B", "NIR"],
        "target_channels": ["R", "G", "B"],
        "changed_keys": list(required),
    }
    return result, metadata
