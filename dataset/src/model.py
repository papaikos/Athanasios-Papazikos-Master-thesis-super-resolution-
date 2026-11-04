"""Pinned full SEN2SR loader and progressive-unfreezing policy."""

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import urllib.request
import torch
from src.rgb_adaptation import adapt_rgbn_state_dict

DEFAULT_LOCK = Path(__file__).resolve().parents[1] / "configs/model-lock.json"
HEAD = ("norm", "conv_after_body", "conv_before_upsample", "upsample", "conv_last")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(directory, lock):
    for name, asset in lock["assets"].items():
        path = Path(directory) / name
        if not path.is_file() or file_hash(path) != asset["sha256"]:
            raise ValueError(f"SHA-256 mismatch or missing model asset: {path}")


def download_assets(directory, lock):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{lock['repository']}/resolve/{lock['revision']}/{lock['variant']}"
    for name, asset in lock["assets"].items():
        destination = directory / name
        if destination.exists():
            if file_hash(destination) != asset["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {destination}")
            continue
        temporary = directory / (name + ".partial")
        urllib.request.urlretrieve(f"{base}/{name}", temporary)
        if file_hash(temporary) != asset["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {temporary}")
        temporary.replace(destination)
    verify_assets(directory, lock)


def adapt_model_rgb(net, source_sha256):
    """Adapt only released input/output convolutions; retain the full body/constraint."""
    backbone = net.sr_model
    state, metadata = adapt_rgbn_state_dict(
        backbone.state_dict(), source_sha256=source_sha256
    )
    for name, inputs, outputs in (
        ("conv_first", 3, backbone.conv_first.out_channels),
        ("conv_last", backbone.conv_last.in_channels, 3),
    ):
        old = getattr(backbone, name)
        if not isinstance(old, torch.nn.Conv2d) or old.groups != 1:
            raise ValueError("RGB adaptation requires standard released convolutions")
        new = torch.nn.Conv2d(
            inputs,
            outputs,
            old.kernel_size,
            old.stride,
            old.padding,
            old.dilation,
            bias=old.bias is not None,
            padding_mode=old.padding_mode,
        ).to(old.weight)
        setattr(backbone, name, new)
    backbone.load_state_dict(state, strict=True)
    net.rgb_adaptation = metadata
    return net


def load_full_model(directory, device="cuda:0", lock_path=DEFAULT_LOCK, channels=4):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "Full SEN2SR pilot requires CUDA; no CPU/Lite substitute is permitted"
        )
    if channels not in (3, 4):
        raise ValueError("Only RGB or RGBN channels are supported")
    lock = json.loads(Path(lock_path).read_text())
    verify_assets(directory, lock)
    distribution = importlib.metadata.distribution("sen2sr")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != lock["sen2sr_commit"]:
        raise RuntimeError("Install SEN2SR from the exact locked git commit")
    spec = importlib.util.spec_from_file_location(
        "pinned_sen2sr_loader", Path(directory) / "load.py"
    )
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    net = loader.trainable_model(Path(directory), device=device)
    backbone = net.sr_model
    if (
        type(backbone).__name__ != "MambaSR"
        or len(backbone.layers) != 6
        or any(len(group.residual_group.blocks) != 8 for group in backbone.layers)
        or backbone.upsampler != "pixelshuffle"
    ):
        raise ValueError("The released full six-group MambaSR architecture is required")
    if channels == 3:
        adapt_model_rgb(net, lock["assets"]["model.safetensor"]["sha256"])
    net.float()
    net.hard_constraint.requires_grad_(False)
    mask = net.hard_constraint.low_pass_mask
    if mask.shape[-2:] != (512, 512):
        raise ValueError("Released Fourier mask must be 512 x 512")
    net.hard_constraint.low_pass_mask = mask.detach().to(
        device=device, dtype=torch.float32
    )
    # Keep the released composite forward and its pre-constraint clamp unchanged.
    # Only the FFT boundary is widened to FP32 under optional backbone autocast.
    original = net.hard_constraint.forward

    def constraint_fp32(lr, sr):
        with torch.autocast(device_type="cuda", enabled=False):
            return original(lr.float(), sr.float())

    net.hard_constraint.forward = constraint_fp32
    return net


def selected_modules(net, stage):
    if stage not in ("A", "B"):
        raise ValueError("Only registered stages A and B are implemented")
    modules = [getattr(net.sr_model, name) for name in HEAD]
    if stage == "B":
        if len(net.sr_model.layers) != 6:
            raise ValueError("Stage B requires all six groups")
        modules.extend(net.sr_model.layers[4:6])
    return modules


def select_stage(net, stage):
    net.requires_grad_(False)
    for module in selected_modules(net, stage):
        module.requires_grad_(True)
    return [
        name for name, parameter in net.named_parameters() if parameter.requires_grad
    ]


def training_mode(net, stage):
    # Called after any blanket train() so frozen stochastic layers stay disabled.
    net.train()
    net.eval()
    for module in selected_modules(net, stage):
        module.train()


def validate_pair(lr, hr, channels=4):
    if (
        channels not in (3, 4)
        or lr.ndim != 4
        or lr.shape[1:] != (channels, 128, 128)
        or hr.shape != (lr.shape[0], channels, 512, 512)
    ):
        raise ValueError(
            f"Expected Bx{channels}x128x128 input and Bx{channels}x512x512 output"
        )
    if not torch.isfinite(lr).all() or not torch.isfinite(hr).all():
        raise ValueError("Nonfinite image values")
