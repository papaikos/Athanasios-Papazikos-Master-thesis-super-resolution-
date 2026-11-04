"""Bounded full-model RGBN reconstruction pilot. No cloud resource creation."""

import argparse
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
from torch.nn import functional as F
from src.model import (
    DEFAULT_LOCK,
    file_hash,
    download_assets,
    load_full_model,
    select_stage,
    training_mode,
    validate_pair,
)


def degrade_tensor(hr):
    """data.degrade: sigma2/radius8, SciPy reflect (edge-repeat), then area4."""
    coordinate = torch.arange(-8, 9, device=hr.device, dtype=torch.float32)
    kernel = torch.exp(-coordinate.square() / 8)
    kernel = kernel / kernel.sum()
    channels = hr.shape[1]
    h, w = hr.shape[-2:]
    iy = torch.arange(-8, h + 8, device=hr.device).remainder(2 * h)
    ix = torch.arange(-8, w + 8, device=hr.device).remainder(2 * w)
    iy = torch.where(iy < h, iy, 2 * h - 1 - iy)
    ix = torch.where(ix < w, ix, 2 * w - 1 - ix)
    image = hr.float().index_select(-2, iy).index_select(-1, ix)
    image = F.conv2d(
        image, kernel.view(1, 1, 1, 17).expand(channels, 1, 1, 17), groups=channels
    )
    image = F.conv2d(
        image, kernel.view(1, 1, 17, 1).expand(channels, 1, 17, 1), groups=channels
    )
    return F.avg_pool2d(image, 4)


def masked_mean(values, mask):
    mask = mask.expand_as(values)
    return torch.where(mask, values, 0).sum() / mask.sum().clamp_min(1)


def reconstruction_loss(pred, target, clean_lr, hr_mask, lr_mask):
    pred, target, clean_lr = pred.float(), target.float(), clean_lr.float()
    hr_mask, lr_mask = hr_mask.bool(), lr_mask.bool()
    if not hr_mask.any():
        raise ValueError("Empty HR loss mask")
    char = masked_mean(torch.sqrt((pred - target).square() + 1e-6), hr_mask)
    dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]) - (
        target[:, :, :, 1:] - target[:, :, :, :-1]
    )
    dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]) - (
        target[:, :, 1:, :] - target[:, :, :-1, :]
    )
    grad = 0.5 * (
        masked_mean(dx.abs(), hr_mask[:, :, :, 1:] & hr_mask[:, :, :, :-1])
        + masked_mean(dy.abs(), hr_mask[:, :, 1:, :] & hr_mask[:, :, :-1, :])
    )
    # Erode by the blur support, including invalid image boundaries; all 4x4
    # aggregation contributors must be safe. This matches the saved LR mask.
    unsafe = F.max_pool2d(
        F.pad((~hr_mask).float(), (8, 8, 8, 8), value=1), 17, stride=1
    )
    safe_lr = F.avg_pool2d(unsafe, 4) == 0
    consistency = masked_mean(
        (degrade_tensor(pred) - clean_lr).abs(), lr_mask & safe_lr
    )
    pn, tn = pred.norm(dim=1, keepdim=True), target.norm(dim=1, keepdim=True)
    spectral_mask = hr_mask & (pn >= 0.02) & (tn >= 0.02)
    cosine = (pred * target).sum(dim=1, keepdim=True) / (pn * tn).clamp_min(1e-8)
    angle = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
    spectral = masked_mean(angle, spectral_mask)
    return {
        "total": char + 0.05 * grad + 0.1 * consistency + 0.01 * spectral,
        "charbonnier": char,
        "gradient": grad,
        "consistency": consistency,
        "spectral": spectral,
        "spectral_masked_fraction": 1 - spectral_mask.sum() / hr_mask.sum(),
    }


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path, net, optimizer, scaler, *, step, stage, metadata, monitor=None
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": 1,
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "stage": stage,
        "metadata": metadata,
        "monitor": monitor or {},
        "rng": rng_state(),
    }
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(path, net, optimizer, scaler, metadata):
    # Local trusted training artifacts only: RNG includes Python/NumPy state.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["format"] != 1 or state["metadata"] != metadata:
        raise ValueError("Checkpoint provenance/configuration differs from this run")
    net.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    return state


def read_pairs(index, split):
    index = Path(index)
    rows = json.loads(index.read_text())
    result = []
    site_splits = {}
    for row in rows:
        if "rejected" in row:
            continue
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError("Invalid geographic split")
        previous = site_splits.setdefault(row["site_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("Geographic site occurs in multiple splits")
        if row["split"] != split:
            continue
        path = Path(row["path"])
        if not path.is_absolute():
            path = index.parent / path
        if file_hash(path) != row["sha256"]:
            raise ValueError(f"Pair SHA-256 mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            provenance = json.loads(str(data["provenance"]))
        if provenance["split"] != split or provenance["site_id"] != row["site_id"]:
            raise ValueError(f"Pair/index provenance disagreement: {path}")
        if "native_covered_bytes" in row and row[
            "native_covered_bytes"
        ] != provenance.get("native_covered_bytes"):
            raise ValueError(f"Pair/index native coverage disagreement: {path}")
        if "source_id" in row and (
            row["source_id"] != provenance.get("source_id")
            or row["cell"] != provenance.get("cell")
        ):
            raise ValueError(f"Pair/index identity disagreement: {path}")
        result.append(path)
    if not result:
        raise ValueError(f"No verified {split} pairs")
    return result


class PairCoverage:
    """Deterministic no-replacement first cycle, counted after optimizer success only.

    Byte counts describe the distinct native cells represented by consumed pairs,
    not full TIFF sizes. The staging adapter verifies these values against receipts.
    """

    def __init__(self, index, paths):
        index = Path(index)
        rows = [
            r
            for r in json.loads(index.read_text())
            if r.get("split") == "train" and "rejected" not in r
        ]
        self.paths = []
        self.bytes = []
        identities = set()
        for row in rows:
            path = Path(row["path"])
            path = path if path.is_absolute() else index.parent / path
            identity = (row["source_id"], tuple(row["cell"]))
            if identity in identities or path in self.paths:
                raise ValueError("Duplicate training pair identity or path")
            identities.add(identity)
            amount = row["native_covered_bytes"]
            if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
                raise ValueError(
                    "Pair requires positive verified native coverage bytes"
                )
            self.paths.append(path)
            self.bytes.append(amount)
        if not self.paths or self.paths != list(paths):
            raise ValueError("Coverage index differs from verified training pairs")
        self.order = list(range(len(self.paths)))
        random.Random(17).shuffle(self.order)
        self.keys = self.order
        self.usage_sha256 = file_hash(index)
        self.prefix_bytes = [0]
        for i in self.order:
            self.prefix_bytes.append(self.prefix_bytes[-1] + self.bytes[i])

    def choose(self, draw):
        return self.paths[self.order[draw % len(self.order)]]

    def status(self, draws):
        if draws < 0:
            raise ValueError("Negative optimizer draws")
        consumed = min(draws, len(self.paths))
        return {
            "optimizer_draws": draws,
            "unique_pairs_consumed": consumed,
            "total_train_pairs": len(self.paths),
            "accepted_consumed_train_bytes": self.prefix_bytes[consumed],
            "accepted_available_train_bytes": self.prefix_bytes[-1],
            "all_pairs_consumed": consumed == len(self.paths),
            "accounting": "unique_native_cells_in_successful_optimizer_updates",
        }


def load_pair(path, device, channels=4):
    with np.load(path, allow_pickle=False) as data:
        provenance = json.loads(str(data["provenance"]))
        if (
            provenance["lr_origin"] != "synthetic_hr_only"
            or provenance["gaussian_sigma_hr"] != 2
            or provenance["gaussian_truncate"] != 4
            or provenance.get("gaussian_mode") != "reflect"
        ):
            raise ValueError(
                "Pair degradation differs from registered sigma2/radius8 loss"
            )
        tensors = [
            torch.from_numpy(data[name].copy()).unsqueeze(0).to(device)
            for name in ("lr", "hr", "hr_mask", "lr_mask")
        ]
    tensors[2], tensors[3] = (
        tensors[2].unsqueeze(1).bool(),
        tensors[3].unsqueeze(1).bool(),
    )
    validate_pair(tensors[0], tensors[1], channels=channels)
    if tensors[2].shape != (1, 1, 512, 512) or tensors[3].shape != (1, 1, 128, 128):
        raise ValueError("Invalid spatial mask shape")
    return tensors


def optimizer_for(net, stage):
    groups = {}
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue
        lr = (
            (
                1e-5
                if name.startswith(("sr_model.layers.4.", "sr_model.layers.5."))
                else 2e-5
            )
            if stage == "B"
            else 5e-5
        )
        no_decay = (
            param.ndim < 2
            or name.endswith(".bias")
            or getattr(param, "_no_weight_decay", False)
            or any(tag in name for tag in ("norm", "A_logs", "Ds"))
        )
        decay = 0.0 if no_decay else 1e-4
        group = groups.setdefault(
            (lr, decay),
            {"params": [], "lr": lr, "initial_lr": lr, "weight_decay": decay},
        )
        group["params"].append(param)
    return torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.999), eps=1e-8)


def schedule(optimizer, position, length):
    warmup = min(200, max(1, length // 10))
    factor = min(1.0, (position + 1) / warmup)
    if position >= warmup:
        progress = min(1.0, (position - warmup) / max(1, length - warmup))
        factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor


def gradient_evidence(net, stage):
    nonzero, scan_nonzero = False, False
    for name, param in net.named_parameters():
        if not param.requires_grad:
            if param.grad is not None:
                raise RuntimeError(f"Frozen gradient: {name}")
            continue
        if param.grad is None or not torch.isfinite(param.grad).all():
            raise RuntimeError(f"Missing/nonfinite trainable gradient: {name}")
        active = bool(param.grad.detach().abs().sum() > 0)
        nonzero |= active
        if (
            name.startswith(("sr_model.layers.4.", "sr_model.layers.5."))
            and "x_proj_weight" in name
        ):
            scan_nonzero |= active
    if not nonzero or (stage == "B" and not scan_nonzero):
        raise RuntimeError("No nonzero gradient in required head/selective-scan path")
    return {"nonzero_gradient": nonzero, "nonzero_scan_gradient": scan_nonzero}


@torch.no_grad()
def evaluate(net, paths, device, amp, deadline=float("inf"), channels=4):
    net.eval()
    results = {"model": [], "bicubic": []}
    for path in paths:
        if time.monotonic() >= deadline:
            raise TimeoutError("Validation reached the run deadline")
        lr, hr, hm, lm = load_pair(path, device, channels=channels)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            pred = net(lr)
        validate_pair(lr, pred, channels=channels)
        bicubic = F.interpolate(
            lr, size=hr.shape[-2:], mode="bicubic", align_corners=False, antialias=True
        )
        for name, image in (("model", pred), ("bicubic", bicubic)):
            results[name].append(
                float(reconstruction_loss(image, hr, lr, hm, lm)["total"])
            )
    return {name: sum(values) / len(values) for name, values in results.items()}


def keep_training(stale, pair_status):
    return stale < 3 or (
        pair_status is not None and not pair_status["all_pairs_consumed"]
    )


def rgb_preflight(net, pair, output, device, deadline):
    """Exercise actual full RGB A/B backward and resume, then restore released init.

    This is a diagnostic only: no preflight sample counts toward optimizer coverage.
    FP32 avoids introducing an unverified mixed precision path into the real run.
    """
    initial = {
        key: value.detach().cpu().clone() for key, value in net.state_dict().items()
    }
    mask = net.hard_constraint.low_pass_mask.clone()
    original_rng = rng_state()
    evidence = {
        "precision": "float32",
        "channels": 3,
        "optimizer_coverage_draws": 0,
        "stages": [],
    }
    temporary = Path(output) / "preflight-checkpoint.pt"
    try:
        lr, hr, hm, lm = load_pair(pair, device, channels=3)
        for stage in ("A", "B"):
            if time.monotonic() >= deadline:
                raise TimeoutError("RGB preflight reached bounded runtime")
            select_stage(net, stage)
            training_mode(net, stage)
            optimizer = optimizer_for(net, stage)
            optimizer.zero_grad(set_to_none=True)
            before = net.sr_model.conv_last.weight.detach().clone()
            pred = net(lr)
            validate_pair(lr, pred, channels=3)
            loss = reconstruction_loss(pred, hr, lr, hm, lm)["total"]
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite RGB preflight loss")
            loss.backward()
            gradients = gradient_evidence(net, stage)
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.requires_grad],
                1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            if torch.equal(before, net.sr_model.conv_last.weight):
                raise RuntimeError("RGB preflight parameter did not update")
            meta = {"diagnostic": "RGB_FP32_A_B", "stage": stage}
            save_checkpoint(
                temporary, net, optimizer, None, step=1, stage=stage, metadata=meta
            )
            expected = net.sr_model.conv_last.weight.detach().clone()
            with torch.no_grad():
                net.sr_model.conv_last.weight.zero_()
            restored = load_checkpoint(temporary, net, optimizer, None, meta)
            if restored["step"] != 1 or not torch.equal(
                expected, net.sr_model.conv_last.weight
            ):
                raise RuntimeError("RGB preflight checkpoint roundtrip failed")
            if not torch.equal(mask, net.hard_constraint.low_pass_mask):
                raise RuntimeError("RGB preflight Fourier mask changed")
            evidence["stages"].append(
                {
                    "stage": stage,
                    "loss": float(loss.detach()),
                    "output_shape": list(pred.shape),
                    "checkpoint_roundtrip": True,
                    **gradients,
                }
            )
            optimizer.zero_grad(set_to_none=True)
    finally:
        net.load_state_dict(initial, strict=True)
        restore_rng(original_rng)
        net.zero_grad(set_to_none=True)
        temporary.unlink(missing_ok=True)
    evidence["released_initialization_restored"] = all(
        torch.equal(value.detach().cpu(), initial[key])
        for key, value in net.state_dict().items()
    )
    if not evidence["released_initialization_restored"]:
        raise RuntimeError("RGB preflight failed to restore released initialization")
    (Path(output) / "rgb-preflight.json").write_text(json.dumps(evidence, indent=2))
    return evidence


def validate_gpu(expected, count, name):
    names = {"t4": {"Tesla T4", "NVIDIA Tesla T4"}, "l4": {"NVIDIA L4"}}
    if count != 1 or name not in names.get(expected, set()):
        raise RuntimeError(
            f"Expected exactly one {expected.upper()} GPU; got {count} {name}"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--assets", type=Path, default=Path("checkpoints/released"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-runtime-seconds", required=True, type=float)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--gpu", choices=("t4", "l4"), default="t4")
    parser.add_argument(
        "--smoke", action="store_true", help="Diagnostic 100 updates; A1 then B99"
    )
    parser.add_argument(
        "--training-only-smoke",
        action="store_true",
        help="Engineering-only smoke: skip baseline/validation; no quality claim",
    )
    parser.add_argument(
        "--steps", type=int, default=2000, help="Cumulative cap, initially <=2000"
    )
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument(
        "--amp", action="store_true", help="Optional; first pilot defaults FP32"
    )
    parser.add_argument("--channels", type=int, choices=(3, 4), default=4)
    parser.add_argument(
        "--pair-coverage",
        action="store_true",
        help="Verified RGB per-pair native-cell coverage",
    )
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    if args.training_only_smoke and not args.smoke:
        parser.error("--training-only-smoke requires --smoke (100-update cap)")
    if (
        not math.isfinite(args.max_runtime_seconds)
        or args.max_runtime_seconds <= 0
        or args.accumulation < 1
        or not 1 <= args.steps <= 2000
    ):
        parser.error(
            "Require positive runtime/accumulation and 1..2000 cumulative updates"
        )
    if args.channels == 3 and (not args.pair_coverage or args.accumulation != 8):
        parser.error("RGB requires --pair-coverage and effective batch eight")
    if args.channels == 3 and args.amp:
        parser.error(
            "RGB run requires FP32 until mixed precision equivalence is verified"
        )
    if args.pair_coverage and args.channels != 3:
        parser.error("--pair-coverage is reserved for verified RGB pairs")
    started = time.monotonic()
    deadline = started + args.max_runtime_seconds
    steps = 100 if args.smoke else args.steps
    a_steps = 1 if args.smoke else min(1000, steps)
    paths = read_pairs(args.pairs, "train")
    validation = (
        [] if args.training_only_smoke else read_pairs(args.pairs, "validation")
    )
    coverage = None
    if not args.smoke:
        from src.full_dataset import SourceCoverage

        coverage = (
            PairCoverage(args.pairs, paths)
            if args.pair_coverage
            else SourceCoverage(args.pairs)
        )
        if steps * args.accumulation < len(coverage.keys):
            parser.error("Update budget cannot cover every counted training source")
    lock = json.loads(args.lock.read_text())
    if args.download:
        download_assets(args.assets, lock)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    device = "cuda:0"
    validate_gpu(args.gpu, torch.cuda.device_count(), torch.cuda.get_device_name())
    net = load_full_model(args.assets, device, args.lock, channels=args.channels)
    if args.channels == 3 and not args.resume:
        args.output.mkdir(parents=True, exist_ok=True)
        rgb_preflight(net, paths[0], args.output, device, deadline)
    mask_before = net.hard_constraint.low_pass_mask.clone()
    metadata = {
        "lock": lock,
        "pairs_sha256": file_hash(args.pairs),
        "amp": args.amp,
        "expected_gpu": args.gpu,
        "accumulation": args.accumulation,
        "a_steps": a_steps,
        "steps": steps,
        "seed": 17,
        "scope": "A=head+norm+body_tail; B=A+groups4,5",
        "loss": "char1_grad.05_consistency.1_spectral.01",
    }
    if args.channels == 3:
        metadata.update(channels=3, rgb_adaptation=net.rgb_adaptation)
    if coverage is not None:
        metadata.update(
            source_usage_sha256=coverage.usage_sha256,
            sampler="pair-coverage-first-v1"
            if args.pair_coverage
            else "source-coverage-first-v1",
        )
    if args.training_only_smoke:
        metadata.update(
            engineering_only=True,
            no_validation=True,
            quality_acceptance="not_evaluated",
        )
    step = 0
    stage = "A"
    monitor = {"best": float("inf"), "stale": 0}
    if args.resume:
        header = torch.load(args.resume, map_location="cpu", weights_only=False)
        stage = header["stage"]
    names = select_stage(net, stage)
    optimizer = optimizer_for(net, stage)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    if args.resume:
        state = load_checkpoint(args.resume, net, optimizer, scaler, metadata)
        step = state["step"]
        monitor = state["monitor"]
    if coverage is not None:
        monitor["source_coverage"] = coverage.status(step * args.accumulation)
    args.output.mkdir(parents=True, exist_ok=True)
    if coverage is not None:
        (args.output / "source-coverage.json").write_text(
            json.dumps(monitor["source_coverage"], indent=2)
        )
    evidence = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "mamba_ssm": importlib.metadata.version("mamba-ssm"),
        "sen2sr_location": str(
            importlib.metadata.distribution("sen2sr").locate_file("")
        ),
        "metadata": metadata,
        "trainable_names": names,
        "trainable_count": sum(p.numel() for p in net.parameters() if p.requires_grad),
    }
    (args.output / "environment.json").write_text(json.dumps(evidence, indent=2))
    if args.training_only_smoke:
        (args.output / "engineering-status.json").write_text(
            json.dumps(
                {
                    "engineering_only": True,
                    "no_validation": True,
                    "released_baseline": "skipped",
                    "validation": "skipped",
                    "quality_acceptance": "not_evaluated",
                    "update_cap": steps,
                },
                indent=2,
            )
        )
    stop = [False]

    def request_stop(signum, frame):
        stop[0] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    checkpoint = args.output / "latest.pt"
    save_checkpoint(
        checkpoint,
        net,
        optimizer,
        scaler,
        step=step,
        stage=stage,
        metadata=metadata,
        monitor=monitor,
    )
    last_save = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    # Record the unchanged full-model validation once, before any updates.
    if not args.resume and not args.training_only_smoke:
        try:
            baseline = evaluate(
                net, validation, device, args.amp, deadline, channels=args.channels
            )
        except TimeoutError:
            result = {
                "status": "deadline_during_baseline",
                "source_coverage": monitor.get("source_coverage"),
                "step": step,
                "checkpoint": str(checkpoint),
            }
            (args.output / "result.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result))
            return
        (args.output / "released-baseline.json").write_text(
            json.dumps(baseline, indent=2)
        )
    while (
        step < steps
        and keep_training(
            monitor["stale"],
            monitor.get("source_coverage") if args.pair_coverage else None,
        )
        and not stop[0]
        and time.monotonic() < deadline
    ):
        next_stage = "A" if step < a_steps else "B"
        if next_stage != stage:
            stage = next_stage
            names = select_stage(net, stage)
            optimizer = optimizer_for(net, stage)
            (args.output / f"trainable-{stage}.json").write_text(
                json.dumps(names, indent=2)
            )
        training_mode(net, stage)
        schedule(
            optimizer,
            step if stage == "A" else step - a_steps,
            a_steps if stage == "A" else steps - a_steps,
        )
        optimizer.zero_grad(set_to_none=True)
        totals = {}
        update_started = time.monotonic()
        update_rng = rng_state()
        before = net.sr_model.conv_last.weight.detach().clone()
        for draw_in_update in range(args.accumulation):
            if stop[0] or time.monotonic() >= deadline:
                stop[0] = True
                break
            chosen = (
                coverage.choose(step * args.accumulation + draw_in_update)
                if coverage is not None
                else random.choice(paths)
            )
            lr, hr, hm, lm = load_pair(chosen, device, channels=args.channels)
            with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
                pred = net(lr)
            validate_pair(lr, pred, channels=args.channels)
            parts = reconstruction_loss(pred, hr, lr, hm, lm)
            if not torch.isfinite(parts["total"]):
                raise RuntimeError("Nonfinite training loss")
            scaler.scale(parts["total"] / args.accumulation).backward()
            for key, value in parts.items():
                totals[key] = (
                    totals.get(key, 0.0) + float(value.detach()) / args.accumulation
                )
        if stop[0]:
            optimizer.zero_grad(set_to_none=True)
            restore_rng(update_rng)
            break
        scaler.unscale_(optimizer)
        gradients = gradient_evidence(net, stage)
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in net.parameters() if p.requires_grad],
            1.0,
            error_if_nonfinite=True,
        )
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        if torch.equal(before, net.sr_model.conv_last.weight):
            raise RuntimeError("Head parameter did not update")
        if not torch.equal(mask_before, net.hard_constraint.low_pass_mask):
            raise RuntimeError("Fourier mask changed")
        step += 1
        if coverage is not None:
            monitor["source_coverage"] = coverage.status(step * args.accumulation)
            (args.output / "source-coverage.json").write_text(
                json.dumps(monitor["source_coverage"], indent=2)
            )
        row = {
            "step": step,
            "stage": stage,
            "elapsed_seconds": time.monotonic() - started,
            "update_seconds": time.monotonic() - update_started,
            "loss": totals,
            "gradient_norm": float(norm),
            "lr_shape": list(lr.shape),
            "output_shape": list(pred.shape),
            "output_dtype": str(pred.dtype),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            **gradients,
        }
        if coverage is not None:
            row["source_coverage"] = monitor["source_coverage"]
        if args.training_only_smoke:
            row.update(engineering_only=True, validation_status="skipped_no_validation")
        if not args.training_only_smoke and step % (50 if args.smoke else 500) == 0:
            try:
                row["validation"] = evaluate(
                    net, validation, device, args.amp, deadline, channels=args.channels
                )
                score = row["validation"]["model"]
                improved = score < monitor["best"]
                monitor = {
                    **monitor,
                    "best": min(score, monitor["best"]),
                    "stale": 0 if improved else monitor["stale"] + 1,
                }
                if improved:
                    save_checkpoint(
                        args.output / "best-reconstruction.pt",
                        net,
                        optimizer,
                        scaler,
                        step=step,
                        stage=stage,
                        metadata=metadata,
                        monitor=monitor,
                    )
            except TimeoutError:
                stop[0] = True
                row["validation_status"] = "deadline"
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if time.monotonic() - last_save >= 300:
            save_checkpoint(
                checkpoint,
                net,
                optimizer,
                scaler,
                step=step,
                stage=stage,
                metadata=metadata,
                monitor=monitor,
            )
            last_save = time.monotonic()
    save_checkpoint(
        checkpoint,
        net,
        optimizer,
        scaler,
        step=step,
        stage=stage,
        metadata=metadata,
        monitor=monitor,
    )
    status = (
        "step_cap"
        if step == steps
        else (
            "validation_patience"
            if not keep_training(
                monitor["stale"],
                monitor.get("source_coverage") if args.pair_coverage else None,
            )
            else "time_or_signal_stop"
        )
    )
    result = {"status": status, "step": step, "checkpoint": str(checkpoint)}
    if coverage is not None:
        result["source_coverage"] = coverage.status(step * args.accumulation)
    if args.training_only_smoke:
        result.update(
            engineering_only=True,
            no_validation=True,
            quality_acceptance="not_evaluated",
        )
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
