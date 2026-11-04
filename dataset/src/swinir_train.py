"""Bounded full-frame SwinIR-M RGB x4 fine-tuning; no acquisition or inference gallery."""

import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import torch

from src.model import file_hash
from src.swinir_model import load_swinir
from src.train import PairCoverage, evaluate, load_pair, read_pairs, reconstruction_loss


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def save(path, net, optimizer, scaler, step, best, metadata):
    temporary = path.with_suffix(".tmp")
    torch.save(
        dict(
            model=net.state_dict(),
            optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),
            step=step,
            best=best,
            metadata=metadata,
        ),
        temporary,
    )
    os.replace(temporary, path)


def checkpoint_roundtrip(output, net, optimizer, scaler, step, best, metadata):
    """Prove atomic serialization/reload before any substantive optimizer update."""
    path = output / ".checkpoint-roundtrip.tmp"
    try:
        save(path, net, optimizer, scaler, step, best, metadata)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if (
            loaded["metadata"] != metadata
            or loaded["step"] != step
            or loaded["best"] != best
        ):
            raise RuntimeError("Checkpoint metadata roundtrip mismatch")
        current = net.state_dict()
        if current.keys() != loaded["model"].keys() or any(
            not torch.equal(value.detach().cpu(), loaded["model"][key])
            for key, value in current.items()
        ):
            raise RuntimeError("Checkpoint model tensor roundtrip mismatch")
        net.load_state_dict(loaded["model"], strict=True)
        optimizer.load_state_dict(loaded["optimizer"])
        scaler.load_state_dict(loaded["scaler"])
        atomic_json(
            output / "checkpoint-roundtrip.json",
            dict(passed=True, strict_reload=True, tensor_equality=True, step=step),
        )
    finally:
        path.unlink(missing_ok=True)


def preflight(net, pair, output, device, amp=True, amp_dtype=torch.float16):
    lr, hr, hm, lm = load_pair(pair, device, channels=3)
    net.eval()
    with torch.no_grad():
        reference = net(lr).float()
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
            mixed = net(lr).float()
    delta = (mixed - reference).abs()
    comparison = dict(
        precision=("bf16" if amp_dtype == torch.bfloat16 else "fp16") if amp else "fp32",
        reference_finite=bool(torch.isfinite(reference).all()),
        candidate_finite=bool(torch.isfinite(mixed).all()),
        comparison_mean_abs=float(delta.mean()),
        comparison_max_abs=float(delta.max()),
    )
    atomic_json(output / "precision-check.json", comparison)
    if (not comparison["reference_finite"] or not comparison["candidate_finite"]
            or delta.mean() > 0.003 or delta.max() > 0.03):
        raise RuntimeError("SwinIR numerical equivalence gate failed; see precision-check.json")
    net.train()
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
        prediction = net(lr)
    loss = reconstruction_loss(prediction, hr, lr, hm, lm)["total"]
    if not torch.isfinite(loss):
        raise RuntimeError("SwinIR nonfinite preflight loss")
    loss.backward()
    gradients = [p.grad for p in net.parameters() if p.requires_grad]
    if not gradients or any(
        g is None or not torch.isfinite(g).all() for g in gradients
    ):
        raise RuntimeError("SwinIR full-model finite-gradient gate failed")
    if not any(torch.count_nonzero(g) for g in gradients):
        raise RuntimeError("SwinIR all gradients are zero")
    net.zero_grad(set_to_none=True)
    atomic_json(
        output / "preflight.json",
        dict(
            passed=True,
            strict_pretrained_load=True,
            parameter_count=sum(p.numel() for p in net.parameters()),
            **comparison,
            loss=float(loss.detach()),
            optimizer_draws=0,
            lr_shape=list(lr.shape),
            hr_shape=list(hr.shape),
            gpu=torch.cuda.get_device_name(0),
        ),
    )


def main(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--assets", type=Path, default=Path("assets/swinir"))
    p.add_argument("--lock", type=Path, default=Path("configs/swinir-lock.json"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--max-seconds", type=float, required=True)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--download-assets", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    args = p.parse_args(argv)
    if (
        args.steps != 2000
        or args.accumulation != 8
        or not 300 <= args.max_seconds <= 86400
    ):
        p.error(
            "Require 2000 updates, effective batch eight, bounded 300..86400 seconds"
        )
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or "L4" not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("Exactly one L4 GPU is required")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "last.pt").exists() and not args.resume:
        raise RuntimeError("Existing training state: explicit resume required")
    started = time.monotonic()
    deadline = started + args.max_seconds
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    random.seed(17)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train = read_pairs(args.index, "train")
    validation = read_pairs(args.index, "validation")
    coverage = PairCoverage(args.index, train)
    if len(train) > args.steps * 8:
        raise ValueError("Planned draws cannot cover every pair")
    device = torch.device("cuda")
    net, lock = load_swinir(args.assets, args.lock, args.download_assets)
    net.to(device)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.learning_rate, betas=(0.9, 0.99)
    )
    amp = args.precision != "fp32"
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
    metadata = dict(
        architecture="official-SwinIR-M-classical-RGB-x4",
        model_lock=lock,
        index_sha256=file_hash(args.index),
        full_pair_shape=[128, 512],
        loss="shared masked Charbonnier + gradient + LR consistency + spectral",
        effective_batch=8,
        target_steps=2000,
        precision=args.precision,
        validation_precision="fp32",
    )
    step, best = 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["metadata"] != metadata:
            raise ValueError("Resume metadata differs")
        net.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        step, best = state["step"], state["best"]

    def status(kind, **extra):
        value = dict(
            status=kind,
            step=step,
            target_steps=args.steps,
            elapsed_seconds=time.monotonic() - started,
            coverage=coverage.status(step * 8),
            **extra,
        )
        atomic_json(args.output / "status.json", value)
        return value

    try:
        status("preflight")
        preflight(net, train[0], args.output, device, amp=amp, amp_dtype=amp_dtype)
        checkpoint_roundtrip(args.output, net, optimizer, scaler, step, best, metadata)
        # Baseline is diagnostic and never counts toward optimizer coverage.
        if step == 0:
            with torch.no_grad():
                baseline = evaluate(
                    net, validation, device, False, deadline - 120, channels=3
                )
            atomic_json(args.output / "baseline.json", baseline)
        failures = 0
        while step < args.steps and not stopped and time.monotonic() < deadline - 120:
            net.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for draw in range(8):
                if stopped or time.monotonic() >= deadline - 120:
                    break
                lr, hr, hm, lm = load_pair(
                    coverage.choose(step * 8 + draw), device, channels=3
                )
                with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
                    pred = net(lr)
                loss = reconstruction_loss(pred, hr, lr, hm, lm)["total"]
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite reconstruction loss")
                scaler.scale(loss / 8).backward()
                total += float(loss.detach()) / 8
            else:
                scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                if not scaler.is_enabled() and not torch.isfinite(grad):
                    raise RuntimeError("Nonfinite unscaled gradient; optimizer update rejected")
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if not torch.isfinite(grad) or scaler.get_scale() < old_scale:
                    failures += 1
                    if failures > 8:
                        raise RuntimeError("Repeated skipped optimizer steps")
                    continue
                failures = 0
                step += 1
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate * (
                        0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps))
                    )
                with (args.output / "metrics.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            dict(
                                step=step,
                                loss=total,
                                elapsed_seconds=time.monotonic() - started,
                            )
                        )
                        + "\n"
                    )
                status("running")
                if step % 25 == 0 or step == args.steps:
                    save(
                        args.output / "last.pt",
                        net,
                        optimizer,
                        scaler,
                        step,
                        best,
                        metadata,
                    )
                if step % 100 == 0 or step == args.steps:
                    with torch.no_grad():
                        metric = evaluate(
                            net, validation, device, False, deadline - 120, channels=3
                        )
                    with (args.output / "validation.jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(step=step, **metric)) + "\n")
                    if metric["model"] < best:
                        best = metric["model"]
                        save(
                            args.output / "best-reconstruction.pt",
                            net,
                            optimizer,
                            scaler,
                            step,
                            best,
                            metadata,
                        )
                    save(
                        args.output / "last.pt",
                        net,
                        optimizer,
                        scaler,
                        step,
                        best,
                        metadata,
                    )
                continue
            break
        save(args.output / "last.pt", net, optimizer, scaler, step, best, metadata)
        complete = (
            step == args.steps and coverage.status(step * 8)["all_pairs_consumed"]
        )
        result = status(
            "completed" if complete else "deadline", best_validation_loss=best
        )
        atomic_json(args.output / "result.json", result)
        return 0 if complete else 2
    except Exception as exc:
        save(args.output / "last.pt", net, optimizer, scaler, step, best, metadata)
        result = status("failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(args.output / "result.json", result)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
