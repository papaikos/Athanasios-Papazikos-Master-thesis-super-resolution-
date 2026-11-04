"""Bounded sequential P2 diagnostics; never launches a VM or training candidate.

Run on the already-authorized idle single T4. Every CUDA phase is a fresh child
process with an enforced timeout. A new output directory stores the policy before
measurements; exit0 means these diagnostics passed, not complete scientific P2.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from src.model import (
    DEFAULT_LOCK,
    file_hash,
    load_full_model,
    select_stage,
    training_mode,
    validate_pair,
)
from src.train import (
    gradient_evidence,
    load_checkpoint,
    load_pair,
    optimizer_for,
    read_pairs,
    reconstruction_loss,
    rng_state,
    save_checkpoint,
    schedule,
)

DEFAULT_POLICY = Path(__file__).resolve().parents[1] / "configs/t4-diagnostic.json"
PHASES = ("fp32", "amp", "control", "restart", "scan_cuda", "scan_reference")
SCAN_NAMES = ("u", "delta", "A", "B", "C", "D")


def validate_device(visible_count, name, capability, physical_names):
    if (
        visible_count != 1
        or "T4" not in name
        or tuple(capability) != (7, 5)
        or len(physical_names) != 1
        or "T4" not in physical_names[0]
    ):
        raise ValueError(
            "Require exactly one physical/visible T4 with capability (7,5)"
        )


def device_evidence():
    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; no CPU/Lite substitution")
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    rows = [line for line in inventory.stdout.splitlines() if line.strip()]
    name, capability = torch.cuda.get_device_name(), torch.cuda.get_device_capability()
    validate_device(
        torch.cuda.device_count(),
        name,
        capability,
        [line.split(",")[0] for line in rows],
    )
    return {
        "device": name,
        "capability": capability,
        "physical_inventory": rows,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "spot_identity": "must be joined from controller evidence",
    }


def compare_tensor(reference, candidate, rtol, atol):
    if reference.shape != candidate.shape:
        return {"passed": False, "reason": "shape mismatch"}
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        return {"passed": False, "reason": "nonfinite values"}
    ref, other = reference.detach().double().cpu(), candidate.detach().double().cpu()
    delta = other - ref
    return {
        "passed": bool(torch.allclose(other, ref, rtol=rtol, atol=atol)),
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "relative_l2": float(delta.norm() / ref.norm().clamp_min(1e-8)),
        "rtol": rtol,
        "atol": atol,
    }


def compare_tree(reference, candidate, rtol, atol):
    failures = []

    def visit(ref, other, path):
        if isinstance(ref, torch.Tensor):
            exact = (
                path.startswith("rng")
                or path.startswith("scaler")
                or path.endswith(".step")
                or not ref.is_floating_point()
            )
            result = (
                compare_tensor(ref, other, 0 if exact else rtol, 0 if exact else atol)
                if isinstance(other, torch.Tensor)
                else {"passed": False}
            )
            if not result["passed"]:
                failures.append({"path": path, **result})
        elif isinstance(ref, np.ndarray):
            if not isinstance(other, np.ndarray) or not np.array_equal(ref, other):
                failures.append({"path": path})
        elif isinstance(ref, dict):
            if not isinstance(other, dict) or ref.keys() != other.keys():
                failures.append({"path": path, "reason": "key mismatch"})
                return
            for key in ref:
                visit(ref[key], other[key], f"{path}.{key}" if path else str(key))
        elif isinstance(ref, (tuple, list)):
            if type(other) is not type(ref) or len(ref) != len(other):
                failures.append({"path": path, "reason": "sequence mismatch"})
                return
            for index, value in enumerate(ref):
                visit(value, other[index], f"{path}.{index}")
        elif ref != other:
            failures.append({"path": path, "reason": "scalar/state mismatch"})

    visit(reference, candidate, "")
    return {
        "passed": not failures,
        "failures": failures[:30],
        "failure_count": len(failures),
    }


def frozen_hash(net):
    digest = hashlib.sha256()
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def snapshot_checkpoint(source, destination):
    digest = file_hash(source)
    shutil.copyfile(source, destination)
    if file_hash(destination) != digest or file_hash(source) != digest:
        raise ValueError(
            "Checkpoint changed during diagnostic snapshot; use an idle pilot"
        )
    return digest


def run_process(command, timeout, log_path):
    started = time.monotonic()
    if timeout <= 0:
        return {"status": "inconclusive", "reason": "global deadline"}
    with Path(log_path).open("w") as log:
        try:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "inconclusive",
                "reason": "phase timeout; child killed",
                "seconds": time.monotonic() - started,
            }
    return {
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "seconds": time.monotonic() - started,
    }


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare(args, policy, restart=False):
    paths = read_pairs(args.pairs, "train")
    source = args.output / "restart-base.pt" if restart else args.checkpoint
    header = torch.load(source, map_location="cpu", weights_only=False)
    lock = json.loads(args.lock.read_text())
    metadata = header["metadata"]
    if (
        header["stage"] != "B"
        or metadata["lock"] != lock
        or metadata["pairs_sha256"] != file_hash(args.pairs)
    ):
        raise ValueError(
            "Require a stage B checkpoint with exactly matching model/data provenance"
        )
    if metadata["accumulation"] != 1:
        raise ValueError(
            "This bounded diagnostic requires the accumulation1 pilot checkpoint"
        )
    net = load_full_model(args.assets, "cuda:0", args.lock)
    select_stage(net, "B")
    released_frozen = frozen_hash(net)
    optimizer = optimizer_for(net, "B")
    scaler = torch.amp.GradScaler("cuda", enabled=metadata["amp"])
    state = load_checkpoint(source, net, optimizer, scaler, metadata)
    if frozen_hash(net) != released_frozen:
        raise ValueError("Checkpoint's frozen prefix differs from released weights")
    if not restart:
        # Deliberately registered diagnostic precision, not production continuation.
        metadata = dict(
            metadata, amp=True, diagnostic_policy_sha256=file_hash(args.policy)
        )
        scaler = torch.amp.GradScaler("cuda", init_scale=policy["amp"]["initial_scale"])
    elif metadata.get("diagnostic_policy_sha256") != file_hash(args.policy):
        raise ValueError("Restart diagnostic policy changed")
    return net, optimizer, scaler, paths, state["step"], metadata


@contextmanager
def observe_precision(net, capture_path=None):
    """Observe real scan and FFT tensors without changing their computation."""
    evidence = {
        "scan_calls": 0,
        "scan_dtypes": set(),
        "fft_calls": 0,
        "fft_dtypes": set(),
        "ifft_dtypes": set(),
        "backbone_dtypes": set(),
    }
    restorations = []
    captured = [False]
    target = "sr_model.layers.4.residual_group.blocks.0.self_attention"
    for name, module in net.named_modules():
        if not hasattr(module, "selective_scan"):
            continue
        original = module.selective_scan

        def observed(*values, _original=original, _name=name, **kwargs):
            evidence["scan_calls"] += 1
            evidence["scan_dtypes"].update(
                str(x.dtype)
                for x in (*values, *kwargs.values())
                if isinstance(x, torch.Tensor)
            )
            if capture_path is not None and _name == target and not captured[0]:
                torch.save(
                    {
                        "args": cpu_tree(values),
                        "kwargs": cpu_tree(kwargs),
                        "module": _name,
                    },
                    capture_path,
                )
                captured[0] = True
            return _original(*values, **kwargs)

        module.selective_scan = observed
        restorations.append((module, "selective_scan", original))
    for function_name, key in (("fftn", "fft_dtypes"), ("ifft2", "ifft_dtypes")):
        original = getattr(torch.fft, function_name)

        def observed_fft(value, *args, _original=original, _key=key, **kwargs):
            evidence[_key].add(str(value.dtype))
            evidence["fft_calls"] += 1
            return _original(value, *args, **kwargs)

        setattr(torch.fft, function_name, observed_fft)
        restorations.append((torch.fft, function_name, original))
    handle = net.sr_model.register_forward_hook(
        lambda module, values, output: evidence["backbone_dtypes"].add(
            str(output.dtype)
        )
    )
    try:
        yield evidence
    finally:
        handle.remove()
        for owner, name, original in restorations:
            setattr(owner, name, original)
        for key, value in evidence.items():
            if isinstance(value, set):
                evidence[key] = sorted(value)
    if evidence["scan_dtypes"] != ["torch.float32"] or not evidence["scan_calls"]:
        raise ValueError("Actual scan inputs were not all FP32")
    if evidence["fft_dtypes"] != ["torch.float32"] or evidence["ifft_dtypes"] != [
        "torch.complex64"
    ]:
        raise ValueError("Actual FFT execution did not use FP32/complex64")
    if capture_path is not None and not captured[0]:
        raise ValueError("Failed to observe the first trainable selective scan")


def parameter_gradients(net):
    return {
        name: value.grad.detach().cpu().clone()
        for name, value in net.named_parameters()
        if value.requires_grad
    }


def precision_phase(args, policy):
    net, optimizer, scaler, paths, _, _ = prepare(args, policy)
    # Matched deterministic weights, sample and RNG; enable autograd in eval mode.
    net.eval()
    seed_all(policy["seed"])
    amp = args.phase == "amp"
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp, init_scale=policy["amp"]["initial_scale"]
    )
    lr, hr, hm, lm = load_pair(paths[0], "cuda:0")
    frozen_before = frozen_hash(net)
    capture = args.output / "scan-inputs.pt" if not amp else None
    with observe_precision(net, capture) as dtypes:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            pred = net(lr)
        validate_pair(lr, pred)
        parts = reconstruction_loss(pred, hr, lr, hm, lm)
        scaler.scale(parts["total"]).backward()
        scaler.unscale_(optimizer)
        gradient_evidence(net, "B")
    if frozen_before != frozen_hash(net):
        raise ValueError("Frozen weights changed during precision probe")
    if amp and dtypes["backbone_dtypes"] != ["torch.float16"]:
        raise ValueError(
            "AMP probe did not execute the reconstruction backbone in FP16"
        )
    torch.save(
        {"output": pred.detach().cpu(), "gradients": parameter_gradients(net)},
        args.output / f"{args.phase}.pt",
    )
    return {
        "status": "passed",
        "dtypes": dtypes,
        "frozen_unchanged": True,
        "sample": str(paths[0]),
        "sample_sha256": file_hash(paths[0]),
        "loss": float(parts["total"].detach()),
        "output_shape": list(pred.shape),
    }


def one_update(net, optimizer, scaler, paths, step, metadata):
    training_mode(net, "B")
    schedule(
        optimizer, step - metadata["a_steps"], metadata["steps"] - metadata["a_steps"]
    )
    optimizer.zero_grad(set_to_none=True)
    frozen_before = frozen_hash(net)
    mask_before = net.hard_constraint.low_pass_mask.detach().cpu().clone()
    head_before = net.sr_model.conv_last.weight.detach().cpu().clone()
    path = random.choice(paths)
    lr, hr, hm, lm = load_pair(path, "cuda:0")
    with torch.autocast("cuda", dtype=torch.float16, enabled=metadata["amp"]):
        pred = net(lr)
    validate_pair(lr, pred)
    parts = reconstruction_loss(pred, hr, lr, hm, lm)
    scaler.scale(parts["total"]).backward()
    scaler.unscale_(optimizer)
    gradient_evidence(net, "B")
    gradients = parameter_gradients(net)
    torch.nn.utils.clip_grad_norm_(
        [p for p in net.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True
    )
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    if torch.equal(head_before, net.sr_model.conv_last.weight.detach().cpu()):
        raise ValueError("Diagnostic optimizer step did not update the head")
    if frozen_before != frozen_hash(net) or not torch.equal(
        mask_before, net.hard_constraint.low_pass_mask.detach().cpu()
    ):
        raise ValueError("Frozen weights/Fourier mask changed")
    return {
        "output": pred.detach().cpu(),
        "gradients": gradients,
        "sampled_pairs": [str(path)],
        "sample_sha256": file_hash(path),
        "loss": parts["total"].detach().cpu(),
        "model": cpu_tree(net.state_dict()),
        "optimizer": cpu_tree(optimizer.state_dict()),
        "scaler": scaler.state_dict(),
        "rng": rng_state(),
        "step": step + 1,
        "stage": "B",
        "metadata": metadata,
        "frozen_hash": frozen_before,
        "mask": mask_before,
    }


def resume_phase(args, policy):
    restart = args.phase == "restart"
    net, optimizer, scaler, paths, step, metadata = prepare(args, policy, restart)
    if not restart:
        # A real update populates/grows the AMP scaler and inherited optimizer.
        first = one_update(net, optimizer, scaler, paths, step, metadata)
        step = first["step"]
        del first
        save_checkpoint(
            args.output / "restart-base.pt",
            net,
            optimizer,
            scaler,
            step=step,
            stage="B",
            metadata=metadata,
        )
    result = one_update(net, optimizer, scaler, paths, step, metadata)
    torch.save(result, args.output / f"{args.phase}.pt")
    return {
        "status": "passed",
        "pid": os.getpid(),
        "step": result["step"],
        "frozen_unchanged": True,
        "actual_updates": 1 if restart else 2,
    }


def scan_phase(args, policy):
    import mamba_ssm.ops.selective_scan_interface as interface

    expected = policy["reference"]
    if (
        importlib.metadata.version("mamba-ssm").split("+")[0] != expected["version"]
        or file_hash(interface.__file__) != expected["interface_sha256"]
    ):
        raise ValueError(
            "Installed selective-scan reference differs from the pinned source"
        )
    capture = torch.load(
        args.output / "scan-inputs.pt", map_location="cpu", weights_only=False
    )
    raw = dict(zip(SCAN_NAMES, capture["args"]))
    raw.update(capture["kwargs"])
    if (
        raw.get("z") is not None
        or raw.get("return_last_state") is not False
        or raw.get("delta_softplus") is not True
        or raw["B"].ndim != 4
        or raw["C"].ndim != 4
    ):
        raise ValueError(
            "Observed scan does not match the registered variable-B/C no-z path"
        )
    reference = args.phase == "scan_reference"
    estimated = (
        raw["u"].numel()
        * raw["A"].shape[-1]
        * 4
        * policy["reference_memory"]["estimated_working_set_multiplier"]
    )
    free, _ = torch.cuda.mem_get_info()
    if reference and estimated > free - policy["reference_memory"]["reserve_bytes"]:
        return {
            "status": "inconclusive",
            "reason": "Full-shaped reference estimated memory exceeds headroom",
            "estimated_bytes": estimated,
            "free_bytes": free,
        }
    values = {
        key: value.to("cuda:0").detach().requires_grad_(True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in raw.items()
    }
    operator = (
        interface.selective_scan_ref if reference else interface.selective_scan_fn
    )
    out = operator(**values)
    generator = torch.Generator(device="cpu").manual_seed(policy["seed"])
    upstream_gradient = torch.randn(
        out.shape, generator=generator, dtype=torch.float32
    ).to("cuda:0")
    out.backward(upstream_gradient)
    gradients = {
        key: value.grad.detach().cpu()
        for key, value in values.items()
        if isinstance(value, torch.Tensor)
    }
    if not torch.isfinite(out).all() or any(
        not torch.isfinite(value).all() for value in gradients.values()
    ):
        raise ValueError("Nonfinite scan reference/kernel result")
    torch.save(
        {"output": out.detach().cpu(), "gradients": gradients},
        args.output / f"{args.phase}.pt",
    )
    return {
        "status": "passed",
        "shapes": {
            key: list(value.shape)
            for key, value in raw.items()
            if isinstance(value, torch.Tensor)
        },
        "reference_sha256": expected["interface_sha256"],
        "estimated_bytes": estimated,
    }


def worker(args, policy):
    started = time.monotonic()
    try:
        device = device_evidence()
        seed_all(policy["seed"])
        torch.cuda.reset_peak_memory_stats()
        if args.phase in ("fp32", "amp"):
            result = precision_phase(args, policy)
        elif args.phase in ("control", "restart"):
            result = resume_phase(args, policy)
        else:
            result = scan_phase(args, policy)
        torch.cuda.synchronize()
        result.update(
            device=device,
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
        )
    except torch.cuda.OutOfMemoryError as error:
        result = {
            "status": "inconclusive",
            "reason": "CUDA OOM; no reduced-shape fallback",
            "error": str(error),
        }
    except Exception as error:
        result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    result.update(seconds=time.monotonic() - started, pid=os.getpid(), phase=args.phase)
    write_json(args.output / f"{args.phase}.json", result)
    return 0 if result["status"] == "passed" else 2


def compare_amp(fp32, amp, policy):
    bounds = policy["amp"]
    output = compare_tensor(
        fp32["output"],
        amp["output"],
        bounds["output"]["rtol"],
        bounds["output"]["atol"],
    )
    output["passed"] &= (
        output.get("relative_l2", float("inf")) <= bounds["output"]["relative_l2_max"]
    )
    if fp32["gradients"].keys() != amp["gradients"].keys():
        return {"passed": False, "reason": "gradient scope mismatch", "output": output}
    squared_error, squared_reference, max_error, max_reference = 0.0, 0.0, 0.0, 0.0
    for name, ref in fp32["gradients"].items():
        other = amp["gradients"][name]
        if (
            ref.shape != other.shape
            or not torch.isfinite(ref).all()
            or not torch.isfinite(other).all()
        ):
            return {
                "passed": False,
                "reason": f"nonfinite/shape gradient: {name}",
                "output": output,
            }
        ref, other = ref.double(), other.double()
        difference = other - ref
        squared_error += float(difference.square().sum())
        squared_reference += float(ref.square().sum())
        max_error = max(max_error, float(difference.abs().max()))
        max_reference = max(max_reference, float(ref.abs().max()))
    relative = math.sqrt(squared_error) / max(
        math.sqrt(squared_reference), bounds["denominator_floor"]
    )
    max_scaled = max_error / max(max_reference, bounds["denominator_floor"])
    return {
        "passed": output["passed"]
        and relative <= bounds["gradient_relative_l2_max"]
        and max_scaled <= bounds["gradient_max_error_over_reference_max"],
        "output": output,
        "gradient_relative_l2": relative,
        "gradient_max_scaled_error": max_scaled,
    }


def comparisons(output, policy, phases):
    result = {}

    def ready(*names):
        return all(phases.get(name, {}).get("status") == "passed" for name in names)

    def artifact(name):
        return torch.load(output / f"{name}.pt", map_location="cpu", weights_only=False)

    if ready("fp32", "amp"):
        result["fp32_amp"] = compare_amp(artifact("fp32"), artifact("amp"), policy)
    if ready("control", "restart"):
        result["fresh_process_resume"] = compare_tree(
            artifact("control"), artifact("restart"), **policy["resume"]
        )
        result["fresh_process_resume"]["distinct_pids"] = (
            phases["control"]["pid"] != phases["restart"]["pid"]
        )
        result["fresh_process_resume"]["passed"] &= result["fresh_process_resume"][
            "distinct_pids"
        ]
    if ready("scan_cuda", "scan_reference"):
        cuda, ref = artifact("scan_cuda"), artifact("scan_reference")
        measured = {
            "output": compare_tensor(
                ref["output"], cuda["output"], **policy["reference"]["output"]
            )
        }
        for name, bounds in policy["reference"]["gradients"].items():
            measured[name] = compare_tensor(
                ref["gradients"][name], cuda["gradients"][name], **bounds
            )
        result["scan_reference"] = {
            "passed": all(value["passed"] for value in measured.values()),
            "measurements": measured,
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trusted stage B accumulation1 pilot checkpoint",
    )
    parser.add_argument("--assets", type=Path, default=Path("checkpoints/released"))
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New diagnostic directory; never overwrites a run",
    )
    parser.add_argument("--max-runtime-seconds", type=float, required=True)
    parser.add_argument("--phase", choices=PHASES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_runtime_seconds) or args.max_runtime_seconds <= 0:
        parser.error("Require a finite positive runtime limit")
    started = time.monotonic()
    deadline = started + args.max_runtime_seconds
    policy = json.loads(args.policy.read_text())
    if args.phase:
        return worker(args, policy)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output / "registered-policy.json"
    snapshot.write_bytes(args.policy.read_bytes())
    source_checkpoint = args.output / "source-checkpoint.pt"
    checkpoint_hash = snapshot_checkpoint(args.checkpoint, source_checkpoint)
    report = {
        "status": "inconclusive",
        "p2_complete": False,
        "requires_joined_evidence": [
            "100 real optimizer updates from the separate pilot",
            "controller-observed single-T4 Spot identity and cost forecast",
            "affordable experiment forecast from measured memory/throughput",
        ],
        "policy_sha256": file_hash(snapshot),
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": checkpoint_hash,
        "pairs_sha256": file_hash(args.pairs),
        "scope": "bounded diagnostic subset; not the 100-update smoke or scientific acceptance",
        "phases": {},
        "comparisons": {},
    }
    report_path = args.output / "t4-compatibility.json"
    write_json(report_path, report)
    for phase in PHASES:
        dependencies = {
            "restart": "control",
            "scan_cuda": "fp32",
            "scan_reference": "scan_cuda",
            "control": "amp",
        }
        if (
            phase in dependencies
            and report["phases"].get(dependencies[phase], {}).get("status") != "passed"
        ):
            report["phases"][phase] = {
                "status": "inconclusive",
                "reason": "prerequisite phase did not pass",
            }
            continue
        remaining = deadline - time.monotonic()
        command = [
            sys.executable,
            "-m",
            "src.diagnose_t4",
            "--phase",
            phase,
            "--pairs",
            str(args.pairs.resolve()),
            "--checkpoint",
            str(source_checkpoint.resolve()),
            "--assets",
            str(args.assets.resolve()),
            "--lock",
            str(args.lock.resolve()),
            "--policy",
            str(snapshot.resolve()),
            "--output",
            str(args.output.resolve()),
            "--max-runtime-seconds",
            str(max(0.001, remaining)),
        ]
        status = run_process(
            command,
            min(remaining, policy["phase_seconds"][phase]),
            args.output / f"{phase}.log",
        )
        phase_path = args.output / f"{phase}.json"
        report["phases"][phase] = (
            json.loads(phase_path.read_text())
            if phase_path.exists() and status["status"] != "inconclusive"
            else status
        )
        write_json(report_path, report)
    report["comparisons"] = comparisons(args.output, policy, report["phases"])
    failed = any(
        value["status"] == "failed" for value in report["phases"].values()
    ) or any(not value["passed"] for value in report["comparisons"].values())
    complete = len(report["comparisons"]) == 3 and all(
        value["status"] == "passed" for value in report["phases"].values()
    )
    report["status"] = (
        "failed" if failed else ("passed" if complete else "inconclusive")
    )
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path)}))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
