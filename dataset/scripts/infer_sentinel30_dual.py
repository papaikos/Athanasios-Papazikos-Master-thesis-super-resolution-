"""Apply both completed full RGB x4 models to exactly 30 native Sentinel-2 patches."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
import torch

from PIL import Image
from src.model import DEFAULT_LOCK, file_hash, load_full_model
from src.swinir_model import load_swinir


def validate_rgb(array, size):
    if array.shape != (3, size, size) or array.dtype != np.float32:
        raise ValueError("Expected float32 CHW RGB at the registered patch size")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("Input RGB must be finite and within [0,1]")


def rgb_image(array):
    return Image.fromarray(
        np.rint(np.clip(array, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    )


def validate_inputs(manifest, directory):
    rows = manifest.get("examples", [])
    if len(rows) != 30:
        raise ValueError("Exactly 30 Sentinel-2 inputs are required")
    ids, hashes, arrays = set(), set(), []
    directory = Path(directory).resolve()
    for row in rows:
        identifier, digest = row["id"], row["sha256"]
        if (
            not re.fullmatch(r"[A-Za-z0-9_-]+", identifier)
            or identifier in ids
            or digest in hashes
        ):
            raise ValueError("Input IDs and file hashes must be unique")
        provenance = row["provenance"]
        if (
            provenance.get("collection") != "COPERNICUS/S2_SR_HARMONIZED"
            or provenance.get("native_pixel_m") != 10
        ):
            raise ValueError("Only native 10 m Sentinel-2 imagery is accepted")
        path = (directory / row["path"]).resolve()
        if not path.is_relative_to(directory) or file_hash(path) != digest:
            raise ValueError("Input path or checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            array = data["lr"].copy()
        validate_rgb(array, 128)
        ids.add(identifier)
        hashes.add(digest)
        arrays.append(array)
    # Different NPZ metadata/compression cannot disguise identical pixel content.
    import hashlib

    pixel_hashes = {hashlib.sha256(a.tobytes()).hexdigest() for a in arrays}
    if len(pixel_hashes) != 30:
        raise ValueError("Duplicate RGB pixel arrays")
    return arrays


def validate_checkpoint(state, model):
    if state.get("step") != 2000 or not state.get("model"):
        raise ValueError("Only completed 2000-update checkpoints are accepted")
    metadata = state.get("metadata", {})
    if model == "sen2sr":
        if state.get("format") != 1 or metadata.get("channels") != 3:
            raise ValueError("Expected full SEN2SR RGB3 checkpoint")
    elif model == "swinir":
        if metadata.get("architecture") != "official-SwinIR-M-classical-RGB-x4":
            raise ValueError("Expected official full SwinIR-M RGB x4 checkpoint")
    else:
        raise ValueError("Unknown model")


def save_example(output, row, lr, sen, swin):
    validate_rgb(lr, 128)
    for array in (sen, swin):
        if (
            array.shape != (3, 512, 512)
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise ValueError("Model output must be finite float32 RGB 512x512")
    directory = output / "examples" / row["id"]
    directory.mkdir(parents=True, exist_ok=True)
    arrays_path = directory / "arrays.npz"
    np.savez_compressed(arrays_path, lr=lr, sen2sr=sen, swinir=swin)
    images, hashes = {}, {"arrays.npz": file_hash(arrays_path)}
    for name, array in [("original", lr), ("sen2sr", sen), ("swinir", swin)]:
        path = directory / f"{name}.png"
        rgb_image(array).save(path)
        images[name] = path.relative_to(output).as_posix()
        hashes[name] = file_hash(path)
    return dict(
        id=row["id"],
        title=row["title"],
        provenance=row["provenance"],
        input_sha256=row["sha256"],
        images=images,
        arrays=arrays_path.relative_to(output).as_posix(),
        output_hashes=hashes,
        output_ranges={
            name: [float(a.min()), float(a.max())]
            for name, a in [("sen2sr", sen), ("swinir", swin)]
        },
    )


def atomic_json(path, value):
    temp = path.with_suffix(".json.partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "sen-checkpoint", "swin-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sen-assets", type=Path, default=Path("checkpoints/released"))
    parser.add_argument("--swin-assets", type=Path, default=Path("assets/swinir"))
    parser.add_argument(
        "--swin-lock", type=Path, default=Path("configs/swinir-lock.json")
    )
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Full SEN2SR requires real CUDA execution")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError(
            "Output directory must be empty to prevent stale completion or mixed results"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    arrays = validate_inputs(manifest, args.manifest.parent)
    models = {}
    states = {}
    for name, path in [
        ("sen2sr", args.sen_checkpoint),
        ("swinir", args.swin_checkpoint),
    ]:
        # These are trusted local artifacts produced by this project's training jobs.
        state = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint(state, name)
        states[name] = state
        models[name] = dict(
            checkpoint_sha256=file_hash(path),
            checkpoint_step=state["step"],
            metadata=state["metadata"],
        )
    if states["sen2sr"]["metadata"].get("lock") != json.loads(DEFAULT_LOCK.read_text()):
        raise ValueError("Trained SEN2SR checkpoint and installed lock differ")
    sen = load_full_model(
        args.sen_assets, device=args.device, lock_path=DEFAULT_LOCK, channels=3
    )
    sen.load_state_dict(states.pop("sen2sr")["model"], strict=True)
    sen.float().eval().requires_grad_(False)
    swin, lock = load_swinir(args.swin_assets, args.swin_lock)
    if states["swinir"]["metadata"].get("model_lock") != lock:
        raise ValueError("Trained SwinIR checkpoint and installed lock differ")
    swin.load_state_dict(states.pop("swinir")["model"], strict=True)
    swin.to(args.device).float().eval().requires_grad_(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        input_manifest_sha256=file_hash(args.manifest),
        models=models,
        precision="fp32",
        device=args.device,
        gpu=torch.cuda.get_device_name(args.device),
        display="Fixed RGB 0..1 clipped only for PNG; no enhancement",
        original_pixel_m=10,
        sr_grid_pixel_m=2.5,
        warning="SR detail is inferred; no matched high-resolution ground truth",
        examples=[],
    )
    for row, lr in zip(manifest["examples"], arrays):
        tensor = torch.from_numpy(lr.copy()).unsqueeze(0).to(args.device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            sen_array = sen(tensor).float().squeeze(0).cpu().numpy()
            swin_array = swin(tensor).float().squeeze(0).cpu().numpy()
        result["examples"].append(
            save_example(args.output, row, lr, sen_array, swin_array)
        )
        print(
            json.dumps(dict(completed=len(result["examples"]), total=30, id=row["id"])),
            flush=True,
        )
    atomic_json(args.output / "manifest.json", result)
    atomic_json(
        args.output / "completion.json",
        dict(
            status="completed",
            examples=30,
            png_images=90,
            manifest_sha256=file_hash(args.output / "manifest.json"),
            completed_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )


if __name__ == "__main__":
    main()
