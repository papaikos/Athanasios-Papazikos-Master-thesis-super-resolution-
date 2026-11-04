"""Reproducible full-model RGB comparison sheets; no training or imagery synthesis."""

import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import random
import subprocess
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.model import download_assets, file_hash, load_full_model, DEFAULT_LOCK
from src.satellite_training_data import seal, stage_dataset


def test_selection(manifest, count=15, seed=20260926):
    if manifest.get("sha256") != seal(manifest)["sha256"]:
        raise ValueError("Frozen manifest checksum mismatch")
    population = [(i, j) for i, row in enumerate(manifest["sources"])
                  if row["split"] == "test" for j in range(row["pairs"])]
    return random.Random(seed).sample(population, count)


def stage_tests(manifest, output, fetch, count=15, seed=20260926):
    selection = test_selection(manifest, count, seed)
    selected_sources = sorted({i for i, _ in selection})
    sources = [manifest["sources"][i] for i in selected_sources]
    subset = seal(dict(version=1, sources=sources, totals={"test": {
        "pairs": sum(r["pairs"] for r in sources),
        "accepted_source_bytes": sum(r["accepted_source_bytes"] for r in sources)}}))
    rows = stage_dataset(subset, output, fetch)
    grouped = {}
    for row in rows:
        key = json.dumps(row["source_object"], sort_keys=True)
        grouped.setdefault(key, []).append(row)
    selected = []
    for source_index, ordinal in selection:
        key = json.dumps(manifest["sources"][source_index]["source_object"], sort_keys=True)
        selected.append(sorted(grouped[key], key=lambda r: r["path"])[ordinal])
    return selected


def validate_rgb(array, size):
    if array.shape != (3, size, size) or array.dtype != np.float32:
        raise ValueError("Expected float32 CHW RGB at the registered patch size")
    if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
        raise ValueError("Input RGB must be finite and within [0,1]")


def sentinel_array(row, directory):
    if "download_url" in row:
        with urlopen(row["download_url"], timeout=180) as response:
            payload = response.read(16_000_001)
        if len(payload) > 16_000_000:
            raise ValueError("Unexpectedly large Sentinel patch")
        if hashlib.sha256(payload).hexdigest() != row["download_sha256"]:
            raise ValueError("Sentinel download checksum mismatch")
        data = np.load(BytesIO(payload), allow_pickle=False)
        lr = np.stack([data[band] for band in ("TCI_R", "TCI_G", "TCI_B")]).astype(np.float32) / 255
        validate_rgb(lr, 128)
        return lr
    path = directory / row["path"]
    if row.get("sha256") and file_hash(path) != row["sha256"]:
        raise ValueError("Sentinel input hash mismatch")
    with np.load(path, allow_pickle=False) as data:
        lr = data["lr"].copy()
    validate_rgb(lr, 128)
    return lr


def psnr(prediction, reference):
    error = float(np.mean((prediction.astype(np.float64) - reference) ** 2))
    return None if error == 0 else -10 * math.log10(error)


def rgb_image(array):
    return Image.fromarray(np.rint(np.clip(array, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8))


def font(size):
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def render_sheet(examples, path, sentinel=False):
    if len(examples) != 15:
        raise ValueError("The comparison sheet requires exactly fifteen examples")
    width, height = 3600, 2400 if not sentinel else 3700
    canvas = Image.new("RGB", (width, height), "#101722")
    draw = ImageDraw.Draw(canvas)
    title = "SEN2SR RGB x4 · 15 Sentinel-2 maritime examples" if sentinel else "SEN2SR RGB x4 · 15 random held-out SkySat test examples"
    subtitle = ("Original 10 m RGB | Model output on 2.5 m grid · No matched HR reference; detail is inferred"
                if sentinel else "LR input | Bicubic | Model output | HR reference · Shared RGB 0–1; no contrast enhancement")
    draw.text((60, 30), title, font=font(47), fill="white")
    draw.text((60, 98), subtitle, font=font(25), fill="#bac3d1")
    group_w = 1160
    row_h = 680 if sentinel else 420
    tile = 548 if sentinel else 270
    labels = ["Original · 10 m", "SR · 2.5 m grid"] if sentinel else ["LR input · 10 m", "Bicubic", "Model · 2.5 m grid", "HR · 2.5 m"]
    keys = ["lr", "sr"] if sentinel else ["lr", "bicubic", "sr", "hr"]
    for i, item in enumerate(examples):
        x, y = 60 + (i % 3) * group_w, 180 + (i // 3) * row_h
        draw.text((x, y), f"{i + 1:02d} · {item['title']}", font=font(27), fill="white")
        for j, (key, label) in enumerate(zip(keys, labels)):
            xx = x + j * (tile + 8)
            draw.text((xx, y + 45), label, font=font(21), fill="#78d9cd" if key == "sr" else "#cfd5df")
            im = Image.open(item["images"][key]).convert("RGB")
            canvas.paste(im.resize((tile, tile), Image.Resampling.NEAREST if key == "lr" else Image.Resampling.LANCZOS), (xx, y + 80))
        caption = item.get("caption", "")
        draw.text((x, y + 88 + tile), caption, font=font(21), fill="#bac3d1")
    footer = "All display panels use the same fixed 0–1 scale. Full RGB SEN2SR; best validation checkpoint."
    draw.text((60, height - 65), footer, font=font(23), fill="#bac3d1")
    canvas.save(path)


def infer_example(net, lr, output, identifier, hr=None):
    validate_rgb(lr, 128)
    if hr is not None:
        validate_rgb(hr, 512)
    tensor = torch.from_numpy(lr.copy()).unsqueeze(0).cuda()
    with torch.inference_mode():
        sr = net(tensor).float().squeeze(0).cpu().numpy()
        bicubic = torch.nn.functional.interpolate(tensor, size=(512, 512), mode="bicubic", align_corners=False).clamp(0, 1).squeeze(0).cpu().numpy()
    if sr.shape != (3, 512, 512) or not np.isfinite(sr).all():
        raise ValueError("Full-model output failed shape/finite validation")
    directory = output / identifier
    directory.mkdir(parents=True, exist_ok=True)
    arrays = dict(lr=lr, sr=sr, bicubic=bicubic)
    if hr is not None:
        arrays["hr"] = hr
    np.savez_compressed(directory / "arrays.npz", **arrays)
    images = {}
    for key, value in arrays.items():
        destination = directory / f"{key}.png"
        rgb_image(value).save(destination)
        images[key] = str(destination)
    metrics = {}
    if hr is not None:
        metrics = {"sr_psnr_db": psnr(np.clip(sr, 0, 1), hr), "bicubic_psnr_db": psnr(bicubic, hr),
                   "metric_protocol": "RGB, full 512x512 valid HR, peak 1, clipped predictions; no border shave"}
    return dict(images=images, metrics=metrics, output_min=float(sr.min()), output_max=float(sr.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--sentinel-manifest", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path("checkpoints/released"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gcloud", default="gcloud")
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if file_hash(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint hash mismatch")
    manifest = json.loads(args.manifest.read_text())
    def fetch(uri, target):
        subprocess.run([args.gcloud, "storage", "cp", uri, str(target)], check=True)
    selected = stage_tests(manifest, args.output / "heldout-data", fetch, seed=args.seed)
    download_assets(args.assets, json.loads(DEFAULT_LOCK.read_text()))
    net = load_full_model(args.assets, channels=3)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != 1 or checkpoint["metadata"].get("channels") != 3:
        raise ValueError("Expected trained full RGB3 checkpoint")
    net.load_state_dict(checkpoint["model"], strict=True)
    net.eval()
    results = dict(checkpoint_sha256=args.checkpoint_sha256, checkpoint_step=checkpoint["step"],
                   frozen_manifest_sha256=manifest["sha256"], seed=args.seed, heldout=[], sentinel=[])
    for i, row in enumerate(selected):
        path = args.output / "heldout-data" / row["path"]
        if file_hash(path) != row["sha256"]:
            raise ValueError("Held-out pair changed after staging")
        with np.load(path, allow_pickle=False) as data:
            item = infer_example(net, data["lr"], args.output / "heldout", f"{i+1:02d}", data["hr"])
        item.update(title=f"SkySat · test {i+1:02d}", provenance=row)
        score, baseline = item["metrics"]["sr_psnr_db"], item["metrics"]["bicubic_psnr_db"]
        item["caption"] = f"PSNR {score:.2f} dB · Δ vs bicubic {score-baseline:+.2f} dB" if score is not None and baseline is not None else "Exact reconstruction; see metrics.json"
        results["heldout"].append(item)
        print(json.dumps({"heldout_done": i+1}), flush=True)
    sentinel = json.loads(args.sentinel_manifest.read_text())
    if len(sentinel["examples"]) != 15:
        raise ValueError("Exactly fifteen Sentinel-2 examples required")
    for i, row in enumerate(sentinel["examples"]):
        lr = sentinel_array(row, args.sentinel_manifest.parent)
        item = infer_example(net, lr, args.output / "sentinel", f"{i+1:02d}")
        public_row = {key: value for key, value in row.items() if key not in ("url", "download_url")}
        item.update(title=row["title"], provenance=public_row, caption="No matched HR reference · no reference metrics")
        results["sentinel"].append(item)
        print(json.dumps({"sentinel_done": i+1}), flush=True)
    render_sheet(results["heldout"], args.output / "heldout-comparison.png")
    render_sheet(results["sentinel"], args.output / "sentinel-comparison.png", sentinel=True)
    for items in (results["heldout"], results["sentinel"]):
        for item in items:
            item["images"] = {key: Path(path).relative_to(args.output).as_posix() for key, path in item["images"].items()}
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
