"""Fixed validation panels with display ranges chosen before model predictions."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .checkpointing import atomic_json
from .evaluate import rng_state, restore_rng


def write_panels(model, dataset, device, root, step):
    root = Path(root)
    if not hasattr(dataset, "members"):
        raise ValueError("Manifest-backed validation membership required for visual selection")
    indices = sorted(range(len(dataset)), key=lambda i: Path(dataset.members[i][1]).stem)[:3]
    selection = {"scope": "fixed main-validation diagnostics; no selection by predictions",
                 "sample_ids": [Path(dataset.members[i][1]).stem for i in indices],
                 "rgb_display_range": [0, 1], "error_display_range": [0, 0.25],
                 "coordinates": "full frame", "prediction_clip": [0, 1]}
    selection_path = root / "config" / "fixed-visual-identities.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Fixed visual identities or display ranges changed")
    atomic_json(selection_path, selection)
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    modes = {module: module.training for module in model.modules()}
    rng = rng_state()
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    files = []
    try:
        model.eval()
        with torch.inference_mode():
            for index in indices:
                pair = dataset[index]
                x = pair["lr"][None].to(device)
                hr = pair["hr"].float().cpu()
                raw = model(x)[0].float().cpu()
                if raw.shape != hr.shape or not torch.isfinite(raw).all():
                    raise ValueError("Malformed or nonfinite visual prediction")
                sr = raw.clamp(0, 1)
                bicubic = torch.nn.functional.interpolate(
                    x.float(), size=hr.shape[-2:], mode="bicubic",
                    align_corners=False, antialias=False)[0].cpu().clamp(0, 1)
                error = (sr - hr).abs().mean(0, keepdim=True).expand_as(hr).div(0.25).clamp(0, 1)
                height, width = hr.shape[-2:]
                canvas = Image.new("RGB", (width * 4, height + 48), "white")
                draw = ImageDraw.Draw(canvas)
                for column, (label, pixels) in enumerate([
                    ("Bicubic", bicubic), ("SwinIR", sr), ("HR", hr), ("Mean |RGB error| 0..0.25", error)]):
                    array = np.rint(pixels.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
                    canvas.paste(Image.fromarray(array), (column * width, 24))
                    draw.text((column * width + 2, 2), label, fill="black")
                sample_id = str(pair["sample_id"])
                if Path(sample_id).name != sample_id or "/" in sample_id or "\\" in sample_id:
                    raise ValueError("Unsafe visual sample identity")
                draw.text((2, height + 28), f"step={step}; id={sample_id}; RGB 0..1", fill="black")
                path = output / f"validation-step-{step}-{sample_id}.png"
                canvas.save(path)
                files.append(path)
        if any(not torch.equal(before[key], value.detach().cpu()) for key, value in model.state_dict().items()):
            raise ValueError("Visual inference mutated model state")
    finally:
        for module, training in modes.items():
            module.training = training
        restore_rng(rng)
    return files
