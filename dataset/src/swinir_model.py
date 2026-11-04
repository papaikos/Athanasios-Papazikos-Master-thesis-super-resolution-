"""Pinned official full SwinIR-M classical RGB x4; strict pretrained loading."""

import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.request


def verify_asset(path, expected):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"SwinIR asset integrity mismatch: {path}")
    return path


def load_swinir(assets, lock_path, download=False):
    import torch

    assets = Path(assets)
    lock = json.loads(Path(lock_path).read_text())
    assets.mkdir(parents=True, exist_ok=True)
    for item in lock["assets"]:
        path = assets / item["name"]
        if not path.exists() and download:
            temp = path.with_suffix(path.suffix + ".partial")
            urllib.request.urlretrieve(item["url"], temp)
            verify_asset(temp, item["sha256"])
            temp.replace(path)
        verify_asset(path, item["sha256"])
    spec = importlib.util.spec_from_file_location(
        "pinned_network_swinir", assets / "network_swinir.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.SwinIR(**lock["architecture"])
    weights = torch.load(
        assets / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
        map_location="cpu",
        weights_only=True,
    )
    net.load_state_dict(weights.get("params", weights), strict=True)
    return net, lock
