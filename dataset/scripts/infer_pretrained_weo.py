"""Run the pinned, unmodified WEO-SAS SEN2SR full ten-band pipeline.

Download uses the immutable HF revision; strict state loading is upstream default.
No training or synthetic weight fallback is supported. Input manifest is produced
by acquire_pretrained_sentinel15.py. Display RGB is reflectance / 0.3, uniformly.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

REPO = "WEO-SAS/sen2sr"
REVISION = "5262f02bd454915de7a77cb8627f581255186719"
SEN2SR_COMMIT = "8e21bb669bcc6e8eb953a1fb24dfc5bb59dc18a4"
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
FILES = ["README.md", "base.py", "config.json", "model.py", "sen2sr_pt.py",
         "predictor.py", "load.py", "mlm.json", "model.safetensor",
         "hard_constraint.safetensor", "sr_model.safetensor",
         "sr_hard_constraint.safetensor", "f2_model.safetensor",
         "f2_hard_constraint.safetensor"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def download(directory):
    from huggingface_hub import snapshot_download
    directory = Path(snapshot_download(REPO, revision=REVISION,
                                       allow_patterns=FILES, local_dir=str(directory)))
    receipt = {"repository": REPO, "revision": REVISION,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "files": {name: {"sha256": sha(directory / name),
                                "bytes": (directory / name).stat().st_size} for name in FILES}}
    write_json(directory / "download-receipt.json", receipt)
    return directory


def verify_model(directory):
    receipt = json.loads((directory / "download-receipt.json").read_text("utf-8"))
    if receipt["repository"] != REPO or receipt["revision"] != REVISION:
        raise ValueError("Unrecognized checkpoint revision")
    for name in FILES:
        if sha(directory / name) != receipt["files"][name]["sha256"]:
            raise ValueError(f"Checkpoint/source checksum mismatch: {name}")
    cfg = json.loads((directory / "config.json").read_text("utf-8"))
    if (cfg.get("variant"), cfg.get("srmodel_type"), cfg.get("in_channels"),
        cfg.get("scaling_factor")) != ("mamba-main", "referencex4", 10, 4):
        raise ValueError("Expected full pretrained mamba-main referencex4 pipeline")
    return receipt


def validate_inputs(manifest, directory):
    import numpy as np
    rows = manifest.get("examples", [])
    if manifest.get("status") != "complete" or len(rows) != 15:
        raise ValueError("Require exactly 15 completed input examples")
    result, identifiers, pixels = [], set(), set()
    directory = directory.resolve()
    for row in rows:
        key = row["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key) or key in identifiers:
            raise ValueError("Invalid or duplicate example ID")
        identifiers.add(key)
        path = (directory / row["path"]).resolve()
        if not path.is_relative_to(directory) or sha(path) != row["sha256"]:
            raise ValueError("Input checksum/path mismatch")
        prov = row["provenance"]
        if prov.get("collection") != "COPERNICUS/S2_SR_HARMONIZED" or prov.get("input_grid_m") != 10:
            raise ValueError("Expected Sentinel-2 surface reflectance on preserved 10 m grid")
        with np.load(path, allow_pickle=False) as z:
            names = z["bands"].tolist()
            if len(names) != 10 or set(names) != set(BANDS) or names != prov["bands"]:
                raise ValueError("Invalid ten-band names/order")
            order = [names.index(b) for b in BANDS]
            array = z["reflectance"][order].copy()
            valid = z["valid_mask"][order].copy()
            scl = z["scl"].copy()
            tci = z["tci"].copy()
        if array.shape != (10, 128, 128) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("Expected finite float32 ten-band 128x128 reflectance")
        if valid.shape != array.shape or valid.dtype != np.bool_ or valid.all(axis=0).mean() < .999:
            raise ValueError("Invalid or insufficient valid spectral coverage")
        if scl.shape != (128, 128) or tci.shape != (3, 128, 128):
            raise ValueError("Invalid ancillary input shapes")
        if tci.dtype != np.float32 or not np.isfinite(tci).all() or tci.min() < 0 or tci.max() > 1:
            raise ValueError("Original TCI must be normalized float32 RGB in [0,1]")
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        if digest in pixels:
            raise ValueError("Duplicate spectral pixels")
        pixels.add(digest)
        result.append((row, array, valid, scl, tci))
    return result


def image(array, white):
    import numpy as np
    from PIL import Image
    return Image.fromarray(np.rint(np.clip(array / white, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--download", action="store_true")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--inputs", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.download or args.download_only:
        download(args.model_dir)
    weights = verify_model(args.model_dir)
    if args.download_only:
        print(json.dumps(weights), flush=True)
        return
    if args.inputs is None or args.output is None:
        p.error("--inputs and --output required for inference")
    import importlib.metadata
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Full Mamba inference requires compatible CUDA; no surrogate CPU fallback")
    dist = importlib.metadata.distribution("sen2sr")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != SEN2SR_COMMIT:
        raise RuntimeError("sen2sr source must be installed at the pinned commit")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(17)
    manifest_path = args.inputs / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    inputs = validate_inputs(manifest, args.inputs)
    sys.path.insert(0, str(args.model_dir.resolve()))
    spec = importlib.util.spec_from_file_location("weo_official_model", args.model_dir / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Official Model builds all three pretrained backbones + Fourier constraints.
    # Every upstream load_state_dict uses its strict=True default.
    model = module.Model(local_dir=str(args.model_dir.resolve()))
    network = model._model._predictor.model.eval().float()
    if any(v.dtype != torch.float32 for v in network.parameters()):
        raise RuntimeError("All inference model parameters must be FP32")
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"status": "running", "repository": REPO, "revision": REVISION,
               "variant": "mamba-main", "model_files": weights["files"],
               "input_manifest_sha256": sha(manifest_path), "input_bands": BANDS,
               "output_bands": BANDS, "input_grid_m": 10, "output_grid_m": 2.5,
               "native_input_m": [10, 10, 10, 20, 20, 20, 10, 20, 20, 20],
               "display": "RGB=[B4,B3,B2]; reflectance/0.3 clipped only for PNG; no gamma/stretch",
               "precision": "FP32; TF32 disabled; no AMP", "weights_modified": False,
               "strict_loading": True, "sen2sr_commit": SEN2SR_COMMIT,
               "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
               "started_utc": datetime.now(timezone.utc).isoformat(), "examples": []}
    write_json(args.output / "result.json", receipt)
    for row, lr, valid, scl, tci in inputs:
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            sr = model.predict(lr)
        torch.cuda.synchronize()
        if sr.shape != (10, 512, 512) or sr.dtype != np.float32 or not np.isfinite(sr).all():
            raise RuntimeError(f"Invalid full-pipeline output for {row['id']}")
        output = args.output / "examples" / row["id"]
        output.mkdir(parents=True, exist_ok=True)
        metadata = dict(row["provenance"], output_bands=BANDS, model_revision=REVISION,
                        output_grid_m=2.5, display_white_reflectance=.3)
        np.savez_compressed(output / "arrays.npz", original=lr, sen2sr=sr, valid_mask=valid,
                            scl=scl, original_tci=tci, bands=np.array(BANDS),
                            provenance=np.array(json.dumps(metadata)))
        image(lr[[2, 1, 0]], .3).save(output / "original.png")
        image(sr[[2, 1, 0]], .3).save(output / "weo_sen2sr.png")
        image(tci, 1).save(output / "original_tci.png")
        record = {"id": row["id"], "title": row["title"], "source": row,
                  "elapsed_seconds": time.monotonic() - start,
                  "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                  "input_minmax": [float(lr.min()), float(lr.max())],
                  "output_minmax": [float(sr.min()), float(sr.max())],
                  "rgb_display_clipped_fraction": float(((sr[[2, 1, 0]] < 0) | (sr[[2, 1, 0]] > .3)).mean()),
                  "files": {f.name: {"path": f.relative_to(args.output).as_posix(),
                                      "sha256": sha(f)} for f in sorted(output.iterdir()) if f.is_file()}}
        receipt["examples"].append(record)
        write_json(args.output / "result.json", receipt)
        print(json.dumps({"completed": len(receipt["examples"]), "id": row["id"],
                          "seconds": record["elapsed_seconds"]}), flush=True)
    receipt.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(), count=15)
    write_json(args.output / "result.json", receipt)


if __name__ == "__main__":
    main()
