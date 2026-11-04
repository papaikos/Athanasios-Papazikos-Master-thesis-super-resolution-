import hashlib
import numpy as np
import pytest
from scripts.infer_sentinel30_dual import (
    validate_inputs,
    validate_checkpoint,
    save_example,
)


def make_inputs(tmp_path):
    rows = []
    for i in range(30):
        path = tmp_path / f"{i}.npz"
        np.savez_compressed(path, lr=np.full((3, 128, 128), i / 30, dtype=np.float32))
        rows.append(
            dict(
                id=f"id-{i}",
                title=str(i),
                path=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                provenance=dict(
                    collection="COPERNICUS/S2_SR_HARMONIZED", native_pixel_m=10
                ),
            )
        )
    return dict(examples=rows)


def test_inputs_require_30_unique_verified_native_sentinel(tmp_path):
    manifest = make_inputs(tmp_path)
    assert len(validate_inputs(manifest, tmp_path)) == 30
    manifest["examples"][1]["sha256"] = manifest["examples"][0]["sha256"]
    with pytest.raises(ValueError):
        validate_inputs(manifest, tmp_path)


def test_wrong_count_and_resolution_rejected(tmp_path):
    manifest = make_inputs(tmp_path)
    with pytest.raises(ValueError):
        validate_inputs(dict(examples=manifest["examples"][:29]), tmp_path)
    manifest["examples"][0]["provenance"]["native_pixel_m"] = 2.5
    with pytest.raises(ValueError):
        validate_inputs(manifest, tmp_path)


def test_incomplete_or_wrong_checkpoint_rejected():
    sen = dict(format=1, step=2000, metadata=dict(channels=3), model={"x": 1})
    validate_checkpoint(sen, "sen2sr")
    sen["step"] = 1999
    with pytest.raises(ValueError):
        validate_checkpoint(sen, "sen2sr")
    with pytest.raises(ValueError):
        validate_checkpoint(dict(step=2000, model={}, metadata={}), "swinir")


def test_native_original_and_outputs_same_fixed_display(tmp_path):
    from PIL import Image

    lr = np.full((3, 128, 128), 0.5, dtype=np.float32)
    sr = np.full((3, 512, 512), 1.5, dtype=np.float32)
    item = save_example(
        tmp_path, dict(id="port", title="Port", provenance={}, sha256="abc"), lr, sr, sr
    )
    assert Image.open(tmp_path / item["images"]["original"]).size == (128, 128)
    assert Image.open(tmp_path / item["images"]["sen2sr"]).getpixel((0, 0)) == (
        255,
        255,
        255,
    )
    with np.load(tmp_path / item["arrays"]) as data:
        assert data["sen2sr"].max() == 1.5
        assert np.array_equal(data["lr"], lr)
    sr[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        save_example(tmp_path, dict(id="bad"), lr, sr, sr)
