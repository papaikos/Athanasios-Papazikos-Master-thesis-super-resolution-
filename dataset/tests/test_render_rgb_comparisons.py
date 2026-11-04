import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.satellite_training_data import seal

spec = importlib.util.spec_from_file_location("comparison", Path(__file__).parents[1] / "scripts/render_rgb_comparisons.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_selection_only_test_deterministic_without_replacement():
    manifest = seal({"sources": [{"split": "train", "pairs": 100}, {"split": "test", "pairs": 20}, {"split": "validation", "pairs": 10}]})
    selected = comparison.test_selection(manifest)
    assert selected == comparison.test_selection(manifest)
    assert len(set(selected)) == 15
    assert all(source == 1 for source, _ in selected)
    manifest["sources"][1]["pairs"] = 21
    with pytest.raises(ValueError, match="checksum"):
        comparison.test_selection(manifest)


def test_rgb_scaling_and_metrics_are_fixed():
    image = comparison.rgb_image(np.full((3, 5, 5), 0.25, dtype=np.float32))
    assert np.asarray(image).min() == np.asarray(image).max() == 64
    assert comparison.psnr(np.zeros((3, 5, 5)), np.ones((3, 5, 5))) == 0
    assert comparison.psnr(np.zeros((3, 5, 5)), np.zeros((3, 5, 5))) is None
    with pytest.raises(ValueError):
        comparison.validate_rgb(np.full((3, 128, 128), np.nan, dtype=np.float32), 128)


def test_sheet_layout_real_dimensions(tmp_path):
    patch = tmp_path / "patch.png"
    Image.new("RGB", (512, 512), "#8090a0").save(patch)
    examples = [{"title": f"Example {i}", "images": {key: str(patch) for key in ("lr", "sr", "hr", "bicubic")}, "caption": "Known test caption"} for i in range(15)]
    comparison.render_sheet(examples, tmp_path / "test.png")
    comparison.render_sheet(examples, tmp_path / "sentinel.png", sentinel=True)
    assert Image.open(tmp_path / "test.png").size == (3600, 2400)
    assert Image.open(tmp_path / "sentinel.png").size == (3600, 3700)


def test_sentinel_direct_google_download_preserves_tci(tmp_path, monkeypatch):
    import hashlib
    from io import BytesIO
    data = np.zeros((128, 128), dtype=[(band, "u1") for band in ("TCI_R", "TCI_G", "TCI_B")])
    data["TCI_R"] = 255
    stream = BytesIO()
    np.save(stream, data)
    monkeypatch.setattr(comparison, "urlopen", lambda url, timeout: BytesIO(stream.getvalue()))
    row = {"download_url": "https://example.invalid/test", "download_sha256": hashlib.sha256(stream.getvalue()).hexdigest()}
    result = comparison.sentinel_array(row, tmp_path)
    assert result.shape == (3, 128, 128)
    assert result[0].min() == 1 and result[1:].max() == 0
    row["download_sha256"] = "wrong"
    with pytest.raises(ValueError, match="checksum"):
        comparison.sentinel_array(row, tmp_path)
