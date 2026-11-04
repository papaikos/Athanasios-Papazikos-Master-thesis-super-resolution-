import numpy as np
import pytest

from scripts.expand_rgb_cloud import (
    count_valid_rgb,
    unique_accounting,
    validate_native_grid,
    selected_candidates,
    qa_object,
)


def test_valid_bytes_exclude_any_zero_and_masked_pixel():
    rgb = np.ones((3, 2, 3), dtype=np.uint8)
    rgb[1, 0, 0] = 0
    masks = np.full(rgb.shape, 255, dtype=np.uint8)
    masks[2, 1, 2] = 0
    assert count_valid_rgb(rgb, masks) == 4
    with pytest.raises(ValueError, match="uint8"):
        count_valid_rgb(rgb.astype(np.float32), masks)


def test_duplicate_content_and_baseline_crc_not_counted():
    objects = [
        dict(sha256="a", crc32c="old", size=100, valid_pixels=10, category="port"),
        dict(sha256="b", crc32c="new", size=110, valid_pixels=20, category="port"),
        dict(sha256="b", crc32c="new", size=110, valid_pixels=20, category="shoreline"),
    ]
    result = unique_accounting(objects, {(100, "old")}, 84_026_880_000)
    assert result["added_useful_bytes"] == 60
    assert result["total_useful_bytes"] == 84_026_880_060
    assert result["useful_bytes_by_category"] == {"port": 60}


def test_native_grid_rejects_degrees_or_resampled_coarse_grid():
    grid = dict(
        crsCode="EPSG:32644",
        dimensions=dict(width=10, height=10),
        affineTransform=dict(scaleX=0.8, scaleY=-0.8, translateX=0, translateY=8),
    )
    validate_native_grid(grid)
    with pytest.raises(ValueError, match="UTM"):
        validate_native_grid({**grid, "crsCode": "EPSG:4326"})
    with pytest.raises(ValueError, match="2.5"):
        validate_native_grid(
            {**grid, "affineTransform": {**grid["affineTransform"], "scaleX": 10}}
        )


def test_ports_first_and_unique_across_rois_and_catalogs(monkeypatch):
    candidates = [
        dict(asset=dict(id="a", properties=dict(catalogID="same"))),
        dict(asset=dict(id="b", properties=dict(catalogID="same"))),
    ]
    monkeypatch.setattr(
        "scripts.expand_rgb_cloud.intake.candidate_assets", lambda *args: candidates
    )
    rois = [
        dict(id="shore", category="shoreline"),
        dict(id="port", category="port_marina_vicinity"),
    ]
    selected = list(selected_candidates([], [], rois))
    assert [(c["asset"]["id"], r["id"]) for c, r in selected] == [("a", "port")]


def test_cloud_qa_checks_native_raster_and_pins_download(tmp_path):
    import shutil
    import rasterio
    from rasterio.transform import from_origin
    from types import SimpleNamespace

    original = tmp_path / "original.tif"
    rgb = np.ones((3, 2, 3), dtype=np.uint8)
    rgb[0, 0, 0] = 0
    with rasterio.open(
        original,
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=3,
        dtype="uint8",
        crs="EPSG:32644",
        transform=from_origin(0, 2, 1, 1),
        nodata=0,
    ) as dst:
        dst.write(rgb)

    def download(path, **kwargs):
        assert kwargs == dict(if_generation_match=7, checksum="crc32c")
        shutil.copyfile(original, path)

    blob = SimpleNamespace(
        generation=7,
        download_to_filename=download,
        bucket=SimpleNamespace(name="b"),
        name="rgb.tif",
        size=original.stat().st_size,
        crc32c="crc",
    )
    row = dict(
        source_id="s",
        asset_id="a",
        category="port",
        grid=dict(
            crsCode="EPSG:32644",
            affineTransform=dict(scaleX=1, scaleY=-1, translateX=0, translateY=2),
        ),
    )
    result = qa_object(blob, row, tmp_path)
    assert result["valid_pixels"] == 5
    assert result["generation"] == "7"
    assert not (tmp_path / "source.tif").exists()


def test_resume_excludes_success_rejection_and_error():
    from scripts.expand_rgb_cloud import attempted_assets

    assert attempted_assets(
        dict(
            sources=[dict(asset_id="a")],
            rejects=[dict(asset_id="b")],
            errors=[dict(asset_id="c")],
        )
    ) == {"a", "b", "c"}


def test_existing_preview_requires_identical_content():
    from google.api_core.exceptions import PreconditionFailed
    from scripts.expand_rgb_cloud import upload_preview

    class Blob:
        generation = 7

        def upload_from_string(self, *args, **kwargs):
            assert kwargs["if_generation_match"] == 0
            raise PreconditionFailed("already exists")

        def reload(self):
            pass

        def download_as_bytes(self, **kwargs):
            assert kwargs["if_generation_match"] == 7
            return b"same"

    upload_preview(Blob(), b"same")
    with pytest.raises(ValueError, match="different"):
        upload_preview(Blob(), b"changed")


def test_screen_continues_after_candidate_failure_and_resumes_without_replay(
    monkeypatch, tmp_path
):
    import json
    from scripts import expand_rgb_cloud as runner

    saved = {}

    class Blob:
        def __init__(self, name):
            self.name = name

        def exists(self):
            return self.name in saved

        def download_as_text(self):
            return json.dumps(saved[self.name])

    class Bucket:
        def blob(self, name):
            return Blob(name)

    candidates = [
        (dict(asset=dict(id=id)), dict(id="port")) for id in ("failed", "good")
    ]
    calls = []

    def screen(ee, candidate, *args):
        id = candidate["asset"]["id"]
        calls.append(id)
        if id == "failed":
            raise RuntimeError("candidate HTTP failure")
        return dict(asset_id=id)

    def publish(bucket, prefix, name, content, immutable=False):
        key = prefix + "/" + name
        if immutable:
            assert key not in saved
        saved[key] = json.loads(json.dumps(content))

    monkeypatch.setattr(runner, "load_inputs", lambda c: ([], []))
    monkeypatch.setattr(runner, "selected_candidates", lambda *args: iter(candidates))
    monkeypatch.setattr(runner, "screen_one", screen)
    monkeypatch.setattr(runner, "publish", publish)
    config = dict(rois=[], output_prefix="run")
    first = runner.run_screen(None, Bucket(), config, tmp_path)
    assert first["status"] == "screened_with_errors"
    assert len(first["errors"]) == len(first["sources"]) == 1
    runner.run_screen(None, Bucket(), config, tmp_path)
    assert calls == ["failed", "good"]
