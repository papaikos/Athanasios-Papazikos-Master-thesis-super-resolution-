import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from scripts import maritime_reuse_candidates as reuse
from test_maritime_cloud_inventory import cloud_fixture


def raster(tmp_path, name="source", left=100, data_value=1, mask=False, hole=None):
    path = tmp_path / (name + ".tif")
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(
        path, "w", driver="GTiff", width=3300, height=3300, count=3,
        dtype="uint8", crs="EPSG:32644", transform=from_origin(left, 3000, .8, .8),
        nodata=None if mask else 0, photometric="RGB") as ds:
        ds.write(np.full((3, 3300, 3300), data_value, dtype=np.uint8))
        if mask:
            validity = np.full((3300, 3300), 255, dtype=np.uint8)
            if hole:
                validity[hole] = 0
            ds.write_mask(validity)
    return path


def metadata():
    return dict(asset_id="asset", catalog_id="acquisition", source_id="source",
                export_recipe=dict(unmask=0, noData=0), timestamp="2015-01-01T00:00:00Z",
                site_id="harbour", license="CC-BY-4.0")


def test_two_overlapping_crops_have_same_native_world_core_identity(tmp_path):
    a = raster(tmp_path)
    b = raster(tmp_path, name="crop", left=500)
    with rasterio.open(a) as first, rasterio.open(b) as second:
        one = list(reuse.supported_windows(first, metadata()))
        two = list(reuse.supported_windows(second, metadata()))
    assert len(one) == len(two) == 1
    assert one[0]["global_core_identity"] == two[0]["global_core_identity"]
    assert one[0]["core_bounds"] == [1280, 1280, 2560, 2560]
    assert one[0]["core_window"] == [1475, 550, 1600, 1600]
    assert one[0]["context_window"] == [1425, 500, 1700, 1700]
    assert reuse.overlaps_existing(two[0], [one[0]])


def test_conservative_valid_support_and_invalid_halo(tmp_path):
    with rasterio.open(raster(tmp_path)) as ds:
        candidates = list(reuse.supported_windows(ds, metadata()))
    assert candidates[0]["mask_provenance"] == "legacy_nodata_zero_conservative"
    assert candidates[0]["original_mask_known"] is False
    with rasterio.open(raster(tmp_path, name="hole", mask=True, hole=(500, 1475))) as ds:
        source = {**metadata(), "mask_provenance": "original_explicit_dataset_mask"}
        assert list(reuse.supported_windows(ds, source)) == []


def test_valid_dark_zero_pixels_are_not_screened_by_intensity(tmp_path):
    path = raster(tmp_path, data_value=0, mask=True)
    with rasterio.open(path) as ds:
        source = {**metadata(), "mask_provenance": "original_explicit_dataset_mask"}
        assert len(list(reuse.supported_windows(ds, source))) == 1
        assert list(reuse.supported_windows(ds, metadata())) == []


def test_lossless_rgb_mask_and_native_png_exact_coverage(tmp_path):
    pytest.importorskip("PIL.Image")
    path = raster(tmp_path)
    with rasterio.open(path) as ds:
        candidate = next(reuse.supported_windows(ds, metadata()))
        artifacts = reuse.materialize(ds, candidate, tmp_path / "candidate")
        expected = ds.read(window=rasterio.windows.Window(*candidate["context_window"]))
    with rasterio.open(artifacts["rgb"]) as output:
        assert output.shape == (1700, 1700)
        np.testing.assert_array_equal(output.read(), expected)
        assert output.bounds == rasterio.coords.BoundingBox(1240, 1240, 2600, 2600)
    with rasterio.open(artifacts["mask"]) as output:
        assert output.count == 1
        assert np.all(output.read() == 255)
    from PIL import Image
    assert Image.open(artifacts["native_preview"]).size == (1600, 1600)
    assert Image.open(artifacts["overview"]).size == (400, 400)


def test_world_snap_never_moves_more_than_half_native_pixel(tmp_path):
    path = raster(tmp_path, left=100.3)
    with rasterio.open(path) as ds:
        candidate = next(reuse.supported_windows(ds, metadata()))
        assert max(abs(value) for value in candidate["native_snap_metres"]) <= .4 + 1e-6


def test_approved_assets_are_explicit_and_unrecognized_schema_rejected():
    assert reuse.approved_asset_ids({"asset_ids": ["a", "b"]}) == {"a", "b"}
    with pytest.raises(ValueError, match="asset_ids"):
        reuse.approved_asset_ids({"sites": ["near-port"]})


def configure_cloud(monkeypatch, tmp_path, source_count=1):
    args, source_blob, objects = cloud_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(reuse, "require_cloud_vm", lambda: "123")
    raw = raster(tmp_path, name="native")
    row = json.loads((tmp_path / "manifest.json").read_text())[0]
    row.update(bytes=raw.stat().st_size, export_recipe=dict(unmask=0, noData=0))
    source_blob.size = row["bytes"]

    def download(path, **kwargs):
        assert kwargs == dict(if_generation_match=7, checksum="crc32c")
        source_blob.downloads += 1
        shutil.copyfile(raw, path)
    source_blob.download_to_filename = download
    rows = [row]
    if source_count == 2:
        rows.append({**row, "uri": "gs://input/raw/duplicate-crop.tif"})
        objects[("input", "raw/duplicate-crop.tif")] = source_blob
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps(rows))
    approved = tmp_path / "approved.json"
    approved.write_text(json.dumps(dict(asset_ids=["a"])))
    return SimpleNamespace(sources=str(manifest), approved_assets=str(approved),
        output_prefix="gs://output/control/runs/reuse", scratch=args.scratch,
        max_candidates=10), source_blob, objects


def test_cloud_resume_restores_counter_and_skips_pinned_raw_download(monkeypatch, tmp_path):
    pytest.importorskip("PIL.Image")
    args, source, objects = configure_cloud(monkeypatch, tmp_path, source_count=2)
    first = reuse.run(args)
    assert first["candidate_count"] == 1
    assert first["processed_source_count"] == 2
    assert source.downloads == 2
    assert reuse.run(args) == first
    assert source.downloads == 2
    reports = [json.loads(blob.data) for (_, name), blob in objects.items() if "/source-reports/" in name]
    assert sum(report["skipped_overlaps"] for report in reports) == 1
    assert len(list((tmp_path / "scratch").iterdir())) == 1


def test_candidate_cap_marks_run_incomplete_and_is_stable_on_resume(monkeypatch, tmp_path):
    pytest.importorskip("PIL.Image")
    args, source, _ = configure_cloud(monkeypatch, tmp_path, source_count=2)
    args.max_candidates = 1
    first = reuse.run(args)
    assert first["candidate_count"] == 1
    assert first["processed_source_count"] == 1
    assert first["exhaustive"] is False
    assert first["semantic_accepted_count"] == first["vessel_confirmation_count"] == 0
    assert reuse.run(args) == first
    assert source.downloads == 1


def test_corrupted_candidate_artifact_blocks_resume(monkeypatch, tmp_path):
    pytest.importorskip("PIL.Image")
    args, _, objects = configure_cloud(monkeypatch, tmp_path)
    reuse.run(args)
    artifact = next(blob for (_, name), blob in objects.items() if name.endswith("native-core.png"))
    artifact.data = b"x" * artifact.size
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        reuse.run(args)


def test_rejected_header_is_recorded_with_raw_sha_and_other_sources_continue(monkeypatch, tmp_path):
    pytest.importorskip("PIL.Image")
    args, source, objects = configure_cloud(monkeypatch, tmp_path, source_count=2)
    original = reuse.validate_header
    calls = 0
    def validate(ds):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("Strict north-up metric 0.8 m native grid required")
        return original(ds)
    monkeypatch.setattr(reuse, "validate_header", validate)
    summary = reuse.run(args)
    assert summary["candidate_count"] == 1
    assert summary["processed_source_count"] == 2
    rejected = [json.loads(blob.data) for (_, name), blob in objects.items()
                if "/source-reports/" in name and json.loads(blob.data)["status"] == "rejected_header"]
    assert len(rejected) == 1
    assert rejected[0]["source_sha256"]
    assert rejected[0]["candidates"] == []


def producer_mask(tmp_path, *, left=100, value=255, hole=None):
    path = tmp_path / "producer-mask.tif"
    values = np.full((1, 3300, 3300), value, dtype=np.uint8)
    if hole:
        values[0, hole[0], hole[1]] = 0
    with rasterio.open(path, "w", driver="GTiff", width=3300, height=3300,
                       count=1, dtype="uint8", crs="EPSG:32644",
                       transform=from_origin(left, 3000, .8, .8), nodata=0) as ds:
        ds.write(values)
    return path


def test_original_producer_mask_preserves_black_rgb_despite_rgb_nodata(tmp_path):
    path = raster(tmp_path, data_value=0)
    mask = producer_mask(tmp_path)
    source = {**metadata(), "mask_provenance": "producer_original_min_rgb_mask"}
    with rasterio.open(path) as ds, rasterio.open(mask) as validity:
        candidates = list(reuse.supported_windows(ds, source, original_mask_ds=validity))
    assert len(candidates) == 1
    assert candidates[0]["original_mask_known"] is True
    assert candidates[0]["mask_provenance"] == "producer_original_min_rgb_mask"


@pytest.mark.parametrize("kwargs,message", [({"left": 100.8}, "exactly match"),
                                            ({"value": 128}, "binary")])
def test_producer_mask_grid_and_values_are_strict(tmp_path, kwargs, message):
    path = raster(tmp_path)
    mask = producer_mask(tmp_path, **kwargs)
    source = {**metadata(), "mask_provenance": "producer_original_min_rgb_mask"}
    with rasterio.open(path) as ds, rasterio.open(mask) as validity:
        with pytest.raises(ValueError, match=message):
            list(reuse.supported_windows(ds, source, original_mask_ds=validity))


def test_original_producer_halo_hole_is_rejected_and_edge_losses_counted(tmp_path):
    path = raster(tmp_path)
    mask = producer_mask(tmp_path, hole=(500, 1475))
    source = {**metadata(), "mask_provenance": "producer_original_min_rgb_mask"}
    stats = {}
    with rasterio.open(path) as ds, rasterio.open(mask) as validity:
        assert list(reuse.supported_windows(ds, source, original_mask_ds=validity, statistics=stats)) == []
    assert stats["invalid_core_or_halo_mask"] == 1
    assert stats["partial_world_cores"] > 0


def test_native_rgb_materialize_uses_producer_mask_support(tmp_path):
    pytest.importorskip("PIL.Image")
    path = raster(tmp_path, data_value=0)
    mask = producer_mask(tmp_path)
    source = {**metadata(), "mask_provenance": "producer_original_min_rgb_mask"}
    with rasterio.open(path) as ds, rasterio.open(mask) as validity:
        candidate = next(reuse.supported_windows(ds, source, original_mask_ds=validity))
        artifacts = reuse.materialize(ds, candidate, tmp_path / "output", original_mask_ds=validity)
    with rasterio.open(artifacts["rgb"]) as rgb, rasterio.open(artifacts["mask"]) as support:
        assert not np.any(rgb.read())
        assert np.all(support.read() == 255)


def test_scheduler_document_sources_normalized_and_review_group_metadata_joined():
    pin = dict(uri="gs://bucket/asset/rgb.tif", generation="7", bytes=10, crc32c="crc",
        asset_id="a", source_id="s", catalog_id="c",
        native_plan=dict(crs="EPSG:32644", crs_transform=[.8, 0, 0, 0, -.8, 2560],
                         native_bbox=[0, 0, 2560, 2560]),
        mask_provenance="producer_original_min_rgb_mask",
        producer_mask=dict(uri="gs://bucket/asset/mask.tif", generation="8", bytes=10, crc32c="mask-crc"))
    approval = dict(asset_ids=["a"], review_metadata=dict(groups=[
        dict(group_id="harbour", asset_ids=["a"], country=["Greece"], properties=dict(example="value"))]))
    rows = reuse.source_rows(dict(assets={"a": dict(source_objects=[pin])}), approval)
    assert rows[0]["producer_mask"]["generation"] == "8"
    assert rows[0]["site_id"] == "harbour"
    assert rows[0]["country"] == ["Greece"]
    assert rows[0]["approval_group"]["properties"]["example"] == "value"


def test_paired_pinned_mask_download_and_resume_preserve_valid_black_pixels(monkeypatch, tmp_path):
    pytest.importorskip("PIL.Image")
    from test_maritime_cloud_inventory import FakeBlob
    args, source, objects = configure_cloud(monkeypatch, tmp_path)
    raw = raster(tmp_path, name="black-original", data_value=0)
    mask = producer_mask(tmp_path)
    rows = json.loads((tmp_path / "sources.json").read_text())
    rows[0].update(bytes=raw.stat().st_size, mask_provenance="producer_original_min_rgb_mask")
    source.size = rows[0]["bytes"]
    def raw_download(path, **kwargs):
        assert kwargs == dict(if_generation_match=7, checksum="crc32c")
        source.downloads += 1
        shutil.copyfile(raw, path)
    source.download_to_filename = raw_download
    mask_blob = FakeBlob()
    mask_blob.generation = 8
    mask_blob.size = mask.stat().st_size
    mask_blob.crc32c = "mask-crc"
    mask_blob.downloads = 0
    def mask_download(path, **kwargs):
        assert kwargs == dict(if_generation_match=8, checksum="crc32c")
        mask_blob.downloads += 1
        shutil.copyfile(mask, path)
    mask_blob.download_to_filename = mask_download
    objects[("input", "raw/producer-mask.tif")] = mask_blob
    rows[0]["producer_mask"] = dict(uri="gs://input/raw/producer-mask.tif", generation="8",
        bytes=mask_blob.size, crc32c="mask-crc", sha256=reuse.file_sha256(mask))
    (tmp_path / "sources.json").write_text(json.dumps(dict(source_objects=rows)))
    first = reuse.run(args)
    assert first["candidate_count"] == 1
    assert first["candidates"][0]["producer_mask_sha256"] == reuse.file_sha256(mask)
    assert first["candidates"][0]["original_mask_known"] is True
    assert reuse.run(args) == first
    assert source.downloads == mask_blob.downloads == 1
    assert len(list((tmp_path / "scratch").iterdir())) == 1


def test_wrong_producer_mask_generation_fails_before_download(tmp_path):
    blob = SimpleNamespace(generation=9, size=10, crc32c="crc", reload=lambda: None)
    client = SimpleNamespace(bucket=lambda bucket: SimpleNamespace(blob=lambda name, generation: blob))
    pin = dict(uri="gs://bucket/mask.tif", generation="8", bytes=10, crc32c="crc")
    with pytest.raises(ValueError, match="metadata mismatch"):
        reuse.download_pinned(client, pin, tmp_path / "mask.tif")
    assert not (tmp_path / "mask.tif").exists()


def test_complete_world_core_at_shard_edge_is_counted_without_fake_halo(tmp_path):
    path = tmp_path / "edge.tif"
    with rasterio.open(path, "w", driver="GTiff", width=1600, height=1600,
                       count=3, dtype="uint8", crs="EPSG:32644", photometric="RGB",
                       transform=from_origin(0, 1280, .8, .8), nodata=0) as ds:
        ds.write(np.ones((3, 1600, 1600), dtype=np.uint8))
    statistics = {}
    with rasterio.open(path) as ds:
        assert list(reuse.supported_windows(ds, metadata(), statistics=statistics)) == []
    assert statistics["missing_real_halo"] == 1
    assert statistics["fully_supported_cores"] == 0


def test_missing_approved_assets_require_explicit_partial_screening_flag(monkeypatch, tmp_path):
    pytest.importorskip("PIL.Image")
    args, source, _ = configure_cloud(monkeypatch, tmp_path)
    (tmp_path / "approved.json").write_text(json.dumps(dict(asset_ids=["a", "missing-acquisition"])))
    with pytest.raises(ValueError, match="Approved assets missing"):
        reuse.run(args)
    assert source.downloads == 0
    args.allow_missing_approved = True
    result = reuse.run(args)
    assert result["candidate_count"] == 1
    assert result["missing_approved_assets"] == ["missing-acquisition"]
    assert result["exhaustive"] is False
    assert result["available_selected_sources_scan_complete"] is True
    assert result["configuration"]["allow_missing_approved"] is True


def test_partial_screening_flag_changes_configuration_fingerprint(monkeypatch, tmp_path):
    pytest.importorskip("PIL.Image")
    args, _, _ = configure_cloud(monkeypatch, tmp_path)
    args.allow_missing_approved = True
    assert reuse.run(args)["exhaustive"] is False
    args.allow_missing_approved = False
    with pytest.raises(ValueError, match="configuration conflict"):
        reuse.run(args)


def test_empty_scheduler_sources_can_be_recorded_only_in_partial_mode(monkeypatch, tmp_path):
    pytest.importorskip("PIL.Image")
    args, source, _ = configure_cloud(monkeypatch, tmp_path)
    (tmp_path / "sources.json").write_text(json.dumps(dict(source_objects=[])))
    args.allow_missing_approved = True
    summary = reuse.run(args)
    assert summary["candidate_count"] == 0
    assert summary["missing_approved_assets"] == ["a"]
    assert summary["exhaustive"] is False
    assert source.downloads == 0
