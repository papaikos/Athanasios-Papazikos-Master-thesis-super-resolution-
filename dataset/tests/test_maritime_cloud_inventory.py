import json
import shutil
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from scripts import maritime_cloud_inventory as inventory


def make_source(tmp_path, *, width=3300, height=3300, mask=True, hole=None,
                resolution=0.8, nodata=None, crs="EPSG:32644", dtype="uint8",
                count=3, transform=None, reverse_rgb=False, data_value=0, mask_value=255):
    path = tmp_path / "source.tif"
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False), rasterio.open(path, "w", driver="GTiff", width=width, height=height,
                       count=count, dtype=dtype, crs=crs,
                       transform=transform or from_origin(100, 3000, resolution, resolution),
                       nodata=nodata) as ds:
        # Real dark water is valid when the original mask says so.
        ds.write(np.full((count, height, width), data_value, dtype=dtype))
        if count == 3:
            rgb = (rasterio.enums.ColorInterp.red,
                   rasterio.enums.ColorInterp.green,
                   rasterio.enums.ColorInterp.blue)
            ds.colorinterp = rgb[::-1] if reverse_rgb else rgb
        if mask:
            validity = np.full((height, width), mask_value, dtype=np.uint8)
            if hole:
                validity[hole] = 0
            ds.write_mask(validity)
    return path


def test_native_alignment_edge_exclusion_and_dark_valid_water(tmp_path):
    result = inventory.inspect_raster(make_source(tmp_path),
                                      original_mask_provenance="original_explicit_dataset_mask")
    assert result["complete_core_count"] == 4
    assert result["geometric_candidate_count"] == 1
    assert result["mask_qualified_candidate_count"] == 1
    assert result["accepted_semantic_count"] == 0
    candidate = result["candidates"][0]
    assert candidate["core_window"] == [1600, 1600, 1600, 1600]
    assert candidate["context_window"] == [1550, 1550, 1700, 1700]
    assert candidate["core_transform"] == pytest.approx([0.8, 0, 1380, 0, -0.8, 1720])
    assert result["source_sha256"]


def test_hole_in_halo_rejects_otherwise_valid_core(tmp_path):
    result = inventory.inspect_raster(make_source(tmp_path, hole=(1550, 1600)),
                                      original_mask_provenance="original_explicit_dataset_mask")
    assert result["mask_qualified_candidate_count"] == 0
    assert result["candidates"][0]["rejection_reasons"] == ["invalid_core_or_halo_mask"]


@pytest.mark.parametrize("resolution,crs", [(0.800002, "EPSG:32644"),
                                            (0.8, "EPSG:4326")])
def test_strict_native_grid_rejected(tmp_path, resolution, crs):
    with pytest.raises(ValueError, match="native grid"):
        inventory.inspect_raster(make_source(tmp_path, resolution=resolution, crs=crs))


@pytest.mark.parametrize("kwargs,message", [
    ({"dtype": "uint16"}, "uint8 RGB"),
    ({"count": 4}, "uint8 RGB"),
    ({"reverse_rgb": True}, "RGB band order"),
    ({"transform": rasterio.Affine(0.8, 0.01, 0, 0, -0.8, 3000)}, "native grid"),
])
def test_strict_header_rejects_band_dtype_order_and_rotation(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        inventory.inspect_raster(make_source(tmp_path, width=100, height=100, **kwargs))


def test_grid_tolerance_accepts_small_rounding_without_resampling(tmp_path):
    result = inventory.inspect_raster(make_source(tmp_path, width=100, height=100,
                                                  resolution=0.8000005))
    assert result["native_transform"][0] == 0.8000005


def test_metric_unit_name_does_not_reject_valid_metre_grid():
    ds = SimpleNamespace(transform=from_origin(100, 3000, .8, .8),
        crs=SimpleNamespace(is_projected=True, linear_units="m", linear_units_factor=("m", 1.0)),
        count=3, dtypes=("uint8",)*3,
        colorinterp=(rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green, rasterio.enums.ColorInterp.blue))
    inventory.validate_header(ds)
    ds.crs.linear_units_factor = ("foot", .3048)
    with pytest.raises(ValueError, match="native grid"):
        inventory.validate_header(ds)


def test_unmasked_original_is_not_claimed_valid(tmp_path):
    result = inventory.inspect_raster(make_source(tmp_path, mask=False))
    assert result["mask_provenance"] == "unresolved_all_valid"
    assert result["mask_qualified_candidate_count"] == 0
    assert "unresolved_original_mask" in result["candidates"][0]["rejection_reasons"]


def test_legacy_zero_nodata_is_conservative_not_known_original(tmp_path):
    result = inventory.inspect_raster(make_source(tmp_path, mask=False, nodata=0))
    assert result["mask_provenance"] == "legacy_nodata_zero_conservative"
    assert result["original_mask_known"] is False
    assert result["mask_qualified_candidate_count"] == 0


def test_metadata_only_does_not_claim_mask_validity_or_file_sha(tmp_path):
    result = inventory.inspect_raster(make_source(tmp_path), metadata_only=True)
    assert result["source_sha256"] is None
    assert result["mask_qualified_candidate_count"] is None
    assert result["candidates"][0]["mask_valid"] is None


def test_manifest_requires_pinned_objects_and_consistent_sharded_source_ids():
    row = dict(uri="gs://bucket/raw/source.tif", generation="7", bytes=50,
               crc32c="AAAAAA==", asset_id="a", source_id="s", catalog_id="c",
               grid={}, geometry={})
    assert inventory.validate_manifest([row]) == [row]
    with pytest.raises(ValueError, match="generation"):
        inventory.validate_manifest([{**row, "generation": "0"}])
    with pytest.raises(ValueError, match="duplicate"):
        inventory.validate_manifest([row, row])
    shard = {**row, "uri": "gs://bucket/raw/source-shard2.tif"}
    assert inventory.validate_manifest([row, shard]) == [row, shard]
    with pytest.raises(ValueError, match="inconsistent asset"):
        inventory.validate_manifest([row, {**shard, "asset_id": "other"}])


def test_dataset_mask_without_original_provenance_remains_unresolved(tmp_path):
    report = inventory.inspect_raster(make_source(tmp_path))
    assert report["mask_provenance"] == "explicit_dataset_mask_unverified"
    assert report["original_mask_known"] is False
    assert report["mask_qualified_candidate_count"] == 0


def test_documented_unmask_zero_recipe_proves_only_conservative_support(tmp_path):
    path = make_source(tmp_path, mask=False, nodata=0, data_value=1)
    recipe = dict(bands=["R", "G", "B"], dtype="uint8", unmask=0,
                  noData=0, cloudOptimized=False)
    report = inventory.inspect_raster(path, export_recipe=recipe)
    assert report["conservative_mask_qualified_candidate_count"] == 1
    assert report["mask_qualified_candidate_count"] == 0
    assert report["original_mask_known"] is False
    assert report["accepted_semantic_count"] == 0
    assert report["candidates"][0]["conservative_mask_support"] is True
    undocumented = inventory.inspect_raster(path)
    assert undocumented["conservative_mask_qualified_candidate_count"] == 0


def test_fractional_original_mask_is_not_full_valid_support(tmp_path):
    path = make_source(tmp_path, mask_value=128)
    report = inventory.inspect_raster(path, original_mask_provenance="original_explicit_dataset_mask")
    assert report["mask_qualified_candidate_count"] == 0
    assert report["candidates"][0]["mask_valid"] is False


class FakeBlob:
    def __init__(self):
        self.data = None
        self.generation = 1
        self.size = None
        self.crc32c = None

    def exists(self):
        return self.data is not None

    def upload_from_string(self, data, **kwargs):
        assert kwargs["if_generation_match"] == 0
        assert kwargs["checksum"] == "crc32c"
        self.data = data
        self.size = len(data)

    def download_as_bytes(self, **kwargs):
        assert kwargs["if_generation_match"] == 1
        assert kwargs["checksum"] == "crc32c"
        return self.data

    def reload(self):
        pass


def test_create_only_output_reuses_matching_and_fails_on_conflict():
    blob = FakeBlob()
    receipt = inventory.create_only(blob, b'{"a":1}')
    assert receipt["generation"] == "1"
    assert inventory.create_only(blob, b'{"a":1}') == receipt
    with pytest.raises(ValueError, match="conflict"):
        inventory.create_only(blob, b'{"a":2}')


def cloud_fixture(monkeypatch, tmp_path):
    original = make_source(tmp_path, width=1700, height=1700)
    row = dict(uri="gs://input/raw/source.tif", generation="7",
               bytes=original.stat().st_size, crc32c="source-crc", asset_id="a",
               source_id="s", catalog_id="c", grid={}, geometry={})
    source = FakeBlob()
    source.generation = 7
    source.size = row["bytes"]
    source.crc32c = row["crc32c"]
    source.downloads = 0

    def download(path, **kwargs):
        assert kwargs == dict(if_generation_match=7, checksum="crc32c")
        source.downloads += 1
        shutil.copyfile(original, path)
        # Synthetic TIFF masks can be external on some GDAL versions.
        sidecar = original.with_suffix(".tif.msk")
        if sidecar.exists():
            shutil.copyfile(sidecar, path + ".msk")

    source.download_to_filename = download
    objects = {("input", "raw/source.tif"): source}

    def blob(bucket, name, generation=None):
        result = objects.setdefault((bucket, name), FakeBlob())
        if generation is not None:
            assert int(generation) == result.generation
        return result

    client = SimpleNamespace(bucket=lambda bucket: SimpleNamespace(
        blob=lambda name, generation=None: blob(bucket, name, generation)))
    google = ModuleType("google")
    auth = ModuleType("google.auth")
    auth.default = lambda **kwargs: (SimpleNamespace(), "project")
    cloud = ModuleType("google.cloud")
    storage = ModuleType("google.cloud.storage")
    storage.Client = lambda **kwargs: client
    google.auth = auth
    google.cloud = cloud
    cloud.storage = storage
    for name, module in (("google", google), ("google.auth", auth),
                         ("google.cloud", cloud), ("google.cloud.storage", storage)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(inventory, "require_cloud_vm", lambda: "123")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([row]))
    args = SimpleNamespace(manifest=str(manifest), output_prefix="gs://output/control/runs/test",
                           scratch=str(tmp_path / "scratch"), max_sources=1,
                           metadata_only=False, preview_samples=0)
    return args, source, objects


def test_pinned_download_restart_skips_verified_source_and_scratch_is_bounded(monkeypatch, tmp_path):
    args, source, objects = cloud_fixture(monkeypatch, tmp_path)
    first = inventory.run(args)
    assert source.downloads == 1
    assert inventory.run(args) == first
    assert source.downloads == 1
    assert list((tmp_path / "scratch").iterdir()) == [tmp_path / "scratch" / "checkpoint.json"]
    receipt = next(blob for (bucket, name), blob in objects.items() if "/receipts/" in name)
    assert json.loads(receipt.data)["accepted_semantic_count"] == 0


def test_resume_fails_on_corrupted_output(monkeypatch, tmp_path):
    args, source, objects = cloud_fixture(monkeypatch, tmp_path)
    inventory.run(args)
    report = next(blob for (bucket, name), blob in objects.items() if "/sources/" in name)
    report.data = b"x" * report.size
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        inventory.run(args)
    assert source.downloads == 1


def test_changed_source_metadata_fails_before_download(monkeypatch, tmp_path):
    args, source, _ = cloud_fixture(monkeypatch, tmp_path)
    source.crc32c = "wrong"
    with pytest.raises(ValueError, match="source object metadata mismatch"):
        inventory.run(args)
    assert source.downloads == 0


def test_changed_run_config_fails_before_reusing_receipts(monkeypatch, tmp_path):
    args, source, _ = cloud_fixture(monkeypatch, tmp_path)
    inventory.run(args)
    args.preview_samples = 1
    with pytest.raises(ValueError, match="receipt conflicts"):
        inventory.run(args)
    assert source.downloads == 1


def test_vm_guard_fails_closed_without_gce_metadata(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("not on GCE")
    monkeypatch.setattr(inventory, "urlopen", unavailable)
    with pytest.raises(RuntimeError, match="Google Compute Engine VM"):
        inventory.require_cloud_vm()


def test_preview_limit_counts_pairs_and_native_detail_is_unresampled(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    path = make_source(tmp_path, width=4900, height=4900)
    report = inventory.inspect_raster(path, preview_directory=tmp_path, preview_samples=3)
    assert len(report["preview_files"]) == 6
    details = [path for path in report["preview_files"] if "native-detail" in path]
    assert all(Image.open(path).size == (256, 256) for path in details)
