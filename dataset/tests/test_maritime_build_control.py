import pytest

from scripts.maritime_build_control import join_sources, source_prefixes


def test_source_join_pins_objects_without_inheriting_historical_splits():
    manifests = [{"prefix": "skysat/raw-v2", "sources": [
        {"source_id": "s", "asset_id": "asset", "split": "test", "group_id": "g"}]}]
    objects = [{"name": "skysat/raw-v2/s/rgb.tif", "generation": "12",
                "size": "123", "crc32c": "crc"}]
    rows = join_sources(manifests, objects, "bucket", {"asset": {
        "properties": {"catalogID": "catalog", "collectionStartTime": "2015-01-01"}}})
    assert rows[0]["uri"] == "gs://bucket/skysat/raw-v2/s/rgb.tif"
    assert rows[0]["generation"] == "12"
    assert rows[0]["catalog_id"] == "catalog"
    assert rows[0]["historical_split"] == "test"
    assert "split" not in rows[0]


def test_unmapped_objects_and_duplicates_do_not_become_sources():
    objects = [{"name": "skysat/raw/unknown/rgb.tif", "generation": "12", "size": "123"}]
    assert join_sources([], objects, "bucket", {}) == []
    manifest = {"prefix": "skysat/raw", "sources": [{"source_id": "s", "asset_id": "a"}]}
    objects = [{"name": "skysat/raw/s/rgb.tif", "generation": "12", "size": "123"}] * 2
    with pytest.raises(ValueError, match="Duplicate"):
        join_sources([manifest], objects, "bucket", {})


def test_raw_inventory_prefixes_are_narrow_and_include_legacy():
    prefixes = source_prefixes()
    assert len(prefixes) == 13
    assert "skysat/raw/" in prefixes
    assert all(p.endswith("/") and "prepared" not in p for p in prefixes)
