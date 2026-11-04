import hashlib
import json
from pathlib import Path

import pytest

from scripts import maritime_catalog_screen as screen


def asset(name, bounds, catalog=None, cloud=10):
    w, s, e, n = bounds
    return {"id": name, "properties": {"catalogID": catalog, "cloudcoverAreaPercentage": cloud},
            "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]},
            "split": "test", "accepted": True}


def test_transitive_overlap_catalog_union_stable_hash_without_labels():
    rows = [asset("a", [0, 0, 1, 1]), asset("b", [1, 0, 2, 1]),
            asset("c", [2, 0, 3, 1], "catalog"), asset("d", [9, 9, 10, 10], "catalog"),
            asset("e", [20, 20, 21, 21])]
    groups = screen.geographic_groups(rows)
    assert groups == screen.geographic_groups(list(reversed(rows)))
    joined = next(g for g in groups if len(g["asset_ids"]) == 4)
    assert joined["group_id"] == hashlib.sha256(b"a|b|c|d").hexdigest()[:16]
    assert joined["bbox"] == [0, 0, 10, 10]
    assert "split" not in joined
    assert "accepted" not in joined


def test_duplicate_ids_rejected_and_missing_catalog_not_unioned():
    rows = [asset("a", [0, 0, 1, 1]), asset("b", [3, 3, 4, 4])]
    assert len(screen.geographic_groups(rows)) == 2
    with pytest.raises(ValueError, match="Duplicate"):
        screen.geographic_groups(rows + [rows[0]])


def test_representative_prefers_low_cloud_then_area_then_id():
    rows = [asset("a", [0, 0, 1, 1], cloud=None), asset("b", [0, 0, 1, 1], cloud=2),
            asset("c", [0, 0, 2, 2], cloud=2), asset("d", [0, 0, 2, 2], cloud=2)]
    assert screen.geographic_groups(rows)[0]["representative"]["id"] == "c"


@pytest.mark.parametrize("stats,kind", [
    ({"observed_water_fraction": 0, "country_land_fraction": 0.5}, "coast_candidate"),
    ({"observed_water_fraction": 0.1, "country_land_fraction": 1}, "mixed_observed_water_land_candidate"),
    ({"observed_water_fraction": 0, "country_land_fraction": 0}, "water_or_ocean_candidate"),
    ({"observed_water_fraction": 0, "country_land_fraction": 1}, "reference_inconclusive_review_required"),
    ({}, "reference_inconclusive_review_required"),
])
def test_reference_rank_never_accepts_or_rejects_semantics(stats, kind):
    assert screen.candidate_rank(stats)[0] == kind


class Store:
    def __init__(self):
        self.objects = {}

    def read_json(self, name):
        return json.loads(self.objects[name]) if name in self.objects else None

    def create(self, name, payload, content_type):
        if name in self.objects:
            assert self.objects[name] == payload
        self.objects[name] = payload
        return {"uri": "gs://test/" + name, "generation": "1", "sha256": screen.sha(payload)}


def test_zero_water_still_previewed_and_immutable_receipts_resume(monkeypatch, tmp_path, capsys):
    groups = screen.geographic_groups([asset("a", [0, 0, 1, 1])])
    store, calls = Store(), []
    monkeypatch.setattr(screen, "screen_metadata", lambda ee, g: {
        "observed_water_fraction": 0, "country_land_fraction": 1, "countries": ["Example"]})
    monkeypatch.setattr(screen, "fetch_preview", lambda ee, a: calls.append(a["id"]) or b"PNG")
    first = screen.run_groups(None, store, groups, "recipe", tmp_path)
    second = screen.run_groups(None, store, groups, "recipe", tmp_path)
    assert first == second and calls == ["a"]
    assert first["preview_count"] == 1
    row = first["groups"][0]
    assert row["status"] == screen.STATUS
    assert "split" not in row and "accepted" not in row
    assert "non-commercial" in row["ancillary_references"]["countries_license"]
    assert json.loads((tmp_path / "checkpoint.json").read_text())["completed_groups"] == [row["group_id"]]
    assert "gs://" not in capsys.readouterr().out
    with pytest.raises(ValueError, match="mismatch"):
        screen.run_groups(None, store, groups, "different_recipe", tmp_path)


def test_metadata_failure_does_not_block_preview_and_error_urls_are_redacted(monkeypatch, tmp_path, capsys):
    attempts = []

    def fail(*args):
        attempts.append(1)
        raise RuntimeError("https://example.test/?token=secret")

    monkeypatch.setattr(screen.time, "sleep", lambda delay: None)
    monkeypatch.setattr(screen, "screen_metadata", fail)
    monkeypatch.setattr(screen, "fetch_preview", lambda ee, a: b"PNG")
    store = Store()
    summary = screen.run_groups(None, store, screen.geographic_groups([asset("a", [0, 0, 1, 1])]), "r", tmp_path)
    assert len(attempts) == 3
    assert summary["error_group_count"] == 1 and summary["preview_count"] == 1
    assert "secret" not in json.dumps(summary) + capsys.readouterr().out


def test_preview_failure_is_checkpointed_and_not_counted(monkeypatch, tmp_path):
    monkeypatch.setattr(screen.time, "sleep", lambda delay: None)
    monkeypatch.setattr(screen, "screen_metadata", lambda ee, g: {})
    monkeypatch.setattr(screen, "fetch_preview", lambda ee, a: (_ for _ in ()).throw(TimeoutError()))
    store = Store()
    summary = screen.run_groups(None, store, screen.geographic_groups([asset("a", [0, 0, 1, 1])]), "r", tmp_path)
    assert summary["preview_count"] == 0
    assert summary["groups"][0]["preview_status"] == "error"
    assert "summary.json" not in store.objects


def test_failed_checkpoint_recovers_and_retry_limit_is_persisted(monkeypatch, tmp_path):
    monkeypatch.setattr(screen.time, 'sleep', lambda delay: None)
    monkeypatch.setattr(screen, 'screen_metadata', lambda ee, g: {})
    monkeypatch.setattr(screen, 'fetch_preview', lambda ee, a: (_ for _ in ()).throw(TimeoutError()))
    store = Store()
    groups = screen.geographic_groups([asset('a', [0,0,1,1])])
    first = screen.run_groups(None, store, groups, 'r', tmp_path)
    assert first['error_group_count'] == 1
    monkeypatch.setattr(screen, 'fetch_preview', lambda ee, a: b'PNG')
    second = screen.run_groups(None, store, groups, 'r', tmp_path)
    assert second['error_group_count'] == 0 and second['preview_count'] == 1
    assert screen.run_groups(None, store, groups, 'r', tmp_path) == second
    store = Store()
    monkeypatch.setattr(screen, 'fetch_preview', lambda ee, a: (_ for _ in ()).throw(TimeoutError()))
    for _ in range(3):
        screen.run_groups(None, store, groups, 'r', tmp_path)
    monkeypatch.setattr(screen, 'fetch_preview', lambda *a: pytest.fail('retry limit ignored'))
    final = screen.run_groups(None, store, groups, 'r', tmp_path)
    assert final['error_group_count'] == 1


def test_repository_inventory_has_full_catalog_group_contract():
    inventory = Path(__file__).resolve().parents[1] / "reports/skysat-inventory.json"
    if not inventory.exists():
        pytest.skip("Catalog inventory is not packaged in the test environment")
    assets = json.loads(inventory.read_text(encoding="utf-8"))
    assert len(assets) == 1094
    assert len(screen.geographic_groups(assets)) == 132


class EE:
    """Record server expression calls without initializing or contacting EE."""
    def __init__(self):
        self.calls = []
        self.Geometry = self
        self.Reducer = self

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if name == "getInfo":
                return {"observed_water_fraction": 0.2, "country_land_fraction": 0.5, "countries": []}
            if name == "getThumbURL":
                return "https://example.test/preview?token=secret"
            return self
        return method


def test_water_fraction_boolean_mask_is_unweighted_and_ocean_unmasked():
    ee = EE()
    stats = screen.screen_metadata(ee, {"bbox": [0, 0, 1, 1]})
    assert stats["observed_water_fraction"] == 0.2
    names = [c[0] for c in ee.calls]
    assert names[names.index("select"):names.index("rename") + 1] == [
        "select", "mask", "gt", "unmask", "rename"]
    assert ("gt", (0,), {}) in ee.calls
    assert ("unmask", (0, False), {}) in ee.calls
    assert "unweighted" in names
    assert ("FeatureCollection", (screen.COUNTRIES,), {}) in ee.calls


def test_preview_uses_single_original_rgb_acquisition_at_1920(monkeypatch):
    ee = EE()

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, limit):
            return b"\x89PNG\r\n\x1a\nsmall"

    monkeypatch.setattr(screen, "urlopen", lambda url, timeout: Response())
    assert screen.fetch_preview(ee, asset("original", [0, 0, 1, 1])).startswith(b"\x89PNG")
    assert ("Image", ("original",), {}) in ee.calls
    assert ("select", (["R", "G", "B"],), {}) in ee.calls
    thumb = next(c for c in ee.calls if c[0] == "getThumbURL")[1][0]
    assert thumb["dimensions"] == 1920
    assert not any(c[0] in {"mosaic", "median", "ImageCollection", "Export"} for c in ee.calls)


def test_cloud_guard_fails_closed_without_metadata_server(monkeypatch):
    monkeypatch.setattr(screen, "urlopen", lambda *a, **k: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(RuntimeError, match="Google Compute Engine"):
        screen.require_cloud_vm()
