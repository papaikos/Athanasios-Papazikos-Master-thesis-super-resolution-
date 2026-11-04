import importlib.util
from pathlib import Path
import pytest

SPEC = importlib.util.spec_from_file_location(
    "skysat_acquire", Path("scripts/skysat_acquire.py")
)


def api():
    assert SPEC.loader is not None and Path(SPEC.origin).exists(), (
        "SkySat acquisition controller missing"
    )
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module


def asset(name, x, catalog=None):
    grid = {
        "crsCode": "EPSG:32610",
        "dimensions": {"width": 100, "height": 100},
        "affineTransform": {
            "scaleX": 0.8,
            "scaleY": -0.8,
            "translateX": 10,
            "translateY": 20,
        },
    }
    return {
        "id": name,
        "properties": {"catalogID": catalog or name},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[x, 0], [x + 1, 0], [x + 1, 1], [x, 1], [x, 0]]],
        },
        "bands": [{"id": b, "grid": grid} for b in "RGB"],
    }


def test_transitive_overlap_and_catalog_never_cross_splits():
    m = api().build_manifest(
        [
            asset("a", 0),
            asset("b", 0.9),
            asset("c", 1.8),
            asset("d", 20, "a"),
            asset("e", 30),
            asset("f", 40),
        ],
        "bucket",
        "raw",
        180000,
    )
    rows = {r["asset_id"]: r for r in m["sources"]}
    assert len({rows[k]["group_id"] for k in "abcd"}) == 1
    assert len({r["split"] for r in rows.values()}) == 3
    assert m["estimated_native_bytes"] == 180000
    assert (
        api().build_manifest(
            list(
                reversed(
                    [
                        asset("a", 0),
                        asset("b", 0.9),
                        asset("c", 1.8),
                        asset("d", 20, "a"),
                        asset("e", 30),
                        asset("f", 40),
                    ]
                )
            ),
            "bucket",
            "raw",
            180000,
        )
        == m
    )


def test_budget_and_native_export_contract():
    mod = api()
    m = mod.build_manifest(
        [asset("a", 0), asset("b", 20), asset("c", 40), asset("d", 60)],
        "bucket",
        "raw",
        90000,
    )
    assert len(m["sources"]) == 3
    kwargs = mod.export_arguments(m, m["sources"][0])
    assert kwargs["crsTransform"] == [0.8, 0, 10, 0, -0.8, 20]
    assert "scale" not in kwargs and "dimensions" not in kwargs
    assert kwargs["region"]["type"] == "Polygon"
    with pytest.raises(ValueError):
        mod.build_manifest([asset("a", 0)], "bucket", "raw", 180000000001)


def test_state_rejects_manifest_change_and_deadline_extension(tmp_path):
    mod = api()
    m = mod.build_manifest(
        [asset("a", 0), asset("b", 20), asset("c", 40)], "bucket", "raw", 90000
    )
    p = tmp_path / "state.json"
    mod.load_state(p, m, "2026-09-26T00:00:00Z")
    with pytest.raises(ValueError):
        mod.load_state(p, m, "2026-09-27T00:00:00Z")


def test_unknown_submission_blocks_duplicate_and_deadline_cancels(tmp_path):
    from types import SimpleNamespace

    mod = api()
    m = mod.build_manifest(
        [asset("a", 0), asset("b", 20), asset("c", 40)], "bucket", "raw", 90000
    )
    p = tmp_path / "state.json"
    s = mod.load_state(p, m, "2026-09-26T00:00:00Z")
    s["tasks"] = {
        m["sources"][0]["source_id"]: {"state": "SUBMITTING", "task_id": None},
        m["sources"][1]["source_id"]: {"state": "RUNNING", "task_id": "owned"},
    }
    mod.write_json(p, s)
    cancelled = []
    ee = SimpleNamespace(
        data=SimpleNamespace(
            getTaskStatus=lambda ids: [
                {"id": task_id, "state": "RUNNING"} for task_id in ids
            ],
            getTaskList=lambda: [{"state": "RUNNING"}],
            cancelTask=cancelled.append,
        )
    )
    result = mod.tick(ee, m, p, "2026-09-26T00:00:00Z", 0, now="2026-09-25T00:00:00Z")
    assert len(result["tasks"]) == 2
    mod.tick(ee, m, p, "2026-09-26T00:00:00Z", 0, now="2026-09-26T00:00:00Z")
    assert cancelled == ["owned"]


def test_windows_gcloud_executable_is_explicit(monkeypatch):
    mod = api()
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    assert mod.resolve_gcloud("C:/sdk/gcloud.cmd") == "C:/sdk/gcloud.cmd"
    with pytest.raises(ValueError):
        mod.resolve_gcloud(None)


def test_native_wkt_is_preserved():
    mod = api()
    assets = [asset("a", 0), asset("b", 20), asset("c", 40)]
    for a in assets:
        for band in a["bands"]:
            band["grid"].pop("crsCode", None)
            band["grid"]["crsWkt"] = (
                'PROJCS["UTM Zone 10, Northern Hemisphere", UNIT["m", 1.0]]'
            )
    m = mod.build_manifest(assets, "bucket", "raw", 90000)
    assert mod.export_arguments(m, m["sources"][0])["crs"].startswith("PROJCS")


@pytest.mark.parametrize("limit", [2, 6])
def test_submission_persists_ids_and_never_exceeds_limit(tmp_path, limit):
    from types import SimpleNamespace

    mod = api()
    m = mod.build_manifest(
        [asset(str(i), i * 10) for i in range(8)], "bucket", "raw", 240000
    )
    p = tmp_path / "state.json"
    started = []
    tasks = []

    class Image:
        def select(self, bands):
            assert bands == ["R", "G", "B"]
            return self

        def toUint8(self):
            return self

        def unmask(self, value):
            assert value == 0
            self.nodata_applied = True
            return self

    class Task:
        id = None

        def start(self):
            import json

            saved = json.loads(p.read_text())
            assert any(
                t.get("request_id") == self._request_id for t in saved["tasks"].values()
            )
            self.id = "operation-" + str(len(started))
            started.append(self.id)

    class Geometry:
        def __init__(self, geojson):
            assert geojson["type"] == "Polygon"

    def create(**kwargs):
        assert isinstance(kwargs["region"], Geometry)
        assert kwargs["image"].nodata_applied
        assert kwargs["formatOptions"]["noData"] == 0
        task = Task()
        tasks.append(task)
        return task

    ee = SimpleNamespace(
        Image=lambda name: Image(),
        Geometry=Geometry,
        batch=SimpleNamespace(
            Export=SimpleNamespace(image=SimpleNamespace(toCloudStorage=create))
        ),
        data=SimpleNamespace(
            newTaskId=lambda: ["request-" + str(len(tasks))],
            getTaskList=lambda: [{"state": "RUNNING"} for _ in started],
            getTaskStatus=lambda ids: [
                {"id": task_id, "state": "RUNNING"} for task_id in ids
            ],
        ),
    )
    mod.tick(
        ee,
        m,
        p,
        "2026-09-26T00:00:00Z",
        0,
        now="2026-09-25T00:00:00Z",
        max_pending=limit,
    )
    mod.tick(
        ee,
        m,
        p,
        "2026-09-26T00:00:00Z",
        0,
        now="2026-09-25T00:01:00Z",
        max_pending=limit,
    )
    assert len(started) == limit


def test_storage_accounting_rejects_unrecognized_metadata(monkeypatch):
    from types import SimpleNamespace
    import json

    mod = api()
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=json.dumps([{"metadata": {"size": "123"}}, {"size": 7}]),
        ),
    )
    assert mod.storage_bytes("bucket", "raw") == 130
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stderr="", stdout='[{"name":"unknown"}]'
        ),
    )
    with pytest.raises(ValueError):
        mod.storage_bytes("bucket", "raw")


@pytest.mark.parametrize("limit", [0, 1, 7, True])
def test_pending_limit_invalid_before_state_mutation(tmp_path, limit):
    mod = api()
    with pytest.raises(ValueError):
        mod.tick(
            None,
            {},
            tmp_path / "absent.json",
            "2026-09-26T00:00:00Z",
            0,
            max_pending=limit,
        )
    assert not (tmp_path / "absent.json").exists()


def test_batched_status_checks_identity_and_maps_out_of_order(tmp_path):
    from types import SimpleNamespace

    mod = api()
    m = mod.build_manifest(
        [asset("a", 0), asset("b", 20), asset("c", 40)], "bucket", "raw", 90000
    )
    path = tmp_path / "state.json"
    state = mod.load_state(path, m, "2026-09-26T00:00:00Z")
    state["tasks"] = {
        "a": {"task_id": "a", "state": "RUNNING"},
        "b": {"task_id": "b", "state": "READY"},
    }
    mod.write_json(path, state)
    calls = []

    def status(ids):
        calls.append(ids)
        return [{"id": "b", "state": "RUNNING"}, {"id": "a", "state": "COMPLETED"}]

    ee = SimpleNamespace(
        data=SimpleNamespace(
            getTaskStatus=status,
            getTaskList=lambda: [{"state": "RUNNING"}, {"state": "READY"}],
        )
    )
    result = mod.tick(ee, m, path, state["deadline"], 0, now="2026-09-25T00:00:00Z")
    assert calls == [["a", "b"]]
    assert result["tasks"]["a"]["state"] == "COMPLETED"
    ee.data.getTaskStatus = lambda ids: [{"id": "alien", "state": "COMPLETED"}]
    with pytest.raises(ValueError):
        mod.tick(ee, m, path, state["deadline"], 0, now="2026-09-25T00:00:00Z")


def test_nodata_recipe_is_frozen_and_old_state_can_only_resume_for_cancellation(
    tmp_path,
):
    import copy

    mod = api()
    manifest = mod.build_manifest(
        [asset("a", 0), asset("b", 20), asset("c", 40)], "bucket", "raw-v2", 90000
    )
    assert manifest["schema"] == 2
    assert manifest["export_recipe"] == {
        "bands": ["R", "G", "B"],
        "dtype": "uint8",
        "unmask": 0,
        "noData": 0,
        "cloudOptimized": False,
    }
    assert (
        mod.export_arguments(manifest, manifest["sources"][0])["formatOptions"][
            "noData"
        ]
        == 0
    )
    old = copy.deepcopy(manifest)
    old["schema"] = 1
    old.pop("export_recipe")
    old["sha256"] = mod.digest({k: v for k, v in old.items() if k != "sha256"})
    path = tmp_path / "old-state.json"
    mod.load_state(path, old, "2026-09-26T00:00:00Z")
    with pytest.raises(ValueError):
        mod.load_state(path, manifest, "2026-09-26T00:00:00Z")
    with pytest.raises(ValueError):
        mod.export_arguments(old, old["sources"][0])
    wrong = copy.deepcopy(manifest)
    wrong["export_recipe"]["noData"] = 255
    with pytest.raises(ValueError):
        mod.export_arguments(wrong, wrong["sources"][0])


def test_storage_audit_prefix_includes_both_export_recipes():
    mod = api()
    manifest = mod.build_manifest(
        [asset("a", 0), asset("b", 20), asset("c", 40)],
        "bucket",
        "skysat/raw-v2",
        90000,
    )
    assert manifest["storage_audit_prefix"] == "skysat"
    with pytest.raises(ValueError):
        mod.build_manifest(
            [asset("a", 0), asset("b", 20), asset("c", 40)],
            "bucket",
            "/absolute",
            90000,
        )


def test_incremental_preserves_global_splits_excludes_prior_assets_and_catalogs():
    mod = api()
    assets = [asset(str(i), i * 10) for i in range(10)]
    # Same acquisition identity joins distant frames into one geographic component.
    assets.append(asset("duplicate", 120, "0"))
    first = mod.build_manifest(assets, "bucket", "skysat/first", 90000)
    before = mod.digest(first)
    second = mod.build_manifest(
        assets, "bucket", "skysat/second", 120000, prior_manifests=[first]
    )
    assert mod.digest(first) == before
    assert second["estimated_native_bytes"] <= 120000
    assert {r["split"] for r in second["sources"]} == {"train"}
    assert second["prior_manifests"] == [
        {"sha256": first["sha256"], "bucket": "bucket", "prefix": "skysat/first"}
    ]
    old_ids = {r["asset_id"] for r in first["sources"]}
    old_catalog = {a["properties"]["catalogID"] for a in assets if a["id"] in old_ids}
    catalogs = {a["id"]: a["properties"]["catalogID"] for a in assets}
    assert all(
        r["asset_id"] not in old_ids and catalogs[r["asset_id"]] not in old_catalog
        for r in second["sources"]
    )
    all_rows = mod.build_manifest(assets, "bucket", "skysat/all", 330000)["sources"]
    assignments = {r["asset_id"]: (r["group_id"], r["split"]) for r in all_rows}
    assert all(
        (r["group_id"], r["split"]) == assignments[r["asset_id"]]
        for r in second["sources"]
    )
    assert second == mod.build_manifest(
        list(reversed(assets)),
        "bucket",
        "skysat/second",
        120000,
        prior_manifests=[first],
    )


def test_incremental_rejects_corrupted_or_reassigned_parent():
    import copy

    mod = api()
    assets = [asset(str(i), i * 10) for i in range(8)]
    first = mod.build_manifest(assets, "bucket", "skysat/first", 90000)
    bad = copy.deepcopy(first)
    bad["sources"][0]["group_id"] = "changed"
    with pytest.raises(ValueError):
        mod.build_manifest(assets, "bucket", "skysat/new", 90000, prior_manifests=[bad])
    bad["sha256"] = mod.digest({k: v for k, v in bad.items() if k != "sha256"})
    with pytest.raises(ValueError):
        mod.build_manifest(assets, "bucket", "skysat/new", 90000, prior_manifests=[bad])
    with pytest.raises(ValueError):
        mod.build_manifest(
            assets, "bucket", "skysat/first", 90000, prior_manifests=[first]
        )
