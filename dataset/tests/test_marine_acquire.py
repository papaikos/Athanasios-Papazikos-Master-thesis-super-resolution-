import json
from pathlib import Path
import pytest
from src.marine_acquire import (audit_objects, digest, footprint_key, preparation_manifest,
                                scheduler_tick, validate_config, validate_manifest, upgrade_source,
                                retry_failed_references)


def test_deadline_requires_initialized_earth_engine_transport(monkeypatch):
    import sys
    import types
    from src import marine_acquire

    events = []

    def deadline(milliseconds):
        assert events == ["initialized"]
        assert milliseconds == 60000
        events.append("deadline")

    ee = types.SimpleNamespace(
        Initialize=lambda **kwargs: events.append("initialized"),
        data=types.SimpleNamespace(setDeadline=deadline),
    )
    credentials = types.ModuleType("google.auth.credentials")
    credentials.Credentials = type("Credentials", (), {})
    monkeypatch.setitem(sys.modules, "ee", ee)
    monkeypatch.setitem(sys.modules, "google.auth.credentials", credentials)
    monkeypatch.setattr(marine_acquire.subprocess, "check_output", lambda *a, **k: "test-token")
    assert marine_acquire.initialize_marine("test-project", gcloud_auth=True) is ee
    assert events == ["initialized", "deadline"]


def fixture():
    config = json.loads(Path("configs/marine-source-v1.json").read_text())
    rows = []
    for i, split in enumerate(("train", "validation")):
        rows.append({"source_id": f"s{i}", "footprint_key": f"f{i}", "naip_id": f"n{i}", "split": split,
                     "bounds_wgs84": {"coordinates": [[[0,0],[1,0],[1,1],[0,1],[0,0]]]},
                     "exports": {"native": {"prefix": f"marine100/raw/s{i}/native", "estimated_max_bytes": 60000000000},
                                 "reference": {"prefix": f"marine100/raw/s{i}/reference"}}})
    m = {"configuration": config, "sources": rows}
    m["sha256"] = digest(m)
    return m


def rehash(m):
    m.pop("sha256", None)
    m["sha256"] = digest(m)
    return m


def obj(i, kind, size):
    return {"uri": f"gs://my-rgb-sr-pilot-bucket/marine100/raw/s{i}/{kind}.tif",
            "generation": "123", "size": size, "crc32c": f"abc{i}{kind}"}


def test_footprint_excludes_date_resolution():
    assert footprint_key("USDA/NAIP/DOQQ/m_3008501_ne_16_060_20191111") == footprint_key("m_3008501_ne_17_100_20221111")
    with pytest.raises(ValueError):
        footprint_key("arbitrary")


def test_duplicate_footprints_across_years_rejected():
    m = fixture()
    m["sources"][1]["footprint_key"] = "f0"
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest(rehash(m))


def test_manifest_tamper_and_size_caps():
    m = fixture()
    m["sources"][0]["exports"]["native"]["estimated_max_bytes"] = 200000000000
    with pytest.raises(ValueError, match="budget"):
        validate_manifest(rehash(m))
    m = fixture()
    m["sources"][0]["naip_id"] = "tamper"
    with pytest.raises(ValueError, match="hash"):
        validate_manifest(m)


def test_only_actual_used_training_native_bytes_count():
    m = fixture()
    objects = [obj(0, "native", 100000000000), obj(1, "native", 1000000000), obj(0, "reference", 999999)]
    assert audit_objects(m, objects)["unique_train_source_bytes_used"] == 0
    a = audit_objects(m, objects, ["s0", "s1"])
    assert a["unique_train_source_bytes_used"] == 100000000000
    assert a["meets_100gb"]
    assert a["native_acquired_bytes"] == 101000000000


def test_duplicate_object_generation_and_overrun_rejected():
    m = fixture()
    with pytest.raises(ValueError, match="Duplicate object"):
        audit_objects(m, [obj(0,"native", 10), obj(0,"native", 10)])
    with pytest.raises(ValueError, match="budget"):
        audit_objects(m, [obj(0,"native", 150000000001)])
    with pytest.raises(ValueError, match="unknown"):
        audit_objects(m, [], ["bogus"])


def test_adapter_excludes_missing_reference():
    m = fixture()
    a = audit_objects(m, [obj(0,"native",10),obj(0,"reference",20)])
    output = preparation_manifest(m,a)
    assert len(output["records"]) == 1
    assert output["records"][0]["files"]["naip"]["generation"] == "123"
    assert len(output["excluded"]) == 1


def test_two_task_limit_configuration():
    c = fixture()["configuration"]
    c["max_active_tasks"] = 3
    with pytest.raises(ValueError):
        validate_config(c)


@pytest.mark.parametrize("cap,other_live", [(None, 0), (6, 4)])
def test_uncertain_submission_never_duplicates(tmp_path, cap, other_live):
    m = fixture()
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"manifest_sha256": m["sha256"], "tasks": {
        "s0:native": {"task_id":"t1","state":"SUBMITTING"},
        "s0:reference": {"task_id":"t2","state":"SUBMITTING"}}}))
    class Data:
        @staticmethod
        def getTaskStatus(task_ids):
            return [{"state":"UNKNOWN"} for _ in task_ids]
        @staticmethod
        def getTaskList():
            return [{"state": "RUNNING"}] * other_live
    class EE:
        data = Data
    state = scheduler_tick(EE, m, path, submit=True, max_pending_tasks=cap)
    assert len(state["tasks"]) == 2


def test_request_uuid_not_confused_with_server_operation(tmp_path, monkeypatch):
    calls = []
    class Task:
        id = None
        name = None
        def start(self):
            assert self.id is None
            calls.append(self._request_id)
            self.id = "server-operation-" + str(len(calls))
            self.name = "projects/project/operations/" + self.id
    class Data:
        @staticmethod
        def getTaskList():
            return []
        @staticmethod
        def newTaskId():
            return ["request-uuid-" + str(len(calls))]
    class EE:
        data = Data
    monkeypatch.setattr("src.marine_acquire.export_task", lambda *args: Task())
    state = scheduler_tick(EE, fixture(), tmp_path / "state.json", submit=True)
    assert len(calls) == 2
    for task in state["tasks"].values():
        assert task["task_id"].startswith("server-operation-")
        assert task["request_id"].startswith("request-uuid-")
        assert task["task_id"] != task["request_id"]


@pytest.mark.parametrize("cap", [0, 1, 7, -1, 2.5, True, "6"])
def test_invalid_pending_cap_rejected_before_state_mutation(tmp_path, cap):
    path = tmp_path / "tasks.json"
    with pytest.raises(ValueError, match="pending"):
        scheduler_tick(None, fixture(), path, submit=True, max_pending_tasks=cap)
    assert not path.exists()


@pytest.mark.parametrize("cap,expected", [(None, 2), (2, 2), (6, 6)])
def test_pending_queue_bound_preserves_existing_ids_and_manifest(tmp_path, monkeypatch, cap, expected):
    import copy
    from types import SimpleNamespace
    m = fixture()
    m["sources"] = []
    for i in range(5):
        row = copy.deepcopy(fixture()["sources"][0])
        row.update(source_id=f"s{i}", footprint_key=f"f{i}", naip_id=f"n{i}")
        row["exports"]["native"]["estimated_max_bytes"] = 100
        m["sources"].append(row)
    rehash(m)
    original = copy.deepcopy(m)
    live = {"existing-operation": "RUNNING"}
    calls = []
    class Task:
        def start(self):
            calls.append(self._request_id)
            self.id = f"new-operation-{len(calls)}"
            self.name = f"projects/project/operations/{self.id}"
            live[self.id] = "READY"
    ee = SimpleNamespace(data=SimpleNamespace(
        getTaskStatus=lambda ids: [{"state": live[i]} for i in ids],
        getTaskList=lambda: [{"state": state} for state in live.values()],
        newTaskId=lambda: [f"request-{len(calls)}"],
    ))
    monkeypatch.setattr("src.marine_acquire.export_task", lambda *args: Task())
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"manifest_sha256": m["sha256"], "tasks": {
        "s0:native": {"task_id": "existing-operation", "request_id": "existing-request", "state": "RUNNING"}}}))
    state = scheduler_tick(ee, m, path, submit=True, max_pending_tasks=cap)
    assert len(state["tasks"]) == expected
    assert state["tasks"]["s0:native"]["task_id"] == "existing-operation"
    assert state["tasks"]["s0:native"]["request_id"] == "existing-request"
    assert len(calls) == expected - 1
    resumed = scheduler_tick(ee, m, path, submit=True, max_pending_tasks=cap)
    assert {key: task["task_id"] for key, task in resumed["tasks"].items()} == {
        key: task["task_id"] for key, task in state["tasks"].items()}
    assert len(calls) == expected - 1
    assert m == original


def test_mask_upgrade_budget_and_exclusion_from_source_bytes():
    m = fixture()
    for row in m["sources"]:
        row["native_dimensions"] = [100, 100]
        row["native_gsd_m"] = 0.6
        upgrade_source(row)
        assert row["estimated_total_raw_bytes"] > row["exports"]["native"]["estimated_max_bytes"]
    rehash(m)
    audit = audit_objects(m, [obj(0,"native",10),obj(0,"nativevalid",100),obj(0,"reference",500)], ["s0"])
    assert audit["unique_train_source_bytes_used"] == 10
    adapter = preparation_manifest(m,audit)
    assert adapter["records"][0]["files"]["nativevalid"]["bytes"] == 100


@pytest.mark.parametrize("objects,error,should_fail", [
    ([], "Requested 25,769,803,776 bytes exceeds 17,179,869,184 bytes", False),
    ([{"uri":"existing.tif"}], "Requested 25769803776 exceeds 17179869184", True),
    ([], "Different failure", True)])
def test_targeted_reference_retry_preserves_failure_and_refuses_overwrite(tmp_path, monkeypatch, objects, error, should_fail):
    m = fixture()
    source_id = "naip_55431267fb767401b72e"
    m["sources"][0]["source_id"] = source_id
    rehash(m)
    key = source_id + ":reference"
    old = {"task_id":"old-operation", "state":"FAILED", "source_id":source_id,"kind":"reference"}
    initial = {"manifest_sha256":m["sha256"],"tasks":{key:old}}
    path = tmp_path / "state.json"
    path.write_text(json.dumps(initial))
    class Task:
        id = None
        name = None
        def start(self):
            self.id = "new-operation"
            self.name = "projects/project/operations/new-operation"
    class Data:
        @staticmethod
        def getTaskStatus(ids):
            return [{"state":"FAILED","error_message":error}]
        @staticmethod
        def getTaskList():
            return []
        @staticmethod
        def newTaskId():
            return ["request-uuid"]
    class EE:
        data = Data
    monkeypatch.setattr("src.marine_acquire.export_task",lambda *args:Task())
    if should_fail:
        with pytest.raises(ValueError):
            retry_failed_references(EE,m,path,[source_id],lambda prefix:objects)
        assert json.loads(path.read_text()) == initial
    else:
        state = retry_failed_references(EE,m,path,[source_id],lambda prefix:objects)
        record = state["tasks"][key]
        assert record["task_id"] == "new-operation"
        assert record["attempts"][0]["task_id"] == "old-operation"
        assert record["state"] == "READY"
        with pytest.raises(ValueError, match="first verified"):
            retry_failed_references(EE,m,path,[source_id],lambda prefix:[])
