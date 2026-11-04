from datetime import datetime, timedelta, timezone
import pytest
from scripts.marine_acquisition_loop import (all_terminal, cancel_owned, completed_sources,
                                             normalize_objects, parse_deadline, inventory_snapshot,
                                             append_only_state, ensure_ee_client, repair_accounting)
from src.marine_acquire import audit_objects, digest, scheduler_tick, prepare_extension
import json
import copy
from pathlib import Path


def test_controller_reuses_refreshing_client_across_ticks(monkeypatch):
    calls, events = [], []
    client = object()
    def initialize(project, gcloud_auth):
        calls.append((project, gcloud_auth))
        return client
    monkeypatch.setattr("scripts.marine_acquisition_loop.initialize_marine", initialize)
    cached = None
    for _ in range(5):
        cached = ensure_ee_client(cached, "project", events.append)
    assert cached is client
    assert calls == [("project", True)]
    assert events == ["ee_initialize_start", "ee_initialize_done"]


def manifest():
    config = json.loads(Path("configs/marine-source-v1.json").read_text())
    row = {"source_id":"s1", "footprint_key":"f1", "naip_id":"n1", "split":"train",
           "bounds_wgs84":{"coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]},
           "exports":{"native":{"prefix":"marine100/raw/s1/native", "estimated_max_bytes":100},
                      "reference":{"prefix":"marine100/raw/s1/reference"}}}
    m = {"configuration":config,"sources":[row]}
    m["sha256"] = digest(m)
    return m


def test_metadata_normalizes_cli_and_json_api_without_size_guessing():
    raw = {"bucket":"my-rgb-sr-pilot-bucket", "name":"marine100/raw/s1/native.tif",
           "size":"456", "generation":"999", "crc32c":"ABCD"}
    assert normalize_objects({"items":[raw]})[0]["size"] == 456
    cli = {"url":"gs://my-rgb-sr-pilot-bucket/marine100/raw/s1/native.tif",
           "metadata":raw}
    assert normalize_objects([cli]) == normalize_objects([raw])
    with pytest.raises(ValueError):
        normalize_objects([dict(raw, size=None)])
    with pytest.raises(ValueError):
        normalize_objects([dict(raw, name="sar/private.tif")])


def test_generation_suffixed_uri_uses_verified_metadata_identity():
    uri = "gs://my-rgb-sr-pilot-bucket/marine100/raw/s1/native.tif"
    obj = {"uri":uri+"#1789651022509638", "generation":"1789651022509638",
           "size":516538188,"crc32c":"4NeI1A=="}
    normalized = normalize_objects([obj])[0]
    assert normalized["uri"] == uri
    assert normalized["generation"] == "1789651022509638"
    assert normalize_objects([normalized]) == [normalized]
    for suffix in ("1789651022509639", "latest", ""):
        with pytest.raises(ValueError,match="generation suffix"):
            normalize_objects([dict(obj,uri=uri+"#"+suffix)])


def test_accounting_repair_requires_completed_tasks_and_does_not_mark_usage():
    m = manifest()
    objects = [{"uri":f"gs://my-rgb-sr-pilot-bucket/marine100/raw/s1/{k}.tif#1",
                "size":50,"generation":"1","crc32c":"a"} for k in ("native","reference")]
    state = {"manifest_sha256":m["sha256"],"tasks":{
        "s1:native":{"state":"COMPLETED"},"s1:reference":{"state":"RUNNING"}}}
    result = repair_accounting(m,state,objects,final_inventory=True)
    assert result["audit"]["native_acquired_bytes"] == 50
    assert result["audit"]["unique_train_source_bytes_used"] == 0
    assert not result["adapter"]["records"]
    assert result["adapter"]["status"] == "partial_read_only"
    state["tasks"]["s1:reference"]["state"] = "COMPLETED"
    result = repair_accounting(m,state,objects,final_inventory=True)
    assert len(result["adapter"]["records"]) == 1
    assert result["adapter"]["status"] == "final"
    with pytest.raises(ValueError,match="hash mismatch"):
        repair_accounting(m,dict(state,manifest_sha256="wrong"),objects)


def test_deadline_is_explicit_absolute_and_max_12_hours():
    now = datetime.now(timezone.utc)
    assert parse_deadline((now + timedelta(hours=12)).isoformat(), now)
    with pytest.raises(ValueError):
        parse_deadline((now + timedelta(hours=12,seconds=1)).isoformat(), now)
    with pytest.raises(ValueError):
        parse_deadline("2026-09-17T12:00:00", now)


def test_partial_adapter_only_completed_exports():
    m = manifest()
    objects = [{"uri":f"gs://my-rgb-sr-pilot-bucket/marine100/raw/s1/{k}.tif",
                "size":50,"generation":"1","crc32c":"a"} for k in ("native","reference")]
    audit = audit_objects(m,objects)
    state = {"tasks":{"s1:native":{"state":"COMPLETED"},"s1:reference":{"state":"RUNNING"}}}
    assert not completed_sources(m,state,audit)["records"]
    state["tasks"]["s1:reference"]["state"] = "COMPLETED"
    assert len(completed_sources(m,state,audit)["records"]) == 1
    assert all_terminal(m,state)
    del state["tasks"]["s1:reference"]
    assert not all_terminal(m,state)


def test_cancellation_only_owned_nonterminal_ids():
    calls = []
    class Data:
        @staticmethod
        def cancelTask(task_id):
            calls.append(task_id)
    class EE:
        data = Data
    state = {"tasks":{"x":{"task_id":"own-active","state":"READY"},
                      "y":{"task_id":"own-done","state":"COMPLETED"}}}
    assert cancel_owned(EE,state,lambda *args, **kwargs: None) == []
    assert calls == ["own-active"]
    assert state["tasks"]["x"]["state"] == "CANCEL_REQUESTED"


def test_scheduler_batches_only_nonterminal_tasks(tmp_path):
    m = manifest()
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({"manifest_sha256":m["sha256"],"tasks":{
        "s1:native":{"task_id":"a","state":"COMPLETED"},
        "s1:reference":{"task_id":"b","state":"RUNNING"}}}))
    calls = []
    class Data:
        @staticmethod
        def getTaskStatus(ids):
            calls.append(ids)
            return [{"state":"COMPLETED"} for _ in ids]
        @staticmethod
        def getTaskList():
            return []
    class EE:
        data = Data
    assert all_terminal(m,scheduler_tick(EE,m,path))
    assert calls == [["b"]]


def test_append_only_transition_preserves_ids_and_rejects_changes():
    previous = manifest()
    incoming = copy.deepcopy(previous)
    row = copy.deepcopy(incoming["sources"][0])
    row.update(source_id="s2", footprint_key="f2", naip_id="n2")
    incoming["sources"].append(row)
    incoming.pop("sha256")
    incoming["sha256"] = digest(incoming)
    state = {"manifest_sha256":previous["sha256"], "tasks":{
        "s1:native":{"task_id":"real-operation", "request_id":"uuid", "state":"COMPLETED"}}}
    migrated = append_only_state(previous,incoming,state)
    assert migrated["tasks"] == state["tasks"]
    assert migrated["manifest_sha256"] == incoming["sha256"]
    assert state["manifest_sha256"] == previous["sha256"]
    for field, value in (("split","test"),("naip_id","different-scene")):
        changed = copy.deepcopy(incoming)
        changed["sources"][0][field] = value
        changed.pop("sha256")
        changed["sha256"] = digest(changed)
        with pytest.raises(ValueError,match="changed existing"):
            append_only_state(previous,changed,state)
    with pytest.raises(ValueError,match="removed"):
        append_only_state(incoming,previous,{"manifest_sha256":incoming["sha256"],"tasks":{}})


def test_progress_snapshot_stable_hash_mask_upgrade_and_final_distinction(tmp_path):
    m = manifest()
    row = m["sources"][0]
    row.update(native_dimensions=[100,100],native_gsd_m=0.6)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(m["configuration"]))
    target = tmp_path / "inventory.json"
    progress = {"configuration_sha256":digest(m["configuration"]),"sources":[row], "seen":["f1"]}
    progress_path = target.with_suffix(".progress.json")
    progress_path.write_text(json.dumps(progress))
    snapshot, final = inventory_snapshot(target,config)
    assert not final
    assert "nativevalid" in snapshot["sources"][0]["exports"]
    progress["seen"].append("another-rejected-footprint")
    progress_path.write_text(json.dumps(progress))
    assert inventory_snapshot(target,config)[0]["sha256"] == snapshot["sha256"]
    final_manifest = copy.deepcopy(snapshot)
    final_manifest["inventory_status"] = "final"
    final_manifest.pop("sha256")
    final_manifest["sha256"] = digest(final_manifest)
    target.write_text(json.dumps(final_manifest))
    incoming, final = inventory_snapshot(target,config)
    assert final
    migrated = append_only_state(snapshot,incoming,{"manifest_sha256":snapshot["sha256"],"tasks":{}})
    assert migrated["manifest_sha256"] == final_manifest["sha256"]


def test_append_only_rejects_config_change_and_unknown_tasks():
    previous = manifest()
    incoming = copy.deepcopy(previous)
    incoming["configuration"]["max_sources"] -= 1
    incoming.pop("sha256")
    incoming["sha256"] = digest(incoming)
    with pytest.raises(ValueError,match="configuration"):
        append_only_state(previous,incoming,{"manifest_sha256":previous["sha256"],"tasks":{}})
    with pytest.raises(ValueError,match="absent"):
        append_only_state(previous,previous,{"manifest_sha256":previous["sha256"],"tasks":{"unknown:native":{}}})


def test_explicit_extension_preserves_rows_tasks_and_deadline(tmp_path):
    previous = manifest()
    previous["sources"][0].update(native_dimensions=[24760,20000], native_gsd_m=0.3)
    previous.pop("sha256")
    previous["sha256"] = digest(previous)
    deadline = datetime.now(timezone.utc) + timedelta(hours=1)
    target = tmp_path / "inventory-v2.json"
    config = tmp_path / "config-v2.json"
    record_path = tmp_path / "extension.json"
    record = prepare_extension(previous,target,config,record_path,deadline.isoformat())
    assert not target.exists()
    progress = json.loads(target.with_suffix(".progress.json").read_text())
    assert progress["sources"] == previous["sources"]
    assert progress["seen"] == ["f1"]
    incoming = copy.deepcopy(previous)
    incoming["configuration"] = json.loads(config.read_text())
    incoming.pop("sha256")
    incoming["sha256"] = digest(incoming)
    state = {"manifest_sha256":previous["sha256"],"tasks":{"s1:native":{"task_id":"operation-real","state":"COMPLETED"}}}
    migrated = append_only_state(previous,incoming,state,record,deadline)
    assert migrated["tasks"] == state["tasks"]
    assert migrated["inventory_migrations"][0]["deadline_utc"] == deadline.isoformat()
    with pytest.raises(ValueError,match="deadline"):
        append_only_state(previous,incoming,state,record,deadline+timedelta(seconds=1))
    with pytest.raises(FileExistsError):
        prepare_extension(previous,target,config,record_path,deadline.isoformat())
    changed = copy.deepcopy(incoming)
    changed["configuration"]["max_sources"] -= 1
    changed.pop("sha256")
    changed["sha256"] = digest(changed)
    with pytest.raises(ValueError,match="beyond"):
        append_only_state(previous,changed,state,record,deadline)
