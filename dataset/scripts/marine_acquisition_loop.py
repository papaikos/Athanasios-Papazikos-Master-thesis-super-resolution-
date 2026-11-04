"""Bounded single-writer EE acquisition loop; no billing or IAM operations.

Run as python -m scripts.marine_acquisition_loop --deadline <UTC ISO timestamp>.
The absolute deadline persists across restarts and can never be extended.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import time

from src.marine_acquire import (BUCKET, PREFIX, audit_objects, preparation_manifest,
                                scheduler_tick, validate_manifest, write_json, initialize_marine,
                                export_kinds, upgrade_source, digest, retry_failed_references,
                                KNOWN_REFERENCE_SIZE_FAILURES)
from src.cloud_config import GCP_PROJECT

CONTROL = f"gs://{BUCKET}/marine100/control/v1"
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}


def ensure_ee_client(client, project, log):
    """Reuse EE's expensive client; its credentials refresh tokens automatically."""
    if client is None:
        log("ee_initialize_start")
        client = initialize_marine(project, gcloud_auth=True)
        log("ee_initialize_done")
    return client


def inventory_snapshot(manifest_path, config_path):
    """Read one atomic progress checkpoint; accepted rows only, stable digest."""
    path = Path(manifest_path)
    if path.exists():
        manifest = json.loads(path.read_text())
        validate_manifest(manifest)
        return manifest, True
    progress_path = path.with_suffix(".progress.json")
    if not progress_path.exists():
        return None, False
    config = json.loads(Path(config_path).read_text())
    progress = json.loads(progress_path.read_text())
    if progress["configuration_sha256"] != digest(config):
        raise ValueError("Progress/config hash mismatch")
    rows = [upgrade_source(copy.deepcopy(row)) for row in progress["sources"]]
    if not rows:
        return None, False
    manifest = {"schema": 1, "configuration": config, "sources": rows,
                "inventory_status": "accepted_progress_snapshot",
                "reference_band_order": ["R", "G", "B", "N", "SCL", "valid"],
                "native_band_order": ["R", "G", "B", "N"]}
    manifest["sha256"] = digest(manifest)
    validate_manifest(manifest)
    return manifest, False


def append_only_state(previous, incoming, state, extension=None, deadline=None):
    """Narrow loop-owned migration; the global scheduler hash guard stays intact."""
    validate_manifest(previous)
    validate_manifest(incoming)
    if state["manifest_sha256"] != previous["sha256"]:
        raise ValueError("State/previous snapshot hash mismatch")
    if previous["configuration"] != incoming["configuration"]:
        if extension is None:
            raise ValueError("Append-only inventory cannot change configuration")
        body = {k:v for k,v in extension.items() if k != "sha256"}
        if extension.get("sha256") != digest(body):
            raise ValueError("Extension record hash mismatch")
        expected_config = copy.deepcopy(previous["configuration"])
        if expected_config["max_native_bytes"] != 150_000_000_000:
            raise ValueError("Extension only permits 150-to-300 GB")
        expected_config["max_native_bytes"] = 300_000_000_000
        if incoming["configuration"] != expected_config:
            raise ValueError("Extension changed fields beyond raw-byte cap")
        if (extension["kind"] != "explicit_raw_budget_150_to_300_gb"
                or extension["previous_configuration"] != previous["configuration"]
                or extension["previous_configuration_sha256"] != digest(previous["configuration"])
                or extension["new_configuration_sha256"] != digest(incoming["configuration"])):
            raise ValueError("Extension configuration evidence mismatch")
        if deadline is None or datetime.fromisoformat(extension["deadline_utc"]) != deadline:
            raise ValueError("Extension must preserve exact existing absolute deadline")
        base_rows = extension["previous_sources"]
        if (base_rows[:len(previous["sources"])] != previous["sources"]
                or incoming["sources"][:len(base_rows)] != base_rows):
            raise ValueError("Extension changed base source records")
    old_rows, new_rows = previous["sources"], incoming["sources"]
    if len(new_rows) < len(old_rows) or new_rows[:len(old_rows)] != old_rows:
        raise ValueError("Append-only inventory removed, reordered, or changed existing sources")
    expected = {f"{row['source_id']}:{kind}" for row in old_rows for kind in export_kinds(row)}
    if not set(state["tasks"]) <= expected:
        raise ValueError("State contains tasks absent from previous snapshot")
    result = copy.deepcopy(state)
    result["manifest_sha256"] = incoming["sha256"]
    if extension is not None and previous["configuration"] != incoming["configuration"]:
        result.setdefault("inventory_migrations", []).append({
            "previous_manifest_sha256": previous["sha256"], "new_manifest_sha256": incoming["sha256"],
            "extension_sha256": extension["sha256"], "deadline_utc": deadline.isoformat()})
    return result


def utc_now():
    return datetime.now(timezone.utc)


def parse_deadline(value, now):
    deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("Deadline must include UTC offset")
    if (deadline - now).total_seconds() > 12 * 3600:
        raise ValueError("Absolute deadline may be at most 12 hours ahead")
    return deadline


def normalize_objects(payload):
    """Normalize gcloud storage ls --json or raw JSON API metadata, fail closed."""
    if isinstance(payload, dict):
        payload = payload.get("items", [payload])
    if not isinstance(payload, list):
        raise ValueError("GCS listing must be a JSON object list")
    result = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Malformed object metadata")
        metadata = item.get("metadata", item)
        bucket = metadata.get("bucket", item.get("bucket"))
        name = metadata.get("name", item.get("name"))
        uri = item.get("url") or item.get("storage_url") or item.get("uri")
        if not uri and bucket and name:
            uri = f"gs://{bucket}/{name}"
        # Some CLI versions use bucket-name/object-name vocabulary.
        if not uri and item.get("bucket_name") and item.get("object_name"):
            uri = f"gs://{item['bucket_name']}/{item['object_name']}"
        if not uri or not uri.startswith(f"gs://{BUCKET}/{PREFIX}/"):
            raise ValueError("Unexpected GCS object outside exact marine raw prefix")
        generation = metadata.get("generation", item.get("generation"))
        size = metadata.get("size", item.get("size"))
        crc32c = metadata.get("crc32c", metadata.get("crc32c_hash", item.get("crc32c_hash")))
        if size is None or int(size) < 0 or not generation or not crc32c:
            raise ValueError(f"Missing exact size/generation/checksum for {uri}")
        if "#" in uri:
            bare_uri, _, suffix = uri.rpartition("#")
            if not suffix.isdecimal() or suffix != str(generation):
                raise ValueError("GCS URI generation suffix disagrees with object metadata")
            uri = bare_uri
        result.append({"uri": uri, "generation": str(generation), "size": int(size), "crc32c": crc32c})
    return result


def gcloud(arguments, timeout=90):
    return subprocess.run(["gcloud", *arguments], check=True, capture_output=True,
                          text=True, timeout=timeout)


def read_objects():
    # --json is required; storage ls explicitly does not support --format=json.
    try:
        response = gcloud(["storage", "ls", f"gs://{BUCKET}/{PREFIX}/**", "--json"])
    except subprocess.CalledProcessError as exc:
        if "matched no objects" in exc.stderr.lower() or "no urls matched" in exc.stderr.lower():
            return []
        raise
    return normalize_objects(json.loads(response.stdout))


def completed_sources(manifest, state, audit):
    candidate = preparation_manifest(manifest, audit)
    accepted = []
    for row in candidate["records"]:
        if all(state["tasks"].get(f"{row['source_id']}:{kind}", {}).get("state") == "COMPLETED"
               for kind in export_kinds(row)):
            accepted.append(row)
    candidate["records"] = accepted
    candidate["status"] = "partial_read_only"
    return candidate


def all_terminal(manifest, state):
    expected = {f"{r['source_id']}:{kind}" for r in manifest["sources"] for kind in export_kinds(r)}
    return set(state["tasks"]) == expected and all(t["state"] in TERMINAL for t in state["tasks"].values())


def repair_accounting(manifest, state, object_listing, final_inventory=False):
    """Rebuild accounting from saved evidence without submitting or altering tasks."""
    validate_manifest(manifest)
    if state.get("manifest_sha256") != manifest["sha256"]:
        raise ValueError("Accounting task state/inventory hash mismatch")
    objects = normalize_objects(object_listing)
    if sum(item["size"] for item in objects) > manifest["configuration"]["max_native_bytes"]:
        raise ValueError("Accounting raw storage budget exceeded")
    audit = audit_objects(manifest, objects)
    candidate = completed_sources(manifest, state, audit)
    if final_inventory:
        if manifest.get("inventory_status") == "accepted_progress_snapshot":
            raise ValueError("Progress snapshot cannot authorize final accounting")
        if all_terminal(manifest, state):
            candidate["status"] = "final"
    return {"objects": objects, "audit": audit, "adapter": candidate}


def cancel_owned(ee, state, log):
    # Never cancel another workload. Task IDs were persisted before submission.
    errors = []
    for task in state.get("tasks", {}).values():
        if task["state"] not in TERMINAL:
            if not task.get("task_id"):
                errors.append({"request_id": task.get("request_id"), "error": "Submission uncertain; server operation ID unavailable; manual resolution required"})
                continue
            try:
                ee.data.cancelTask(task["task_id"])
                task["state"] = "CANCEL_REQUESTED"
                log("cancel_requested", task_id=task["task_id"])
            except Exception as exc:
                errors.append({"task_id": task["task_id"], "error": str(exc)})
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/marine100/inventory.json")
    parser.add_argument("--config", default="configs/marine-source-v1.json")
    parser.add_argument("--directory", default="data/marine100/acquisition")
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--project", default=GCP_PROJECT)
    parser.add_argument("--max-pending-tasks", type=int, choices=range(2, 7), default=2,
                        help="Bound READY/RUNNING export queue; does not change EE execution quota")
    parser.add_argument("--extension-record", help="Explicit immutable 150-to-300 GB migration record")
    parser.add_argument("--retry-known-reference-size-failures", action="store_true",
                        help="One targeted retry of the two previously verified reference tile-size failures")
    args = parser.parse_args()
    extension = json.loads(Path(args.extension_record).read_text()) if args.extension_record else None
    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=True)
    # OS-held lock disappears after crashes; no stale PID lock can start duplicates.
    import fcntl
    lock = (root / ".loop.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = parse_deadline(args.deadline, utc_now())
    control_path = root / "loop-state.json"
    control = json.loads(control_path.read_text()) if control_path.exists() else {
        "created_utc": utc_now().isoformat(), "deadline_utc": deadline.isoformat(), "status": "waiting_inventory"}
    if datetime.fromisoformat(control["deadline_utc"]) != deadline:
        raise ValueError("Persisted acquisition deadline cannot change")
    write_json(control_path, control)
    log_path = root / "loop.jsonl"

    def log(event, **fields):
        row = {"utc": utc_now().isoformat(), "event": event, **fields}
        with log_path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    def sync():
        files = [str(p) for p in root.iterdir() if p.is_file() and p.suffix in {".json", ".jsonl"}]
        gcloud(["storage", "cp", *files, CONTROL + "/", "--quiet"], timeout=90)

    def interrupt(signum, frame):
        raise TimeoutError(f"Acquisition interrupted or deadline reached: {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    remaining = (deadline - utc_now()).total_seconds()
    state, ee, manifest = {"tasks": {}}, None, None
    state_path = root / "tasks.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    signal.alarm(max(1, int(remaining)))
    try:
        log("controller_started", deadline_utc=deadline.isoformat(), max_pending_tasks=args.max_pending_tasks)
        if remaining <= 0:
            raise TimeoutError("Absolute acquisition deadline already expired")
        while utc_now() < deadline:
            incoming, final_inventory = inventory_snapshot(args.manifest, args.config)
            if incoming is None:
                control["status"] = "waiting_inventory"
                write_json(control_path, control)
                log("waiting_inventory")
                sync()
                time.sleep(min(60, max(0, (deadline-utc_now()).total_seconds())))
                continue
            if manifest is None or manifest["sha256"] != incoming["sha256"]:
                snapshots = root / "snapshots"
                snapshot_path = snapshots / (incoming["sha256"] + ".json")
                if not snapshot_path.exists():
                    write_json(snapshot_path, incoming, exclusive=True)
                elif json.loads(snapshot_path.read_text()) != incoming:
                    raise ValueError("Immutable snapshot content changed")
                # Remote snapshot precedes the task-state transition and submissions.
                log("snapshot_sync_start", snapshot_sha256=incoming["sha256"], sources=len(incoming["sources"]))
                gcloud(["storage", "cp", str(snapshot_path),
                        CONTROL + "/snapshots/" + snapshot_path.name, "--quiet"])
                log("snapshot_sync_done")
                if state_path.exists():
                    state = json.loads(state_path.read_text())
                    previous_hash = state.get("manifest_sha256")
                    if previous_hash and previous_hash != incoming["sha256"]:
                        previous_path = snapshots / (previous_hash + ".json")
                        if previous_path.exists():
                            previous = json.loads(previous_path.read_text())
                        else:
                            # Upgrade a pre-streaming controller only with its exact saved inventory.
                            previous = json.loads((root / "inventory.json").read_text())
                        state = append_only_state(previous, incoming, state, extension, deadline)
                        write_json(state_path, state)
                    elif not previous_hash and state.get("tasks"):
                        raise ValueError("Nonempty task state lacks inventory identity")
                    elif not previous_hash:
                        state = {"manifest_sha256": incoming["sha256"], "tasks": {}}
                        write_json(state_path, state)
                manifest = incoming
                write_json(root / "inventory.json", manifest)
                control["manifest_sha256"] = manifest["sha256"]
                control["inventory_final"] = final_inventory
                write_json(control_path, control)
                if extension is not None:
                    write_json(root / "extension-record.json", extension)
            # Credentials refresh automatically; reconstructing EE each tick is costly.
            ee = ensure_ee_client(ee, args.project, log)
            log("task_status_start")
            state = scheduler_tick(ee, manifest, state_path, submit=False)
            log("task_status_done", tasks=len(state["tasks"]))
            log("object_listing_start")
            objects = read_objects()
            log("object_listing_done", objects=len(objects))
            write_json(root / "objects.json", objects)
            # Include references and partial artifacts in total raw storage cap.
            raw_bytes = sum(o["size"] for o in objects)
            if raw_bytes >= manifest["configuration"]["max_native_bytes"]:
                raise RuntimeError(f"Raw prefix storage cap reached: {manifest['configuration']['max_native_bytes']} bytes")
            audit = audit_objects(manifest, objects)
            write_json(root / "audit.json", audit)
            candidate = completed_sources(manifest, state, audit)
            write_json(root / "sources.partial.json", candidate)
            snapshot_done = all_terminal(manifest, state)
            known_retry_pending = args.retry_known_reference_size_failures and any(
                state["tasks"].get(f"{source_id}:reference", {}).get("state") == "FAILED"
                and not state["tasks"][f"{source_id}:reference"].get("attempts")
                for source_id in KNOWN_REFERENCE_SIZE_FAILURES)
            done = final_inventory and snapshot_done and not known_retry_pending
            available_train_bytes = sum(o["size"] for o in audit["objects"] if o["kind"] == "native" and o["split"] == "train")
            status = "acquisition_exhausted_100gb_unproven" if done else ("waiting_more_inventory" if snapshot_done else "exporting")
            control.update(status=status, inventory_final=final_inventory, raw_bytes=raw_bytes,
                           available_train_native_bytes=available_train_bytes,
                           required_unique_train_bytes=100000000000, training_size_requirement_verified=False,
                           completed_source_pairs=len(candidate["records"]), task_count=len(state["tasks"]))
            write_json(control_path, control)
            if done:
                candidate["status"] = "final"
                final = root / "sources.final.json"
                if not final.exists():
                    write_json(final, candidate, exclusive=True)
                log("all_tasks_terminal", **control)
                sync()
                break
            # Before each submission, preserve the current bounded state remotely.
            sync()
            if utc_now() >= deadline:
                raise TimeoutError("Deadline reached before submission")
            if args.retry_known_reference_size_failures:
                retry_ids = [source_id for source_id in sorted(KNOWN_REFERENCE_SIZE_FAILURES)
                             if state["tasks"].get(f"{source_id}:reference", {}).get("state") == "FAILED"
                             and not state["tasks"][f"{source_id}:reference"].get("attempts")]
                if retry_ids:
                    log("known_reference_retry_check", source_ids=retry_ids)
                    state = retry_failed_references(ee, manifest, state_path, retry_ids)
            state = scheduler_tick(ee, manifest, state_path, submit=True, max_pending_tasks=args.max_pending_tasks)
            log("tick", **control)
            sync()
            time.sleep(min(60, max(0, (deadline-utc_now()).total_seconds())))
        else:
            raise TimeoutError("Absolute acquisition deadline reached")
    except BaseException as exc:
        # Preserve exit evidence and request remote task cancellation on deadline/error.
        signal.alarm(0)
        if state_path.exists():
            state = json.loads(state_path.read_text())
        if ee is None and state.get("tasks"):
            ee = ensure_ee_client(ee, args.project, log)
        errors = cancel_owned(ee, state, log) if ee is not None else []
        control.update(status="stopped", reason=str(exc), cancellation_errors=errors)
        write_json(state_path, state)
        write_json(control_path, control)
        log("stopped", reason=str(exc), cancellation_errors=errors)
        try:
            sync()
        except Exception as sync_error:
            log("final_sync_failed", error=str(sync_error))
        raise
    finally:
        signal.alarm(0)
        lock.close()


if __name__ == "__main__":
    main()
