"""Native coastal NAIP acquisition, resumable bounded Earth Engine exports.

Inventory is immutable; task state and generation-pinned object audits are separate.
No authentication, billing, VM provisioning, or automatic failed-task retries.
"""
import argparse
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from src.cloud_config import GCP_PROJECT, GCS_PILOT_BUCKET

BUCKET = GCS_PILOT_BUCKET
PREFIX = "marine100/raw"
ACTIVE = {"READY", "RUNNING", "SUBMITTING", "UNSUBMITTED", "CANCEL_REQUESTED"}
KNOWN_REFERENCE_SIZE_FAILURES = {"naip_55431267fb767401b72e", "naip_eef54ccb408744b6e606"}


def export_kinds(row):
    return [kind for kind in ("native", "reference", "nativevalid") if kind in row["exports"]]


def upgrade_source(row):
    """Upgrade pre-export progress rows; never rewrite a completed inventory."""
    native = row["exports"]["native"]["estimated_max_bytes"]
    pixels = row["native_dimensions"][0] * row["native_dimensions"][1]
    row["exports"]["nativevalid"] = {"prefix": f"{PREFIX}/{row['source_id']}/nativevalid",
                                      "estimated_max_bytes": math.ceil(pixels * 1.01) + 1048576}
    # Whole NAIP footprint on a 10m grid, six float32 bands, conservative edge reserve.
    ref_pixels = math.ceil(pixels * row["native_gsd_m"] ** 2 / 100 * 1.05)
    row["exports"]["reference"]["estimated_max_bytes"] = ref_pixels * 24 + 1048576
    row["estimated_total_raw_bytes"] = native + sum(row["exports"][k]["estimated_max_bytes"] for k in ("reference", "nativevalid"))
    return row


def initialize_marine(project, gcloud_auth=False):
    """Refresh gcloud credentials during long inventories instead of freezing token."""
    if not gcloud_auth:
        from .export_ee import initialize
        return initialize(project)
    import ee
    from google.auth.credentials import Credentials

    class GcloudCredentials(Credentials):
        def refresh(self, request):
            self.token = subprocess.check_output(
                ["gcloud", "auth", "print-access-token"], text=True, timeout=60
            ).strip()
            if not self.token:
                raise ValueError("gcloud returned empty access token")
            # google-auth compares expiry with naive UTC; refresh proactively.
            self.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=40)

    credentials = GcloudCredentials()
    credentials.refresh(None)
    ee.Initialize(credentials=credentials, project=project)
    ee.data.setDeadline(60000)
    return ee


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def footprint_key(asset_id):
    # USDA quarter-quad name excludes date, resolution, and UTM zone.
    name = asset_id.rsplit("/", 1)[-1].lower()
    match = re.match(r"[a-z]_(\d{7,8})_(ne|nw|se|sw)_", name)
    if not match:
        raise ValueError(f"Unknown NAIP footprint naming: {asset_id}")
    return "_".join(match.groups())


def validate_config(config):
    if config["bucket"] != BUCKET or config["prefix"] != PREFIX:
        raise ValueError("Only dedicated marine bucket prefix allowed")
    if not 1 <= config["max_sources"] <= 600:
        raise ValueError("Source cap must be <=600")
    if not 1 <= config["max_active_tasks"] <= 2:
        raise ValueError("At most two active export tasks")
    if not 100_000_000_000 <= config["max_native_bytes"] <= 300_000_000_000:
        raise ValueError("Raw byte budget must be 100..300 decimal GB")
    if not 1 <= config["max_candidates_per_region"] <= 100:
        raise ValueError("Bounded region inventory required")
    if len(config["regions"]) > 60:
        raise ValueError("At most sixty predefined coastal regions")
    for region in config["regions"]:
        if region["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        if not 1 <= region["radius_km"] <= 40:
            raise ValueError("Coastal search radius cap 40 km")


def validate_manifest(manifest):
    config = manifest["configuration"]
    validate_config(config)
    sources = manifest["sources"]
    if len(sources) > config["max_sources"]:
        raise ValueError("Source cap exceeded")
    for field in ("source_id", "footprint_key", "naip_id"):
        if len({row[field] for row in sources}) != len(sources):
            raise ValueError(f"Duplicate {field}")
    if sum(s.get("estimated_total_raw_bytes", s["exports"]["native"]["estimated_max_bytes"]) for s in sources) > config["max_native_bytes"]:
        raise ValueError("Native upper-bound budget exceeded")
    body = {k: v for k, v in manifest.items() if k != "sha256"}
    if manifest["sha256"] != digest(body):
        raise ValueError("Immutable manifest hash mismatch")


def prepare_extension(previous, next_manifest, next_config, record_path, deadline):
    """Stage an explicit 150->300GB version without editing old immutable files."""
    validate_manifest(previous)
    if previous["configuration"]["max_native_bytes"] != 150_000_000_000:
        raise ValueError("Only the explicit 150-to-300 GB extension is supported")
    stamp = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Existing absolute deadline must include timezone")
    for row in previous["sources"]:
        if max(row["native_dimensions"]) > 32768:
            raise ValueError("Existing source exceeds single-TIFF bound")
    config = copy.deepcopy(previous["configuration"])
    config["max_native_bytes"] = 300_000_000_000
    validate_config(config)
    target = Path(next_manifest)
    progress_path = target.with_suffix(".progress.json")
    paths = [target, progress_path, Path(next_config), Path(record_path)]
    if len({p.resolve() for p in paths}) != len(paths) or any(p.exists() for p in paths):
        raise FileExistsError("Extension requires distinct unused version paths")
    progress = {"configuration_sha256": digest(config), "sources": copy.deepcopy(previous["sources"]),
                "rejected": copy.deepcopy(previous.get("rejected", [])),
                "seen": sorted(row["footprint_key"] for row in previous["sources"])}
    record = {"schema": 1, "kind": "explicit_raw_budget_150_to_300_gb",
              "previous_inventory_sha256": previous["sha256"],
              "previous_configuration": previous["configuration"],
              "previous_configuration_sha256": digest(previous["configuration"]),
              "previous_sources": previous["sources"], "new_configuration_sha256": digest(config),
              "deadline_utc": stamp.isoformat(), "next_manifest": str(target),
              "next_config": str(next_config), "created_utc": datetime.now(timezone.utc).isoformat()}
    record["sha256"] = digest(record)
    write_json(next_config, config, exclusive=True)
    write_json(progress_path, progress, exclusive=True)
    write_json(record_path, record, exclusive=True)
    return record


def inventory(ee, config, output):
    validate_config(config)
    if Path(output).exists():
        raise FileExistsError("Inventory immutable; use existing manifest to resume")
    progress_path = Path(output).with_suffix(".progress.json")
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
    if progress and progress["configuration_sha256"] != digest(config):
        raise ValueError("Inventory progress configuration differs")
    sources = [upgrade_source(row) for row in progress["sources"]] if progress else []
    rejected = progress["rejected"] if progress else []
    seen = set(progress["seen"]) if progress else set()
    budget = sum(s["estimated_total_raw_bytes"] for s in sources)
    if budget > config["max_native_bytes"]:
        raise ValueError("Upgraded progress plus mask/reference exceeds raw budget")

    def checkpoint():
        write_json(progress_path, {"configuration_sha256": digest(config), "sources": sources,
                                   "rejected": rejected, "seen": sorted(seen)})
    water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(50).unmask(0)
    for region in config["regions"]:
        print(json.dumps({"inventory_region": region["id"], "accepted": len(sources),
                          "seen_footprints": len(seen), "native_upper_bytes": budget}), flush=True)
        roi = ee.Geometry.Point(region["center"]).buffer(region["radius_km"] * 1000)
        collection = (ee.ImageCollection("USDA/NAIP/DOQQ")
                      .filterBounds(roi).filterDate(config["start"], config["end"])
                      .map(lambda im: im.set("has_nir", im.bandNames().contains("N")))
                      .filter(ee.Filter.eq("has_nir", True))
                      .sort("system:time_start", False).limit(min(region.get("max_candidates", config["max_candidates_per_region"]), config["max_candidates_per_region"])))
        ids = collection.aggregate_array("system:id").getInfo()
        for asset in ids:
            if len(sources) >= config["max_sources"]:
                break
            key = footprint_key(asset)
            if key in seen:
                continue
            seen.add(key)
            image = ee.Image(asset)
            info = image.select(["R", "G", "B", "N"]).getInfo()
            band = info["bands"][0]
            transform = band["crs_transform"]
            dimensions = band["dimensions"]
            if max(dimensions) > 32768:
                rejected.append({"naip_id": asset, "reason": "exceeds_32768_single_tiff_bound"})
                continue
            native_gsd = max(math.hypot(transform[0], transform[3]), math.hypot(transform[1], transform[4]))
            if native_gsd > 2.5 or native_gsd < 0.1:
                rejected.append({"naip_id": asset, "reason": "unsupported_native_gsd"})
                continue
            # Require the actual scene to overlap this coastal ROI, with water.
            water_fraction = water.reduceRegion(ee.Reducer.mean(), image.geometry(), 120,
                                                maxPixels=1000000).getInfo().get("occurrence")
            if water_fraction is None or not config["min_water_fraction"] <= water_fraction <= config["max_water_fraction"]:
                rejected.append({"naip_id": asset, "reason": "coastal_water_fraction", "water_fraction": water_fraction})
                continue
            stamp = info["properties"]["system:time_start"]
            date = datetime.fromtimestamp(stamp / 1000, timezone.utc)
            gap = timedelta(days=config["max_date_gap_days"])
            s2s = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                   .filterBounds(image.geometry().centroid(1))
                   .filterDate((date-gap).isoformat(), (date+gap).isoformat())
                   .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 20))
                   .map(lambda im: im.set("date_gap", ee.Number(im.get("system:time_start")).subtract(stamp).abs()))
                   .sort("date_gap").limit(1))
            s2_ids = s2s.aggregate_array("system:id").getInfo()
            if not s2_ids:
                rejected.append({"naip_id": asset, "reason": "no_near_date_s2"})
                continue
            # Native raster only, no upsampled padding/overviews. 1% TIFF overhead reserve.
            upper = math.ceil(dimensions[0] * dimensions[1] * 4 * 1.01) + 1048576
            if budget + upper > config["max_native_bytes"]:
                continue
            s2 = ee.Image(s2_ids[0])
            source_id = "naip_" + hashlib.sha256(asset.encode()).hexdigest()[:20]
            row = dict(source_id=source_id, footprint_key=key, split=region["split"],
                       geography_group=region["id"], naip_id=asset, naip_time_ms=stamp,
                       naip_projection={"crs": band["crs"], "transform": transform},
                       native_dimensions=dimensions, native_gsd_m=native_gsd,
                       bounds_wgs84=image.geometry().bounds(1).getInfo(), water_fraction=water_fraction,
                       s2_id=s2_ids[0], s2_time_ms=s2.get("system:time_start").getInfo(),
                       s2_projection=s2.select("B4").projection().getInfo(),
                       exports={"native": {"prefix": f"{PREFIX}/{source_id}/native", "estimated_max_bytes": upper},
                                "reference": {"prefix": f"{PREFIX}/{source_id}/reference"}})
            upgrade_source(row)
            if budget + row["estimated_total_raw_bytes"] > config["max_native_bytes"]:
                continue
            sources.append(row)
            budget += row["estimated_total_raw_bytes"]
            checkpoint()
            print(json.dumps({"accepted": len(sources), "native_upper_bytes": budget, "source_id": source_id}), flush=True)
        checkpoint()
        if len(sources) >= config["max_sources"]:
            break
    manifest = {"schema": 1, "configuration": config, "sources": sources, "rejected": rejected,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "reference_band_order": ["R", "G", "B", "N", "SCL", "valid"],
                "native_band_order": ["R", "G", "B", "N"]}
    manifest["sha256"] = digest(manifest)
    validate_manifest(manifest)
    write_json(output, manifest, exclusive=True)
    return manifest


def export_task(ee, row, kind):
    if max(row["native_dimensions"]) > 32768:
        raise ValueError("Source exceeds single-TIFF dimension bound")
    naip = ee.Image(row["naip_id"])
    if kind in {"native", "nativevalid"}:
        image = naip.select(["R", "G", "B", "N"]).toUint8()
        if kind == "nativevalid":
            image = image.mask().reduce(ee.Reducer.min()).unmask(0).rename("nativevalid").toUint8()
        projection = row["naip_projection"]
        max_pixels = row["native_dimensions"][0] * row["native_dimensions"][1] * 1.02
    else:
        s2 = ee.Image(row["s2_id"])
        rgbn = s2.select(["B4", "B3", "B2", "B8"], ["R", "G", "B", "N"]).multiply(0.0001)
        valid = rgbn.mask().reduce(ee.Reducer.min()).rename("valid")
        image = rgbn.addBands(s2.select("SCL")).addBands(valid).toFloat()
        projection = row["s2_projection"]
        max_pixels = 4000000
    return ee.batch.Export.image.toCloudStorage(
        image=image, description=f"marine100_{row['source_id']}_{kind}", bucket=BUCKET,
        fileNamePrefix=row["exports"][kind]["prefix"], region=naip.geometry(),
        crs=projection["crs"], crsTransform=projection["transform"],
        maxPixels=math.ceil(max_pixels), fileFormat="GeoTIFF", fileDimensions=16384 if kind == "reference" else 32768,
        shardSize=256, skipEmptyTiles=True, formatOptions={"cloudOptimized": False})


def reference_prefix_objects(prefix):
    response = subprocess.run(["gcloud", "storage", "ls", f"gs://{BUCKET}/{prefix}**", "--json"],
                              capture_output=True, text=True, timeout=90)
    if response.returncode:
        if "matched no objects" in response.stderr.lower() or "no urls matched" in response.stderr.lower():
            return []
        raise RuntimeError(response.stderr)
    result = json.loads(response.stdout)
    if not isinstance(result, list):
        raise ValueError("Expected exact GCS JSON object list")
    return result


def retry_failed_references(ee, manifest, state_path, source_ids, list_objects=reference_prefix_objects):
    """One explicit retry of the two observed EE float32 tile-size failures."""
    if not source_ids or len(source_ids) != len(set(source_ids)) or not set(source_ids) <= KNOWN_REFERENCE_SIZE_FAILURES:
        raise ValueError("Retry restricted to the two verified reference size failures")
    validate_manifest(manifest)
    path = Path(state_path)
    state = json.loads(path.read_text())
    if state["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("Retry state/inventory mismatch")
    rows = {row["source_id"]: row for row in manifest["sources"]}
    eligible = []
    for source_id in source_ids:
        row = rows[source_id]
        old = state["tasks"][f"{source_id}:reference"]
        if old["state"] != "FAILED" or not old.get("task_id") or old.get("attempts"):
            raise ValueError("Only first verified failed reference may be retried")
        status = ee.data.getTaskStatus([old["task_id"]])[0]
        compact_error = re.sub(r"[\s,]", "", status.get("error_message", ""))
        if status["state"] != "FAILED" or not all(size in compact_error for size in ("25769803776", "17179869184")):
            raise ValueError("Server failure is not the known reference tile-size error")
        if list_objects(row["exports"]["reference"]["prefix"]):
            raise ValueError("Reference prefix has objects; retry would overwrite output")
        eligible.append((row, old, status))
    active = sum(task.get("state") in ACTIVE for task in ee.data.getTaskList())
    unknown = sum(task["state"] in {"UNKNOWN", "SUBMITTING", "UNSUBMITTED"} for task in state["tasks"].values())
    if active + unknown + len(eligible) > manifest["configuration"]["max_active_tasks"]:
        return state  # No mutation; an explicit caller may retry when slots free.
    for row, old, status in eligible:
        task = export_task(ee, row, "reference")
        request_id = ee.data.newTaskId()[0]
        task._request_id = request_id
        record = {"request_id": request_id, "task_id": None, "state": "SUBMITTING",
                  "source_id": row["source_id"], "kind": "reference",
                  "retry_reason": "float32_reference_fileDimensions_32768_to_16384",
                  "attempts": [dict(old, verified_failure=status)]}
        state["tasks"][f"{row['source_id']}:reference"] = record
        write_json(path, state)
        task.start()
        if not task.id:
            raise ValueError("Retry submission uncertain; operation ID missing")
        record.update(task_id=task.id, operation_name=task.name, state="READY")
        write_json(path, state)
    return state


def scheduler_tick(ee, manifest, state_path, submit=False, *, max_pending_tasks=None):
    # Queue depth is separate from Earth Engine's server-side execution quota.
    # Keep the immutable inventory's original default when no override is given.
    if max_pending_tasks is not None and (type(max_pending_tasks) is not int or not 2 <= max_pending_tasks <= 6):
        raise ValueError("max_pending_tasks must be an integer from 2 to 6")
    validate_manifest(manifest)
    pending_limit = manifest["configuration"]["max_active_tasks"] if max_pending_tasks is None else max_pending_tasks
    path = Path(state_path)
    state = json.loads(path.read_text()) if path.exists() else {"manifest_sha256": manifest["sha256"], "tasks": {}}
    if state["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("Task state belongs to different inventory")
    # Persist IDs before submission. Unknown/SUBMITTING remains blocking on resume.
    pending = [task for task in state["tasks"].values()
               if task.get("task_id") and task["state"] not in {"COMPLETED", "FAILED", "CANCELLED"}]
    for offset in range(0, len(pending), 100):
        batch = pending[offset:offset + 100]
        statuses = ee.data.getTaskStatus([task["task_id"] for task in batch])
        if len(statuses) != len(batch):
            raise ValueError("Incomplete EE task status response")
        for task, status in zip(batch, statuses):
            task.update(state=status["state"], error_message=status.get("error_message"))
    write_json(path, state)
    # Count project-wide ready/running tasks to avoid competing exports.
    live = sum(t.get("state") in ACTIVE for t in ee.data.getTaskList())
    pending_unknown = sum(t["state"] in {"UNKNOWN", "UNSUBMITTED", "SUBMITTING"} for t in state["tasks"].values())
    slots = max(0, pending_limit - live - pending_unknown)
    if submit:
        for row in manifest["sources"]:
            for kind in export_kinds(row):
                key = f"{row['source_id']}:{kind}"
                if key in state["tasks"] or slots <= 0:
                    continue
                task = export_task(ee, row, kind)
                # Idempotency request UUID and returned operation ID are distinct.
                # Leave task.id unset so EE.start captures the server operation.
                request_id = ee.data.newTaskId()[0]
                task._request_id = request_id
                state["tasks"][key] = {"request_id": request_id, "task_id": None,
                                       "state": "SUBMITTING", "source_id": row["source_id"], "kind": kind}
                write_json(path, state)
                task.start()
                if not task.id:
                    raise ValueError("EE export did not return an operation ID")
                state["tasks"][key].update(task_id=task.id, operation_name=task.name)
                state["tasks"][key]["state"] = "READY"
                write_json(path, state)
                slots -= 1
    return state


def audit_objects(manifest, objects, used_source_ids=()):
    validate_manifest(manifest)
    by_prefix = {f"gs://{BUCKET}/{r['exports'][kind]['prefix']}": (r, kind)
                 for r in manifest["sources"] for kind in export_kinds(r)}
    used = set(used_source_ids)
    known = {r["source_id"] for r in manifest["sources"]}
    if not used <= known:
        raise ValueError("Used ledger has unknown source IDs")
    seen, acquired, train_used, rows = set(), 0, 0, []
    for obj in objects:
        uri = obj["uri"]
        if uri in seen:
            raise ValueError("Duplicate object URI (including multiple generations)")
        seen.add(uri)
        if int(obj["size"]) <= 0 or not obj.get("generation") or not obj.get("crc32c"):
            raise ValueError("Object size/generation/checksum required")
        matches = [r for prefix, r in by_prefix.items() if uri == prefix + ".tif" or (uri.startswith(prefix + "-") and uri.endswith(".tif"))]
        if not matches:
            continue
        source, kind = matches[0]
        size = int(obj["size"])
        acquired += size if kind == "native" else 0
        admitted = kind == "native" and source["split"] == "train" and source["source_id"] in used
        train_used += size if admitted else 0
        rows.append(dict(obj, source_id=source["source_id"], kind=kind, split=source["split"], used_for_training=admitted))
    if acquired > manifest["configuration"]["max_native_bytes"]:
        raise ValueError("Actual native GCS byte budget exceeded; stop acquisition")
    return {"manifest_sha256": manifest["sha256"], "native_acquired_bytes": acquired,
            "unique_train_source_bytes_used": train_used, "meets_100gb": train_used >= 100_000_000_000,
            "objects": rows, "warning": "Used ledger must come from admitted sources actually consumed by trainer; downloads alone do not qualify."}


def preparation_manifest(manifest, audit):
    rows, excluded = [], []
    for row in manifest["sources"]:
        objects = {kind: [o for o in audit["objects"] if o["source_id"] == row["source_id"] and o["kind"] == kind]
                   for kind in export_kinds(row)}
        if any(len(v) != 1 for v in objects.values()):
            excluded.append({"source_id": row["source_id"], "reason": "missing_or_sharded_export"})
            continue
        coordinates = row["bounds_wgs84"]["coordinates"][0]
        lon = (min(p[0] for p in coordinates) + max(p[0] for p in coordinates)) / 2
        lat = (min(p[1] for p in coordinates) + max(p[1] for p in coordinates)) / 2
        names = [("naip", "native"), ("s2", "reference")]
        if "nativevalid" in objects:
            names.append(("nativevalid", "nativevalid"))
        files = {name: {"uri": objects[kind][0]["uri"], "generation": str(objects[kind][0]["generation"]),
                        "bytes": int(objects[kind][0]["size"]), "crc32c": objects[kind][0]["crc32c"]}
                 for name, kind in names}
        rows.append(dict(row, id=row["source_id"], lon=lon, lat=lat, files=files))
    return {"schema": 1, "inventory_sha256": manifest["sha256"], "records": rows, "excluded": excluded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inventory", "tick", "status", "audit", "extend", "retry-reference-size"])
    parser.add_argument("--config", default="configs/marine-source-v1.json")
    parser.add_argument("--manifest", default="data/marine100/inventory.json")
    parser.add_argument("--state", default="data/marine100/tasks.json")
    parser.add_argument("--project", default=GCP_PROJECT)
    parser.add_argument("--gcloud-auth", action="store_true")
    parser.add_argument("--objects", help="JSON list of uri, generation, size, crc32c from GCS metadata")
    parser.add_argument("--used", help="JSON array of unique source IDs actually used by trainer")
    parser.add_argument("--output", default="data/marine100/audit.json")
    parser.add_argument("--next-manifest")
    parser.add_argument("--next-config")
    parser.add_argument("--extension-record")
    parser.add_argument("--deadline", help="Existing controller absolute deadline; never extended")
    parser.add_argument("--source-id", action="append", help="Exact verified source for targeted reference-size retry")
    args = parser.parse_args()
    if args.command == "extend":
        if not all((args.next_manifest, args.next_config, args.extension_record, args.deadline)):
            parser.error("extend requires --next-manifest --next-config --extension-record --deadline")
        result = prepare_extension(json.loads(Path(args.manifest).read_text()), args.next_manifest,
                                   args.next_config, args.extension_record, args.deadline)
    elif args.command == "audit":
        manifest = json.loads(Path(args.manifest).read_text())
        result = audit_objects(manifest, json.loads(Path(args.objects).read_text()),
                               json.loads(Path(args.used).read_text()) if args.used else [])
        write_json(args.output, result)
        write_json(Path(args.output).with_suffix(".sources.json"), preparation_manifest(manifest, result))
    else:
        ee = initialize_marine(args.project, args.gcloud_auth)
        if args.command == "inventory":
            result = inventory(ee, json.loads(Path(args.config).read_text()), args.manifest)
        elif args.command == "retry-reference-size":
            result = retry_failed_references(ee, json.loads(Path(args.manifest).read_text()), args.state, args.source_id or [])
        else:
            result = scheduler_tick(ee, json.loads(Path(args.manifest).read_text()), args.state, args.command == "tick")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
