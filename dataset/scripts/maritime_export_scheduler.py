"""GCE-only, persisted Earth Engine export controller for approved acquisitions.

Uses the original RGB projection and separate producer validity masks. A lost
start response blocks new submissions until the server task list identifies it.
References: developers.google.com/earth-engine/apidocs/export-image-tocloudstorage
and developers.google.com/earth-engine/reference/rest/v1/projects.image/export.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from rasterio.crs import CRS
from rasterio.errors import CRSError
from rasterio.transform import Affine

from scripts.maritime_cloud_inventory import (create_only, digest_bytes, file_sha256,
    json_bytes, parse_gs, require_cloud_vm, validate_manifest)
from scripts.maritime_reuse_candidates import approved_asset_ids

ACTIVE = {"READY", "RUNNING", "CANCEL_REQUESTED"}
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}
MAX_ATTEMPTS = 3
FILE_SIZE = 8192


def tile_count(width, height):
    return math.ceil(width / FILE_SIZE) * math.ceil(height / FILE_SIZE)


def initial_state(approval, sources, output_prefix, project):
    approved = sorted(approved_asset_ids(approval))
    if isinstance(sources, dict):
        sources = sources.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Source inventory must be an array or a document with sources")
    if sources:
        validate_manifest(sources)
    bucket, prefix = parse_gs(output_prefix.rstrip("/"))
    for row in sources:
        old_bucket, old_name = parse_gs(row["uri"])
        if bucket == old_bucket and old_name.startswith(prefix + "/"):
            raise ValueError("Export prefix overlaps existing raw imagery")
    existing = {row["asset_id"] for row in sources}
    config = dict(project=project, approval_sha256=digest_bytes(json_bytes(approval)),
                  source_inventory_sha256=digest_bytes(json_bytes(sources)),
                  output_prefix=output_prefix.rstrip("/"), worker_sha256=file_sha256(__file__),
                  max_active=2, max_attempts=MAX_ATTEMPTS, file_dimensions=FILE_SIZE)
    return dict(schema_version=1, revision=0, configuration=config,
                configuration_sha256=digest_bytes(json_bytes(config)), status="exporting",
                assets={asset: dict(state="SKIPPED_EXISTING" if asset in existing else "PENDING",
                                    tasks={}, history=[], source_objects=[])
                        for asset in approved})


def native_projection(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("Native projection must declare CRS or WKT and affine transform")
    definition = metadata.get("crs") or metadata.get("wkt")
    affine = metadata.get("transform")
    if not isinstance(definition, str) or not definition or not isinstance(affine, (list, tuple)) or len(affine) != 6:
        raise ValueError("Native projection must declare CRS or WKT and six affine terms")
    try:
        crs = CRS.from_user_input(definition)
    except CRSError as exc:
        raise ValueError("Invalid native CRS or WKT") from exc
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in affine):
        raise ValueError("Native affine must contain six finite numeric terms")
    return crs, tuple(affine)


def asset_plan(ee, asset_id):
    image = ee.Image(asset_id)
    projections = [image.select(band).projection().getInfo() for band in ("R", "G", "B")]
    typed = [native_projection(projection) for projection in projections]
    if typed[1:] != [typed[0], typed[0]]:
        raise ValueError("Original RGB band projections differ")
    projection = projections[0]
    affine = projection["transform"]
    crs, _ = typed[0]
    # Native WKT can call its metre unit 'm', 'Meter', or 'metre'. The
    # conversion factor is authoritative; a display label is not a unit test.
    metric = crs.is_projected and math.isclose(crs.linear_units_factor[1], 1.0,
                                             rel_tol=0, abs_tol=1e-12)
    if not metric or len(affine) != 6 or not all(
        math.isfinite(value) for value in affine) or not (
        abs(affine[0] - .8) <= 1e-6 and abs(affine[4] + .8) <= 1e-6
        and affine[1] == affine[3] == 0):
        raise ValueError("Original north-up metric 0.8 metre projection required")
    definition = projection.get("crs") or projection["wkt"]
    # A band's Projection includes its pixel-to-CRS affine. bounds() in that
    # projection returns pixel coordinates. Request the base CRS with identity
    # transform so coordinates really are metres, as the export region declares.
    base_projection = ee.Projection(definition)
    geometry = image.geometry().bounds(maxError=1, proj=base_projection)
    rings = geometry.coordinates().getInfo()
    if (not isinstance(rings, list) or len(rings) != 1 or not isinstance(rings[0], list)
        or len(rings[0]) < 4 or any(not isinstance(point, (list, tuple)) or len(point) != 2
            or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in point)
            for point in rings[0])):
        raise ValueError("Original footprint must have a finite bounded native rectangle")
    points = rings[0]
    bbox = [min(point[0] for point in points), min(point[1] for point in points),
            max(point[0] for point in points), max(point[1] for point in points)]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("Original footprint must have a nonempty bounded native rectangle")
    inverse = ~Affine(*affine)
    col1, row1 = inverse * (bbox[0], bbox[3])
    col2, row2 = inverse * (bbox[2], bbox[1])
    width = math.ceil(col2 - 1e-6) - math.floor(col1 + 1e-6)
    height = math.ceil(row2 - 1e-6) - math.floor(row1 + 1e-6)
    if width <= 0 or height <= 0 or width * height > 10**13:
        raise ValueError("Original footprint pixel dimensions exceed export safety limit")
    catalog = image.get("catalogID").getInfo()
    timestamp = image.get("system:time_start").getInfo()
    return dict(crs=definition,
                native_wkt=projection.get("wkt"), native_projection=projection,
                crs_transform=affine, native_bbox=bbox,
                geometry_coordinate_space="native_base_crs_metres",
                width=width, height=height, max_pixels=width * height + width + height,
                expected_file_tiles=tile_count(width, height), catalog_id=catalog or asset_id,
                acquisition_utc=datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat()
                    if isinstance(timestamp, (int, float)) else None,
                mask_provenance="producer_original_min_rgb_mask",
                mask_recipe="original_RGB.mask().reduce(min).eq(1).multiply(255).uint8().unmask(0,sameFootprint=True)",
                rgb_recipe="original_RGB.uint8().unmask(0,sameFootprint=True)",
                license="CC-BY-4.0", provider="Google Earth Engine",
                collection="SKYSAT/GEN-A/PUBLIC/ORTHO/RGB")


def export_images(ee, asset_id):
    original = ee.Image(asset_id).select(["R", "G", "B"])
    mask = original.mask().reduce(ee.Reducer.min()).eq(1).multiply(255).toUint8().rename("validity")
    return (original.toUint8().unmask(0, sameFootprint=True),
            mask.unmask(0, sameFootprint=True))


def permission_error(message):
    return any(word in (message or "").lower() for word in (
        "permission", "forbidden", "unauthorized", "not authorized", "access denied", "403", "401"))


def reconcile(ee, state, save):
    if state.get("status") == "blocked_permission":
        raise RuntimeError("Persisted Earth Engine permission failure; no retry permitted")
    # Query the full server list BEFORE any decisions about retries or capacity.
    remote = ee.data.getTaskList()
    if not isinstance(remote, list) or any("id" not in task or "state" not in task for task in remote):
        raise RuntimeError("Incomplete Earth Engine task list")
    for asset in state["assets"].values():
        for task in asset["tasks"].values():
            if task["state"] in TERMINAL:
                continue
            matches = [item for item in remote if item["id"] == task.get("task_id")
                       or item["id"] == task["request_id"]
                       or item.get("description") == task["description"]]
            if len(matches) != 1:
                state["status"] = "blocked_uncertain_start"
                save(state)
                raise RuntimeError("Earth Engine start/status remains uncertain; no submission permitted")
            item = matches[0]
            if item["state"] not in ACTIVE | TERMINAL:
                state["status"] = "blocked_uncertain_start"
                save(state)
                raise RuntimeError("Earth Engine task state remains uncertain")
            task.update(task_id=item["id"], state=item["state"],
                        error_message=item.get("error_message"))
            if permission_error(task.get("error_message")):
                state["status"] = "blocked_permission"
                save(state)
                raise RuntimeError("Earth Engine export permission failure; no retry permitted")
    save(state)
    return sum(task["state"] not in TERMINAL for task in remote)


def tick(ee, state, save, pin_outputs, *, deadline=None):
    active = reconcile(ee, state, save)
    state["status"] = "exporting"
    bucket, prefix = parse_gs(state["configuration"]["output_prefix"])
    for asset_id, asset in state["assets"].items():
        if deadline is not None and time.monotonic() >= deadline:
            break
        if asset["state"] in {"SKIPPED_EXISTING", "COMPLETED", "FAILED_EXHAUSTED", "REJECTED_NATIVE_GRID"}:
            continue
        for task in asset["tasks"].values():
            if task["state"] == "COMPLETED" and "objects" not in task:
                task["objects"] = pin_outputs(task)
                if not task["objects"]:
                    raise RuntimeError("Completed export has no pinned TIFF objects")
                save(state)
        if len(asset["tasks"]) == 2 and all(task["state"] == "COMPLETED" for task in asset["tasks"].values()):
            rgb = asset["tasks"]["rgb"]
            mask = asset["tasks"]["mask"]
            def suffixes(task):
                return {parse_gs(ref["uri"])[1][len(task["file_prefix"]):]: ref for ref in task["objects"]}
            rgb_objects, masks = suffixes(rgb), suffixes(mask)
            if set(rgb_objects) != set(masks):
                raise RuntimeError("RGB and original-mask exported tile grids differ")
            asset["source_objects"] = [dict(**ref, asset_id=asset_id,
                source_id="skysat_export_" + digest_bytes(asset_id.encode())[:20],
                catalog_id=asset["plan"]["catalog_id"], acquisition_utc=asset["plan"]["acquisition_utc"],
                producer_mask=masks[suffix], mask_provenance=asset["plan"]["mask_provenance"],
                original_mask_recipe=asset["plan"]["mask_recipe"], native_plan=asset["plan"],
                license="CC-BY-4.0", provider="Google Earth Engine")
                for suffix, ref in sorted(rgb_objects.items())]
            asset["state"] = "COMPLETED"
            save(state)
            continue
        if active >= 2:
            continue
        if "plan" not in asset:
            try:
                asset["plan"] = asset_plan(ee, asset_id)
            except ValueError as exc:
                asset.update(state="REJECTED_NATIVE_GRID", error_message=str(exc))
                save(state)
                continue
            save(state)
        images = None
        for kind in ("rgb", "mask"):
            if deadline is not None and time.monotonic() >= deadline:
                break
            previous = asset["tasks"].get(kind)
            if previous and previous["state"] in ACTIVE | {"COMPLETED"}:
                continue
            if previous and previous["attempt"] >= MAX_ATTEMPTS:
                asset["state"] = "FAILED_EXHAUSTED"
                save(state)
                break
            if active >= 2:
                break
            attempt = 1 if previous is None else previous["attempt"] + 1
            if images is None:
                images = dict(zip(("rgb", "mask"), export_images(ee, asset_id)))
            key = digest_bytes(asset_id.encode())[:24]
            description = f"maritime_{digest_bytes(prefix.encode())[:12]}_{key}_{kind}_{attempt}"
            file_prefix = f"{prefix}/assets/{key}/{kind}-attempt-{attempt}/{kind}"
            plan = asset["plan"]
            region = ee.Geometry.Rectangle(plan["native_bbox"], proj=plan["crs"], geodesic=False)
            export = ee.batch.Export.image.toCloudStorage(image=images[kind].clip(region),
                description=description, bucket=bucket, fileNamePrefix=file_prefix,
                region=region, crs=plan["crs"], crsTransform=plan["crs_transform"],
                maxPixels=plan["max_pixels"], fileDimensions=FILE_SIZE, shardSize=256,
                fileFormat="GeoTIFF", skipEmptyTiles=False,
                formatOptions={"cloudOptimized": False, "noData": 0})
            request_id = ee.data.newTaskId()[0]
            if previous:
                asset["history"].append(previous)
            task = dict(attempt=attempt, request_id=request_id, task_id=None,
                        description=description, file_prefix=file_prefix, state="SUBMITTING")
            asset["tasks"][kind] = task
            asset["state"] = "EXPORTING"
            save(state)  # Authoritative checkpoint exists before the start RPC.
            export._request_id = request_id  # Same pinned-ID mechanism as skysat_acquire.py.
            try:
                export.start()
            except Exception as exc:
                task["state"] = "START_UNCERTAIN"
                task["error_type"] = type(exc).__name__
                state["status"] = "blocked_permission" if permission_error(str(exc)) else "blocked_uncertain_start"
                save(state)
                raise RuntimeError("Earth Engine start uncertain or permission failure; reconcile before proceeding") from exc
            if not export.id:
                task["state"] = "START_UNCERTAIN"
                state["status"] = "blocked_uncertain_start"
                save(state)
                raise RuntimeError("Earth Engine start uncertain: missing task ID")
            task.update(task_id=export.id, state="READY")
            active += 1
            save(state)
    statuses = {asset["state"] for asset in state["assets"].values()}
    if statuses <= {"COMPLETED", "SKIPPED_EXISTING", "FAILED_EXHAUSTED", "REJECTED_NATIVE_GRID"}:
        state["status"] = "complete" if statuses <= {"COMPLETED", "SKIPPED_EXISTING"} else "complete_with_failures"
    save(state)
    return state


class CloudJournal:
    """Sequential create-only snapshots; local state is a readable mirror."""
    def __init__(self, client, output_prefix, scratch):
        self.client = client
        bucket, self.prefix = parse_gs(output_prefix.rstrip("/"))
        self.bucket = client.bucket(bucket)
        self.bucket_name = bucket
        self.scratch = Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)

    def load(self, expected):
        blobs = list(self.client.list_blobs(self.bucket_name, prefix=f"{self.prefix}/control/journal/"))
        if not blobs:
            return expected
        latest = max(blobs, key=lambda blob: blob.name)
        latest.reload()
        state = json.loads(latest.download_as_bytes(if_generation_match=int(latest.generation), checksum="crc32c"))
        if state["configuration_sha256"] != expected["configuration_sha256"]:
            raise ValueError("Scheduler checkpoint configuration conflict")
        return state

    def save(self, state):
        state["revision"] += 1
        data = json_bytes(state)
        create_only(self.bucket.blob(f"{self.prefix}/control/journal/{state['revision']:08d}.json"), data)
        temporary = self.scratch / "export-state.json.tmp"
        temporary.write_bytes(data)
        temporary.replace(self.scratch / "export-state.json")

    def pin_outputs(self, task):
        pins = []
        for blob in self.client.list_blobs(self.bucket_name, prefix=task["file_prefix"]):
            if not blob.name.endswith(".tif"):
                continue
            blob.reload()
            if not blob.generation or not blob.crc32c or not blob.size:
                raise ValueError("Export object lacks generation, size or CRC32C")
            pins.append(dict(uri=f"gs://{self.bucket_name}/{blob.name}",
                             generation=str(blob.generation), bytes=int(blob.size), crc32c=blob.crc32c))
        return sorted(pins, key=lambda ref: ref["uri"])


def run(args):
    require_cloud_vm()
    import ee
    import google.auth
    from google.cloud import storage
    if not 0 < args.time_limit_minutes <= 40 or not 1 <= args.poll_seconds <= 60:
        raise ValueError("Require time limit 0..40 minutes and poll interval 1..60 seconds")
    approval = json.loads(Path(args.approved_assets).read_text(encoding="utf-8-sig"))
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8-sig"))
    expected = initial_state(approval, sources, args.output_prefix, args.project)
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    ee.Initialize(credentials=credentials, project=args.project)
    ee.data.setDeadline(60000)
    journal = CloudJournal(storage.Client(project=args.project, credentials=credentials), args.output_prefix, args.scratch)
    state = journal.load(expected)
    journal.save(state)
    deadline = time.monotonic() + args.time_limit_minutes * 60
    while time.monotonic() < deadline:
        tick(ee, state, journal.save, journal.pin_outputs, deadline=deadline)
        print(json.dumps(dict(status=state["status"], revision=state["revision"],
                              asset_state_counts={key:sum(row['state']==key for row in state['assets'].values())
                                  for key in sorted({row['state'] for row in state['assets'].values()})})))
        if state["status"].startswith("complete"):
            return state
        time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
    state["status"] = "paused_poll_time_limit_remote_tasks_continue"
    journal.save(state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-assets", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--time-limit-minutes", type=float, default=40)
    parser.add_argument("--poll-seconds", type=float, default=30)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
