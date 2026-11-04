"""Bounded native RGB SkySat exports. No provisioning or billing operations.

Bounding-grid estimates are storage reservations, NEVER accepted training bytes.
Run with ``python -m scripts.skysat_acquire`` from the repository root.
"""

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import time

from src.marine_acquire import digest, write_json

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED"}
ACTIVE = {
    "READY",
    "RUNNING",
    "SUBMITTING",
    "UNSUBMITTED",
    "CANCEL_REQUESTED",
    "UNKNOWN",
}
RAW_CAP = 200_000_000_000
EXPORT_RECIPE = {
    "bands": ["R", "G", "B"],
    "dtype": "uint8",
    "unmask": 0,
    "noData": 0,
    "cloudOptimized": False,
}

GCLOUD = "gcloud"


def resolve_gcloud(explicit):
    result = explicit or shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if not result:
        raise ValueError("gcloud executable unavailable; provide --gcloud path")
    return result


def initialize(project, gcloud_auth):
    if not gcloud_auth:
        from src.export_ee import initialize as initialize_default

        return initialize_default(project)
    import ee
    from google.auth.credentials import Credentials

    class GcloudCredentials(Credentials):
        def refresh(self, request):
            self.token = subprocess.check_output(
                [GCLOUD, "auth", "print-access-token"], text=True, timeout=60
            ).strip()
            if not self.token:
                raise ValueError("Empty gcloud token")
            self.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
                minutes=40
            )

    credentials = GcloudCredentials()
    credentials.refresh(None)
    ee.Initialize(credentials=credentials, project=project)
    ee.data.setDeadline(60000)
    return ee


def bbox(geometry):
    points = []

    def visit(value):
        if isinstance(value[0], (float, int)):
            points.append(value)
        else:
            for item in value:
                visit(item)

    visit(geometry["coordinates"])
    return [
        min(p[0] for p in points),
        min(p[1] for p in points),
        max(p[0] for p in points),
        max(p[1] for p in points),
    ]


def build_manifest(
    assets, bucket, prefix, max_native_bytes=180_000_000_000, *, prior_manifests=()
):
    if not 0 < max_native_bytes <= 180_000_000_000:
        raise ValueError("Native reservation must be positive and at most 180 GB")
    if (
        not bucket
        or "/" in bucket
        or not prefix.strip("/")
        or ".." in prefix
        or prefix.startswith("/")
    ):
        raise ValueError("Dedicated bucket and nonempty relative prefix required")
    assets = sorted(assets, key=lambda a: a["id"])
    if len({a["id"] for a in assets}) != len(assets):
        raise ValueError("Duplicate asset IDs")
    boxes = [bbox(a["geometry"]) for a in assets]
    parent = list(range(len(assets)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(assets):
        for j in range(i):
            b = assets[j]
            overlap = not (
                boxes[i][2] < boxes[j][0]
                or boxes[j][2] < boxes[i][0]
                or boxes[i][3] < boxes[j][1]
                or boxes[j][3] < boxes[i][1]
            )
            catalog = a.get("properties", {}).get("catalogID")
            if overlap or (
                catalog and catalog == b.get("properties", {}).get("catalogID")
            ):
                parent[root(i)] = root(j)
    groups = {}
    for i, a in enumerate(assets):
        groups.setdefault(root(i), []).append(a["id"])
    if len(groups) < 3:
        raise ValueError("At least three geographically independent groups required")
    group_ids = {
        k: hashlib.sha256("|".join(v).encode()).hexdigest()[:16]
        for k, v in groups.items()
    }
    # Stable geographic components, including dates and duplicate catalog IDs, never split.
    ordered = sorted(group_ids.values())
    splits = {
        g: ("validation" if n == 0 else "test" if n == 1 else "train")
        for n, g in enumerate(ordered)
    }
    candidates = []
    for i, a in enumerate(assets):
        bands = {b["id"]: b for b in a["bands"]}
        if not all(b in bands for b in "RGB"):
            continue
        grid = bands["R"]["grid"]
        if any(bands[b]["grid"] != grid for b in "GB"):
            raise ValueError("RGB grids differ")
        transform = grid["affineTransform"]
        crs = grid.get("crsCode") or grid.get("crsWkt", "")
        metric_utm = crs.startswith(("EPSG:326", "EPSG:327")) or (
            crs.startswith('PROJCS["UTM Zone ') and 'UNIT["m", 1.0]' in crs
        )
        if (
            not metric_utm
            or not 0 < abs(transform["scaleX"]) <= 2.5
            or not 0 < abs(transform["scaleY"]) <= 2.5
        ):
            raise ValueError("Native metre UTM grid <=2.5 m required")
        estimated = grid["dimensions"]["width"] * grid["dimensions"]["height"] * 3
        group = group_ids[root(i)]
        candidates.append(
            {
                "asset_id": a["id"],
                "source_id": "skysat_"
                + hashlib.sha256(a["id"].encode()).hexdigest()[:20],
                "group_id": group,
                "split": splits[group],
                "geometry": a["geometry"],
                "grid": grid,
                "estimated_native_bytes": estimated,
                "cloud_percent": a.get("properties", {}).get(
                    "cloudcoverAreaPercentage", 100
                ),
            }
        )
    # Recompute components over the FULL inventory before excluding previous batches.
    prior_manifests = list(prior_manifests)
    by_asset = {row["asset_id"]: row for row in candidates}
    catalogs = {a["id"]: a.get("properties", {}).get("catalogID") for a in assets}
    excluded = set()
    excluded_catalogs = set()
    prior_refs = []
    parent_digests = {m.get("sha256") for m in prior_manifests}
    if len(parent_digests) != len(prior_manifests):
        raise ValueError("Duplicate prior manifests")
    for previous in prior_manifests:
        if digest({k: v for k, v in previous.items() if k != "sha256"}) != previous.get(
            "sha256"
        ):
            raise ValueError("Prior manifest checksum mismatch")
        if previous["bucket"] != bucket or previous["prefix"] == prefix.strip("/"):
            raise ValueError("Incremental exports require same bucket and a new prefix")
        if any(
            ref["sha256"] not in parent_digests
            for ref in previous.get("prior_manifests", [])
        ):
            raise ValueError(
                "Supply every ancestor manifest to avoid duplicate acquisition"
            )
        for row in previous["sources"]:
            current = by_asset.get(row["asset_id"])
            if current is None or any(
                row[key] != current[key]
                for key in ("source_id", "group_id", "split", "grid", "geometry")
            ):
                raise ValueError(
                    "Prior source identity or global geographic split changed"
                )
            excluded.add(row["asset_id"])
            if catalogs[row["asset_id"]]:
                excluded_catalogs.add(catalogs[row["asset_id"]])
        prior_refs.append(
            {
                "sha256": previous["sha256"],
                "bucket": previous["bucket"],
                "prefix": previous["prefix"],
            }
        )
    if prior_manifests:
        prior_splits = {
            row["split"] for previous in prior_manifests for row in previous["sources"]
        }
        if not {"validation", "test"} <= prior_splits:
            raise ValueError(
                "Incremental training batch requires existing held-out manifests"
            )
        candidates = [
            row
            for row in candidates
            if row["split"] == "train"
            and row["asset_id"] not in excluded
            and catalogs[row["asset_id"]] not in excluded_catalogs
        ]
    # Reserve one low-cloud source per held-out split, then prioritize low-cloud TRAIN.
    selected = []
    total = 0
    for split in ("train",) if prior_manifests else ("validation", "test", "train"):
        rows = sorted(
            (r for r in candidates if r["split"] == split),
            key=lambda r: (r["cloud_percent"], r["asset_id"]),
        )
        for row in rows:
            if total + row["estimated_native_bytes"] <= max_native_bytes:
                selected.append(row)
                total += row["estimated_native_bytes"]
                if split != "train":
                    break
    required_splits = {"train"} if prior_manifests else {"train", "validation", "test"}
    if {r["split"] for r in selected} != required_splits:
        raise ValueError(
            "Budget cannot accommodate independent train and held-out sources"
        )
    manifest = {
        "schema": 2,
        "export_recipe": {**EXPORT_RECIPE, "bands": list(EXPORT_RECIPE["bands"])},
        "validity_limitation": "Nodata zero conservatively excludes otherwise valid pixels containing any zero RGB channel during preparation. Valid RGB values are not shifted or rescaled.",
        "bucket": bucket,
        "prefix": prefix.strip("/"),
        "storage_audit_prefix": prefix.split("/", 1)[0],
        "sources": selected,
        "estimated_native_bytes": total,
        "raw_storage_cap_bytes": RAW_CAP,
        "counting": "Native RGB uint8 bounding-grid reservation only; masked area, duplicates, heldouts and unused data do not count toward training requirement.",
    }
    if prior_manifests:
        manifest["prior_manifests"] = sorted(prior_refs, key=lambda ref: ref["sha256"])
    manifest["sha256"] = digest(manifest)
    return manifest


def export_arguments(manifest, row):
    if manifest.get("schema") != 2 or manifest.get("export_recipe") != EXPORT_RECIPE:
        raise ValueError(
            "New exports require a version 2 nodata manifest and separate state/prefix"
        )
    t = row["grid"]["affineTransform"]
    return dict(
        description=row["source_id"],
        bucket=manifest["bucket"],
        fileNamePrefix=f"{manifest['prefix']}/{row['source_id']}/rgb",
        region=row["geometry"],
        crs=row["grid"].get("crsCode") or row["grid"]["crsWkt"],
        crsTransform=[
            t["scaleX"],
            t.get("shearX", 0),
            t["translateX"],
            t.get("shearY", 0),
            t["scaleY"],
            t["translateY"],
        ],
        maxPixels=10**13,
        fileFormat="GeoTIFF",
        fileDimensions=8192,
        skipEmptyTiles=True,
        formatOptions={"cloudOptimized": False, "noData": 0},
    )


def utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Absolute timezone-aware deadline required")
    return result


def load_state(path, manifest, deadline):
    utc(deadline)
    content = {k: v for k, v in manifest.items() if k != "sha256"}
    if digest(content) != manifest["sha256"]:
        raise ValueError("Manifest checksum mismatch")
    path = Path(path)
    if path.exists():
        state = json.loads(path.read_text())
        if (
            state["manifest_sha256"] != manifest["sha256"]
            or state["deadline"] != deadline
        ):
            raise ValueError("Inventory or absolute deadline changed on resume")
    else:
        state = {
            "manifest_sha256": manifest["sha256"],
            "deadline": deadline,
            "tasks": {},
        }
        write_json(path, state)
    return state


def tick(ee, manifest, path, deadline, raw_bytes, now=None, *, max_pending=2):
    if type(max_pending) is not int or not 2 <= max_pending <= 6:
        raise ValueError("max_pending must be an integer from 2 to 6")
    state = load_state(path, manifest, deadline)
    now = now or datetime.now(timezone.utc).isoformat()
    pending = [
        task
        for task in state["tasks"].values()
        if task.get("task_id") and task["state"] not in TERMINAL
    ]
    if pending:
        ids = [task["task_id"] for task in pending]
        statuses = ee.data.getTaskStatus(ids)
        by_id = {status.get("id"): status for status in statuses}
        if (
            len(statuses) != len(ids)
            or len(by_id) != len(ids)
            or set(by_id) != set(ids)
        ):
            raise ValueError("Incomplete or mismatched task status response")
        for task in pending:
            status = by_id[task["task_id"]]
            task.update(state=status["state"], error=status.get("error_message"))
    state.update(updated_at=now, raw_bytes=raw_bytes)
    stop = utc(now) >= utc(deadline) or raw_bytes >= RAW_CAP
    if stop:
        for task in state["tasks"].values():
            if task["state"] not in TERMINAL and (
                task.get("task_id") or task.get("request_id")
            ):
                ee.data.cancelTask(task.get("task_id") or task["request_id"])
                task["state"] = "CANCEL_REQUESTED"
        state["status"] = (
            "stopped_deadline" if utc(now) >= utc(deadline) else "stopped_storage_cap"
        )
        write_json(path, state)
        return state
    live = sum(t.get("state") in ACTIVE for t in ee.data.getTaskList())
    unknown = sum(
        t["state"] not in TERMINAL and not t.get("task_id")
        for t in state["tasks"].values()
    )
    slots = max(0, max_pending - live - unknown)
    for row in manifest["sources"]:
        if slots <= 0:
            break
        key = row["source_id"]
        if key in state["tasks"]:
            continue
        # Whole-manifest 180GB reservation leaves 20GB for TIFF overhead and bookkeeping.
        arguments = export_arguments(manifest, row)
        arguments["region"] = ee.Geometry(row["geometry"])
        task = ee.batch.Export.image.toCloudStorage(
            image=ee.Image(row["asset_id"]).select(["R", "G", "B"]).toUint8().unmask(0),
            **arguments,
        )
        request_id = ee.data.newTaskId()[0]
        task._request_id = request_id
        state["tasks"][key] = {
            "request_id": request_id,
            "task_id": None,
            "state": "SUBMITTING",
        }
        write_json(
            path, state
        )  # Crash/uncertain submission blocks retries instead of duplicating.
        task.start()
        if not task.id:
            raise ValueError("EE did not return operation ID")
        state["tasks"][key].update(task_id=task.id, state="READY")
        write_json(path, state)
        slots -= 1
    state["status"] = (
        "complete"
        if len(state["tasks"]) == len(manifest["sources"])
        and all(t["state"] in TERMINAL for t in state["tasks"].values())
        else "exporting"
    )
    write_json(path, state)
    return state


def storage_bytes(bucket, prefix):
    result = subprocess.run(
        [GCLOUD, "storage", "ls", f"gs://{bucket}/{prefix}/**", "--json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        if "matched no objects" in result.stderr or "No URLs matched" in result.stderr:
            return 0
        raise RuntimeError(result.stderr)
    objects = json.loads(result.stdout)
    total = 0
    for obj in objects:
        size = obj.get("size", obj.get("metadata", {}).get("size"))
        if size is None or int(size) < 0:
            raise ValueError("Unrecognized GCS object size; refusing further exports")
        total += int(size)
    return total


def main():
    global GCLOUD
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path)
    p.add_argument(
        "--prior-manifest",
        type=Path,
        action="append",
        default=[],
        help="Existing batch manifest; repeat for all ancestors",
    )
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--bucket")
    p.add_argument("--prefix", default="skysat/raw")
    p.add_argument("--max-native-bytes", type=int, default=180_000_000_000)
    p.add_argument("--state", type=Path)
    p.add_argument("--deadline")
    p.add_argument("--project")
    p.add_argument("--gcloud-auth", action="store_true")
    p.add_argument("--submit", action="store_true")
    p.add_argument(
        "--gcloud", help="Explicit gcloud executable, including Windows .cmd"
    )
    p.add_argument(
        "--max-pending",
        type=int,
        choices=range(2, 7),
        default=2,
        help="Total pending EE tasks (READY plus RUNNING), not execution quota",
    )
    p.add_argument("--once", action="store_true")
    p.add_argument("--sync-uri")
    args = p.parse_args()
    if not args.manifest.exists():
        if not args.inventory or not args.bucket:
            p.error("New manifest requires inventory and bucket")
        assets = json.loads(args.inventory.read_text())
        if isinstance(assets, dict):
            assets = assets["assets"]
        write_json(
            args.manifest,
            build_manifest(
                assets,
                args.bucket,
                args.prefix,
                args.max_native_bytes,
                prior_manifests=[
                    json.loads(path.read_text()) for path in args.prior_manifest
                ],
            ),
            exclusive=True,
        )
    manifest = json.loads(args.manifest.read_text())
    if not args.submit:
        print(
            json.dumps({k: v for k, v in manifest.items() if k != "sources"}, indent=2)
        )
        return
    if not args.state or not args.deadline or not args.project:
        p.error("Submission requires state, deadline and project")
    GCLOUD = resolve_gcloud(args.gcloud)
    ee = initialize(args.project, args.gcloud_auth)
    while True:
        state = tick(
            ee,
            manifest,
            args.state,
            args.deadline,
            storage_bytes(
                manifest["bucket"],
                manifest.get("storage_audit_prefix", manifest["prefix"]),
            ),
            max_pending=args.max_pending,
        )
        if args.sync_uri:
            subprocess.run(
                [GCLOUD, "storage", "cp", str(args.state), args.sync_uri],
                check=True,
                timeout=120,
            )
        print(
            json.dumps(
                {
                    "status": state["status"],
                    "tasks": len(state["tasks"]),
                    "raw_bytes": state["raw_bytes"],
                }
            ),
            flush=True,
        )
        if args.once or state["status"] != "exporting":
            break
        time.sleep(
            min(
                30,
                max(
                    0, (utc(args.deadline) - datetime.now(timezone.utc)).total_seconds()
                ),
            )
        )


if __name__ == "__main__":
    main()
