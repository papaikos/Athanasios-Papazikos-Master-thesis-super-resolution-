"""Bounded Earth Engine pilot: dry-run, access-check, inventory, download, prepare.

No EE imports/authentication in dry-run or prepare. Authentication must already
exist; this utility never enables billing or provisions compute. Only the explicit
export-gcs command submits bounded exports into the dedicated private bucket.
Run from repository root: python -m src.export_ee --help.
"""

import argparse
from datetime import datetime, timezone, timedelta
import json
import os
import re
from pathlib import Path
from urllib.request import urlopen
import numpy as np
from .data import (
    BANDS,
    S2_BANDS,
    aligned_grids,
    validate_sites,
    stable_land,
    fit_calibration,
    fit_acquisition_group,
    calibration_group_assignments,
    apply_calibration,
    save_pair,
    sha256,
)
from .cloud_config import GCS_PILOT_BUCKET


def configuration(path, limit=None):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    validate_sites(config["sites"])
    count = config["max_patches"] if limit is None else limit
    if not 1 <= count <= min(100, len(config["sites"])):
        raise ValueError("Pilot limit must be 1..100 and at most one patch per site")
    config["sites"] = config["sites"][:count]
    return config


def initialize(project, gcloud_auth=False):
    if not project:
        raise ValueError("--project is required for authenticated operations")
    import ee

    if gcloud_auth:
        import subprocess
        from google.oauth2.credentials import Credentials

        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"], text=True
        ).strip()
        ee.Initialize(credentials=Credentials(token), project=project)
    else:
        ee.Initialize(project=project)
    return ee


def candidate_centers(x, y):
    # Ordered by minimal shift; all stay within 320 m of the named harbor point.
    return [
        (x + dx, y + dy, dx, dy)
        for dx, dy in ((0, 0), (320, 0), (-320, 0), (0, 320), (0, -320))
    ]


def candidates(ee, config):
    """Single-scene selection only; no temporal or spatial mosaics."""
    for site in config["sites"]:
        point = ee.Geometry.Point([site["lon"], site["lat"]])
        collection = (
            ee.ImageCollection("USDA/NAIP/DOQQ")
            .filterBounds(point)
            .filterDate(config["naip_start"], config["naip_end"])
            .map(lambda im: im.set("has_nir", im.bandNames().contains("N")))
            .filter(ee.Filter.eq("has_nir", True))
            .sort("system:time_start", False)
            .limit(config["max_candidates_per_site"])
        )
        naip_ids = collection.aggregate_array("system:id").getInfo()
        found = False
        for naip_id in naip_ids:
            naip = ee.Image(naip_id)
            stamp = naip.get("system:time_start").getInfo()
            acquired = datetime.fromtimestamp(stamp / 1000, timezone.utc)
            gap = timedelta(days=config["max_date_gap_days"])
            s2s = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(point)
                .filterDate((acquired - gap).isoformat(), (acquired + gap).isoformat())
                .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 20))
                .map(
                    lambda im: im.set(
                        "date_gap",
                        ee.Number(im.get("system:time_start")).subtract(stamp).abs(),
                    )
                )
                .sort("date_gap")
                .limit(1)
            )
            ids = s2s.aggregate_array("system:id").getInfo()
            if not ids:
                continue
            s2 = ee.Image(ids[0])
            projection = s2.select("B4").projection().getInfo()
            x, y = point.transform(projection["crs"], 1).coordinates().getInfo()
            chosen = None
            coverage = (
                naip.select(list(BANDS))
                .mask()
                .reduce(ee.Reducer.min())
                .unmask(0, False)
            )
            for cx, cy, dx, dy in candidate_centers(x, y):
                grids = aligned_grids(projection["transform"], cx, cy)
                t = grids["lr"]
                region = ee.Geometry.Rectangle(
                    [t[2], t[5] - 1280, t[2] + 1280, t[5]],
                    proj=projection["crs"],
                    geodesic=False,
                )
                if not naip.geometry().contains(region, 1).getInfo():
                    continue
                coarse = coverage.reduceResolution(
                    reducer=ee.Reducer.min(), bestEffort=False, maxPixels=4096
                ).reproject(crs=projection["crs"], crsTransform=t)
                stats = coarse.reduceRegion(
                    reducer=ee.Reducer.min(),
                    geometry=region,
                    crs=projection["crs"],
                    crsTransform=t,
                    maxPixels=16384,
                ).getInfo()
                if not stats or any(
                    value is None or value < 0.999999 for value in stats.values()
                ):
                    continue
                chosen = (grids, cx, cy, dx, dy)
                break
            if chosen is None:
                continue
            grids, cx, cy, dx, dy = chosen
            yield dict(
                site,
                naip_id=naip_id,
                s2_id=ids[0],
                naip_time_ms=stamp,
                s2_time_ms=s2.get("system:time_start").getInfo(),
                naip_projection=naip.select("R").projection().getInfo(),
                s2_projection=projection,
                crs=projection["crs"],
                grids=grids,
                selected_center_projected=[cx, cy],
                center_shift_m=[dx, dy],
                coverage_preflight="full_footprint_and_min_source_mask_10m",
            )
            found = True
            break
        if not found:
            yield dict(site, rejected="No bounded RGBN NAIP/near-date S2 candidate")


def download_image(image, path, crs, transform, size):
    url = image.getDownloadURL(
        {
            "crs": crs,
            "crs_transform": transform,
            "dimensions": [size, size],
            "format": "GEO_TIFF",
        }
    )
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(url, timeout=180) as response, partial.open("wb") as output:
            size_bytes = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > 32 * 1024 * 1024:
                    raise ValueError("Download exceeds bounded 32 MiB limit")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return sha256(path)


def checkpoint_manifest(output, manifest):
    partial = output / "manifest.json.part"
    with partial.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(output / "manifest.json")


def image_specs(ee, record):
    naip = ee.Image(record["naip_id"]).select(list(BANDS))
    s2 = ee.Image(record["s2_id"])
    # Fractional valid coverage is exported explicitly. unmask(sameFootprint=False)
    # makes off-scene pixels invalid instead of silently renormalizing coverage.
    coverage = naip.mask().reduce(ee.Reducer.min()).unmask(0, False)
    mean = naip.reduceResolution(
        reducer=ee.Reducer.mean(), bestEffort=False, maxPixels=4096
    )
    coverage_mean = coverage.reduceResolution(
        reducer=ee.Reducer.mean(), bestEffort=False, maxPixels=4096
    )
    s2_rgbn = s2.select(list(S2_BANDS)).multiply(0.0001)
    s2_valid = (
        s2.select(list(S2_BANDS)).mask().reduce(ee.Reducer.min()).unmask(0, False)
    )
    return [
        (
            "naip_hr",
            mean.addBands(coverage_mean.rename("valid")).toFloat(),
            "hr",
            512,
        ),
        (
            "naip_lr",
            mean.addBands(coverage_mean.rename("valid")).toFloat(),
            "lr",
            128,
        ),
        (
            "s2_lr",
            s2_rgbn.addBands(s2.select("SCL"))
            .addBands(s2_valid.rename("valid"))
            .toFloat(),
            "lr",
            128,
        ),
    ]


def download(ee, records, output):
    output.mkdir(parents=True, exist_ok=True)
    validate_sites(records)
    manifest = {
        "schema_version": 1,
        "bands": list(BANDS),
        "s2_bands": list(S2_BANDS),
        "source": "Google Earth Engine",
        "status": "partial",
        "records": records,
    }
    checkpoint_manifest(output, manifest)
    for record in records:
        if "rejected" in record:
            continue
        specs = image_specs(ee, record)
        record.setdefault("files", {})
        for name, image, grid, size in specs:
            path = output / (record["id"] + "_" + name + ".tif")
            if name in record["files"]:
                item = record["files"][name]
                if (
                    item["path"] != path.name
                    or not path.exists()
                    or sha256(path) != item["sha256"]
                ):
                    raise ValueError("Resume checksum/path mismatch: " + str(path))
                print(f"Verified existing {record['id']} {name}", flush=True)
                continue
            print(f"Downloading {record['id']} {name} ({size}x{size})", flush=True)
            try:
                checksum = download_image(
                    image, path, record["crs"], record["grids"][grid], size
                )
            except Exception as error:
                # Persist exception type only: HTTP error text can contain signed URLs.
                record["download_error"] = {"file": name, "type": type(error).__name__}
                checkpoint_manifest(output, manifest)
                raise
            record["files"][name] = {"path": path.name, "sha256": checksum}
            record.pop("download_error", None)
            checkpoint_manifest(output, manifest)
            print(f"Saved {path.name} sha256={checksum}", flush=True)
    manifest["status"] = "complete"
    checkpoint_manifest(output, manifest)


GCS_BUCKET = GCS_PILOT_BUCKET
GCS_PREFIX = "pilot/raw"


def gcs_export_parameters(record, name, grid, size):
    transform = record["grids"][grid]
    a, b, x, d, e, y = transform
    if b != 0 or d != 0 or size not in (128, 512):
        raise ValueError("Export requires bounded north-up pilot grids")
    inset_x, inset_y = abs(a) * 0.01, abs(e) * 0.01
    bounds = [x + inset_x, y + e * size + inset_y, x + a * size - inset_x, y - inset_y]
    return {
        "description": record["id"] + "_" + name,
        "bucket": GCS_BUCKET,
        "fileNamePrefix": GCS_PREFIX + "/" + record["id"] + "_" + name,
        "crs": record["crs"],
        "crsTransform": transform,
        "maxPixels": size * size,
        "shardSize": 128,
        "fileDimensions": size,
        "fileFormat": "GeoTIFF",
        "formatOptions": {"cloudOptimized": True},
    }, bounds


def export_gcs(ee, records, output):
    if not 1 <= len(records) <= 6:
        raise ValueError("GCS pilot requires at most six sites / eighteen exports")
    validate_sites(records)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(
            "Task manifest exists; use export-gcs-status, never resubmit blindly"
        )
    manifest = {
        "schema_version": 1,
        "source": "Google Earth Engine to GCS",
        "status": "export_pending",
        "bucket": GCS_BUCKET,
        "prefix": GCS_PREFIX,
        "bands": list(BANDS),
        "s2_bands": list(S2_BANDS),
        "records": records,
        "tasks": [],
    }
    checkpoint_manifest(output, manifest)
    for record in records:
        if "rejected" in record:
            continue
        for name, image, grid, size in image_specs(ee, record):
            params, bounds = gcs_export_parameters(record, name, grid, size)
            region = ee.Geometry.Rectangle(bounds, proj=record["crs"], geodesic=False)
            task = ee.batch.Export.image.toCloudStorage(
                image=image, region=region, **params
            )
            item = {
                "id": task.id,
                "site_id": record["id"],
                "name": name,
                "grid": grid,
                "size": size,
                "state": "SUBMITTING",
                "parameters": params,
                "bounds": bounds,
                "gcs_uri": "gs://"
                + GCS_BUCKET
                + "/"
                + params["fileNamePrefix"]
                + ".tif",
            }
            manifest["tasks"].append(item)
            checkpoint_manifest(output, manifest)
            print(f"Submitting {record['id']} {name} task={task.id}", flush=True)
            try:
                task.start()
            except Exception:
                item["state"] = "SUBMISSION_UNCERTAIN"
                checkpoint_manifest(output, manifest)
                raise
            item["id"] = task.id
            item["state"] = "SUBMITTED"
            checkpoint_manifest(output, manifest)
    manifest["submission_complete"] = True
    checkpoint_manifest(output, manifest)
    return manifest


def export_gcs_status(ee, manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    if not 0 < len(tasks) <= 18 or manifest["bucket"] != GCS_BUCKET:
        raise ValueError("Not a bounded pilot task manifest")
    if any(not task.get("id") for task in tasks):
        raise RuntimeError(
            "Submission needs reconciliation with ee.batch.Task.list(); do not resubmit"
        )
    statuses = {
        item["id"]: item
        for item in ee.data.getTaskStatus([task["id"] for task in tasks])
    }
    for task in tasks:
        status = statuses[task["id"]]
        task["state"] = status["state"]
        if "error_message" in status:
            task["error_message"] = status["error_message"]
    if manifest.get("submission_complete") and all(
        task["state"] == "COMPLETED" for task in tasks
    ):
        manifest["status"] = "export_complete"
    elif any(task["state"] in ("FAILED", "CANCELLED") for task in tasks):
        manifest["status"] = "export_failed"
    checkpoint_manifest(manifest_path.parent, manifest)
    return [
        {
            "id": task["id"],
            "site_id": task["site_id"],
            "name": task["name"],
            "state": task["state"],
        }
        for task in tasks
    ]


def export_gcs_retry(ee, manifest_path):
    """Retry only fresh Hueneme failures from the observed one-pixel boundary error."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest["tasks"]
    if manifest.get("bucket") != GCS_BUCKET or not 0 < len(tasks) <= 18:
        raise ValueError("Not a bounded pilot manifest")
    if any(not task.get("id") for task in tasks):
        raise RuntimeError(
            "Submission needs reconciliation; do not retry uncertain tasks"
        )
    fresh = {
        status["id"]: status
        for status in ee.data.getTaskStatus([task["id"] for task in tasks])
    }
    eligible = []
    for item in tasks:
        status = fresh[item["id"]]
        message = status.get("error_message", "")
        match = re.search(
            r"specified\s+(\d+)\s+pixels.*?max\s*:?\s*(\d+)", message, re.IGNORECASE
        )
        size = item["size"]
        if (
            item["site_id"] == "port_hueneme"
            and status["state"] == "FAILED"
            and match
            and int(match[1]) == (size + 1) ** 2
            and int(match[2]) == size**2
            and item["name"] in ("naip_hr", "naip_lr", "s2_lr")
            and size in (128, 512)
        ):
            eligible.append((item, status))
    if not eligible or len(eligible) > 3:
        raise ValueError("No eligible bounded Hueneme rounding failures to retry")
    by_id = {record["id"]: record for record in manifest["records"]}
    for item, status in eligible:
        record = by_id[item["site_id"]]
        specs = {
            name: (image, grid, size)
            for name, image, grid, size in image_specs(ee, record)
        }
        image, grid, size = specs[item["name"]]
        params, bounds = gcs_export_parameters(record, item["name"], grid, size)
        region = ee.Geometry.Rectangle(bounds, proj=record["crs"], geodesic=False)
        task = ee.batch.Export.image.toCloudStorage(
            image=image, region=region, **params
        )
        previous = {key: value for key, value in item.items() if key != "attempts"}
        previous.update(state=status["state"], error_message=status["error_message"])
        item.setdefault("attempts", []).append(previous)
        item.update(id=None, state="SUBMITTING", parameters=params, bounds=bounds)
        item.pop("error_message", None)
        manifest["status"] = "export_pending"
        checkpoint_manifest(manifest_path.parent, manifest)
        print(
            f"Retrying verified failed {record['id']} {item['name']} previous={previous['id']}",
            flush=True,
        )
        try:
            task.start()
        except Exception:
            item["state"] = "SUBMISSION_UNCERTAIN"
            checkpoint_manifest(manifest_path.parent, manifest)
            raise
        item.update(id=task.id, state="SUBMITTED")
        checkpoint_manifest(manifest_path.parent, manifest)
    return [
        {"site_id": item["site_id"], "name": item["name"], "id": item["id"]}
        for item, _ in eligible
    ]


def collect_gcs(manifest_path, output):
    """Validate TIFFs copied from GCS inside Cloud Shell; no network/copy here."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "export_complete":
        raise ValueError("Refresh export-gcs-status; all exports must be complete")
    if (output / "manifest.json").exists():
        raise FileExistsError(output / "manifest.json")
    records = manifest["records"]
    by_id = {record["id"]: record for record in records}
    for task in manifest["tasks"]:
        record = by_id[task["site_id"]]
        path = output / (task["site_id"] + "_" + task["name"] + ".tif")
        item = {
            "path": path.name,
            "sha256": sha256(path),
            "gcs_uri": task["gcs_uri"],
            "ee_task_id": task["id"],
        }
        read_raster(
            output,
            item,
            record,
            task["grid"],
            task["size"],
            6 if task["name"] == "s2_lr" else 5,
        )
        record.setdefault("files", {})[task["name"]] = item
    manifest["status"] = "complete"
    manifest["export_manifest_sha256"] = sha256(manifest_path)
    checkpoint_manifest(output, manifest)
    return str(output / "manifest.json")


def read_raster(root, item, record, grid, size, bands):
    import rasterio

    path = root / item["path"]
    if sha256(path) != item["sha256"]:
        raise ValueError("Source checksum mismatch: " + str(path))
    with rasterio.open(path) as dataset:
        if dataset.shape != (size, size) or dataset.count != bands:
            raise ValueError("Unexpected raster shape/band contract")
        expected = record["grids"][grid]
        actual = list(dataset.transform)[:6]
        if (
            not np.allclose(actual, expected, rtol=0, atol=1e-6)
            or dataset.crs.to_string() != record["crs"]
        ):
            raise ValueError("Raster grid does not match source manifest")
        array = dataset.read()
        mask = dataset.read_masks().min(axis=0) > 0
        return array, mask


def prepare(manifest_path, output, calibration_mode="pooled"):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status", "complete") != "complete":
        raise ValueError("Pilot download is incomplete; resume download before prepare")
    records = [r for r in manifest["records"] if "rejected" not in r]
    validate_sites(records)
    if calibration_mode not in ("pooled", "acquisition-groups"):
        raise ValueError("Unknown calibration mode")
    assignments = (
        calibration_group_assignments(records)
        if calibration_mode == "acquisition-groups"
        else None
    )
    root = manifest_path.parent
    training_dn, training_sr, training_support, train_sites = [], [], [], []
    loaded = []
    for record in records:
        files = record["files"]
        hr, hm = read_raster(root, files["naip_hr"], record, "hr", 512, 5)
        dn, dm = read_raster(root, files["naip_lr"], record, "lr", 128, 5)
        sr, sm = read_raster(root, files["s2_lr"], record, "lr", 128, 6)
        valid = hm & (hr[4] >= 0.999999) & np.isfinite(hr).all(axis=0)
        support = stable_land(
            sr[4], dm & sm & (dn[4] >= 0.999999) & (sr[5] >= 0.999999)
        )
        loaded.append((record, hr[:4], valid, dn[:4], sr[:4], support))
        if record["split"] == "train":
            training_dn.append(dn[:4])
            training_sr.append(sr[:4])
            training_support.append(support)
            train_sites.append(record["id"])
    if not training_dn:
        raise ValueError("No training scenes to fit calibration")
    if assignments is None:
        calibration = fit_calibration(
            np.concatenate(training_dn, axis=2),
            np.concatenate(training_sr, axis=2),
            np.concatenate(training_support, axis=1),
            split="train",
            site_ids=train_sites,
        )
    else:
        groups = {}
        rejected_groups = {}
        for acquisition in sorted(
            {r["naip_id"] for r in records if r["split"] == "train"}
        ):
            members = [
                item
                for item in loaded
                if item[0]["split"] == "train" and item[0]["naip_id"] == acquisition
            ]
            member_sites = [item[0]["id"] for item in members]
            try:
                groups[acquisition] = fit_acquisition_group(
                    np.concatenate([item[3] for item in members], axis=2),
                    np.concatenate([item[4] for item in members], axis=2),
                    np.concatenate([item[5] for item in members], axis=1),
                    site_ids=member_sites,
                )
            except ValueError as error:
                # A rejected acquisition must not invalidate unrelated groups.
                # Its preassigned evaluation sites remain rejected, never rerouted.
                rejected_groups[acquisition] = {
                    "train_sites": member_sites,
                    "reason": str(error),
                    "diagnostics": getattr(error, "diagnostics", None),
                }
        calibration = {
            "method": "training_acquisition_groups",
            "train_sites": sorted(train_sites),
            "assignment_policy": "nearest_training_site_haversine_id_tiebreak_before_evaluation_pixels",
            "assignments": assignments,
            "groups": groups,
            "rejected_groups": rejected_groups,
        }
    output.mkdir(parents=True, exist_ok=True)
    with (output / "calibration.json").open("x", encoding="utf-8") as stream:
        json.dump(calibration, stream, indent=2)
    outcomes = []
    for record, hr, valid, dn, sr, support in loaded:
        if (
            assignments is not None
            and assignments[record["id"]] in calibration["rejected_groups"]
        ):
            group = assignments[record["id"]]
            outcomes.append(
                {
                    "site_id": record["id"],
                    "split": record["split"],
                    "calibration_group": group,
                    "rejected": "Assigned training acquisition calibration failed",
                    "reason": calibration["rejected_groups"][group]["reason"],
                }
            )
            continue
        mapping = (
            calibration
            if assignments is None
            else calibration["groups"][assignments[record["id"]]]
        )
        # Frozen calibration is evaluated on held-out land but never refitted.
        residual = apply_calibration(dn, mapping) - sr
        stable = support & np.isfinite(residual).all(axis=0)
        good = stable & (np.abs(residual).max(axis=0) <= 0.08)
        if (
            not valid.all()
            or stable.sum() < 512
            or good.sum() < 512
            or good.sum() / max(1, stable.sum()) < 0.7
        ):
            outcomes.append(
                {
                    "site_id": record["id"],
                    "split": record["split"],
                    "rejected": "Coverage or frozen stable-land calibration gate failed",
                }
            )
            continue
        provenance = dict(
            record,
            site_id=record["id"],
            calibration_train_sites=mapping["train_sites"],
            calibration=mapping,
            calibration_mode=calibration_mode,
            calibration_group=assignments[record["id"]]
            if assignments is not None
            else "pooled",
            source_manifest_sha256=sha256(manifest_path),
            calibration_land_pixels=int(good.sum()),
        )
        path = output / (record["id"] + ".npz")
        checksum = save_pair(path, apply_calibration(hr, mapping), valid, provenance)
        outcomes.append(
            {
                "site_id": record["id"],
                "split": record["split"],
                "path": path.name,
                "sha256": checksum,
            }
        )
    with (output / "pairs.json").open("x", encoding="utf-8") as stream:
        json.dump(outcomes, stream, indent=2)
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "dry-run",
            "access-check",
            "inventory",
            "download",
            "prepare",
            "export-gcs",
            "export-gcs-status",
            "export-gcs-retry",
            "collect-gcs",
        ],
    )
    parser.add_argument("--config", type=Path, default=Path("configs/pilot_sites.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--project")
    parser.add_argument(
        "--gcloud-auth",
        action="store_true",
        help="Use existing gcloud OAuth access token; never print token",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--calibration-mode", choices=("pooled", "acquisition-groups"), default="pooled"
    )
    args = parser.parse_args()
    if args.command == "collect-gcs":
        if not args.manifest or not args.output:
            parser.error(
                "collect-gcs requires --manifest and --output containing copied TIFFs"
            )
        print(collect_gcs(args.manifest, args.output))
        return
    if args.command == "export-gcs-retry":
        if not args.manifest:
            parser.error("export-gcs-retry requires --manifest")
        print(
            json.dumps(
                export_gcs_retry(
                    initialize(args.project, args.gcloud_auth), args.manifest
                ),
                indent=2,
            )
        )
        return
    if args.command == "export-gcs-status":
        if not args.manifest:
            parser.error("export-gcs-status requires --manifest")
        print(
            json.dumps(
                export_gcs_status(
                    initialize(args.project, args.gcloud_auth), args.manifest
                ),
                indent=2,
            )
        )
        return
    if args.command == "prepare":
        if not args.manifest or not args.output:
            parser.error("prepare requires --manifest and --output")
        print(
            json.dumps(
                prepare(args.manifest, args.output, args.calibration_mode), indent=2
            )
        )
        return
    config = configuration(args.config, args.limit)
    if args.command == "dry-run":
        print(
            json.dumps(
                {
                    "network": False,
                    "contract": "RGBN 512@2.5m -> 128@10m",
                    "config": config,
                },
                indent=2,
            )
        )
        return
    if args.command in ("download", "export-gcs") and not args.output:
        parser.error("download requires an explicit --output directory")
    ee = initialize(args.project, args.gcloud_auth)
    if args.command == "access-check":
        print(json.dumps({"project": args.project, "ee_probe": ee.Number(1).getInfo()}))
        return
    manifest_path = args.output / "manifest.json" if args.output else None
    if args.command == "download" and manifest_path.exists():
        records = json.loads(manifest_path.read_text(encoding="utf-8"))["records"]
        configured = {site["id"]: site for site in config["sites"]}
        if {record["id"] for record in records} != set(configured):
            raise ValueError(
                "Resume manifest sites differ from configured bounded sites"
            )
        for record in records:
            site = configured[record["id"]]
            if any(record[key] != site[key] for key in ("split", "lon", "lat")):
                raise ValueError(
                    "Resume manifest geographic assignment differs from config"
                )
        print(f"Resuming {manifest_path}", flush=True)
    else:
        records = list(candidates(ee, config))
    if args.command == "inventory":
        print(json.dumps(records, indent=2))
        return
    if args.command == "export-gcs":
        export_gcs(ee, records, args.output)
        return
    download(ee, records, args.output)
    print(str(args.output / "manifest.json"))


if __name__ == "__main__":
    main()
