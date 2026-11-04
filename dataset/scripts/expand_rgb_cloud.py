"""Cloud-only native SkySat RGB screening, reviewed acquisition and source QA.

Run on a Google Compute Engine VM with application-default credentials. Screen
never exports training sources. Acquisition accepts only an explicitly reviewed
schema-2 manifest. No HR/LR pairs or training are produced.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import tempfile
import time
from urllib.request import Request, urlopen

from scripts import marine_skysat_intake as intake
from scripts import skysat_acquire as acquisition
from src.marine_acquire import digest, write_json


def count_valid_rgb(rgb, masks):
    import numpy as np

    if rgb.dtype != np.uint8 or rgb.shape[0] != 3 or masks.shape != rgb.shape:
        raise ValueError("Expected three uint8 bands and matching masks")
    return int(np.count_nonzero(np.all(rgb > 0, axis=0) & np.all(masks > 0, axis=0)))


def unique_accounting(objects, baseline_fingerprints, baseline_bytes):
    seen = set()
    categories = {}
    for obj in objects:
        duplicate = (int(obj["size"]), obj["crc32c"]) in baseline_fingerprints
        duplicate = duplicate or obj["sha256"] in seen
        seen.add(obj["sha256"])
        obj["duplicate"] = duplicate
        obj["useful_bytes"] = 0 if duplicate else 3 * int(obj["valid_pixels"])
        if not duplicate:
            category = obj["category"]
            categories[category] = categories.get(category, 0) + obj["useful_bytes"]
    added = sum(categories.values())
    return dict(
        added_useful_bytes=added,
        total_useful_bytes=baseline_bytes + added,
        useful_bytes_by_category=categories,
    )


def validate_native_grid(grid):
    crs = grid.get("crsCode") or grid.get("crsWkt", "")
    metric = crs.startswith(("EPSG:326", "EPSG:327")) or (
        crs.startswith('PROJCS["UTM Zone ') and 'UNIT["m", 1.0]' in crs
    )
    if not metric:
        raise ValueError("Native metre UTM grid required")
    t = grid["affineTransform"]
    if not (0 < t["scaleX"] <= 2.5 and -2.5 <= t["scaleY"] < 0):
        raise ValueError("Native north-up resolution <=2.5m required")
    if t.get("shearX", 0) or t.get("shearY", 0):
        raise ValueError("Rotated grids unsupported")


def require_cloud_vm():
    request = Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/id",
        headers={"Metadata-Flavor": "Google"},
    )
    try:
        with urlopen(request, timeout=5) as response:
            if response.headers.get("Metadata-Flavor") != "Google":
                raise RuntimeError("Invalid GCE metadata response")
            return response.read().decode().strip()
    except Exception as exc:
        raise RuntimeError("This runner requires a Google Compute Engine VM") from exc


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_inputs(config):
    inventory = read_json(config["inventory"])
    assets = inventory["assets"] if isinstance(inventory, dict) else inventory
    priors = [read_json(path) for path in config["priors"]]
    if not priors:
        raise ValueError("All existing prior manifests are required")
    supplied = {p["sha256"] for p in priors}
    for prior in priors:
        if digest({k: v for k, v in prior.items() if k != "sha256"}) != prior["sha256"]:
            raise ValueError("Prior checksum mismatch")
        if any(
            ref["sha256"] not in supplied for ref in prior.get("prior_manifests", [])
        ):
            raise ValueError("Missing ancestor manifest")
    return assets, priors


def selected_candidates(assets, priors, rois, limit=None):
    """Ports/marinas first, one asset/catalog observation across every ROI."""
    seen_assets, seen_catalogs = set(), set()
    for roi in sorted(rois, key=lambda r: r["category"] == "shoreline"):
        count = 0
        for candidate in intake.candidate_assets(assets, priors, roi):
            asset = candidate["asset"]
            catalog = asset.get("properties", {}).get("catalogID")
            if asset["id"] in seen_assets or (catalog and catalog in seen_catalogs):
                continue
            if limit is not None and count >= limit:
                break
            seen_assets.add(asset["id"])
            if catalog:
                seen_catalogs.add(catalog)
            count += 1
            yield candidate, roi


def publish(bucket, prefix, name, content, *, immutable=False):
    blob = bucket.blob(f"{prefix}/{name}")
    kwargs = {"if_generation_match": 0} if immutable else {}
    blob.upload_from_string(
        json.dumps(content, indent=2), content_type="application/json", **kwargs
    )


def upload_preview(blob, preview):
    from google.api_core.exceptions import PreconditionFailed

    try:
        blob.upload_from_string(
            preview, content_type="image/png", if_generation_match=0
        )
    except PreconditionFailed:
        blob.reload()
        existing = blob.download_as_bytes(if_generation_match=int(blob.generation))
        if hashlib.sha256(existing).digest() != hashlib.sha256(preview).digest():
            raise ValueError("Existing immutable preview has different content")


def attempted_assets(report):
    return {
        row["asset_id"]
        for key in ("sources", "rejects", "errors")
        for row in report[key]
    }


def screening_worker(config, candidate, roi):
    """Spawned processes initialize their own credentials and HTTP transports."""
    import ee
    import google.auth
    from google.cloud import storage

    credentials, _ = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/earthengine",
        ]
    )
    ee.Initialize(credentials=credentials, project=config["project"])
    ee.data.setDeadline(120000)
    client = storage.Client(project=config["project"], credentials=credentials)
    return screen_one(
        ee, candidate, roi, client.bucket(config["bucket"]), config["output_prefix"]
    )


def screen_one(ee, candidate, roi, bucket, prefix):
    asset = candidate["asset"]
    if "SKYSAT" not in asset["id"].upper():
        raise ValueError("Only Google-hosted SkySat assets are allowed")
    bands = {b["id"]: b for b in asset["bands"]}
    grid = bands["R"]["grid"]
    if any(bands[b]["grid"] != grid for b in "GB"):
        raise ValueError("Native RGB grids differ")
    validate_native_grid(grid)
    image = ee.Image(asset["id"]).select(["R", "G", "B"])
    region = ee.Geometry.Rectangle(roi["bbox"], geodesic=False).intersection(
        image.geometry(), 1
    )
    area = region.area(1).getInfo()
    if area < 100000:
        raise ValueError("Less than 0.1km2 footprint overlap")
    geometry = region.getInfo()
    projection = ee.Projection(grid.get("crsCode") or grid["crsWkt"])
    projected = dict(
        type="Polygon",
        coordinates=region.transform(projection, 1)
        .bounds(1, projection)
        .coordinates()
        .getInfo(),
    )
    crop = intake.cropped_grid(grid, acquisition.bbox(projected))
    valid = (
        image.mask().reduce(ee.Reducer.min()).And(image.reduce(ee.Reducer.min()).gt(0))
    )
    stats = (
        valid.unmask(0, False)
        .rename("valid_fraction")
        .reduceRegion(ee.Reducer.mean(), region, scale=30, maxPixels=10000000)
        .getInfo()
    )
    source_id = (
        "skysat_marine_"
        + digest(dict(asset_id=asset["id"], roi=roi["id"], geometry=geometry))[:20]
    )
    url = image.getThumbURL(
        dict(region=region, dimensions=1024, min=0, max=255, format="png")
    )
    with urlopen(url, timeout=120) as response:
        preview = response.read()
    preview_name = f"{prefix}/previews/{source_id}.png"
    upload_preview(bucket.blob(preview_name), preview)
    estimated = crop["dimensions"]["width"] * crop["dimensions"]["height"] * 3
    t = grid["affineTransform"]
    rough = int(
        area
        / abs(t["scaleX"] * t["scaleY"])
        * float(stats.get("valid_fraction") or 0)
        * 3
    )
    return dict(
        asset_id=asset["id"],
        source_asset_parent=asset["id"],
        source_id=source_id,
        catalog_id=asset.get("properties", {}).get("catalogID"),
        group_id=candidate["group_id"],
        split=candidate["split"],
        geometry=geometry,
        grid=crop,
        estimated_native_bytes=estimated,
        rough_valid_bytes=min(estimated, rough),
        category=roi["category"],
        roi=roi,
        screening=stats,
        preview=f"gs://{bucket.name}/{preview_name}",
        preview_sha256=hashlib.sha256(preview).hexdigest(),
        source_asset=asset,
        raw_acquisition_review={"approved": False},
        counting="Rough valid bytes are a coarse30m estimate, not accepted bytes or proven ship pixels",
    )


def run_screen(ee, bucket, config, output):
    assets, priors = load_inputs(config)
    rois = (
        read_json(config["rois"]) if isinstance(config["rois"], str) else config["rois"]
    )
    report = dict(
        schema=1,
        status="screening",
        mode="NO_SOURCE_EXPORTS",
        sources=[],
        rejects=[],
        errors=[],
    )
    workers = config.get("screen_workers", 1)
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("screen_workers must be an integer from1 to4")
    selection = list(
        selected_candidates(assets, priors, rois, config.get("limit_per_roi"))
    )
    selection_digest = digest([(c, r) for c, r in selection])
    snapshots = []
    local = output / "screen-progress.json"
    if local.exists():
        snapshots.append(read_json(local))
    for name in ("screen-progress.json", "screen-partial.json", "screen.json"):
        blob = bucket.blob(f"{config['output_prefix']}/{name}")
        if blob.exists():
            snapshots.append(json.loads(blob.download_as_text()))
    allowed = {candidate["asset"]["id"] for candidate, _ in selection}
    for snapshot in snapshots:
        if (
            snapshot.get("selection_sha256", selection_digest) != selection_digest
            or not attempted_assets(snapshot) <= allowed
        ):
            raise ValueError("Screen selection changed on resume")
    if snapshots:
        report = max(
            snapshots,
            key=lambda item: (
                len(attempted_assets(item)),
                item["status"] in ("screened", "screened_with_errors"),
            ),
        )
        if report["status"] in ("screened", "screened_with_errors"):
            return report
    report.update(
        status="screening", selection_sha256=selection_digest, screen_workers=workers
    )
    report.pop("failure", None)
    attempted = attempted_assets(report)
    pending = [(c, r) for c, r in selection if c["asset"]["id"] not in attempted]

    def checkpoint():
        write_json(output / "screen-progress.json", report)
        publish(bucket, config["output_prefix"], "screen-progress.json", report)
        publish(bucket, config["output_prefix"], "progress.json", report)
        publish(bucket, config["output_prefix"], "screen-partial.json", report)

    def record(candidate, roi, result=None, error=None):
        if error is None:
            report["sources"].append(result)
        else:
            key = "rejects" if isinstance(error, ValueError) else "errors"
            report[key].append(
                dict(asset_id=candidate["asset"]["id"], roi=roi["id"], error=str(error))
            )
        checkpoint()

    try:
        checkpoint()
        if workers == 1:
            for candidate, roi in pending:
                try:
                    result = screen_one(
                        ee, candidate, roi, bucket, config["output_prefix"]
                    )
                except Exception as exc:
                    record(candidate, roi, error=exc)
                else:
                    record(candidate, roi, result=result)
        else:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                futures = {
                    pool.submit(screening_worker, config, c, r): (c, r)
                    for c, r in pending
                }
                for future in as_completed(futures):
                    candidate, roi = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        record(candidate, roi, error=exc)
                    else:
                        record(candidate, roi, result=result)
        report["status"] = "screened_with_errors" if report["errors"] else "screened"
        write_json(output / "screen.json", report)
        publish(bucket, config["output_prefix"], "screen.json", report, immutable=True)
    except Exception as exc:
        report.update(status="failed", failure=str(exc))
        checkpoint()
        raise
    return report


def validate_reviewed_manifest(manifest, config, assets, priors):
    if (
        manifest.get("schema") != 2
        or manifest.get("export_recipe") != acquisition.EXPORT_RECIPE
    ):
        raise ValueError("A schema2 native RGB export recipe is required")
    if (
        digest({k: v for k, v in manifest.items() if k != "sha256"})
        != manifest["sha256"]
    ):
        raise ValueError("Manifest checksum mismatch")
    if manifest["bucket"] != config["bucket"] or not manifest["prefix"].startswith(
        config["output_prefix"] + "/"
    ):
        raise ValueError("Export prefix must be dedicated to this expansion")
    assignments = intake.global_assignments(assets)
    by_id = {a["id"]: a for a in assets}
    excluded = {r["asset_id"] for p in priors for r in p["sources"]}
    catalogs = {
        by_id[i].get("properties", {}).get("catalogID") for i in excluded if i in by_id
    }
    seen, seen_catalogs = set(), set()
    for row in manifest["sources"]:
        asset = by_id[row["asset_id"]]
        catalog = asset.get("properties", {}).get("catalogID")
        if row.get("raw_acquisition_review", {}).get("approved") is not True:
            raise ValueError("Every source requires raw acquisition review")
        if row["asset_id"] in excluded | seen or (
            catalog and catalog in catalogs | seen_catalogs
        ):
            raise ValueError("Duplicate prior or within-run source/catalog")
        if (
            assignments[row["asset_id"]]
            != {"group_id": row["group_id"], "split": "train"}
            or row["split"] != "train"
        ):
            raise ValueError("Heldout or changed geographic assignment")
        if "SKYSAT" not in row["asset_id"].upper():
            raise ValueError("Only SkySat sources allowed")
        validate_native_grid(row["grid"])
        native = asset["bands"][0]["grid"]
        validate_native_grid(native)
        for key in ("scaleX", "scaleY", "shearX", "shearY"):
            if row["grid"]["affineTransform"].get(key, 0) != native[
                "affineTransform"
            ].get(key, 0):
                raise ValueError("Resampling is forbidden")
        for origin, scale in (("translateX", "scaleX"), ("translateY", "scaleY")):
            offset = (
                row["grid"]["affineTransform"][origin]
                - native["affineTransform"][origin]
            ) / native["affineTransform"][scale]
            if abs(offset - round(offset)) > 1e-6:
                raise ValueError("Crop is off native pixel alignment")
        seen.add(row["asset_id"])
        if catalog:
            seen_catalogs.add(catalog)
    if not seen:
        raise ValueError("Empty reviewed source selection")


def baseline_fingerprints(client, priors):
    result = set()
    objects = []
    for prior in priors:
        for blob in client.list_blobs(
            prior["bucket"], prefix=prior["prefix"].rstrip("/") + "/"
        ):
            if blob.name.lower().endswith((".tif", ".tiff")):
                if not blob.crc32c:
                    raise ValueError("Baseline object missing CRC32C")
                result.add((int(blob.size), blob.crc32c))
                objects.append(
                    dict(
                        bucket=prior["bucket"],
                        name=blob.name,
                        generation=str(blob.generation),
                        size=int(blob.size),
                        crc32c=blob.crc32c,
                    )
                )
    return result, objects


def qa_object(blob, row, scratch):
    import rasterio
    from rasterio.windows import Window

    generation = int(blob.generation)
    path = scratch / "source.tif"
    try:
        blob.download_to_filename(
            str(path), if_generation_match=generation, checksum="crc32c"
        )
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha.update(chunk)
        valid = 0
        with rasterio.open(path) as dataset:
            grid = row["grid"]
            t = grid["affineTransform"]
            if dataset.count != 3 or dataset.dtypes != ("uint8",) * 3:
                raise ValueError("Export is not native uint8 RGB")
            if dataset.crs != rasterio.crs.CRS.from_user_input(
                grid.get("crsCode") or grid["crsWkt"]
            ):
                raise ValueError("Export CRS changed")
            transform = dataset.transform
            if (
                abs(transform.a - t["scaleX"]) > 1e-9
                or abs(transform.e - t["scaleY"]) > 1e-9
                or transform.b
                or transform.d
            ):
                raise ValueError("Export resolution changed")
            for value in (
                (transform.c - t["translateX"]) / transform.a,
                (transform.f - t["translateY"]) / transform.e,
            ):
                if abs(value - round(value)) > 1e-6:
                    raise ValueError("Export is off native grid")
            for y in range(0, dataset.height, 1024):
                for x in range(0, dataset.width, 1024):
                    window = Window(
                        x,
                        y,
                        min(1024, dataset.width - x),
                        min(1024, dataset.height - y),
                    )
                    valid += count_valid_rgb(
                        dataset.read(window=window), dataset.read_masks(window=window)
                    )
        return dict(
            bucket=blob.bucket.name,
            name=blob.name,
            generation=str(generation),
            size=int(blob.size),
            crc32c=blob.crc32c,
            sha256=sha.hexdigest(),
            valid_pixels=valid,
            category=row["category"],
            source_id=row["source_id"],
            asset_id=row["asset_id"],
        )
    finally:
        path.unlink(missing_ok=True)


def run_acquire(ee, client, bucket, config, manifest, output, deadline):
    assets, priors = load_inputs(config)
    validate_reviewed_manifest(manifest, config, assets, priors)
    acquisition.utc(deadline)
    prefix = config["output_prefix"]
    state_path = output / "state.json"
    state_blob = bucket.blob(f"{prefix}/state.json")
    if state_blob.exists():
        state_blob.download_to_filename(str(state_path))
    manifest_blob = bucket.blob(f"{prefix}/acquisition-manifest.json")
    if manifest_blob.exists():
        if json.loads(manifest_blob.download_as_text())["sha256"] != manifest["sha256"]:
            raise ValueError("Cloud manifest differs from reviewed manifest")
    else:
        publish(bucket, prefix, "acquisition-manifest.json", manifest, immutable=True)
    report = dict(
        schema=1,
        status="exporting",
        objects=[],
        verified_ship_bytes=None,
        target_coverage_status="candidate_native_rgb_only_pending_later_pair_acceptance",
        limitation="Baseline represented training-source bytes plus new valid native RGB candidate bytes. Valid RGB pixels are not proven ship pixels; vicinity crops can contain land; no HR/LR pairs generated",
    )
    try:
        fingerprints, baseline = baseline_fingerprints(client, priors)
        publish(bucket, prefix, "baseline-objects.json", baseline)
        # The approved manifest, not stored padding or old exports, bounds selection.
        acquisition.RAW_CAP = 2**63 - 1
        while True:
            try:
                state = acquisition.tick(
                    ee, manifest, state_path, deadline, 0, max_pending=6
                )
            finally:
                if state_path.exists():
                    publish(bucket, prefix, "state.json", read_json(state_path))
            if state["status"] != "exporting":
                break
            time.sleep(
                min(
                    30,
                    max(
                        0,
                        (
                            acquisition.utc(deadline) - datetime.now(timezone.utc)
                        ).total_seconds(),
                    ),
                )
            )
        if state["status"] != "complete" or any(
            t["state"] != "COMPLETED" for t in state["tasks"].values()
        ):
            raise RuntimeError("Exports incomplete or failed: " + state["status"])
        report["status"] = "qa"
        with tempfile.TemporaryDirectory(dir=output) as directory:
            for row in manifest["sources"]:
                blobs = [
                    b
                    for b in client.list_blobs(
                        bucket.name,
                        prefix=f"{manifest['prefix']}/{row['source_id']}/rgb",
                    )
                    if b.name.lower().endswith(".tif")
                ]
                if not blobs:
                    raise ValueError(
                        "Completed export has no TIFF objects: " + row["source_id"]
                    )
                for blob in blobs:
                    report["objects"].append(qa_object(blob, row, Path(directory)))
                    report.update(
                        unique_accounting(
                            report["objects"],
                            fingerprints,
                            config.get("baseline_useful_bytes", 84_026_880_000),
                        )
                    )
                    write_json(output / "qa.json", report)
                    publish(bucket, prefix, "qa.json", report)
        report["status"] = "complete"
        report["target_useful_bytes"] = config.get(
            "target_useful_bytes", 150_000_000_000
        )
        report["candidate_threshold_reached"] = (
            report["total_useful_bytes"] >= report["target_useful_bytes"]
        )
    except Exception as exc:
        report.update(status="failed", failure=str(exc))
        raise
    finally:
        write_json(output / "qa.json", report)
        publish(bucket, prefix, "qa.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("screen", "acquire"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest")
    parser.add_argument("--deadline")
    args = parser.parse_args()
    if args.mode == "acquire" and (not args.manifest or not args.deadline):
        parser.error("acquire requires --manifest and --deadline")
    require_cloud_vm()
    import ee
    import google.auth
    from google.cloud import storage

    config = read_json(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    credentials, _ = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/earthengine",
        ]
    )
    ee.Initialize(credentials=credentials, project=config["project"])
    ee.data.setDeadline(120000)
    client = storage.Client(project=config["project"], credentials=credentials)
    bucket = client.bucket(config["bucket"])
    if args.mode == "screen":
        run_screen(ee, bucket, config, args.output_dir)
    else:
        run_acquire(
            ee,
            client,
            bucket,
            config,
            read_json(args.manifest),
            args.output_dir,
            args.deadline,
        )


if __name__ == "__main__":
    main()
