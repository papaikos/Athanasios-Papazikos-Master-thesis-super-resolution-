"""Read-only marine ROI screening; never starts EE exports or provisions compute."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.request import urlopen
from src.cloud_config import GCP_PROJECT

from scripts import skysat_acquire as acquire
from src.marine_acquire import digest, write_json


def cropped_grid(grid, projected_bbox):
    t = grid["affineTransform"]
    if t.get("shearX", 0) or t.get("shearY", 0):
        raise ValueError("Rotated grids need explicit support")
    if not (0 < t["scaleX"] <= 2.5 and -2.5 <= t["scaleY"] < 0):
        raise ValueError("Expected native north-up metre grid <=2.5m")
    x0, y0, x1, y1 = projected_bbox
    col0 = max(0, math.floor((x0 - t["translateX"]) / t["scaleX"]))
    col1 = min(
        grid["dimensions"]["width"], math.ceil((x1 - t["translateX"]) / t["scaleX"])
    )
    row0 = max(0, math.floor((y1 - t["translateY"]) / t["scaleY"]))
    row1 = min(
        grid["dimensions"]["height"], math.ceil((y0 - t["translateY"]) / t["scaleY"])
    )
    if col1 <= col0 or row1 <= row0:
        raise ValueError("ROI outside native grid")
    return {
        **grid,
        "dimensions": {"width": col1 - col0, "height": row1 - row0},
        "affineTransform": {
            **t,
            "translateX": t["translateX"] + col0 * t["scaleX"],
            "translateY": t["translateY"] + row0 * t["scaleY"],
        },
    }


def global_assignments(assets):
    assets = sorted(assets, key=lambda a: a["id"])
    boxes = [acquire.bbox(a["geometry"]) for a in assets]
    parent = list(range(len(assets)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(assets):
        for j in range(i):
            b, c = boxes[i], boxes[j]
            overlap = not (b[2] < c[0] or c[2] < b[0] or b[3] < c[1] or c[3] < b[1])
            catalog = a.get("properties", {}).get("catalogID")
            if overlap or (
                catalog and catalog == assets[j].get("properties", {}).get("catalogID")
            ):
                parent[root(i)] = root(j)
    groups = {}
    for i, a in enumerate(assets):
        groups.setdefault(root(i), []).append(a["id"])
    ids = {
        k: hashlib.sha256("|".join(v).encode()).hexdigest()[:16]
        for k, v in groups.items()
    }
    split = {
        g: ("validation" if n == 0 else "test" if n == 1 else "train")
        for n, g in enumerate(sorted(ids.values()))
    }
    return {
        a["id"]: {"group_id": ids[root(i)], "split": split[ids[root(i)]]}
        for i, a in enumerate(assets)
    }


def candidate_assets(assets, priors, roi):
    assignments = global_assignments(assets)
    used = {r["asset_id"] for p in priors for r in p["sources"]}
    catalogs = {
        a.get("properties", {}).get("catalogID") for a in assets if a["id"] in used
    }
    for p in priors:
        if digest({k: v for k, v in p.items() if k != "sha256"}) != p["sha256"]:
            raise ValueError("Prior manifest checksum mismatch")
        for r in p["sources"]:
            if assignments[r["asset_id"]] != {k: r[k] for k in ("group_id", "split")}:
                raise ValueError("Frozen geographic split changed")
    result = []
    west, south, east, north = roi["bbox"]
    for a in assets:
        b = acquire.bbox(a["geometry"])
        if a["id"] in used or (a.get("properties", {}).get("catalogID") in catalogs):
            continue
        if assignments[a["id"]]["split"] != "train":
            continue
        if b[2] < west or east < b[0] or b[3] < south or north < b[1]:
            continue
        result.append({"asset": a, **assignments[a["id"]]})
    return result


def screen(ee, candidate, roi, output):
    a = candidate["asset"]
    image = ee.Image(a["id"]).select(["R", "G", "B"])
    region = ee.Geometry.Rectangle(roi["bbox"], geodesic=False).intersection(
        image.geometry(), 1
    )
    geometry = region.getInfo()
    if region.area(1).getInfo() < 100000:
        raise ValueError("Less than 0.1km2 footprint overlap")
    native = a["bands"][0]["grid"]
    projection = ee.Projection(native.get("crsCode") or native["crsWkt"])
    projected = {
        "type": "Polygon",
        "coordinates": region.transform(projection, 1)
        .bounds(1, projection)
        .coordinates()
        .getInfo(),
    }
    grid = cropped_grid(native, acquire.bbox(projected))
    valid = (
        image.mask().reduce(ee.Reducer.min()).And(image.reduce(ee.Reducer.min()).gt(0))
    )
    water = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).gte(50)
    )
    stats = (
        valid.unmask(0, False)
        .rename("valid_fraction")
        .addBands(water.updateMask(valid).rename("water_fraction_valid"))
        .reduceRegion(
            ee.Reducer.mean(),
            region,
            scale=roi.get("screen_scale_metres", 30),
            maxPixels=1000000,
        )
        .getInfo()
    )
    source = (
        "skysat_marine_"
        + digest({"asset_id": a["id"], "roi": roi["id"], "geometry": geometry})[:20]
    )
    preview = output / (source + ".png")
    url = image.getThumbURL(
        {
            "region": region,
            "dimensions": roi.get("preview_dimensions", 1024),
            "min": 0,
            "max": 255,
            "format": "png",
        }
    )
    with urlopen(url, timeout=120) as response:
        preview.write_bytes(response.read())
    return {
        "asset_id": a["id"],
        "source_asset_parent": a["id"],
        "source_id": source,
        "group_id": candidate["group_id"],
        "split": candidate["split"],
        "geometry": geometry,
        "grid": grid,
        "estimated_native_bytes": grid["dimensions"]["width"]
        * grid["dimensions"]["height"]
        * 3,
        "cloud_percent": a.get("properties", {}).get("cloudcoverAreaPercentage"),
        "roi": roi,
        "screening": stats,
        "preview": str(preview),
        "preview_sha256": hashlib.sha256(preview.read_bytes()).hexdigest(),
        "water_reference": "JRC/GSW1_4/GlobalSurfaceWater occurrence>=50%; coarse30m screening only; masked ocean is unknown, never reject by this fraction",
        "visual_review": "PENDING",
        "accepted_train_bytes": 0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", default="reports/skysat-inventory.json")
    p.add_argument("--rois", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--limit-per-roi", type=int, default=1)
    p.add_argument("--project", default=GCP_PROJECT)
    p.add_argument("--prior", action="append", default=[])
    p.add_argument("--deadline-utc")
    args = p.parse_args()
    if not 1 <= args.limit_per_roi <= 3:
        raise ValueError("Read-only preview bound is1..3 observations per ROI")
    assets = json.loads(Path(args.inventory).read_text())
    paths = [Path("reports/skysat-acquisition-manifest-v2.json")] + [
        Path(f"reports/skysat-acquisition-manifest-batch{i}.json") for i in range(2, 9)
    ]
    paths.extend(Path(x) for x in args.prior)
    priors = [json.loads(x.read_text()) for x in paths]
    deadline = (
        datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
        if args.deadline_utc
        else datetime.now(timezone.utc) + timedelta(minutes=30)
    )
    if (
        deadline.tzinfo is None
        or not 0 < (deadline - datetime.now(timezone.utc)).total_seconds() <= 7200
    ):
        raise ValueError(
            "Preview deadline must be timezone-aware, future and within2hours"
        )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    acquire.GCLOUD = (
        "C:/Program Files (x86)/Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd"
    )
    ee = acquire.initialize(args.project, True)
    rows = []
    errors = []
    for roi in json.loads(Path(args.rois).read_text()):
        candidates = candidate_assets(assets, priors, roi)
        for candidate in candidates[: args.limit_per_roi]:
            if datetime.now(timezone.utc) >= deadline:
                raise TimeoutError("Read-only preview deadline reached")
            try:
                row = screen(ee, candidate, roi, out)
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "roi": roi["id"],
                            "asset": row["asset_id"],
                            "screening": row["screening"],
                        }
                    ),
                    flush=True,
                )
            except Exception as exc:
                errors.append(
                    {
                        "roi": roi["id"],
                        "asset": candidate["asset"]["id"],
                        "error": str(exc),
                    }
                )
            write_json(
                out / "screening.json",
                {
                    "schema": 1,
                    "mode": "READ_ONLY_NO_EXPORTS",
                    "sources": rows,
                    "errors": errors,
                    "counting": "No accepted bytes. Cropped grid reservations exclude no invalid pixels yet; visual review and pair receipts required.",
                },
            )


if __name__ == "__main__":
    main()
