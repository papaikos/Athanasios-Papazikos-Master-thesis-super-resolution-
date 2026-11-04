"""GCE CPU catalog screening: metadata and small single-acquisition previews only.

Every result remains a candidate requiring semantic review. No raster exports,
dataset splits, or accepted training data are created by this controller.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

COLLECTION = "SKYSAT/GEN-A/PUBLIC/ORTHO/RGB"
WATER = "JRC/GSW1_4/GlobalSurfaceWater"
COUNTRIES = "FAO/GAUL/2015/level0"
STATUS = "candidate_not_semantic_acceptance"
ANCILLARY = {
    "water": WATER,
    "water_method": "occurrence.mask().gt(0).unmask(0, False); unweighted mean; observed water at least once; ocean absence is unknown",
    "countries": COUNTRIES,
    "countries_attribution": "FAO, Global Administrative Unit Layers (GAUL), 2015",
    "countries_license": "Ancillary GAUL reference: non-commercial use only; retain FAO attribution and GAUL license conditions.",
    "countries_license_url": "https://developers.google.com/earth-engine/datasets/catalog/FAO_GAUL_2015_level0",
    "coast_method": "GAUL per-country clipped area / bbox area, 1000 m geometry error; approximate administrative geometry, not a semantic coastline label",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def bbox(geometry):
    points = []

    def visit(value):
        if not value:
            raise ValueError("Empty geometry")
        if isinstance(value[0], (int, float)):
            if len(value) < 2 or not all(math.isfinite(v) for v in value[:2]):
                raise ValueError("Invalid geometry coordinate")
            points.append(value)
        else:
            for child in value:
                visit(child)

    visit(geometry["coordinates"])
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]


def geographic_groups(assets):
    """Same transitive bbox/catalog grouping and ID hash as existing intake."""
    if not isinstance(assets, list) or not assets:
        raise ValueError("Inventory must be a nonempty asset array")
    assets = sorted(assets, key=lambda a: a["id"])
    if len({a["id"] for a in assets}) != len(assets):
        raise ValueError("Duplicate asset ID")
    boxes = [bbox(a["geometry"]) for a in assets]
    parent = list(range(len(assets)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(assets):
        catalog = a.get("properties", {}).get("catalogID")
        for j in range(i):
            b, c = boxes[i], boxes[j]
            overlap = not (b[2] < c[0] or c[2] < b[0] or b[3] < c[1] or c[3] < b[1])
            if overlap or (catalog and catalog == assets[j].get("properties", {}).get("catalogID")):
                parent[root(i)] = root(j)
    members = {}
    for i, asset in enumerate(assets):
        members.setdefault(root(i), []).append(asset)
    groups = []
    for rows in members.values():
        ids = [a["id"] for a in rows]
        bounds = [bbox(a["geometry"]) for a in rows]
        groups.append({"group_id": sha("|".join(ids).encode())[:16], "asset_ids": ids,
                       "bbox": [min(b[0] for b in bounds), min(b[1] for b in bounds),
                                max(b[2] for b in bounds), max(b[3] for b in bounds)],
                       "representative": min(rows, key=representative_key)})
    return sorted(groups, key=lambda g: g["group_id"])


def representative_key(asset):
    cloud = asset.get("properties", {}).get("cloudcoverAreaPercentage")
    if not isinstance(cloud, (int, float)) or not math.isfinite(cloud):
        cloud = 101
    b = bbox(asset["geometry"])
    area = (b[2] - b[0]) * (b[3] - b[1]) * math.cos(math.radians((b[1] + b[3]) / 2))
    return cloud, -area, asset["id"]


def candidate_rank(stats):
    water, land = stats.get("observed_water_fraction"), stats.get("country_land_fraction")
    if land is not None and 0.01 < land < 0.99:
        return "coast_candidate", 3
    if water is not None and 0 < water < 1:
        return "mixed_observed_water_land_candidate", 2
    if (water is not None and water > 0) or (land is not None and land < 0.99):
        return "water_or_ocean_candidate", 1
    return "reference_inconclusive_review_required", 0


def screen_metadata(ee, group):
    region = ee.Geometry.Rectangle(group["bbox"], geodesic=False)
    countries = ee.FeatureCollection(COUNTRIES).filterBounds(region)
    # Clip before aggregation: unioning an entire complex country first is slow.
    clipped = countries.map(lambda feature: feature.set(
        'clip_area', feature.geometry().intersection(region, 1000).area(1000)))
    land_area = ee.Number(clipped.aggregate_sum('clip_area'))
    water = ee.Image(WATER).select("occurrence").mask().gt(0).unmask(0, False).rename("water")
    fraction = water.reduceRegion(reducer=ee.Reducer.mean().unweighted(), geometry=region,
                                  scale=30, maxPixels=100000000, bestEffort=True).get("water")
    return ee.Dictionary({"observed_water_fraction": fraction,
                          "country_land_fraction": land_area.divide(region.area(100)).max(0).min(1),
                          "countries": countries.aggregate_array("ADM0_NAME").distinct().sort()}).getInfo()


def bounded_retry(action):
    for attempt in range(3):  # initial request plus at most two retries
        try:
            return action()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def fetch_preview(ee, asset):
    image = ee.Image(asset["id"]).select(["R", "G", "B"])
    url = image.getThumbURL({"region": image.geometry().bounds(1), "dimensions": 1920,
                             "min": 0, "max": 255, "format": "png"})
    with urlopen(url, timeout=120) as response:
        payload = response.read(32 * 1024 * 1024 + 1)
    if len(payload) > 32 * 1024 * 1024 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Invalid or oversized PNG preview")
    return payload


def require_cloud_vm():
    request = Request("http://metadata.google.internal/computeMetadata/v1/instance/id",
                      headers={"Metadata-Flavor": "Google"})
    try:
        with urlopen(request, timeout=5) as response:
            instance = response.read().decode().strip()
            if response.headers.get("Metadata-Flavor") != "Google" or not instance.isdigit():
                raise ValueError("Invalid VM metadata")
            return instance
    except Exception as exc:
        raise RuntimeError("Catalog screening must run on a Google Compute Engine VM") from exc


class CloudStore:
    def __init__(self, client, prefix):
        uri = urlsplit(prefix)
        if uri.scheme != "gs" or not uri.netloc or not uri.path.strip("/") or uri.query or uri.fragment:
            raise ValueError("Output prefix must be gs://bucket/path")
        self.bucket = client.bucket(uri.netloc)
        self.prefix = uri.path.strip("/")
        self.uri = prefix.rstrip("/")

    def read_json(self, name):
        blob = self.bucket.blob(self.prefix + "/" + name)
        if not blob.exists():
            return None
        blob.reload()
        return json.loads(blob.download_as_bytes(if_generation_match=int(blob.generation)))

    def create(self, name, payload, content_type):
        from google.api_core.exceptions import PreconditionFailed
        blob = self.bucket.blob(self.prefix + "/" + name)
        try:
            blob.upload_from_string(payload, content_type=content_type, if_generation_match=0)
        except PreconditionFailed:
            blob.reload()
            if blob.download_as_bytes(if_generation_match=int(blob.generation)) != payload:
                raise ValueError("Immutable output conflict")
        blob.reload()
        return {"uri": self.uri + "/" + name, "generation": str(blob.generation),
                "bytes": len(payload), "sha256": sha(payload), "crc32c": blob.crc32c}


def run_groups(ee, store, groups, run_identity, scratch):
    rows = []
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    for index, group in enumerate(groups, 1):
        name = "groups/" + group["group_id"] + ".json"
        row = store.read_json(name)
        if row is not None:
            if row.get("run_identity") != run_identity or row.get("asset_ids") != group["asset_ids"]:
                raise ValueError("Checkpoint inventory or recipe mismatch")
        legacy_failure = row is not None and bool(row.get('errors'))
        attempts = []
        for number in range(1, 4):
            attempt = store.read_json(f'attempts/{group["group_id"]}-{number}.json')
            if attempt is None:
                break
            if attempt.get('run_identity') != run_identity or attempt.get('asset_ids') != group['asset_ids']:
                raise ValueError('Attempt checkpoint inventory or recipe mismatch')
            attempts.append(attempt)
        if attempts:
            row = attempts[-1]
        if row is None or (row.get('errors') and len(attempts) + int(legacy_failure) < 3):
            asset = group["representative"]
            row = {"schema": 1, "run_identity": run_identity, "status": STATUS,
                   "group_id": group["group_id"], "asset_ids": group["asset_ids"],
                   "bbox": group["bbox"], "representative_asset_id": asset["id"],
                   "representative_catalog_id": asset.get("properties", {}).get("catalogID"),
                   "representative_bbox": bbox(asset["geometry"]),
                   "cloud_percent": asset.get("properties", {}).get("cloudcoverAreaPercentage"),
                   "ancillary_references": ANCILLARY, "errors": [], "preview_status": "pending"}
            try:
                row["screening"] = bounded_retry(lambda: screen_metadata(ee, group))
            except Exception as exc:
                row["screening"] = {}
                row["errors"].append({"stage": "metadata", "type": type(exc).__name__})
            row["candidate_kind"], row["priority"] = candidate_rank(row["screening"])
            # Even a zero JRC fraction may be ocean: every group gets a preview attempt.
            try:
                payload = bounded_retry(lambda: fetch_preview(ee, asset))
            except Exception as exc:
                row["preview_status"] = "error"
                row["errors"].append({"stage": "preview", "type": type(exc).__name__})
            else:
                # Storage conflicts must fail closed rather than masquerade as a
                # preview fetch error and leave conflicting immutable outputs.
                row["preview"] = bounded_retry(lambda: store.create(
                    "previews/" + group["group_id"] + ".png", payload, "image/png"))
                row["preview_status"] = "created"
            attempt_number = len(attempts) + 1
            store.create(f'attempts/{group["group_id"]}-{attempt_number}.json', canonical(row), 'application/json')
            if not row['errors']:
                completion_name = f'groups/recovered-{group["group_id"]}.json' if legacy_failure else name
                store.create(completion_name, canonical(row), 'application/json')
        rows.append(row)
        checkpoint = scratch / "checkpoint.json.tmp"
        checkpoint.write_bytes(canonical({"run_identity": run_identity, "completed_groups": [r["group_id"] for r in rows]}))
        checkpoint.replace(scratch / "checkpoint.json")
        print(json.dumps({"completed": index, "total": len(groups), "group_id": row["group_id"],
                          "preview_status": row["preview_status"], "errors": row["errors"]}), flush=True)
    summary = {"schema": 1, "run_identity": run_identity, "status": STATUS,
               "asset_count": sum(len(g["asset_ids"]) for g in groups), "group_count": len(groups),
               "preview_count": sum(r["preview_status"] == "created" for r in rows),
               "error_group_count": sum(bool(r["errors"]) for r in rows),
               "ancillary_references": ANCILLARY,
               "groups": sorted(rows, key=lambda r: (-r["priority"], r["group_id"]))}
    if not summary['error_group_count']:
        old_summary = store.read_json('summary.json')
        name = 'summary-recovered.json' if old_summary and old_summary.get('error_group_count') else 'summary.json'
        store.create(name, canonical(summary), 'application/json')
    else:
        store.create('failed-summaries/' + sha(canonical(summary)) + '.json', canonical(summary), 'application/json')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--expected-assets", type=int, default=1094)
    parser.add_argument("--expected-groups", type=int, default=132)
    parser.add_argument('--partition-count', type=int, default=1)
    parser.add_argument('--partition-id', type=int, default=0)
    args = parser.parse_args()
    require_cloud_vm()
    assets = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    if any(not a["id"].startswith(COLLECTION + "/") for a in assets):
        raise ValueError("Inventory contains an asset outside the public RGB catalog")
    groups = geographic_groups(assets)
    if len(assets) != args.expected_assets or len(groups) != args.expected_groups:
        raise ValueError("Inventory asset/group count does not match the full catalog contract")
    if args.partition_count < 1 or not 0 <= args.partition_id < args.partition_count:
        parser.error('Invalid catalog partition')
    groups = [g for i,g in enumerate(groups) if i % args.partition_count == args.partition_id]
    import ee
    import google.auth
    from google.cloud import storage
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    ee.Initialize(credentials=credentials, project=args.project)
    ee.data.setDeadline(45000)
    ee.data.setMaxRetries(0)  # The persisted, bounded application retry is authoritative.
    store = CloudStore(storage.Client(project=args.project, credentials=credentials), args.output_prefix)
    identity = sha(canonical({"inventory": sorted(assets, key=lambda a: a["id"]),
                              "recipe": ANCILLARY, "preview_dimensions": 1920,
                              "project": args.project, "schema": 1}))
    identity = sha(canonical([identity, args.partition_count, args.partition_id]))
    summary = run_groups(ee, store, groups, identity, args.scratch)
    if summary["error_group_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
