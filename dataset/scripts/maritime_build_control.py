"""Metadata-only local control; raster processing belongs on Compute Engine."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from src.cloud_config import GCP_PROJECT, GCS_BUCKET

PROJECT = GCP_PROJECT
BUCKET = GCS_BUCKET


def source_prefixes():
    return (["skysat/raw/", "skysat/raw-v2/"]
            + [f"skysat/raw-batch{i}/" for i in range(2, 9)]
            + [f"skysat/marine-addition/raw-batch{i}/" for i in range(1, 4)]
            + ["skysat/expansion-20260929/raw/"])


def join_sources(manifests, objects, bucket, assets):
    lookup = {}
    for manifest in manifests:
        for row in manifest["sources"]:
            key = manifest["prefix"].rstrip("/") + "/" + row["source_id"] + "/"
            if key in lookup and lookup[key]["asset_id"] != row["asset_id"]:
                raise ValueError("Conflicting source folder identity")
            lookup[key] = {**row, "export_recipe": manifest.get("export_recipe")}
    result, seen = [], set()
    for obj in sorted(objects, key=lambda o: o["name"]):
        if not obj["name"].lower().endswith((".tif", ".tiff")):
            continue
        if obj["name"] in seen:
            raise ValueError("Duplicate object identity")
        seen.add(obj["name"])
        row = lookup.get(obj["name"].rsplit("/", 1)[0] + "/")
        if row is None:
            continue
        asset = assets.get(row["asset_id"], {})
        properties = asset.get("properties", {})
        if not str(obj["generation"]).isdigit() or int(obj["size"]) <= 0:
            raise ValueError("Positive generation-pinned source required")
        result.append({
            "uri": f"gs://{bucket}/{obj['name']}",
            "generation": str(obj["generation"]), "bytes": int(obj["size"]),
            "crc32c": obj.get("crc32c"), "asset_id": row["asset_id"],
            "source_id": row["source_id"], "catalog_id": properties.get("catalogID"),
            "acquisition_utc": properties.get("collectionStartTime"),
            "grid": row.get("grid"), "geometry": row.get("geometry"),
            "site_id": row.get("group_id"), "historical_split": row.get("split"),
            "roi": row.get("roi"), "properties": properties,
            "mask_provenance": "legacy_nodata_zero_conservative",
            "export_recipe": row.get("export_recipe"),
        })
    return result


def client_from_gcloud():
    from google.auth.credentials import Credentials
    from google.cloud import storage
    executable = shutil.which("gcloud.cmd") or shutil.which("gcloud")
    if not executable:
        raise RuntimeError("Google Cloud SDK unavailable")

    class SDKCredentials(Credentials):
        def refresh(self, request):
            from datetime import timedelta
            self.token = subprocess.check_output(
                [executable, "auth", "print-access-token"], text=True, timeout=60).strip()
            self.expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=40)

    return storage.Client(project=PROJECT, credentials=SDKCredentials())


def census(root, output):
    config = json.loads((root / "reports/expansion-20260929/config.json").read_text())
    paths = config["priors"] + ["reports/expansion-20260929/acquisition-manifest.json"]
    manifests = [json.loads((root / p).read_text()) for p in paths]
    assets = json.loads((root / "reports/skysat-inventory.json").read_text())
    client = client_from_gcloud()
    bucket = client.bucket(BUCKET)
    bucket.reload()
    if bucket.location.upper() != "EUROPE-WEST4":
        raise ValueError("Unexpected bucket location")
    objects = []
    for prefix in source_prefixes():
        rows = list(client.list_blobs(BUCKET, prefix=prefix))
        objects.extend({"name": b.name, "generation": str(b.generation),
                        "size": b.size, "crc32c": b.crc32c, "md5_hash": b.md5_hash}
                       for b in rows)
        print(json.dumps({"prefix": prefix, "objects": len(rows)}), flush=True)
    sources = join_sources(manifests, objects, BUCKET, {a["id"]: a for a in assets})
    payload = {"schema": 1, "status": "candidate_inventory_not_semantic_acceptance",
               "project": PROJECT, "bucket": BUCKET, "location": bucket.location,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "sources": sources, "ancestor_manifests": paths,
               "source_objects": len(sources),
               "compressed_source_bytes": sum(s["bytes"] for s in sources),
               "assets": len({s["asset_id"] for s in sources}),
               "historical_sites": len({s["site_id"] for s in sources}),
               "unmapped_objects": [o for o in objects
                                     if o["name"].endswith(".tif")
                                     and f"gs://{BUCKET}/{o['name']}" not in {s["uri"] for s in sources}]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "source-inventory.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output / "raw-objects.json").write_text(json.dumps(objects, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k not in ("sources", "unmapped_objects")}), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    census(args.root, args.output)


if __name__ == "__main__":
    main()
