"""GCE-only census of pinned original SkySat objects; never accepts labels.

Run: python -m scripts.maritime_cloud_inventory --manifest sources.json
     --output-prefix gs://BUCKET/control/runs/RUN --scratch /mnt/work/census
One source is downloaded at a time. Local use is restricted to synthetic tests.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, MaskFlags
from rasterio.windows import Window, bounds, transform

CORE = 1600
HALO = 50
SCHEMA = 1


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha256(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def parse_gs(uri):
    parsed = urlsplit(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("Expected gs://bucket/object URI")
    if parsed.query or parsed.fragment:
        raise ValueError("Pin generation separately from the GCS URI")
    return parsed.netloc, parsed.path.lstrip("/")


def validate_manifest(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty array")
    seen = {}
    objects = set()
    required = {"uri", "generation", "bytes", "crc32c", "asset_id", "source_id",
                "catalog_id", "grid", "geometry"}
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("Source row missing required pinned identity fields")
        parse_gs(row["uri"])
        if not str(row["generation"]).isdigit() or int(row["generation"]) <= 0:
            raise ValueError("Source generation must be a positive integer")
        if isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int) or row["bytes"] <= 0:
            raise ValueError("Source bytes must be a positive integer")
        if any(not isinstance(row[key], str) or not row[key]
               for key in ("crc32c", "source_id", "asset_id", "catalog_id")):
            raise ValueError("Source identities and CRC32C must be nonempty strings")
        if not isinstance(row["grid"], dict) or not isinstance(row["geometry"], dict):
            raise ValueError("Source grid and geometry must be objects")
        identity = (row["uri"], str(row["generation"]))
        if identity in objects:
            raise ValueError("Manifest contains duplicate pinned objects")
        source_identity = (row["asset_id"], row["catalog_id"])
        if row["source_id"] in seen and seen[row["source_id"]] != source_identity:
            raise ValueError("Sharded source_id has inconsistent asset or catalog identity")
        seen[row["source_id"]] = source_identity
        objects.add(identity)
    return rows


def validate_header(ds):
    t = ds.transform
    metric = (ds.crs is not None and ds.crs.is_projected
              and math.isclose(ds.crs.linear_units_factor[1], 1.0, rel_tol=0, abs_tol=1e-12))
    if not metric or not all(math.isfinite(v) for v in tuple(t)[:6]) or not (
        abs(t.a - 0.8) <= 1e-6 and abs(t.e + 0.8) <= 1e-6
        and t.b == 0 and t.d == 0
    ):
        raise ValueError("Strict north-up metric 0.8 m native grid required")
    if ds.count != 3 or ds.dtypes != ("uint8", "uint8", "uint8"):
        raise ValueError("Exactly three uint8 RGB bands required")
    if ds.colorinterp != (ColorInterp.red, ColorInterp.green, ColorInterp.blue):
        raise ValueError("Explicit RGB band order required")


def inspect_raster(path, *, metadata_only=False, preview_directory=None,
                   preview_samples=3, original_mask_provenance=None,
                   export_recipe=None):
    source_sha = None if metadata_only else file_sha256(path)
    with rasterio.open(path) as ds:
        validate_header(ds)
        flags = ds.mask_flag_enums
        explicit = all(MaskFlags.per_dataset in band for band in flags)
        known = explicit and original_mask_provenance == "original_explicit_dataset_mask"
        if explicit:
            provenance = "original_explicit_dataset_mask" if known else "explicit_dataset_mask_unverified"
        elif any(MaskFlags.nodata in band for band in flags) and ds.nodata == 0:
            provenance = "legacy_nodata_zero_conservative"
        else:
            provenance = "unresolved_all_valid" if all(
                MaskFlags.all_valid in band for band in flags) else "unresolved_nodata_mask"
        conservative = provenance == "legacy_nodata_zero_conservative" and (
            isinstance(export_recipe, dict) and export_recipe.get("unmask") == 0
            and export_recipe.get("noData") == 0)
        candidates = []
        previews = []
        for row in range(0, ds.height - CORE + 1, CORE):
            for col in range(0, ds.width - CORE + 1, CORE):
                if col < HALO or row < HALO or col + CORE + HALO > ds.width or row + CORE + HALO > ds.height:
                    continue
                core = Window(col, row, CORE, CORE)
                context = Window(col - HALO, row - HALO, CORE + 2 * HALO, CORE + 2 * HALO)
                valid = None if metadata_only else bool(np.all(ds.read_masks(window=context) == 255))
                reasons = []
                if not known:
                    reasons.append("unresolved_original_mask")
                if valid is False:
                    reasons.append("invalid_core_or_halo_mask")
                candidate = dict(core_window=[col, row, CORE, CORE],
                                 context_window=[col - HALO, row - HALO, CORE + 2 * HALO, CORE + 2 * HALO],
                                 core_transform=list(transform(core, ds.transform))[:6],
                                 core_bounds=list(bounds(core, ds.transform)),
                                 context_bounds=list(bounds(context, ds.transform)),
                                 mask_valid=valid, rejection_reasons=reasons,
                                 conservative_mask_support=conservative and valid is True,
                                 requires_semantic_review=True)
                candidates.append(candidate)
                if preview_directory and not metadata_only and len(previews) // 2 < preview_samples:
                    previews.extend(make_previews(ds, core, preview_directory, len(previews) // 2))
        return dict(schema_version=SCHEMA, width=ds.width, height=ds.height,
                    crs=ds.crs.to_string(), native_transform=list(ds.transform)[:6],
                    native_bounds=list(ds.bounds), band_order=["R", "G", "B"],
                    dtype="uint8", nodata=ds.nodata,
                    mask_flags=[[flag.name for flag in band] for band in flags],
                    mask_provenance=provenance, original_mask_known=known,
                    documented_export_recipe=export_recipe,
                    source_sha256=source_sha, metadata_only=metadata_only,
                    complete_core_count=(ds.width // CORE) * (ds.height // CORE),
                    geometric_candidate_count=len(candidates),
                    mask_qualified_candidate_count=None if metadata_only else sum(
                        candidate["mask_valid"] is True and not candidate["rejection_reasons"]
                        for candidate in candidates),
                    conservative_mask_qualified_candidate_count=None if metadata_only else sum(
                        candidate["conservative_mask_support"] for candidate in candidates),
                    accepted_semantic_count=0, candidates=candidates,
                    preview_files=previews)


def make_previews(ds, core, directory, index):
    try:
        from PIL import Image
    except ImportError:
        return []
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    overview = ds.read(window=core, out_shape=(3, 400, 400))
    detail = ds.read(window=Window(core.col_off + 672, core.row_off + 672, 256, 256))
    paths = []
    for name, pixels in (("overview", overview), ("native-detail", detail)):
        path = directory / f"candidate-{index:03d}-{name}.png"
        Image.fromarray(np.moveaxis(pixels, 0, -1)).save(path)
        paths.append(str(path))
    return paths


def create_only(blob, payload):
    """Immutable byte-identical writes; existing conflicting objects fail closed."""
    if not blob.exists():
        try:
            blob.upload_from_string(payload, if_generation_match=0, checksum="crc32c")
        except Exception as exc:
            # Recover only a create race, never an arbitrary upload failure.
            if getattr(exc, "code", None) != 412:
                raise
    blob.reload()
    actual = blob.download_as_bytes(if_generation_match=int(blob.generation), checksum="crc32c")
    if actual != payload:
        raise ValueError("Immutable output conflict")
    return dict(generation=str(blob.generation), bytes=len(actual),
                crc32c=blob.crc32c, sha256=digest_bytes(actual))


def require_cloud_vm():
    request = Request("http://metadata.google.internal/computeMetadata/v1/instance/id",
                      headers={"Metadata-Flavor": "Google"})
    try:
        with urlopen(request, timeout=5) as response:
            if response.headers.get("Metadata-Flavor") != "Google":
                raise RuntimeError("Invalid metadata server")
            instance = response.read().decode().strip()
            if not instance.isdigit():
                raise RuntimeError("Invalid VM identity")
            return instance
    except Exception as exc:
        raise RuntimeError("Raw census must run on a Google Compute Engine VM") from exc


@contextmanager
def authorized_range_source(row, credentials):
    from google.auth.transport.requests import Request as AuthRequest
    credentials.refresh(AuthRequest())
    bucket, name = parse_gs(row["uri"])
    url = (f"https://storage.googleapis.com/{bucket}/{quote(name, safe='/')}"
           f"?generation={row['generation']}")
    previous = os.environ.get("GDAL_HTTP_HEADERS")
    os.environ["GDAL_HTTP_HEADERS"] = f"Authorization: Bearer {credentials.token}"
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                          CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
                          CPL_CURL_VERBOSE=False, CPL_DEBUG=False):
            yield "/vsicurl/" + url
    finally:
        if previous is None:
            os.environ.pop("GDAL_HTTP_HEADERS", None)
        else:
            os.environ["GDAL_HTTP_HEADERS"] = previous


def verify_output(client, ref):
    bucket, name = parse_gs(ref["uri"])
    blob = client.bucket(bucket).blob(name, generation=int(ref["generation"]))
    blob.reload()
    if str(blob.generation) != str(ref["generation"]) or int(blob.size) != ref["bytes"] or blob.crc32c != ref["crc32c"]:
        raise ValueError("Checkpoint output metadata mismatch")
    data = blob.download_as_bytes(if_generation_match=int(ref["generation"]), checksum="crc32c")
    if digest_bytes(data) != ref["sha256"]:
        raise ValueError("Checkpoint output SHA-256 mismatch")
    return data


def run(args):
    instance = require_cloud_vm()
    import google.auth
    from google.cloud import storage

    rows = validate_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8-sig")))
    if args.max_sources is not None:
        if args.max_sources < 1:
            raise ValueError("max-sources must be positive")
        rows = rows[:args.max_sources]
    credentials, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    client = storage.Client(project=project, credentials=credentials)
    bucket_name, prefix = parse_gs(args.output_prefix.rstrip("/"))
    bucket = client.bucket(bucket_name)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    configuration = dict(schema_version=SCHEMA, core=CORE, halo=HALO,
                         metadata_only=args.metadata_only, preview_samples=args.preview_samples,
                         worker_sha256=file_sha256(__file__))
    config_sha = digest_bytes(json_bytes(configuration))
    completed = []
    for row in rows:
        source_key = digest_bytes(json_bytes(row))
        receipt_name = f"{prefix}/receipts/{source_key}.json"
        receipt_blob = bucket.blob(receipt_name)
        if receipt_blob.exists():
            receipt_blob.reload()
            receipt = json.loads(receipt_blob.download_as_bytes(
                if_generation_match=int(receipt_blob.generation), checksum="crc32c"))
            if receipt.get("source") != row or receipt.get("configuration_sha256") != config_sha:
                raise ValueError("Existing source receipt conflicts with input or worker configuration")
            for ref in receipt["outputs"]:
                verify_output(client, ref)
        else:
            source_bucket, source_name = parse_gs(row["uri"])
            source_blob = client.bucket(source_bucket).blob(source_name, generation=int(row["generation"]))
            source_blob.reload()
            if str(source_blob.generation) != str(row["generation"]) or int(source_blob.size) != row["bytes"] or source_blob.crc32c != row["crc32c"]:
                raise ValueError("Pinned source object metadata mismatch")
            with tempfile.TemporaryDirectory(prefix="source-", dir=scratch) as work:
                if args.metadata_only:
                    with authorized_range_source(row, credentials) as path:
                        report = inspect_raster(path, metadata_only=True,
                            original_mask_provenance=row.get("mask_provenance"),
                            export_recipe=row.get("export_recipe"))
                else:
                    path = Path(work) / "original.tif"
                    source_blob.download_to_filename(str(path), if_generation_match=int(row["generation"]), checksum="crc32c")
                    if path.stat().st_size != row["bytes"]:
                        raise ValueError("Downloaded source byte size mismatch")
                    report = inspect_raster(path, preview_directory=work,
                        preview_samples=args.preview_samples,
                        original_mask_provenance=row.get("mask_provenance"),
                        export_recipe=row.get("export_recipe"))
                outputs = []
                for preview in report.pop("preview_files"):
                    name = f"{prefix}/previews/{source_key}/{Path(preview).name}"
                    ref = create_only(bucket.blob(name), Path(preview).read_bytes())
                    outputs.append(dict(uri=f"gs://{bucket_name}/{name}", **ref))
                report["source"] = row
                report["preview_objects"] = outputs.copy()
                report["configuration"] = configuration
                name = f"{prefix}/sources/{source_key}.json"
                ref = create_only(bucket.blob(name), json_bytes(report))
                outputs.append(dict(uri=f"gs://{bucket_name}/{name}", **ref))
                receipt = dict(schema_version=SCHEMA, source=row,
                               configuration_sha256=config_sha, outputs=outputs,
                               geometric_candidate_count=report["geometric_candidate_count"],
                               mask_qualified_candidate_count=report["mask_qualified_candidate_count"],
                               conservative_mask_qualified_candidate_count=report["conservative_mask_qualified_candidate_count"],
                               accepted_semantic_count=0)
                create_only(receipt_blob, json_bytes(receipt))
        completed.append(dict(uri=f"gs://{bucket_name}/{receipt_name}", source_id=row["source_id"]))
        checkpoint = dict(instance_id=instance, configuration_sha256=config_sha,
                          completed=completed)
        temporary = scratch / "checkpoint.json.tmp"
        temporary.write_bytes(json_bytes(checkpoint))
        temporary.replace(scratch / "checkpoint.json")
        print(json.dumps(dict(source_id=row["source_id"], completed_sources=len(completed),
                              geometric_candidates=receipt["geometric_candidate_count"],
                              mask_qualified_candidates=receipt["mask_qualified_candidate_count"])))
    summary = dict(schema_version=SCHEMA, configuration=configuration,
                   selected_manifest_sha256=digest_bytes(json_bytes(rows)),
                   completed_sources=len(completed), receipts=completed,
                   accepted_semantic_count=0)
    create_only(bucket.blob(f"{prefix}/summary.json"), json_bytes(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--preview-samples", type=int, default=3,
                        help="Bounded overview/native detail pairs per source (default: 3)")
    args = parser.parse_args()
    if args.preview_samples < 0:
        parser.error("preview-samples cannot be negative")
    run(args)


if __name__ == "__main__":
    main()
