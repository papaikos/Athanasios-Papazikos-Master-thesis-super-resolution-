"""GCE-only native mother candidates from explicitly selected original assets.

Outputs require semantic and quality review. No child datasets, ready markers,
vessel confirmations, or accepted useful-byte totals are produced.
"""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import rasterio
from rasterio.enums import MaskFlags
from rasterio.windows import Window, bounds, transform

from scripts.maritime_cloud_inventory import (create_only, digest_bytes,
    file_sha256, json_bytes, parse_gs, require_cloud_vm, validate_header,
    validate_manifest, verify_output)
from scripts.maritime_pilot import core_identity, core_window


def approved_asset_ids(document):
    assets = document.get("asset_ids") if isinstance(document, dict) else None
    if not isinstance(assets, list) or not assets or any(
        not isinstance(asset, str) or not asset for asset in assets):
        raise ValueError("Approval metadata must contain explicit nonempty asset_ids")
    return set(assets)


def source_rows(document, approval):
    if isinstance(document, list):
        rows = document
    elif isinstance(document, dict):
        if "sources" in document:
            rows = document["sources"]
        elif "source_objects" in document:
            rows = document["source_objects"]
        elif isinstance(document.get("assets"), dict):
            rows = [row for asset in document["assets"].values()
                    for row in asset.get("source_objects", [])]
        else:
            raise ValueError("Source document needs sources, source_objects or scheduler assets")
    else:
        raise ValueError("Invalid source document")
    if not isinstance(rows, list):
        raise ValueError("Source rows must be an array")
    groups = approval.get("review_metadata", {}).get("groups", [])
    by_asset = {}
    for group in groups:
        for asset in group.get("asset_ids", []):
            if asset in by_asset and by_asset[asset].get("group_id") != group.get("group_id"):
                raise ValueError("Asset belongs to conflicting approval groups")
            by_asset[asset] = group
    normalized = []
    for original in rows:
        row = dict(original)
        if "producer_mask" in row:
            pin = row["producer_mask"]
            if not isinstance(pin, dict) or not {"uri", "generation", "bytes", "crc32c"}.issubset(pin):
                raise ValueError("Producer mask requires pinned URI, generation, bytes and CRC32C")
            parse_gs(pin["uri"])
            if not str(pin["generation"]).isdigit() or int(pin["generation"]) <= 0 or not pin["crc32c"]:
                raise ValueError("Producer mask requires positive generation and CRC32C")
            if isinstance(pin["bytes"], bool) or not isinstance(pin["bytes"], int) or pin["bytes"] <= 0:
                raise ValueError("Producer mask bytes must be a positive integer")
            plan = row.get("native_plan", {})
            row.setdefault("grid", plan)
            if "geometry" not in row:
                left, bottom, right, top = plan["native_bbox"]
                row["geometry"] = dict(type="Polygon", coordinates=[[
                    [left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]],
                    crs=dict(type="name", properties=dict(name=plan["crs"])))
            row["grid_metadata_scope"] = "producer_plan_tile_header_verified_during_extraction"
        group = by_asset.get(row.get("asset_id"))
        if group:
            row["approval_group"] = group
            row.setdefault("site_id", group.get("group_id"))
            row.setdefault("country", group.get("country"))
        normalized.append(row)
    return validate_manifest(normalized) if normalized else []


def validate_producer_mask(ds, mask):
    if (mask.count != 1 or mask.dtypes != ("uint8",) or mask.width != ds.width
        or mask.height != ds.height or mask.crs != ds.crs or mask.transform != ds.transform):
        raise ValueError("Producer mask must exactly match RGB shape/grid with one uint8 band")
    for _, window in mask.block_windows(1):
        values = mask.read(1, window=window)
        if not np.all((values == 0) | (values == 255)):
            raise ValueError("Original producer mask must be binary 0/255")


def support_mask(ds, window, original_mask_ds=None):
    if original_mask_ds is not None:
        return original_mask_ds.read(window=window)
    return np.min(ds.read_masks(window=window), axis=0, keepdims=True)


def supported_windows(ds, source, original_mask_ds=None, *, statistics=None):
    """One world 1280 m grid, snapped to original pixels without any warp."""
    validate_header(ds)
    flags = ds.mask_flag_enums
    explicit = all(MaskFlags.per_dataset in band for band in flags)
    producer = original_mask_ds is not None
    if producer:
        if source.get("mask_provenance") != "producer_original_min_rgb_mask":
            raise ValueError("Producer mask original provenance must be documented")
        validate_producer_mask(ds, original_mask_ds)
    original_known = producer or (explicit and source.get("mask_provenance") == "original_explicit_dataset_mask")
    recipe = source.get("export_recipe", {})
    if not isinstance(recipe, dict):
        recipe = {}
    conservative = (ds.nodata == 0 and all(MaskFlags.nodata in band for band in flags)
                    and recipe.get("unmask") == 0 and recipe.get("noData") == 0)
    if not (original_known or conservative):
        return
    provenance = "producer_original_min_rgb_mask" if producer else (
        "original_explicit_dataset_mask" if original_known else "legacy_nodata_zero_conservative")
    statistics = statistics if statistics is not None else {}
    for key in ("partial_world_cores", "missing_real_halo", "invalid_core_or_halo_mask", "fully_supported_cores"):
        statistics.setdefault(key, 0)
    extent = ds.bounds
    for y in reversed(range(math.floor(extent.bottom / 1280), math.ceil(extent.top / 1280))):
        for x in range(math.floor(extent.left / 1280), math.ceil(extent.right / 1280)):
            core, shift = core_window(ds.transform, (x, y))
            if max(abs(value) for value in shift) > .4 + 1e-6:
                raise ValueError("World-grid snapping exceeds 0.4 metre")
            col, row = int(core.col_off), int(core.row_off)
            if min(col, row) < 0 or col + 1600 > ds.width or row + 1600 > ds.height:
                statistics["partial_world_cores"] += 1
                continue
            if min(col, row) < 50 or col + 1650 > ds.width or row + 1650 > ds.height:
                statistics["missing_real_halo"] += 1
                continue
            context = Window(col - 50, row - 50, 1700, 1700)
            masks = support_mask(ds, context, original_mask_ds)
            if masks.shape != (1, 1700, 1700) or not np.all(masks == 255):
                statistics["invalid_core_or_halo_mask"] += 1
                continue
            statistics["fully_supported_cores"] += 1
            crs = ds.crs.to_string()
            yield dict(acquisition_id=source["catalog_id"], crs=crs,
                       global_core_identity=core_identity(source["catalog_id"], crs, ds.transform, core),
                       world_grid_cell=[x, y], native_snap_metres=shift,
                       core_window=[col, row, 1600, 1600],
                       context_window=[col - 50, row - 50, 1700, 1700],
                       core_bounds=list(bounds(core, ds.transform)),
                       context_bounds=list(bounds(context, ds.transform)),
                       core_transform=list(transform(core, ds.transform))[:6],
                       context_transform=list(transform(context, ds.transform))[:6],
                       mask_provenance=provenance, original_mask_known=original_known,
                       support_policy="producer_mask_equal_255_core_and_halo" if producer else "all_original_read_masks_equal_255_core_and_halo",
                       source_processing="native_pixels_only_no_warp_no_resampling",
                       status="pending_semantic_and_quality_review",
                       vessel_presence="unconfirmed", semantic_accepted=False)


def overlaps_existing(candidate, existing):
    left, bottom, right, top = candidate["core_bounds"]
    for other in existing:
        if (candidate["acquisition_id"], candidate["crs"]) != (other["acquisition_id"], other["crs"]):
            continue
        a, b, c, d = other["core_bounds"]
        if min(right, c) - max(left, a) > 1e-6 and min(top, d) - max(bottom, b) > 1e-6:
            return True
    return False


def materialize(ds, candidate, directory, original_mask_ds=None):
    from PIL import Image
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    context = Window(*candidate["context_window"])
    rgb = ds.read(window=context)
    masks = support_mask(ds, context, original_mask_ds)
    if rgb.shape != (3, 1700, 1700) or masks.shape != (1, 1700, 1700) or not np.all(masks == 255):
        raise ValueError("Candidate lost full original RGB mask support")
    affine = transform(context, ds.transform)
    profile = dict(driver="GTiff", width=1700, height=1700, dtype="uint8",
                   crs=ds.crs, transform=affine, compress="DEFLATE", tiled=True,
                   blockxsize=256, blockysize=256)
    paths = dict(rgb=directory / "rgb-context.tif", mask=directory / "validity-context.tif",
                 native_preview=directory / "native-core.png", overview=directory / "overview.png")
    with rasterio.open(paths["rgb"], "w", count=3, photometric="RGB", predictor=2, **profile) as dst:
        dst.write(rgb)
    with rasterio.open(paths["mask"], "w", count=1, **profile) as dst:
        dst.write(masks)
    core = np.moveaxis(rgb[:, 50:1650, 50:1650], 0, -1)
    native = Image.fromarray(core)
    native.save(paths["native_preview"])
    native.resize((400, 400), Image.Resampling.BOX).save(paths["overview"])
    return paths


def download_pinned(client, pin, path):
    bucket, name = parse_gs(pin["uri"])
    if not str(pin["generation"]).isdigit() or int(pin["generation"]) <= 0:
        raise ValueError("Pinned raw/mask generation must be positive")
    blob = client.bucket(bucket).blob(name, generation=int(pin["generation"]))
    blob.reload()
    if str(blob.generation) != str(pin["generation"]) or int(blob.size) != pin["bytes"] or blob.crc32c != pin["crc32c"]:
        raise ValueError("Pinned raw/mask source metadata mismatch")
    blob.download_to_filename(str(path), if_generation_match=int(pin["generation"]), checksum="crc32c")
    if path.stat().st_size != pin["bytes"]:
        raise ValueError("Raw/mask source byte size mismatch")
    actual_sha = file_sha256(path)
    if pin.get("sha256") and actual_sha != pin["sha256"]:
        raise ValueError("Pinned raw/mask source SHA-256 mismatch")
    return actual_sha


def run(args):
    instance = require_cloud_vm()
    from google.cloud import storage
    approval = json.loads(Path(args.approved_assets).read_text(encoding="utf-8-sig"))
    sources = source_rows(json.loads(Path(args.sources).read_text(encoding="utf-8-sig")), approval)
    approved = approved_asset_ids(approval)
    selected = [source for source in sources if source["asset_id"] in approved]
    missing = approved - {source["asset_id"] for source in selected}
    allow_missing = getattr(args, "allow_missing_approved", False)
    if (missing or not selected) and not allow_missing:
        raise ValueError("Approved assets missing from pinned source inventory")
    if args.max_candidates < 1:
        raise ValueError("max-candidates must be positive")
    # Fail before any transport when the pinned VM runtime cannot render previews.
    from PIL import Image  # noqa: F401
    client = storage.Client()
    bucket_name, prefix = parse_gs(args.output_prefix.rstrip("/"))
    bucket = client.bucket(bucket_name)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    config = dict(worker_sha256=file_sha256(__file__),
                  source_manifest_sha256=digest_bytes(json_bytes(sources)),
                  approvals_sha256=digest_bytes(json_bytes(approval)),
                  allow_missing_approved=allow_missing,
                  max_candidates=args.max_candidates, core_size=1600, halo=50,
                  world_grid_metres=1280, status="pending_review_only")
    config_sha = digest_bytes(json_bytes(config))
    candidates, reports, support_totals = [], [], {}

    def publish(name, payload):
        return dict(uri=f"gs://{bucket_name}/{prefix}/{name}",
                    **create_only(bucket.blob(f"{prefix}/{name}"), payload))

    def checkpoint():
        snapshot = dict(configuration_sha256=config_sha, instance_id=instance,
                        candidate_count=len(candidates), processed_sources=len(reports),
                        source_reports=reports, status="pending_review_only")
        path = scratch / "checkpoint.json.tmp"
        path.write_bytes(json_bytes(snapshot))
        path.replace(scratch / "checkpoint.json")

    for source in selected:
        if len(candidates) >= args.max_candidates:
            break
        source_key = digest_bytes(json_bytes(source))
        receipt_name = f"receipts/{source_key}.json"
        receipt_blob = bucket.blob(f"{prefix}/{receipt_name}")
        if receipt_blob.exists():
            receipt_blob.reload()
            receipt = json.loads(receipt_blob.download_as_bytes(
                if_generation_match=int(receipt_blob.generation), checksum="crc32c"))
            if receipt.get("configuration_sha256") != config_sha or receipt.get("source") != source:
                raise ValueError("Existing source receipt configuration conflict")
            report = json.loads(verify_output(client, receipt["source_report"]))
            for candidate in report["candidates"]:
                for artifact in candidate["artifacts"].values():
                    verify_output(client, artifact)
                if overlaps_existing(candidate, candidates):
                    raise ValueError("Resumed candidates overlap earlier acquisition cores")
                candidates.append(candidate)
            if len(candidates) > args.max_candidates:
                raise ValueError("Resumed candidate count exceeds configured cap")
            report_ref = receipt["source_report"]
        else:
            report = dict(source=source, candidates=[], status="completed",
                          configuration_sha256=config_sha, skipped_overlaps=0,
                          exhaustive_source_scan=True, support_counts={},
                          halo_boundary_policy="all_1700_pixels_inside_one_original_rgb_and_mask_shard")
            with tempfile.TemporaryDirectory(prefix="raw-", dir=scratch) as work:
                path = Path(work) / "original.tif"
                report["source_sha256"] = download_pinned(client, source, path)
                mask_path = None
                if source.get("producer_mask"):
                    mask_path = Path(work) / "producer-mask.tif"
                    report["producer_mask_sha256"] = download_pinned(client, source["producer_mask"], mask_path)
                with ExitStack() as readers:
                    ds = readers.enter_context(rasterio.open(path))
                    original_mask_ds = readers.enter_context(rasterio.open(mask_path)) if mask_path else None
                    try:
                        validate_header(ds)
                        if original_mask_ds is not None:
                            validate_producer_mask(ds, original_mask_ds)
                    except ValueError as exc:
                        report.update(status="rejected_header", rejection_reason=str(exc))
                    else:
                        report["native_header"] = dict(crs=ds.crs.to_string(),
                            affine=list(ds.transform)[:6], width=ds.width, height=ds.height,
                            mask_flags=[[flag.name for flag in band] for band in ds.mask_flag_enums])
                        for candidate in supported_windows(ds, source, original_mask_ds,
                                                           statistics=report["support_counts"]):
                            if overlaps_existing(candidate, candidates):
                                report["skipped_overlaps"] += 1
                                continue
                            if len(candidates) >= args.max_candidates:
                                report["exhaustive_source_scan"] = False
                                break
                            sample_id = digest_bytes(json_bytes(candidate["global_core_identity"]))
                            paths = materialize(ds, candidate, Path(work) / "candidate", original_mask_ds)
                            candidate["artifacts"] = {kind: publish(
                                f"candidates/{sample_id}/{path.name}", path.read_bytes())
                                for kind, path in paths.items()}
                            candidate.update(sample_id=sample_id, source_object=source,
                                             source_sha256=report["source_sha256"],
                                             acquisition_utc=source.get("acquisition_utc", source.get("timestamp")),
                                             site_id=source.get("site_id"),
                                             country=source.get("country"),
                                             approval_group=source.get("approval_group"),
                                             producer_mask_sha256=report.get("producer_mask_sha256"),
                                             license=source.get("license", "CC-BY-4.0"),
                                             provider=source.get("provider", "Google Earth Engine"))
                            candidates.append(candidate)
                            report["candidates"].append(candidate)
                            if len(candidates) >= args.max_candidates:
                                report["exhaustive_source_scan"] = False
                                break
            report_ref = publish(f"source-reports/{source_key}.json", json_bytes(report))
            receipt = dict(configuration_sha256=config_sha, source=source,
                           source_report=report_ref, candidate_count=len(report["candidates"]))
            publish(receipt_name, json_bytes(receipt))
        reports.append(report_ref)
        for key, count in report.get("support_counts", {}).items():
            support_totals[key] = support_totals.get(key, 0) + count
        checkpoint()
        print(json.dumps(dict(source_id=source["source_id"], source_status=report["status"],
                              candidate_count=len(candidates), processed_sources=len(reports))))
    summary = dict(configuration=config, candidate_count=len(candidates),
                   processed_source_count=len(reports), selected_source_count=len(selected),
                   source_reports=reports, candidates=candidates,
                   support_counts=support_totals,
                   missing_approved_assets=sorted(missing),
                   available_selected_sources_scan_complete=len(reports) == len(selected) and len(candidates) < args.max_candidates,
                   # A reached cap conservatively prevents completeness claims.
                   exhaustive=not allow_missing and not missing and len(reports) == len(selected) and len(candidates) < args.max_candidates,
                   status="pending_semantic_and_quality_review",
                   semantic_accepted_count=0, vessel_confirmation_count=0)
    publish("candidate-manifest.json", json_bytes(summary))
    checkpoint()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--approved-assets", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--allow-missing-approved", action="store_true",
                        help="Explicit partial screening; record unavailable approved assets and never claim archive completeness")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
