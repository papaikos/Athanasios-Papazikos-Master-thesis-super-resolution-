"""Stream generation-pinned coastal RGBN sources into verified synthetic pairs.

The input manifest assigns splits before pixels are read. Only source objects
that contribute accepted training patches count toward the source-byte minimum.
No compute resources or billing settings are changed by this command.
"""

import argparse
from contextlib import contextmanager
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Resampling
from rasterio.io import MemoryFile
from rasterio.shutil import copy as copy_raster
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, transform as window_transform
from rasterio.warp import reproject, transform_bounds

from src.data import (
    apply_calibration,
    area4,
    calibration_group_assignments,
    fit_acquisition_group,
    save_pair,
    sha256,
    stable_land,
    validate_sites,
)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def validate_manifest(manifest):
    records = manifest["records"]
    validate_sites(records)
    identities = set()
    for record in records:
        for kind in ("naip", "s2", "nativevalid"):
            item = record["files"][kind]
            if (
                not item["uri"].startswith("gs://")
                or not str(item["generation"]).isdigit()
                or int(item["bytes"]) <= 0
            ):
                raise ValueError(
                    "Sources must have positive bytes and pinned GCS generations"
                )
        identity = record["files"]["naip"]["uri"]
        if identity in identities:
            raise ValueError("Duplicate source object; cannot count bytes twice")
        identities.add(identity)
    return calibration_group_assignments(records)


def copy_source(item, destination):
    subprocess.run(
        [
            "gcloud",
            "storage",
            "cp",
            item["uri"] + "#" + str(item["generation"]),
            str(destination),
        ],
        check=True,
    )
    if destination.stat().st_size != int(item["bytes"]):
        raise ValueError("Downloaded source size differs from pinned manifest")
    if item.get("sha256") and sha256(destination) != item["sha256"]:
        raise ValueError("Downloaded source checksum mismatch")


def publish(path, prefix, scratch):
    uri = prefix.rstrip("/") + "/" + path.name
    subprocess.run(
        ["gcloud", "storage", "cp", "--if-generation-match=0", str(path), uri],
        # A retry may find the already-published immutable object. In either
        # case, the following download must prove exact byte identity.
        check=False,
    )
    readback = scratch / (path.name + ".readback")
    subprocess.run(["gcloud", "storage", "cp", uri, str(readback)], check=True)
    if sha256(path) != sha256(readback):
        raise ValueError("Uploaded artifact failed SHA-256 readback")
    readback.unlink()
    return uri


def complete_coverage(source, transform, crs):
    """Warp explicit zero/one support, so averages cannot hide missing pixels."""
    bounds = rasterio.transform.array_bounds(512, 512, transform)
    bounds = transform_bounds(crs, source.crs, *bounds, densify_pts=21)
    window = rasterio.windows.from_bounds(*bounds, transform=source.transform)
    left, top = int(np.floor(window.col_off)) - 2, int(np.floor(window.row_off)) - 2
    right = int(np.ceil(window.col_off + window.width)) + 2
    bottom = int(np.ceil(window.row_off + window.height)) + 2
    window = Window(left, top, right - left, bottom - top)
    mask = (source.read(1, window=window, boundless=True, fill_value=0) == 1).astype(
        "float32"
    )
    coverage = np.zeros((512, 512), "float32")
    reproject(
        mask,
        coverage,
        src_transform=window_transform(window, source.transform),
        src_crs=source.crs,
        dst_transform=transform,
        dst_crs=crs,
        resampling=Resampling.average,
        src_nodata=None,
        dst_nodata=None,
    )
    return coverage >= 0.999999


@contextmanager
def rgbn_source(path):
    """Remove erroneous NIR alpha interpretation without changing source bytes."""
    with rasterio.open(path) as source:
        if source.count != 4 or source.colorinterp[3] != ColorInterp.alpha:
            yield source
            return
        # A metadata-only VRT retains georeferencing, nodata and real masks.
        # src_alpha=0 cannot override WarpedVRT's alpha autodetection.
        with MemoryFile() as memory:
            copy_raster(source, memory.name, driver="VRT")
            with rasterio.open(memory.name, "r+") as normalized:
                normalized.colorinterp = (
                    *source.colorinterp[:3],
                    ColorInterp.undefined,
                )
            with memory.open() as normalized:
                yield normalized


def count_diagnostic(diagnostics, key):
    if diagnostics is not None:
        diagnostics[key] = diagnostics.get(key, 0) + 1


def candidate_patches(
    naip_path,
    s2_path,
    nativevalid_path,
    *,
    minimum_water=0.1,
    maximum_water=0.95,
    diagnostics=None,
):
    """Yield nonoverlapping patches in the original Sentinel-2 10 m grid.

    Optional counters record the first rejection per window; an early-stopped
    calibration traversal reports only windows examined, not the entire source.
    """
    with (
        rgbn_source(naip_path) as source,
        rasterio.open(s2_path) as reference,
        rasterio.open(nativevalid_path) as nativevalid,
    ):
        t = reference.transform
        if source.count != 4 or any(dtype != "uint8" for dtype in source.dtypes):
            raise ValueError("NAIP must be native uint8 RGBN")
        if (
            nativevalid.count != 1
            or nativevalid.dtypes != ("uint8",)
            or nativevalid.crs != source.crs
            or nativevalid.transform != source.transform
            or nativevalid.shape != source.shape
        ):
            raise ValueError(
                "Explicit native validity must match the original NAIP grid"
            )
        if reference.count != 6 or not np.allclose(
            [t.a, t.b, t.d, t.e], [10, 0, 0, -10]
        ):
            raise ValueError(
                "S2 must be RGBN reflectance/SCL/valid on north-up 10 m grid"
            )
        # Global pixel anchoring keeps neighboring source footprints on the same
        # disjoint grid, independent of each export's cropped raster origin.
        row_start = int(round(t.f / 10)) % 128
        col_start = -int(round(t.c / 10)) % 128
        for row in range(row_start, reference.height - 127, 128):
            for col in range(col_start, reference.width - 127, 128):
                count_diagnostic(diagnostics, "windows")
                window = Window(col, row, 128, 128)
                sr = reference.read(window=window)
                reference_valid = (
                    reference.read_masks(window=window).min(axis=0) > 0
                ) & (sr[5] >= 0.999999)
                water = float(((sr[4] == 6) & reference_valid).sum() / 16384)
                if not minimum_water <= water <= maximum_water:
                    count_diagnostic(diagnostics, "rejected_water")
                    continue
                lr_transform = window_transform(window, t)
                hr_transform = lr_transform * rasterio.Affine.scale(0.25, 0.25)
                # Alpha includes the original raster mask and out-of-footprint area.
                with WarpedVRT(
                    source,
                    crs=reference.crs,
                    transform=hr_transform,
                    width=512,
                    height=512,
                    resampling=Resampling.average,
                    dtype="float32",
                    add_alpha=True,
                ) as warped:
                    pixels = warped.read()
                    valid = (pixels[4] >= 255 - 1e-5) & np.isfinite(pixels[:4]).all(
                        axis=0
                    )
                if not valid.all():
                    count_diagnostic(diagnostics, "rejected_original_mask_or_footprint")
                    continue
                valid &= complete_coverage(nativevalid, hr_transform, reference.crs)
                if not valid.all():
                    count_diagnostic(diagnostics, "rejected_native_coverage")
                    continue
                hr = pixels[:4]
                dn = area4(hr)
                support = stable_land(sr[4], reference_valid)
                if support.sum() < 1024:
                    count_diagnostic(diagnostics, "rejected_support")
                    continue
                count_diagnostic(diagnostics, "yielded")
                yield {
                    "row": row,
                    "col": col,
                    "hr": hr,
                    "dn": dn,
                    "sr": sr[:4],
                    "support": support,
                    "valid": valid,
                    "water_fraction": water,
                    "crs": reference.crs.to_string(),
                    "hr_transform": list(hr_transform)[:6],
                }


def source_usage(outcomes):
    unique = {}
    for outcome in outcomes:
        if outcome["split"] == "train" and outcome.get("pairs"):
            source = outcome["source"]
            identity = source["uri"]
            if identity in unique and unique[identity] != source:
                raise ValueError("Inconsistent generations of a source object")
            unique[identity] = source
    return {
        "definition": "unique source objects contributing accepted training patches; excludes reference, validation, test and rejected sources",
        "bytes": sum(int(source["bytes"]) for source in unique.values()),
        "sources": list(unique.values()),
    }


def prepare(
    manifest_path,
    output,
    *,
    upload_prefix=None,
    minimum_source_bytes=100_000_000_000,
    calibration_patches=32,
):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    assignments = validate_manifest(manifest)  # frozen before source pixels
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "preparation-state.json"
    digest = sha256(manifest_path)
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "manifest_sha256": digest,
            "assignments": assignments,
            "groups": {},
            "outcomes": [],
        }
    )
    if state["manifest_sha256"] != digest or state["assignments"] != assignments:
        raise ValueError(
            "Cannot resume with different source manifest or split assignments"
        )
    records = sorted(
        manifest["records"], key=lambda r: (r["split"] != "train", r["id"])
    )
    done = {item["id"] for item in state["outcomes"] if item.get("complete")}
    used_cells = {
        tuple(pair["cell"]) for item in state["outcomes"] for pair in item["pairs"]
    }
    for record in records:
        if record["id"] in done:
            continue
        outcome = next(
            (item for item in state["outcomes"] if item["id"] == record["id"]), None
        )
        if outcome is None:
            outcome = {
                "id": record["id"],
                "split": record["split"],
                "source": record["files"]["naip"],
                "pairs": [],
            }
            state["outcomes"].append(outcome)
        # Only this invocation's disposable cache is removed. Accepted local pairs
        # remain unless their uploaded bytes have passed a complete readback.
        with tempfile.TemporaryDirectory(
            prefix="marine-source-", dir=output
        ) as temporary:
            scratch = Path(temporary)
            copy_source(record["files"]["naip"], scratch / "naip.tif")
            copy_source(record["files"]["s2"], scratch / "s2.tif")
            copy_source(record["files"]["nativevalid"], scratch / "nativevalid.tif")
            group = assignments[record["id"]]
            # Separate traversals: calibration may stop early or reuse a mapping.
            # Output counters describe this pass; accepted pairs remain cumulative.
            diagnostics = outcome.setdefault(
                "diagnostics", {"calibration": {}, "output": {}}
            )
            diagnostics["output"] = {}
            if record["split"] == "train" and group not in state["groups"]:
                diagnostics["calibration"] = {}
                calibration = []
                for patch in candidate_patches(
                    scratch / "naip.tif",
                    scratch / "s2.tif",
                    scratch / "nativevalid.tif",
                    diagnostics=diagnostics["calibration"],
                ):
                    calibration.append(
                        {key: patch[key] for key in ("dn", "sr", "support")}
                    )
                    if len(calibration) >= calibration_patches:
                        break
                try:
                    if not calibration:
                        raise ValueError("No complete coastal patches with stable land")
                    state["groups"][group] = fit_acquisition_group(
                        np.concatenate([p["dn"] for p in calibration], axis=2),
                        np.concatenate([p["sr"] for p in calibration], axis=2),
                        np.concatenate([p["support"] for p in calibration], axis=1),
                        site_ids=[record["id"]],
                    )
                except ValueError as error:
                    state["groups"][group] = {"rejected": str(error)}
            mapping = state["groups"].get(
                group, {"rejected": "Assigned training acquisition unavailable"}
            )
            if "rejected" in mapping:
                outcome["rejected"] = mapping["rejected"]
            else:
                for patch in candidate_patches(
                    scratch / "naip.tif",
                    scratch / "s2.tif",
                    scratch / "nativevalid.tif",
                    diagnostics=diagnostics["output"],
                ):
                    cell = (
                        patch["crs"],
                        patch["hr_transform"][2],
                        patch["hr_transform"][5],
                    )
                    if cell in used_cells:
                        count_diagnostic(diagnostics["output"], "rejected_duplicate")
                        continue
                    residual = apply_calibration(patch["dn"], mapping) - patch["sr"]
                    stable = patch["support"] & np.isfinite(residual).all(axis=0)
                    good = stable & (np.abs(residual).max(axis=0) <= 0.08)
                    if good.sum() < 512 or good.sum() / max(1, stable.sum()) < 0.7:
                        count_diagnostic(diagnostics["output"], "rejected_residual")
                        continue
                    pair_id = f"{record['id']}_{patch['row']}_{patch['col']}"
                    # Avoid untrusted source IDs becoming filesystem paths.
                    if not all(c.isalnum() or c in "_-" for c in pair_id):
                        raise ValueError("Unsafe source ID")
                    path = output / (pair_id + ".npz")
                    provenance = {
                        "site_id": pair_id,
                        "split": record["split"],
                        "naip_id": record["naip_id"],
                        "s2_id": record["s2_id"],
                        "calibration_train_sites": mapping["train_sites"],
                        "calibration": mapping,
                        "calibration_group": group,
                        "source_manifest_sha256": digest,
                        "source": outcome["source"],
                        "native_validity_source": record["files"]["nativevalid"],
                        "water_fraction": patch["water_fraction"],
                        "crs": patch["crs"],
                        "hr_transform": patch["hr_transform"],
                    }
                    checksum = save_pair(
                        path,
                        apply_calibration(patch["hr"], mapping),
                        patch["valid"],
                        provenance,
                    )
                    item = {
                        "site_id": pair_id,
                        "split": record["split"],
                        "path": path.name,
                        "cell": list(cell),
                        "sha256": checksum,
                        "bytes": path.stat().st_size,
                    }
                    if upload_prefix:
                        item["uri"] = publish(path, upload_prefix, scratch)
                        path.unlink()
                    count_diagnostic(diagnostics["output"], "accepted")
                    outcome["pairs"].append(item)
                    used_cells.add(cell)
                    write_json(state_path, state)
            outcome["complete"] = True
            state["source_usage"] = source_usage(state["outcomes"])
            write_json(state_path, state)
    pairs = [pair for outcome in state["outcomes"] for pair in outcome["pairs"]]
    index = output / "pairs.json"
    if index.exists():
        if json.loads(index.read_text()) != pairs:
            raise ValueError("Final immutable index differs from current results")
    else:
        write_json(index, pairs)
    usage = source_usage(state["outcomes"])
    usage.update(
        minimum_bytes=minimum_source_bytes,
        requirement_met=usage["bytes"] >= minimum_source_bytes,
        pairs_index_sha256=sha256(index),
        source_manifest_sha256=digest,
    )
    write_json(output / "source-usage.json", usage)
    if upload_prefix:
        with tempfile.TemporaryDirectory(
            prefix="marine-publish-", dir=output
        ) as temporary:
            for path in (index, output / "source-usage.json", state_path):
                publish(path, upload_prefix, Path(temporary))
    return usage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upload-prefix")
    parser.add_argument("--minimum-source-bytes", type=int, default=100_000_000_000)
    args = parser.parse_args()
    result = prepare(
        args.manifest,
        args.output,
        upload_prefix=args.upload_prefix,
        minimum_source_bytes=args.minimum_source_bytes,
    )
    print(json.dumps(result, indent=2))
    if not result["requirement_met"]:
        raise SystemExit(
            "Insufficient unique accepted TRAIN source bytes; do not launch training"
        )


if __name__ == "__main__":
    main()
