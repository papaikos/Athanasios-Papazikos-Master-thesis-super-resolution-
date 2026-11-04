"""RGBN pilot data contracts. Arrays are channel-first; NAIP input is DN."""

import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.optimize import least_squares

BANDS = ("R", "G", "B", "N")
S2_BANDS = ("B4", "B3", "B2", "B8")


class CalibrationError(ValueError):
    """Rejected mapping with measured diagnostics, never an accepted calibration."""

    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(
            "Unstable calibration: " + json.dumps(diagnostics, sort_keys=True)
        )


def aligned_grids(transform, x, y):
    a, b, c, d, e, f = transform
    if not np.allclose([a, b, d, e], [10, 0, 0, -10]):
        raise ValueError("Require north-up 10 m Sentinel-2 projection")
    col, row = int(np.floor((x - c) / a)) - 64, int(np.floor((y - f) / e)) - 64
    origin_x, origin_y = c + col * a, f + row * e
    return {
        "lr": [10, 0, origin_x, 0, -10, origin_y],
        "hr": [2.5, 0, origin_x, 0, -2.5, origin_y],
    }


def validate_sites(sites):
    seen = set()
    source_splits = {}
    for site in sites:
        if site["id"] in seen or site["split"] not in ("train", "validation", "test"):
            raise ValueError("Sites must have unique IDs and one geographic split")
        seen.add(site["id"])
        for kind in ("naip_id", "s2_id"):
            if kind not in site:
                continue
            key = (kind, site[kind])
            if key in source_splits and source_splits[key] != site["split"]:
                raise ValueError("Source scene cannot cross geographic splits")
            source_splits[key] = site["split"]


def stable_land(scl, valid):
    return np.isin(scl, [4, 5]) & np.asarray(valid, bool)


def area4(image):
    c, h, w = image.shape
    if h % 4 or w % 4:
        raise ValueError("Dimensions must be divisible by four")
    return image.reshape(c, h // 4, 4, w // 4, 4).mean(axis=(2, 4))


def degrade(hr, valid):
    hr = np.asarray(hr, np.float32)
    if hr.shape != (4, 512, 512) or valid.shape != (512, 512):
        raise ValueError("Expected RGBN 512 HR and spatial validity mask")
    valid = np.asarray(valid, bool) & np.isfinite(hr).all(axis=0)
    # Truncated sigma=2 kernel has radius 8. Require its entire support.
    safe = binary_erosion(valid, structure=np.ones((17, 17)), border_value=0)
    blurred = gaussian_filter(
        np.where(valid[None], hr, 0), sigma=(0, 2, 2), truncate=4, mode="reflect"
    )
    lr = area4(blurred)
    mask = safe.reshape(128, 4, 128, 4).all(axis=(1, 3))
    return lr.astype(np.float32), mask


def fit_calibration(
    dn, sr, support, *, split, site_ids, min_support=512, max_rmse=0.04
):
    if split != "train" or not site_ids:
        raise ValueError("Calibration fitting requires named training sites only")
    dn, sr = np.asarray(dn, float), np.asarray(sr, float)
    if dn.shape != sr.shape or dn.shape[0] != 4 or support.shape != dn.shape[1:]:
        raise ValueError("RGBN calibration arrays and support must match")
    support = (
        np.asarray(support, bool)
        & np.isfinite(dn).all(axis=0)
        & np.isfinite(sr).all(axis=0)
    )
    if support.sum() < min_support:
        raise ValueError("Insufficient stable-land calibration support")
    gains, offsets, residuals, bands = [], [], [], []
    for band, x, y in zip(BANDS, dn[:, support], sr[:, support]):
        dynamic_range = float(np.percentile(x, 95) - np.percentile(x, 5))
        detail = {
            "band": band,
            "support": int(len(x)),
            "dn_p95_minus_p05": dynamic_range,
        }
        bands.append(detail)
        if dynamic_range < 10:
            detail["rejected"] = "Unstable DN dynamic range"
            continue
        fitted = least_squares(
            lambda p: p[0] * x + p[1] - y,
            [0.002, 0.01],
            bounds=(0, np.inf),
            loss="soft_l1",
            f_scale=0.01,
        )
        residual = fitted.x[0] * x + fitted.x[1] - y
        inliers = np.abs(residual) <= max(
            0.02, 3 * 1.4826 * np.median(np.abs(residual - np.median(residual)))
        )
        detail.update(
            gain=float(fitted.x[0]),
            offset=float(fitted.x[1]),
            inliers=int(inliers.sum()),
            inlier_fraction=float(inliers.mean()),
        )
        if inliers.sum() < min_support or inliers.mean() < 0.7:
            detail["rejected"] = "Unstable calibration inlier support"
            continue
        fitted = least_squares(
            lambda p: p[0] * x[inliers] + p[1] - y[inliers],
            fitted.x,
            bounds=(0, np.inf),
        )
        rmse = float(
            np.sqrt(np.mean((fitted.x[0] * x[inliers] + fitted.x[1] - y[inliers]) ** 2))
        )
        detail.update(
            gain=float(fitted.x[0]),
            offset=float(fitted.x[1]),
            rmse=rmse,
            optimizer_success=bool(fitted.success),
        )
        if not fitted.success or fitted.x[0] <= 1e-8 or rmse > max_rmse:
            detail["rejected"] = "Unstable calibration residuals or gain"
        gains.append(float(fitted.x[0]))
        offsets.append(float(fitted.x[1]))
        residuals.append(rmse)
    if any("rejected" in band for band in bands):
        raise CalibrationError(
            {
                "train_sites": sorted(set(site_ids)),
                "support": int(support.sum()),
                "max_rmse": max_rmse,
                "bands": bands,
            }
        )
    return {
        "gain": gains,
        "offset": offsets,
        "rmse": residuals,
        "train_sites": sorted(set(site_ids)),
        "support": int(support.sum()),
        "method": "nonnegative_affine_soft_l1_then_inliers",
        "diagnostics": bands,
    }


def apply_calibration(dn, calibration):
    return (
        np.asarray(dn, np.float32) * np.asarray(calibration["gain"])[:, None, None]
        + np.asarray(calibration["offset"])[:, None, None]
    ).astype(np.float32)


def calibration_group_assignments(records):
    """Assign unseen acquisitions by geography only, before reading their pixels."""
    training = [r for r in records if r["split"] == "train"]
    if not training:
        raise ValueError("No training scenes to fit calibration")
    for record in records:
        if not (-180 <= record["lon"] <= 180 and -90 <= record["lat"] <= 90):
            raise ValueError("Invalid geographic calibration metadata")

    def distance(a, b):
        lat1, lat2 = np.deg2rad([a["lat"], b["lat"]])
        dlon = np.deg2rad(a["lon"] - b["lon"])
        return float(
            np.sin((lat1 - lat2) / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )

    return {
        record["id"]: record["naip_id"]
        if record["split"] == "train"
        else min(training, key=lambda train: (distance(record, train), train["id"]))[
            "naip_id"
        ]
        for record in records
    }


def fit_acquisition_group(dn, sr, support, *, site_ids):
    """Two-way spatial training checks, then fit a fixed acquisition mapping."""
    rows, cols = np.indices(support.shape)
    partition = ((rows // 32 + cols // 32) % 2).astype(bool)
    checks = []
    for fold in (False, True):
        fitted = fit_calibration(
            dn, sr, support & (partition == fold), split="train", site_ids=site_ids
        )
        heldout = support & (partition != fold)
        residual = apply_calibration(dn, fitted) - sr
        stable = heldout & np.isfinite(residual).all(axis=0)
        good = stable & (np.abs(residual).max(axis=0) <= 0.08)
        detail = {
            "fit_fold": int(fold),
            "heldout_support": int(stable.sum()),
            "heldout_good": int(good.sum()),
            "heldout_good_fraction": float(good.sum() / max(1, stable.sum())),
            "heldout_rmse": [float(np.sqrt(np.mean(r[stable] ** 2))) for r in residual]
            if stable.any()
            else [],
        }
        if (
            stable.sum() < 512
            or good.sum() < 512
            or detail["heldout_good_fraction"] < 0.7
        ):
            raise CalibrationError(
                {
                    "train_sites": site_ids,
                    "rejected": "Spatial heldout training calibration gate failed",
                    "check": detail,
                }
            )
        checks.append(detail)
    result = fit_calibration(dn, sr, support, split="train", site_ids=site_ids)
    result["spatial_training_checks"] = checks
    result["spatial_partition"] = "32px_checkerboard_two_way"
    return result


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_pair(path, hr, valid, provenance):
    required = {"site_id", "split", "naip_id", "s2_id", "calibration_train_sites"}
    if not required <= provenance.keys():
        raise ValueError("Missing pair provenance")
    if (
        provenance["split"] != "train"
        and provenance["site_id"] in provenance["calibration_train_sites"]
    ):
        raise ValueError("Holdout site leaked into calibration")
    lr, lr_valid = degrade(hr, valid)
    hr_valid = np.asarray(valid, bool) & np.isfinite(hr).all(axis=0)
    if not hr_valid.all():
        raise ValueError("Pilot pairs require complete HR coverage for global FFT")
    metadata = dict(
        provenance,
        bands=list(BANDS),
        hr_m=2.5,
        lr_m=10,
        lr_origin="synthetic_hr_only",
        gaussian_sigma_hr=2,
        gaussian_truncate=4,
        gaussian_mode="reflect",
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            hr=np.where(hr_valid[None], hr, 0).astype(np.float32),
            lr=lr,
            hr_mask=hr_valid,
            lr_mask=lr_valid,
            provenance=json.dumps(metadata, sort_keys=True),
        )
    return sha256(path)
