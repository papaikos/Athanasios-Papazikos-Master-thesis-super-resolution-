"""Training-only calibration diagnosis; never creates or admits training pairs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter

from .data import CalibrationError, fit_calibration, stable_land
from .export_ee import read_raster


def diagnose(manifest_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status", "complete") != "complete":
        raise ValueError("Complete downloaded manifest required")
    records = [
        r
        for r in manifest["records"]
        if r.get("split") == "train" and "rejected" not in r
    ]
    groups = []
    for record in records:
        dn, dm = read_raster(
            manifest_path.parent, record["files"]["naip_lr"], record, "lr", 128, 5
        )
        sr, sm = read_raster(
            manifest_path.parent, record["files"]["s2_lr"], record, "lr", 128, 6
        )
        support = stable_land(
            sr[4], dm & sm & (dn[4] >= 0.999999) & (sr[5] >= 0.999999)
        )
        groups.append((record["id"], dn[:4], sr[:4], support))
    if not groups:
        raise ValueError("No training records")
    results = []
    # sigma zero reproduces production; additional blur diagnoses PSF mismatch.
    # These exploratory results do not select a mapping or use holdout geography.
    for sigma in (0, 0.5, 1):
        prepared = []
        for site, dn, sr, support in groups:
            if sigma:
                radius = int(4 * sigma + 0.5)
                support = binary_erosion(support, iterations=radius, border_value=0)
                dn = gaussian_filter(
                    dn, sigma=(0, sigma, sigma), truncate=4, mode="reflect"
                )
            prepared.append((site, dn, sr, support))
        for selected in [[item] for item in prepared] + [prepared]:
            sites = [item[0] for item in selected]
            try:
                result = fit_calibration(
                    np.concatenate([item[1] for item in selected], axis=2),
                    np.concatenate([item[2] for item in selected], axis=2),
                    np.concatenate([item[3] for item in selected], axis=1),
                    split="train",
                    site_ids=sites,
                )
                result = {"accepted": True, **result}
            except CalibrationError as error:
                result = {"accepted": False, **error.diagnostics}
            except ValueError as error:
                result = {"accepted": False, "error": str(error), "train_sites": sites}
            results.append({"extra_blur_sigma_lr": sigma, **result})
    return {"purpose": "training_only_diagnosis_not_pair_admission", "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(diagnose(args.manifest), indent=2))
