import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.data import (
    CalibrationError,
    fit_calibration,
    fit_acquisition_group,
    calibration_group_assignments,
)
from src.diagnose_calibration import diagnose


class CalibrationDiagnosticsTests(unittest.TestCase):
    def test_group_assignments_use_geography_not_pixel_metrics(self):
        records = [
            {
                "id": "south",
                "split": "train",
                "naip_id": "naip/south",
                "lon": -117,
                "lat": 32,
            },
            {
                "id": "north",
                "split": "train",
                "naip_id": "naip/north",
                "lon": -119,
                "lat": 34,
            },
            {
                "id": "sf",
                "split": "validation",
                "naip_id": "naip/sf",
                "lon": -122,
                "lat": 37,
            },
            {
                "id": "texas",
                "split": "test",
                "naip_id": "naip/texas",
                "lon": -94,
                "lat": 29,
            },
        ]
        result = calibration_group_assignments(records)
        self.assertEqual(result["sf"], "naip/north")
        self.assertEqual(result["texas"], "naip/south")
        self.assertEqual(result["north"], "naip/north")

    def test_group_checks_detect_spatially_different_mapping(self):
        rng = np.random.default_rng(11)
        dn = rng.uniform(20, 220, (4, 128, 128))
        support = np.ones((128, 128), bool)
        good = fit_acquisition_group(dn, 0.002 * dn + 0.01, support, site_ids=["train"])
        self.assertEqual(len(good["spatial_training_checks"]), 2)
        self.assertEqual(good["spatial_training_checks"][0]["heldout_support"], 8192)
        rows, cols = np.indices(support.shape)
        sr = 0.002 * dn + 0.01 + 0.2 * ((rows // 32 + cols // 32) % 2)[None]
        with self.assertRaisesRegex(CalibrationError, "Spatial heldout"):
            fit_acquisition_group(dn, sr, support, site_ids=["train"])

    def test_diagnosis_never_reads_holdout_rasters(self):
        records = [
            {"id": split, "split": split, "files": {"naip_lr": "dn", "s2_lr": "sr"}}
            for split in ("train", "validation", "test")
        ]
        rng = np.random.default_rng(19)
        dn = rng.uniform(20, 220, (4, 128, 128))

        def read(root, item, record, grid, size, bands):
            self.assertEqual(record["split"], "train")
            values = np.ones((bands, size, size))
            values[:4] = dn if item == "dn" else 0.002 * dn + 0.01
            if item == "sr":
                values[4] = 4
            return values, np.ones((size, size), bool)

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps({"records": records}), encoding="utf-8")
            with patch(
                "src.diagnose_calibration.read_raster", side_effect=read
            ) as reader:
                result = diagnose(manifest)
            self.assertEqual(reader.call_count, 2)
        self.assertEqual(len(result["results"]), 6)
        self.assertTrue(result["results"][0]["accepted"])

    def test_failure_reports_all_bands_and_keeps_threshold(self):
        rng = np.random.default_rng(731)
        dn = rng.uniform(20, 220, (4, 32, 32))
        sr = 0.002 * dn + 0.01
        sr[0] += rng.uniform(-0.15, 0.15, (32, 32))
        with self.assertRaises(CalibrationError) as caught:
            fit_calibration(
                dn, sr, np.ones((32, 32), bool), split="train", site_ids=["train"]
            )
        diagnostics = caught.exception.diagnostics
        self.assertEqual(diagnostics["max_rmse"], 0.04)
        self.assertEqual(
            [b["band"] for b in diagnostics["bands"]], ["R", "G", "B", "N"]
        )
        self.assertGreater(diagnostics["bands"][0]["rmse"], 0.04)
        self.assertIn("rejected", diagnostics["bands"][0])
        for band in diagnostics["bands"][1:]:
            self.assertNotIn("rejected", band)
            self.assertAlmostEqual(band["gain"], 0.002)
            self.assertEqual(band["inliers"], 1024)

    def test_dynamic_range_failure_preserves_other_band_diagnostics(self):
        rng = np.random.default_rng(5)
        dn = rng.uniform(20, 220, (4, 32, 32))
        dn[1] = 50
        with self.assertRaises(CalibrationError) as caught:
            fit_calibration(
                dn,
                0.002 * dn + 0.01,
                np.ones((32, 32), bool),
                split="train",
                site_ids=["train"],
            )
        self.assertEqual(
            caught.exception.diagnostics["bands"][1]["dn_p95_minus_p05"], 0
        )
        self.assertEqual(len(caught.exception.diagnostics["bands"]), 4)
