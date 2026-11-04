import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.export_ee import prepare


class RejectedGroupTests(unittest.TestCase):
    def records(self):
        return [
            {
                "id": site,
                "split": split,
                "naip_id": "naip/" + site,
                "s2_id": "s2/" + site,
                "lon": -120,
                "lat": lat,
                "files": {name: name for name in ("naip_hr", "naip_lr", "s2_lr")},
            }
            for site, split, lat in [
                ("bad", "train", 30),
                ("good", "train", 40),
                ("near_bad", "validation", 31),
                ("near_good", "validation", 39),
            ]
        ]

    def raster(self, root, item, record, grid, size, bands):
        dn = np.random.default_rng(4).uniform(20, 200, (4, 128, 128))
        if record["id"] == "bad":
            dn[:] = 100  # Insufficient acquisition dynamic range, not a raised mock.
        array = np.ones((bands, size, size))
        if item == "naip_hr":
            array[:4] = dn.repeat(4, axis=1).repeat(4, axis=2)
        elif item == "naip_lr":
            array[:4] = dn
        else:
            array[:4] = 0.002 * dn + 0.01
            array[4] = 4
        return array, np.ones((size, size), bool)

    def test_bad_group_skipped_and_assigned_validation_not_rerouted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"records": self.records()}))
            with patch("src.export_ee.read_raster", side_effect=self.raster):
                outcomes = prepare(manifest, root / "pairs", "acquisition-groups")
            by_site = {row["site_id"]: row for row in outcomes}
            for site in ("good", "near_good"):
                self.assertIn("sha256", by_site[site])
                self.assertTrue((root / "pairs" / by_site[site]["path"]).exists())
            for site in ("bad", "near_bad"):
                self.assertIn("rejected", by_site[site])
                self.assertEqual(by_site[site]["calibration_group"], "naip/bad")
                self.assertIn("dynamic range", by_site[site]["reason"])
                self.assertFalse((root / "pairs" / (site + ".npz")).exists())
            self.assertEqual(by_site["near_bad"]["split"], "validation")
            calibration = json.loads((root / "pairs" / "calibration.json").read_text())
            self.assertEqual(set(calibration["groups"]), {"naip/good"})
            self.assertEqual(calibration["assignments"]["near_bad"], "naip/bad")
            self.assertEqual(
                len(calibration["rejected_groups"]["naip/bad"]["diagnostics"]["bands"]),
                4,
            )

    def test_unexpected_runtime_failure_is_not_treated_as_rejected_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"records": self.records()}))
            with (
                patch("src.export_ee.read_raster", side_effect=self.raster),
                patch(
                    "src.export_ee.fit_acquisition_group",
                    side_effect=RuntimeError("unexpected runtime failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "unexpected"),
            ):
                prepare(manifest, root / "pairs", "acquisition-groups")
