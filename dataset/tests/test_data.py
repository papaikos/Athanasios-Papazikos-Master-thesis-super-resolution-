import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src import data


class DataTests(unittest.TestCase):
    def test_contract_and_grid_origin(self):
        grid = data.aligned_grids([10, 0, 600003, 0, -10, 4200007], 600654, 4199382)
        self.assertEqual(grid["lr"][0], 10)
        self.assertEqual(grid["hr"][0], 2.5)
        self.assertEqual((grid["lr"][2] - 600003) % 10, 0)
        self.assertEqual(grid["hr"][2:], [grid["lr"][2], 0, -2.5, grid["lr"][5]])
        with self.assertRaises(ValueError):
            data.aligned_grids([10, 1, 0, 0, -10, 0], 0, 0)

    def test_degradation_preserves_bands_erodes_invalid_support(self):
        hr = (
            np.broadcast_to(np.arange(1, 5)[:, None, None], (4, 512, 512))
            .copy()
            .astype(float)
        )
        valid = np.ones((512, 512), bool)
        valid[250, 250] = False
        hr[:, 250, 250] = np.nan
        lr, mask = data.degrade(hr, valid)
        self.assertEqual(lr.shape, (4, 128, 128))
        np.testing.assert_allclose(
            lr[:, mask], np.broadcast_to(np.arange(1, 5)[:, None], lr[:, mask].shape)
        )
        self.assertFalse(mask[62, 62])
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[30, 30])

    def test_robust_calibration_and_holdout_rejection(self):
        rng = np.random.default_rng(9)
        dn = rng.uniform(20, 220, (4, 32, 32))
        sr = dn * np.array([0.002, 0.003, 0.001, 0.004])[:, None, None] + 0.02
        sr[:, :2] += 0.3
        support = np.ones((32, 32), bool)
        result = data.fit_calibration(dn, sr, support, split="train", site_ids=["la"])
        np.testing.assert_allclose(
            result["gain"], [0.002, 0.003, 0.001, 0.004], rtol=0.02
        )
        for split in ("validation", "test"):
            with self.assertRaises(ValueError):
                data.fit_calibration(dn, sr, support, split=split, site_ids=["sf"])
        with self.assertRaises(ValueError):
            data.fit_calibration(
                np.ones_like(dn), sr, support, split="train", site_ids=["la"]
            )
        with self.assertRaises(ValueError):
            data.fit_calibration(
                dn, sr, np.zeros_like(support), split="train", site_ids=["la"]
            )

    def test_stable_land_excludes_water_cloud_shadow(self):
        scl = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
        np.testing.assert_array_equal(
            data.stable_land(scl, np.ones_like(scl, bool)), np.isin(scl, [4, 5])
        )

    def test_save_pair_provenance_and_no_real_s2_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pair.npz"
            provenance = {
                "site_id": "sf",
                "split": "validation",
                "naip_id": "a",
                "s2_id": "b",
                "calibration_train_sites": ["la"],
            }
            data.save_pair(
                path, np.ones((4, 512, 512)), np.ones((512, 512), bool), provenance
            )
            with np.load(path, allow_pickle=False) as result:
                self.assertEqual(result["lr"].shape, (4, 128, 128))
                self.assertEqual(
                    json.loads(str(result["provenance"]))["lr_origin"],
                    "synthetic_hr_only",
                )

    def test_constant_valid_chip_has_no_zero_lr_border(self):
        lr, mask = data.degrade(np.full((4, 512, 512), 0.2), np.ones((512, 512), bool))
        np.testing.assert_allclose(lr, 0.2, atol=1e-7)
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[64, 64])

    def test_pilot_pair_rejects_incomplete_hr_for_global_fft(self):
        with tempfile.TemporaryDirectory() as tmp:
            valid = np.ones((512, 512), bool)
            valid[250, 250] = False
            with self.assertRaisesRegex(ValueError, "complete HR"):
                data.save_pair(
                    Path(tmp) / "bad.npz",
                    np.ones((4, 512, 512)),
                    valid,
                    {
                        "site_id": "la",
                        "split": "train",
                        "naip_id": "a",
                        "s2_id": "b",
                        "calibration_train_sites": ["la"],
                    },
                )

    def test_duplicate_site_split_rejected(self):
        with self.assertRaises(ValueError):
            data.validate_sites(
                [{"id": "la", "split": "train"}, {"id": "la", "split": "test"}]
            )


class PilotIntegrationTests(unittest.TestCase):
    def test_prepare_uses_frozen_training_calibration_and_checks_hash(self):
        import rasterio
        from rasterio.transform import Affine
        from src.export_ee import prepare

        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for site, split in [("la", "train"), ("sf", "validation")]:
                grids = data.aligned_grids(
                    [10, 0, 500003, 0, -10, 4200007], 500643, 4199367
                )
                record = {
                    "id": site,
                    "split": split,
                    "lon": -118 if split == "train" else -122,
                    "lat": 34 if split == "train" else 37,
                    "naip_id": "naip/" + site,
                    "s2_id": "s2/" + site,
                    "crs": "EPSG:32611",
                    "grids": grids,
                    "files": {},
                }
                dn = rng.uniform(20, 200, (4, 128, 128)).astype("float32")
                hr = dn.repeat(4, axis=1).repeat(4, axis=2)
                sr = dn * 0.002 + 0.02
                arrays = {
                    "naip_hr": np.concatenate([hr, np.ones((1, 512, 512))]),
                    "naip_lr": np.concatenate([dn, np.ones((1, 128, 128))]),
                    "s2_lr": np.concatenate(
                        [sr, np.full((1, 128, 128), 4), np.ones((1, 128, 128))]
                    ),
                }
                for name, array in arrays.items():
                    grid = "hr" if name == "naip_hr" else "lr"
                    path = root / (site + "_" + name + ".tif")
                    with rasterio.open(
                        path,
                        "w",
                        driver="GTiff",
                        width=array.shape[2],
                        height=array.shape[1],
                        count=array.shape[0],
                        dtype="float32",
                        crs=record["crs"],
                        transform=Affine(*grids[grid]),
                    ) as dst:
                        dst.write(array.astype("float32"))
                    record["files"][name] = {
                        "path": path.name,
                        "sha256": data.sha256(path),
                    }
                records.append(record)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"records": records}), encoding="utf-8")
            result = prepare(manifest, root / "pairs")
            self.assertEqual(len(result), 2)
            self.assertTrue(all("sha256" in item for item in result))
            calibration = json.loads((root / "pairs" / "calibration.json").read_text())
            self.assertEqual(calibration["train_sites"], ["la"])
            grouped = prepare(manifest, root / "grouped", "acquisition-groups")
            self.assertTrue(all("sha256" in item for item in grouped))
            group_calibration = json.loads(
                (root / "grouped" / "calibration.json").read_text()
            )
            self.assertEqual(group_calibration["assignments"]["sf"], "naip/la")
            self.assertEqual(
                len(group_calibration["groups"]["naip/la"]["spatial_training_checks"]),
                2,
            )
            with np.load(root / "pairs" / "sf.npz", allow_pickle=False) as pair:
                meta = json.loads(str(pair["provenance"]))
                self.assertEqual(meta["calibration_train_sites"], ["la"])
                np.testing.assert_allclose(
                    pair["hr"], arrays["naip_hr"][:4] * 0.002 + 0.02, atol=1e-6
                )
            records[0]["files"]["naip_hr"]["sha256"] = "tampered"
            manifest.write_text(json.dumps({"records": records}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                prepare(manifest, root / "bad")

    def test_shared_source_scene_cannot_cross_split(self):
        with self.assertRaisesRegex(ValueError, "Source scene"):
            data.validate_sites(
                [
                    {"id": "la", "split": "train", "naip_id": "same"},
                    {"id": "sf", "split": "validation", "naip_id": "same"},
                ]
            )

    def test_grid_validation_rejects_one_pixel_shift_at_utm_northing(self):
        import rasterio
        from rasterio.transform import Affine
        from src.export_ee import read_raster

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "shifted.tif"
            expected = [10, 0, 500000, 0, -10, 4200000]
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=128,
                height=128,
                count=5,
                dtype="float32",
                crs="EPSG:32611",
                transform=Affine(10, 0, 500000, 0, -10, 4200010),
            ) as dst:
                dst.write(np.ones((5, 128, 128), dtype="float32"))
            with self.assertRaisesRegex(ValueError, "grid"):
                read_raster(
                    root,
                    {"path": path.name, "sha256": data.sha256(path)},
                    {"crs": "EPSG:32611", "grids": {"lr": expected}},
                    "lr",
                    128,
                    5,
                )

    def test_download_resume_preserves_verified_files_after_failure(self):
        from unittest.mock import MagicMock, patch
        from src.export_ee import download

        ee = MagicMock()
        record = {
            "id": "la",
            "split": "train",
            "naip_id": "naip/la",
            "s2_id": "s2/la",
            "crs": "EPSG:32611",
            "grids": {"hr": [2.5, 0, 0, 0, -2.5, 0], "lr": [10, 0, 0, 0, -10, 0]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def interrupted(image, path, crs, transform, size):
                if "naip_lr" in path.name:
                    raise RuntimeError("temporary network failure")
                path.write_bytes(b"verified-raster")
                return data.sha256(path)

            with patch("src.export_ee.download_image", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "network"):
                    download(ee, [record], root)
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(set(manifest["records"][0]["files"]), {"naip_hr"})
            self.assertEqual(manifest["status"], "partial")
            downloaded = []

            def completed(image, path, crs, transform, size):
                downloaded.append(path.name)
                path.write_bytes(b"verified-raster")
                return data.sha256(path)

            with patch("src.export_ee.download_image", side_effect=completed):
                download(ee, manifest["records"], root)
            self.assertEqual(downloaded, ["la_naip_lr.tif", "la_s2_lr.tif"])
            self.assertEqual(
                json.loads((root / "manifest.json").read_text())["status"], "complete"
            )
            (root / "la_naip_hr.tif").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                download(ee, manifest["records"], root)

    def test_prepare_rejects_partial_download_manifest(self):
        from src.export_ee import prepare

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"status": "partial", "records": []}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "download is incomplete"):
                prepare(manifest, root / "pairs")

    def test_gcs_export_config_exact_grid_and_bounded_pixels(self):
        from src.export_ee import gcs_export_parameters, GCS_BUCKET

        record = {
            "id": "la",
            "crs": "EPSG:32611",
            "grids": {
                "hr": [2.5, 0, 500000, 0, -2.5, 4200000],
                "lr": [10, 0, 500000, 0, -10, 4200000],
            },
        }
        for name, grid, size in [("naip_hr", "hr", 512), ("s2_lr", "lr", 128)]:
            params, bounds = gcs_export_parameters(record, name, grid, size)
            self.assertEqual(params["bucket"], GCS_BUCKET)
            self.assertEqual(params["fileNamePrefix"], "pilot/raw/la_" + name)
            self.assertEqual(params["crsTransform"], record["grids"][grid])
            pixel = record["grids"][grid][0]
            np.testing.assert_allclose(
                bounds,
                [
                    500000 + 0.01 * pixel,
                    4198720 + 0.01 * pixel,
                    501280 - 0.01 * pixel,
                    4200000 - 0.01 * pixel,
                ],
                rtol=0,
                atol=1e-8,
            )
            self.assertLess(bounds[0], 500000 + 0.5 * pixel)
            self.assertGreater(bounds[2], 501280 - 0.5 * pixel)
            self.assertLess(bounds[1], 4198720 + 0.5 * pixel)
            self.assertGreater(bounds[3], 4200000 - 0.5 * pixel)
            self.assertEqual(params["maxPixels"], size * size)
            self.assertEqual(params["fileDimensions"] % params["shardSize"], 0)
            self.assertNotIn("scale", params)
            self.assertNotIn("dimensions", params)

    def test_gcs_submission_bounded_and_durable_before_start(self):
        from unittest.mock import MagicMock, patch
        from src.export_ee import export_gcs

        record = {
            "id": "la",
            "split": "train",
            "naip_id": "n",
            "s2_id": "s",
            "crs": "EPSG:32611",
            "grids": {"hr": [2.5, 0, 0, 0, -2.5, 0], "lr": [10, 0, 0, 0, -10, 0]},
        }
        ee = MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = MagicMock()
            task.id = None

            def start():
                manifest = json.loads((root / "manifest.json").read_text())
                self.assertIsNone(manifest["tasks"][0]["id"])
                self.assertEqual(manifest["tasks"][0]["state"], "SUBMITTING")
                raise RuntimeError("submission uncertain")

            task.start.side_effect = start
            ee.batch.Export.image.toCloudStorage.return_value = task
            with patch(
                "src.export_ee.image_specs",
                return_value=[("naip_hr", object(), "hr", 512)],
            ):
                with self.assertRaisesRegex(RuntimeError, "uncertain"):
                    export_gcs(ee, [record], root)
            self.assertEqual(
                json.loads((root / "manifest.json").read_text())["status"],
                "export_pending",
            )
            with self.assertRaises(ValueError):
                export_gcs(
                    ee, [dict(record, id=str(i)) for i in range(7)], root / "too_many"
                )

    def test_gcs_retry_only_fresh_failed_rounding_preserves_attempt(self):
        from unittest.mock import MagicMock, patch
        from src.export_ee import export_gcs_retry, GCS_BUCKET

        record = {
            "id": "port_hueneme",
            "split": "train",
            "naip_id": "n",
            "s2_id": "s",
            "crs": "EPSG:32611",
            "grids": {"hr": [2.5, 0, 0, 0, -2.5, 0], "lr": [10, 0, 0, 0, -10, 0]},
        }
        original = {
            "id": "old",
            "site_id": "port_hueneme",
            "name": "naip_hr",
            "grid": "hr",
            "size": 512,
            "state": "FAILED",
            "parameters": {},
            "bounds": [0, -1280, 1280, 0],
        }
        completed = dict(
            original, id="done", name="naip_lr", state="COMPLETED", size=128, grid="lr"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "bucket": GCS_BUCKET,
                        "tasks": [original, completed],
                        "records": [record],
                        "submission_complete": True,
                    }
                ),
                encoding="utf-8",
            )
            ee = MagicMock()
            ee.data.getTaskStatus.return_value = [
                {
                    "id": "old",
                    "state": "FAILED",
                    "error_message": "Export too large: specified 263169 pixels (max: 262144).",
                },
                {"id": "done", "state": "COMPLETED"},
            ]
            task = MagicMock()
            task.id = None

            def start():
                before = json.loads(path.read_text())["tasks"][0]
                self.assertEqual(before["attempts"][0]["id"], "old")
                self.assertIsNone(before["id"])
                self.assertEqual(before["state"], "SUBMITTING")
                task.id = "new"

            task.start.side_effect = start
            ee.batch.Export.image.toCloudStorage.return_value = task
            with patch(
                "src.export_ee.image_specs",
                return_value=[("naip_hr", object(), "hr", 512)],
            ):
                export_gcs_retry(ee, path)
            saved = json.loads(path.read_text())
            self.assertEqual(saved["tasks"][0]["id"], "new")
            self.assertEqual(saved["tasks"][1], completed)
            self.assertEqual(ee.batch.Export.image.toCloudStorage.call_count, 1)
            ee.data.getTaskStatus.return_value = [
                {"id": "new", "state": "RUNNING"},
                {"id": "done", "state": "COMPLETED"},
            ]
            with self.assertRaisesRegex(ValueError, "No eligible"):
                export_gcs_retry(ee, path)
            self.assertEqual(ee.batch.Export.image.toCloudStorage.call_count, 1)

    def test_harbor_center_shifts_are_bounded_and_nearest_first(self):
        from src.export_ee import candidate_centers

        centers = candidate_centers(500000, 4200000)
        self.assertEqual(centers[0], (500000, 4200000, 0, 0))
        self.assertEqual(len(centers), 5)
        self.assertTrue(all(dx * dx + dy * dy <= 320**2 for _, _, dx, dy in centers))

    def test_uncertain_gcs_task_fails_closed_without_query_or_retry(self):
        from unittest.mock import MagicMock
        from src.export_ee import export_gcs_status, GCS_BUCKET

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "bucket": GCS_BUCKET,
                        "tasks": [{"id": None, "state": "SUBMISSION_UNCERTAIN"}],
                    }
                ),
                encoding="utf-8",
            )
            ee = MagicMock()
            with self.assertRaisesRegex(RuntimeError, "reconciliation"):
                export_gcs_status(ee, path)
            ee.data.getTaskStatus.assert_not_called()

    def test_dry_run_is_offline_and_bounded(self):
        import subprocess
        from src.export_ee import configuration

        result = subprocess.run(
            ["python", "-m", "src.export_ee", "dry-run", "--limit", "2"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertFalse(json.loads(result.stdout)["network"])
        with self.assertRaises(ValueError):
            configuration(Path("configs/pilot_sites.json"), 101)

    def test_gaussian_then_area_impulse_has_unit_energy(self):
        hr = np.zeros((4, 512, 512), dtype=np.float32)
        hr[2, 254, 254] = 16
        lr, mask = data.degrade(hr, np.ones((512, 512), bool))
        self.assertAlmostEqual(float(lr[2].sum()), 1.0, places=6)
        self.assertEqual(float(lr[[0, 1, 3]].sum()), 0.0)
        self.assertGreater(np.count_nonzero(lr[2]), 4)
        self.assertTrue(mask[63, 63])


if __name__ == "__main__":
    unittest.main()
