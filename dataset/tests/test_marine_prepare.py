import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src import marine_prepare as marine


def record(name, split="train", lon=0):
    return {
        "id": name,
        "naip_id": "NAIP/" + name,
        "s2_id": "S2/" + name,
        "split": split,
        "lon": lon,
        "lat": 0,
        "files": {
            kind: {
                "uri": f"gs://bucket/{name}-{kind}.tif",
                "generation": "1",
                "bytes": 100,
            }
            for kind in ("naip", "s2", "nativevalid")
        },
    }


class MarinePrepareTests(unittest.TestCase):
    def test_geographic_assignment_frozen_and_duplicate_object_rejected(self):
        records = [
            record("west", lon=-10),
            record("east", lon=10),
            record("holdout", "test", -9),
        ]
        self.assertEqual(
            marine.validate_manifest({"records": records})["holdout"], "NAIP/west"
        )
        records[2]["files"]["naip"] = records[0]["files"]["naip"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            marine.validate_manifest({"records": records})

    def test_count_only_unique_accepted_train_source_bytes(self):
        outcomes = [
            {"split": split, "source": record(name)["files"]["naip"], "pairs": pairs}
            for name, split, pairs in [
                ("a", "train", [1]),
                ("a", "train", [2]),
                ("b", "validation", [1]),
                ("c", "train", []),
            ]
        ]
        self.assertEqual(marine.source_usage(outcomes)["bytes"], 100)

    def test_unpinned_sources_rejected(self):
        item = record("a")
        item["files"]["naip"]["generation"] = ""
        with self.assertRaisesRegex(ValueError, "pinned"):
            marine.validate_manifest({"records": [item]})

    def test_coastal_grid_and_complete_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # x=1280,y=2560 are global128-pixel grid boundaries.
            profile = dict(
                driver="GTiff",
                crs="EPSG:32610",
                transform=from_origin(1280, 2560, 2.5, 2.5),
                width=512,
                height=512,
                count=4,
                dtype="uint8",
            )
            with rasterio.open(root / "naip.tif", "w", **profile) as dataset:
                dataset.colorinterp = (
                    rasterio.enums.ColorInterp.red,
                    rasterio.enums.ColorInterp.green,
                    rasterio.enums.ColorInterp.blue,
                    rasterio.enums.ColorInterp.undefined,
                )
                dataset.write(np.full((4, 512, 512), 100, dtype="uint8"))
            profile["count"] = 1
            with rasterio.open(root / "valid.tif", "w", **profile) as dataset:
                dataset.write(np.ones((1, 512, 512), dtype="uint8"))
            reference = np.full((6, 128, 128), 0.2, dtype="float32")
            reference[4] = 4
            reference[4, :32] = 6
            reference[5] = 1
            profile.update(
                transform=from_origin(1280, 2560, 10, 10),
                width=128,
                height=128,
                count=6,
                dtype="float32",
            )
            with rasterio.open(root / "s2.tif", "w", **profile) as dataset:
                dataset.write(reference)
            pairs = list(
                marine.candidate_patches(
                    root / "naip.tif", root / "s2.tif", root / "valid.tif"
                )
            )
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["water_fraction"], 0.25)
            self.assertTrue(pairs[0]["valid"].all())
            # RGBN GeoTIFF still reports all-valid; explicit sidecar must reject.
            with rasterio.open(root / "valid.tif", "r+") as dataset:
                mask = np.ones((512, 512), dtype="uint8")
                mask[200:220, 200:220] = 0
                dataset.write(mask, 1)
            self.assertEqual(
                list(
                    marine.candidate_patches(
                        root / "naip.tif", root / "s2.tif", root / "valid.tif"
                    )
                ),
                [],
            )

    def test_rgba_tagged_rgbn_preserves_nir_and_rejects_invalid_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / name for name in ("naip.tif", "s2.tif", "valid.tif")]
            profile = dict(
                driver="GTiff",
                crs="EPSG:32610",
                transform=from_origin(1280, 2560, 2.5, 2.5),
                width=512,
                height=512,
                count=4,
                dtype="uint8",
            )
            pixels = np.full((4, 512, 512), 100, dtype="uint8")
            pixels[3] = np.arange(512, dtype="uint16")[None, :] % 256
            with rasterio.open(paths[0], "w", **profile) as dataset:
                dataset.write(pixels)
                dataset.colorinterp = tuple(
                    rasterio.enums.ColorInterp(i) for i in (3, 4, 5, 6)
                )
            original_digest = marine.sha256(paths[0])
            profile["count"] = 1
            with rasterio.open(paths[2], "w", **profile) as dataset:
                dataset.write(np.ones((1, 512, 512), dtype="uint8"))
            profile.update(
                count=6,
                dtype="float32",
                width=128,
                height=128,
                transform=from_origin(1280, 2560, 10, 10),
            )
            reference = np.full((6, 128, 128), 0.2, dtype="float32")
            reference[4] = 4
            reference[4, :32] = 6
            reference[5] = 1
            with rasterio.open(paths[1], "w", **profile) as dataset:
                dataset.write(reference)
            diagnostics = {}
            pairs = list(marine.candidate_patches(*paths, diagnostics=diagnostics))
            self.assertEqual(diagnostics, {"windows": 1, "yielded": 1})
            self.assertEqual(len(pairs), 1)
            baseline = list(marine.candidate_patches(*paths))
            self.assertEqual(len(baseline), len(pairs))
            for key in ("hr", "dn", "sr", "support", "valid"):
                np.testing.assert_array_equal(pairs[0][key], baseline[0][key])
            np.testing.assert_array_equal(pairs[0]["hr"], pixels)
            self.assertEqual(marine.sha256(paths[0]), original_digest)
            # Explicit validity holes remain authoritative, even with nonzero NIR.
            mask = np.ones((512, 512), dtype="uint8")
            mask[200:220, 200:220] = 0
            with rasterio.open(paths[2], "r+") as dataset:
                dataset.write(mask, 1)
            diagnostics = {}
            self.assertEqual(
                list(marine.candidate_patches(*paths, diagnostics=diagnostics)), []
            )
            self.assertEqual(diagnostics, {"windows": 1, "rejected_native_coverage": 1})
            with rasterio.open(paths[2], "r+") as dataset:
                dataset.write(np.ones((512, 512), dtype="uint8"), 1)
            # A real raster mask must also survive the color interpretation fix.
            with rasterio.open(paths[0], "r+") as dataset:
                dataset.write_mask(mask * 255)
            diagnostics = {}
            self.assertEqual(
                list(marine.candidate_patches(*paths, diagnostics=diagnostics)), []
            )
            self.assertEqual(
                diagnostics, {"windows": 1, "rejected_original_mask_or_footprint": 1}
            )
            with rasterio.open(paths[0], "r+") as dataset:
                dataset.write_mask(np.full((512, 512), 255, dtype="uint8"))
            # Moving the reference beyond the source footprint must reject padding.
            with rasterio.open(paths[1], "r+") as dataset:
                dataset.transform = from_origin(0, 2560, 10, 10)
            diagnostics = {}
            self.assertEqual(
                list(marine.candidate_patches(*paths, diagnostics=diagnostics)), []
            )
            self.assertEqual(
                diagnostics, {"windows": 1, "rejected_original_mask_or_footprint": 1}
            )
            with rasterio.open(paths[1], "r+") as dataset:
                dataset.transform = from_origin(1280, 2560, 10, 10)
                reference[4] = 4
                dataset.write(reference)
            diagnostics = {}
            self.assertEqual(
                list(marine.candidate_patches(*paths, diagnostics=diagnostics)), []
            )
            self.assertEqual(diagnostics, {"windows": 1, "rejected_water": 1})
            with rasterio.open(paths[1], "r+") as dataset:
                reference[4] = 0
                reference[4, :32] = 6
                dataset.write(reference)
            diagnostics = {}
            self.assertEqual(
                list(marine.candidate_patches(*paths, diagnostics=diagnostics)), []
            )
            self.assertEqual(diagnostics, {"windows": 1, "rejected_support": 1})

    def test_pipeline_fits_training_only_and_reuses_frozen_mapping(self):
        rng = np.random.default_rng(3)
        dn = rng.uniform(20, 220, (4, 128, 128)).astype("float32")
        hr = np.repeat(np.repeat(dn, 4, axis=1), 4, axis=2)
        candidate = {
            "row": 0,
            "col": 0,
            "dn": dn,
            "hr": hr,
            "sr": dn * 0.002 + 0.01,
            "support": np.ones((128, 128), bool),
            "valid": np.ones((512, 512), bool),
            "water_fraction": 0.2,
            "crs": "EPSG:32610",
            "hr_transform": [2.5, 0, 0, 0, -2.5, 1280],
        }
        calls = 0

        def candidates(*args, diagnostics=None):
            nonlocal calls
            calls += 1
            item = dict(candidate)
            # First two invocations are training fit + pair, third is holdout.
            if calls >= 3:
                item["hr_transform"] = [2.5, 0, 1280, 0, -2.5, 1280]
            if diagnostics is not None:
                diagnostics["windows"] = 1
                diagnostics["yielded"] = 1
            yield item
            if calls == 2:
                # Output traversal encounters a duplicate and a distinct bad residual.
                if diagnostics is not None:
                    diagnostics["windows"] += 2
                    diagnostics["yielded"] += 2
                yield item
                bad = dict(item, sr=np.full_like(dn, 2.0))
                bad["hr_transform"] = [2.5, 0, 2560, 0, -2.5, 1280]
                yield bad

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {"records": [record("train"), record("valid", "validation", 1)]}
                )
            )
            with (
                patch.object(marine, "copy_source"),
                patch.object(marine, "candidate_patches", side_effect=candidates),
            ):
                usage = marine.prepare(manifest, root / "out", minimum_source_bytes=101)
            self.assertEqual(usage["bytes"], 100)
            self.assertFalse(usage["requirement_met"])
            pairs = json.loads((root / "out/pairs.json").read_text())
            self.assertEqual(len(pairs), 2)
            state = json.loads((root / "out/preparation-state.json").read_text())
            train, heldout = state["outcomes"]
            self.assertEqual(
                train["diagnostics"]["calibration"], {"windows": 1, "yielded": 1}
            )
            self.assertEqual(
                train["diagnostics"]["output"],
                {
                    "windows": 3,
                    "yielded": 3,
                    "accepted": 1,
                    "rejected_duplicate": 1,
                    "rejected_residual": 1,
                },
            )
            self.assertEqual(heldout["diagnostics"]["calibration"], {})
            self.assertEqual(heldout["diagnostics"]["output"]["accepted"], 1)
            with np.load(root / "out" / pairs[1]["path"], allow_pickle=False) as data:
                provenance = json.loads(str(data["provenance"]))
                self.assertEqual(provenance["calibration_train_sites"], ["train"])
                self.assertEqual(provenance["lr_origin"], "synthetic_hr_only")
            self.assertFalse(list((root / "out").glob("marine-source-*")))


if __name__ == "__main__":
    unittest.main()
