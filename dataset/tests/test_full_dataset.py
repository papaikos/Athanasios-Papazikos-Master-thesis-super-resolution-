import hashlib
import json
from pathlib import Path
import tempfile
import shutil
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from src.full_dataset import SourceCoverage, download, read_contract


class FullDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.index = self.root / "pairs.json"
        self.sources = [
            {
                "uri": f"gs://sources/native{i}.tif",
                "generation": "123",
                "bytes": 60_000_000_000,
            }
            for i in range(2)
        ]
        self.rows = []
        for number, split in enumerate(["train", "train", "validation"]):
            name = f"pair{number}"
            source = (
                self.sources[number]
                if number < 2
                else {
                    "uri": "gs://sources/holdout.tif",
                    "generation": "456",
                    "bytes": 20,
                }
            )
            provenance = dict(
                source=source,
                source_manifest_sha256="b" * 64,
                split=split,
                site_id=name,
            )
            path = self.root / (name + ".npz")
            np.savez(path, provenance=np.array(json.dumps(provenance)))
            self.rows.append(
                dict(
                    site_id=name,
                    split=split,
                    path=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    bytes=path.stat().st_size,
                )
            )
        self.write()

    def write(self):
        self.index.write_text(json.dumps(self.rows))
        self.usage = dict(
            pairs_index_sha256=hashlib.sha256(self.index.read_bytes()).hexdigest(),
            source_manifest_sha256="b" * 64,
            bytes=sum(s["bytes"] for s in self.sources),
            sources=self.sources,
        )
        self.write_usage()

    def write_usage(self):
        self.index.with_name("source-usage.json").write_text(json.dumps(self.usage))

    def test_first_draws_cover_sources_and_only_completed_draws_count(self):
        coverage = SourceCoverage(self.index)
        self.assertNotEqual(coverage.choose(0), coverage.choose(1))
        self.assertEqual(coverage.status(0)["unique_train_source_bytes_used"], 0)
        self.assertFalse(coverage.status(1)["source_requirement_met"])
        self.assertEqual(
            coverage.status(2)["unique_train_source_bytes_used"], 120_000_000_000
        )
        self.assertTrue(coverage.status(2)["all_counted_sources_used"])
        self.assertEqual(coverage.status(100)["unique_train_sources_used"], 2)
        # Recreated sampler after checkpoint resumes deterministically from committed draws.
        self.assertEqual(SourceCoverage(self.index).choose(1), coverage.choose(1))

    def test_duplicate_identity_cannot_inflate_source_bytes(self):
        self.sources[1] = self.sources[0].copy()
        self.write()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_contract(self.index)

    def test_uncounted_validation_bytes_and_absent_sources_rejected(self):
        self.sources = self.sources[:1]
        self.write()
        with self.assertRaisesRegex(ValueError, "100 GB"):
            read_contract(self.index)
        self.sources.append(
            {
                "uri": "gs://sources/absent.tif",
                "generation": "2",
                "bytes": 60_000_000_000,
            }
        )
        self.write()
        with self.assertRaisesRegex(ValueError, "not in exact"):
            SourceCoverage(self.index)

    def test_index_tamper_and_missing_validation_rejected(self):
        self.index.write_text(self.index.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            read_contract(self.index)
        self.rows.pop()
        self.write()
        with self.assertRaisesRegex(ValueError, "validation"):
            read_contract(self.index)

    def test_npz_tamper_rejected_before_training(self):
        with (self.root / self.rows[0]["path"]).open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            SourceCoverage(self.index)

    def test_download_checks_disk_budget_and_verifies_copied_pairs(self):
        prefix = "gs://my-rgb-sr-pilot-bucket/prepared/test"
        for row in self.rows:
            row["uri"] = prefix + "/" + row["path"]
        self.write()
        copied = []

        def cloud_copy(command, check):
            self.assertTrue(check)
            copied.append(command[3])
            shutil.copyfile(self.root / command[3].rsplit("/", 1)[1], command[4])

        with patch("src.full_dataset.subprocess.run", side_effect=cloud_copy):
            with patch(
                "src.full_dataset.shutil.disk_usage",
                return_value=SimpleNamespace(free=1),
            ):
                with self.assertRaisesRegex(ValueError, "Insufficient disk"):
                    download(prefix, self.root / "small")
            self.assertEqual(len(copied), 2)  # No imagery copied when it cannot fit.
            with patch(
                "src.full_dataset.shutil.disk_usage",
                return_value=SimpleNamespace(free=20_000_000_000),
            ):
                result = download(prefix, self.root / "large")
            self.assertEqual(result["unique_train_source_bytes_used"], 0)
            self.assertFalse(result["source_requirement_met"])
            self.assertEqual(len(copied), 7)

    def test_generation_mismatch_rejected(self):
        self.sources[0]["generation"] = "999"
        self.write()
        with self.assertRaisesRegex(ValueError, "not in exact"):
            SourceCoverage(self.index)


if __name__ == "__main__":
    unittest.main()
