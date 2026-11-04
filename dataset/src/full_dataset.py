"""Full-run source accounting and verified download of prepared marine pairs."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import subprocess
from src.cloud_config import GCS_PILOT_BUCKET


MIN_SOURCE_BYTES = 100_000_000_000
MAX_SOURCES = 600


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_key(source):
    if (
        not isinstance(source, dict)
        or not re.fullmatch(
            r"gs://[a-z0-9._-]+/[a-zA-Z0-9/_.-]+", source.get("uri", "")
        )
        or not str(source.get("generation", "")).isdigit()
        or isinstance(source.get("bytes"), bool)
        or not isinstance(source.get("bytes"), int)
        or source["bytes"] <= 0
    ):
        raise ValueError("Invalid pinned source identity")
    return source["uri"]


def read_contract(index):
    index = Path(index)
    usage_path = index.with_name("source-usage.json")
    usage = json.loads(usage_path.read_text())
    if usage.get("pairs_index_sha256") != digest(index):
        raise ValueError("Prepared index checksum mismatch")
    if not re.fullmatch("[a-f0-9]{64}", usage.get("source_manifest_sha256", "")):
        raise ValueError("Missing source manifest digest")
    sources = {}
    for source in usage["sources"]:
        key = source_key(source)
        if key in sources:
            raise ValueError("Duplicate source identity in accounting")
        sources[key] = source
    total = sum(source["bytes"] for source in sources.values())
    if (
        not 1 <= len(sources) <= MAX_SOURCES
        or total < MIN_SOURCE_BYTES
        or usage.get("bytes") != total
    ):
        raise ValueError(
            "Require at least 100 GB unique TRAIN source bytes and at most 600 sources"
        )
    rows = json.loads(index.read_text())
    if not rows or len(rows) > 100_000:
        raise ValueError("Invalid prepared pair count")
    paths = set()
    splits = set()
    for row in rows:
        path = row.get("path", "")
        if (
            not re.fullmatch(r"[a-zA-Z0-9_-]+\.npz", path)
            or path in paths
            or row.get("split") not in ("train", "validation", "test")
            or not re.fullmatch("[a-f0-9]{64}", row.get("sha256", ""))
            or not isinstance(row.get("bytes"), int)
            or row["bytes"] <= 0
            or "rejected" in row
        ):
            raise ValueError("Invalid prepared pair record")
        paths.add(path)
        splits.add(row["split"])
    if not {"train", "validation"} <= splits:
        raise ValueError("Full run requires train and validation pairs")
    return rows, sources, usage


class SourceCoverage:
    """First draw covers each source; later draws retain the existing random sampler."""

    def __init__(self, index):
        import numpy as np

        self.index = Path(index)
        rows, self.sources, usage = read_contract(index)
        self.by_source = {key: [] for key in self.sources}
        split_sources = {}
        self.paths = []
        for row in rows:
            path = self.index.parent / row["path"]
            if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ValueError("Prepared pair byte count/checksum mismatch")
            with np.load(path, allow_pickle=False) as archive:
                provenance = json.loads(str(archive["provenance"]))
            source = provenance["source"]
            key = source_key(source)
            if (
                provenance.get("split") != row["split"]
                or provenance.get("site_id") != row["site_id"]
                or provenance.get("source_manifest_sha256")
                != usage["source_manifest_sha256"]
            ):
                raise ValueError("Pair source/index provenance mismatch")
            previous = split_sources.setdefault(key, row["split"])
            if previous != row["split"]:
                raise ValueError("Source leaks across geographic splits")
            if row["split"] == "train":
                if self.sources.get(key) != source:
                    raise ValueError("TRAIN source not in exact accounting manifest")
                self.by_source[key].append(path)
                self.paths.append(path)
        if any(not paths for paths in self.by_source.values()):
            raise ValueError("Counted TRAIN source has no consumed-patch candidate")
        self.keys = sorted(self.sources)
        self.first_pass = [sorted(self.by_source[key])[0] for key in self.keys]
        self.usage_sha256 = digest(self.index.with_name("source-usage.json"))

    def choose(self, draw):
        if draw < len(self.first_pass):
            return self.first_pass[draw]
        return random.choice(self.paths)

    def status(self, completed_draws):
        keys = self.keys[: min(completed_draws, len(self.keys))]
        used = sum(self.sources[key]["bytes"] for key in keys)
        return {
            "completed_training_draws": completed_draws,
            "unique_train_sources_used": len(keys),
            "unique_train_source_bytes_used": used,
            "minimum_source_bytes": MIN_SOURCE_BYTES,
            "source_requirement_met": used >= MIN_SOURCE_BYTES,
            "all_counted_sources_used": len(keys) == len(self.keys),
        }


def download(prefix, output):
    if not re.fullmatch(
        "gs://" + re.escape(GCS_PILOT_BUCKET) + r"/[a-zA-Z0-9/_.-]+", prefix
    ):
        raise ValueError("Prepared data must be in the RGB experiment bucket")
    prefix = prefix.rstrip("/")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def copy(uri, path):
        subprocess.run(["gcloud", "storage", "cp", uri, str(path)], check=True)

    for name in ("pairs.json", "source-usage.json"):
        copy(prefix + "/" + name, output / name)
    rows, _, _ = read_contract(output / "pairs.json")
    if (
        sum(row["bytes"] for row in rows) + 10_000_000_000
        > shutil.disk_usage(output).free
    ):
        raise ValueError(
            "Insufficient disk for prepared pairs and 10 GB runtime reserve"
        )
    for row in rows:
        expected = prefix + "/" + row["path"]
        if row.get("uri") != expected:
            raise ValueError("Prepared pair URI outside exact prepared prefix")
        copy(expected, output / row["path"])
    coverage = SourceCoverage(output / "pairs.json")
    return coverage.status(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(download(args.prefix, args.output)))


if __name__ == "__main__":
    main()
