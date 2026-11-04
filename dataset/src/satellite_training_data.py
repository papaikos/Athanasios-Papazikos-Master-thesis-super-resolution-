"""Verify and stage frozen satellite RGB pairs; never acquires imagery or trains."""

import argparse
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import shutil
import tarfile
import tempfile

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seal(manifest):
    payload = {k: v for k, v in manifest.items() if k != "sha256"}
    return dict(
        payload,
        sha256=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
    )


def stage_dataset(manifest, output, fetch, workers=4):
    """Prefetch at most four pinned shards, consuming in deterministic manifest order."""
    if not 1 <= workers <= 4:
        raise ValueError("Download workers must be between one and four")
    if manifest.get("sha256") != seal(manifest)["sha256"]:
        raise ValueError("Dataset manifest checksum mismatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for filename in ("index.json", "staging-receipt.json"):
        (output / filename).unlink(missing_ok=True)
    shards = iter(
        shard for receipt in manifest["sources"] for shard in receipt["shards"]
    )
    with (
        tempfile.TemporaryDirectory(dir=output) as cache,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        pending = deque()
        serial = 0

        def enqueue():
            nonlocal serial
            shard = next(shards, None)
            if shard is None:
                return
            uri = shard["uri"] + "#" + shard["generation"]
            destination = Path(cache) / f"{serial}.tar"
            serial += 1
            pending.append((uri, destination, pool.submit(fetch, uri, destination)))

        for _ in range(workers):
            enqueue()

        def prefetched_fetch(uri, destination):
            expected, cached, future = pending.popleft()
            if uri != expected:
                raise ValueError("Shard prefetch order mismatch")
            future.result()
            shutil.move(str(cached), str(destination))
            enqueue()

        return _stage_dataset(manifest, output, prefetched_fetch)


def _stage_dataset(manifest, output, fetch):
    if manifest.get("sha256") != seal(manifest)["sha256"]:
        raise ValueError("Dataset manifest checksum mismatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, hashes, identities, cells, groups, shard_uris = (
        [],
        set(),
        set(),
        set(),
        {},
        set(),
    )
    totals = defaultdict(lambda: dict(pairs=0, accepted_source_bytes=0))
    for receipt in manifest["sources"]:
        split = receipt["split"]
        if split not in ("train", "validation", "test"):
            raise ValueError("Invalid split")
        if groups.setdefault(receipt["group_id"], split) != split:
            raise ValueError("Geographic split leakage")
        obj = receipt["source_object"]
        identity = obj["uri"] + "#" + obj["generation"]
        if identity in identities or receipt["source_sha256"] in hashes:
            raise ValueError("Duplicate source identity or native hash")
        identities.add(identity)
        hashes.add(receipt["source_sha256"])
        source_dir = Path("pairs") / hashlib.sha256(identity.encode()).hexdigest()[:24]
        count, covered = 0, 0
        for shard in receipt["shards"]:
            if shard["uri"] in shard_uris:
                raise ValueError("Duplicate shard URI")
            shard_uris.add(shard["uri"])
            with tempfile.TemporaryDirectory(dir=output) as scratch:
                archive = Path(scratch) / "shard.tar"
                fetch(shard["uri"] + "#" + shard["generation"], archive)
                if digest(archive) != shard["sha256"]:
                    raise ValueError("Shard checksum mismatch")
                with tarfile.open(archive) as tar:
                    for member in tar:
                        if (
                            not member.isfile()
                            or not re.fullmatch(r"[0-9]{6}\.npz", member.name)
                            or PurePosixPath(member.name).name != member.name
                            or not 0 < member.size < 8_000_000
                        ):
                            raise ValueError("Unsafe shard member")
                        with np.load(
                            tar.extractfile(member), allow_pickle=False
                        ) as pair:
                            hr, lr, mask = pair["hr"], pair["lr"], pair["mask"]
                            meta = json.loads(str(pair["metadata"]))
                        for key in (
                            "source_id",
                            "group_id",
                            "split",
                            "source_object",
                            "manifest_sha256",
                            "recipe",
                        ):
                            if meta.get(key) != receipt[key]:
                                raise ValueError("Pair provenance differs from receipt")
                        if (
                            hr.shape != (3, 512, 512)
                            or lr.shape != (3, 128, 128)
                            or hr.dtype != np.float32
                            or lr.dtype != np.float32
                            or mask.shape != (128, 128)
                            or mask.dtype != np.bool_
                            or not np.isfinite(hr).all()
                            or not np.isfinite(lr).all()
                            or hr.min() < 0
                            or hr.max() > 1
                            or lr.min() < 0
                            or lr.max() > 1
                        ):
                            raise ValueError("Invalid RGB pair arrays")
                        expected_mask = np.zeros((128, 128), bool)
                        expected_mask[2:-2, 2:-2] = True
                        if not np.array_equal(mask, expected_mask):
                            raise ValueError("Invalid synthetic LR support mask")
                        cell = meta.get("cell")
                        native_bytes = meta.get("native_covered_bytes")
                        if (
                            not isinstance(cell, list)
                            or len(cell) != 2
                            or any(type(c) is not int for c in cell)
                            or type(native_bytes) is not int
                            or native_bytes <= 0
                        ):
                            raise ValueError("Missing native coverage provenance")
                        expected_hr = [
                            2.5,
                            0,
                            cell[0] * 1280,
                            0,
                            -2.5,
                            (cell[1] + 1) * 1280,
                        ]
                        expected_lr = [
                            10,
                            0,
                            cell[0] * 1280,
                            0,
                            -10,
                            (cell[1] + 1) * 1280,
                        ]
                        if (
                            meta.get("transform") != expected_hr
                            or meta.get("lr_transform") != expected_lr
                        ):
                            raise ValueError("Pair grid differs from 2.5m/10m cell")
                        cell_key = (receipt["source_id"], meta["crs"], *cell)
                        if cell_key in cells:
                            raise ValueError("Duplicate source cell")
                        cells.add(cell_key)
                        provenance = dict(
                            meta,
                            site_id=receipt["group_id"],
                            bands=["R", "G", "B"],
                            hr_m=2.5,
                            lr_m=10,
                            lr_origin="synthetic_hr_only",
                            gaussian_sigma_hr=2,
                            gaussian_truncate=4,
                            gaussian_mode="reflect",
                            source_sha256=receipt["source_sha256"],
                        )
                        relative = source_dir / member.name
                        destination = output / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with destination.open("wb") as stream:
                            np.savez(
                                stream,
                                hr=hr,
                                lr=lr,
                                hr_mask=np.ones((512, 512), bool),
                                lr_mask=mask,
                                provenance=json.dumps(provenance, sort_keys=True),
                            )
                        rows.append(
                            dict(
                                path=relative.as_posix(),
                                sha256=digest(destination),
                                site_id=receipt["group_id"],
                                split=split,
                                source_id=receipt["source_id"],
                                source_object=obj,
                                source_sha256=receipt["source_sha256"],
                                cell=cell,
                                crs=meta["crs"],
                                native_covered_bytes=native_bytes,
                            )
                        )
                        covered += native_bytes
                        count += 1
        if count != receipt["pairs"] or covered != receipt["accepted_source_bytes"]:
            raise ValueError("Receipt coverage or pair count mismatch")
        totals[split]["pairs"] += count
        totals[split]["accepted_source_bytes"] += covered
        print(
            json.dumps(dict(staged_pairs=len(rows), source=receipt["source_id"])),
            flush=True,
        )
    if dict(totals) != manifest["totals"]:
        raise ValueError("Aggregate coverage mismatch")
    (output / "index.json").write_text(json.dumps(rows, indent=2))
    (output / "staging-receipt.json").write_text(
        json.dumps(
            dict(
                manifest_sha256=manifest["sha256"],
                index_sha256=digest(output / "index.json"),
                totals=dict(totals),
                optimizer_consumed_bytes=0,
            ),
            indent=2,
        )
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gcloud", default="gcloud")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    def fetch(uri, destination):
        subprocess.run(
            [args.gcloud, "storage", "cp", uri, str(destination)], check=True
        )

    stage_dataset(
        json.loads(Path(args.manifest).read_text()), args.output, fetch, args.workers
    )


if __name__ == "__main__":
    main()
