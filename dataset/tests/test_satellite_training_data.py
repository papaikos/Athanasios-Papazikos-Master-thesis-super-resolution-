import hashlib
import io
import json
import tarfile

import numpy as np
import pytest

from src.satellite_training_data import stage_dataset, seal


def fixture(tmp_path, name="000000.npz", split="train"):
    obj = {"uri": "gs://bucket/source.tif", "generation": "1", "bytes": 10}
    meta = dict(
        source_id="source",
        group_id="group",
        split=split,
        source_object=obj,
        manifest_sha256="a" * 64,
        recipe="rgb-v2",
        cell=[1, 2],
        native_covered_bytes=7680000,
        transform=[2.5, 0, 1280, 0, -2.5, 3840],
        lr_transform=[10, 0, 1280, 0, -10, 3840],
        crs="EPSG:32632",
    )
    mask = np.zeros((128, 128), bool)
    mask[2:-2, 2:-2] = True
    content = io.BytesIO()
    np.savez_compressed(
        content,
        hr=np.ones((3, 512, 512), np.float32) * 0.5,
        lr=np.ones((3, 128, 128), np.float32) * 0.5,
        mask=mask,
        metadata=json.dumps(meta),
    )
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(content.getvalue())
        tar.addfile(info, io.BytesIO(content.getvalue()))
    receipt = dict(
        source_id="source",
        group_id="group",
        split=split,
        source_object=obj,
        source_sha256="b" * 64,
        manifest_sha256="a" * 64,
        recipe="rgb-v2",
        pairs=1,
        accepted_source_bytes=7680000,
        shards=[
            dict(
                uri="gs://bucket/shard.tar",
                generation="2",
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            )
        ],
    )
    manifest = seal(
        dict(
            version=1,
            sources=[receipt],
            totals={split: dict(pairs=1, accepted_source_bytes=7680000)},
        )
    )

    def fetch(uri, dest):
        dest.write_bytes(archive.read_bytes())

    return manifest, fetch


def test_normalizes_rgb_and_keeps_accounting(tmp_path):
    manifest, fetch = fixture(tmp_path)
    rows = stage_dataset(manifest, tmp_path / "out", fetch)
    assert rows[0]["native_covered_bytes"] == 7680000
    with np.load(tmp_path / "out" / rows[0]["path"]) as data:
        assert data["hr"].shape == (3, 512, 512)
        assert data["hr_mask"].all()
        assert json.loads(str(data["provenance"]))["gaussian_mode"] == "reflect"


def test_bad_checksum_rejected(tmp_path):
    manifest, fetch = fixture(tmp_path)
    manifest["sources"][0]["shards"][0]["sha256"] = "0" * 64
    manifest = seal(manifest)
    with pytest.raises(ValueError, match="checksum"):
        stage_dataset(manifest, tmp_path / "out", fetch)


def test_path_traversal_rejected(tmp_path):
    manifest, fetch = fixture(tmp_path, "../escape.npz")
    with pytest.raises(ValueError, match="Unsafe"):
        stage_dataset(manifest, tmp_path / "out", fetch)
    assert not (tmp_path / "escape.npz").exists()


def test_pair_split_disagreement_rejected(tmp_path):
    manifest, fetch = fixture(tmp_path)
    manifest["sources"][0]["split"] = "test"
    manifest = seal(manifest)
    with pytest.raises(ValueError, match="provenance"):
        stage_dataset(manifest, tmp_path / "out", fetch)


def test_native_coverage_mismatch_rejected(tmp_path):
    manifest, fetch = fixture(tmp_path)
    manifest["sources"][0]["accepted_source_bytes"] += 1
    manifest = seal(manifest)
    with pytest.raises(ValueError, match="coverage"):
        stage_dataset(manifest, tmp_path / "out", fetch)


def test_geographic_leakage_rejected(tmp_path):
    manifest, fetch = fixture(tmp_path)
    import copy

    second = copy.deepcopy(manifest["sources"][0])
    second.update(split="validation", source_sha256="c" * 64)
    second["source_object"]["uri"] = "gs://bucket/second.tif"
    manifest["sources"].append(second)
    manifest = seal(manifest)
    with pytest.raises(ValueError, match="split leakage"):
        stage_dataset(manifest, tmp_path / "out", fetch)


def test_tampered_frozen_manifest_rejected(tmp_path):
    manifest, fetch = fixture(tmp_path)
    manifest["totals"]["train"]["pairs"] = 2
    with pytest.raises(ValueError, match="manifest checksum"):
        stage_dataset(manifest, tmp_path / "out", fetch)


def test_prefetch_bounded_and_consumed_in_manifest_order(tmp_path, monkeypatch):
    import threading
    import src.satellite_training_data as module

    barrier = threading.Barrier(4)
    lock = threading.Lock()
    active = 0
    peak = 0
    consumed = []
    shards = [dict(uri=f"gs://bucket/{i}", generation="1") for i in range(8)]
    manifest = seal(dict(sources=[dict(shards=shards)]))

    def fetch(uri, destination):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        if int(uri.rsplit("/", 1)[-1].split("#")[0]) < 4:
            barrier.wait(timeout=5)
        destination.write_text(uri)
        with lock:
            active -= 1

    def consume(manifest, output, getter):
        for shard in shards:
            uri = shard["uri"] + "#1"
            getter(uri, output / "download.tar")
            consumed.append((output / "download.tar").read_text())
        return consumed

    monkeypatch.setattr(module, "_stage_dataset", consume)
    assert stage_dataset(manifest, tmp_path / "out", fetch) == [
        s["uri"] + "#1" for s in shards
    ]
    assert peak == 4


def test_normalized_pairs_load_in_rgb_trainer(tmp_path):
    import torch
    from src.train import read_pairs, load_pair

    manifest, fetch = fixture(tmp_path)
    stage_dataset(manifest, tmp_path / "out", fetch)
    paths = read_pairs(tmp_path / "out" / "index.json", "train")
    lr, hr, hr_mask, lr_mask = load_pair(paths[0], torch.device("cpu"), channels=3)
    assert lr.shape == (1, 3, 128, 128)
    assert hr.shape == (1, 3, 512, 512)
    assert hr_mask.shape == (1, 1, 512, 512)
    assert lr_mask.shape == (1, 1, 128, 128)
