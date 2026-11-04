import pytest

from scripts.prepare_rgb_expansion_cloud import (
    validate_decisions,
    accepted_receipt,
    safe_member,
    validate_acquisition,
)


def test_exhaustive_reasoned_decisions_and_index_identity():
    index = dict(sha256="index", pairs=[dict(pair_id="a"), dict(pair_id="b")])
    decisions = dict(
        schema=1,
        review_index_sha256="index",
        decisions=[
            dict(pair_id="a", decision="accept", reason="Clear harbor"),
            dict(pair_id="b", decision="reject", reason="Cloud"),
        ],
    )
    assert validate_decisions(index, decisions) == {"a"}
    for invalid in (
        {**decisions, "review_index_sha256": "wrong"},
        {**decisions, "decisions": decisions["decisions"][:1]},
        {**decisions, "decisions": decisions["decisions"] * 2},
    ):
        with pytest.raises(ValueError):
            validate_decisions(index, invalid)


def test_zero_acceptance_removes_shards_and_native_bytes():
    receipt = dict(
        pairs=5,
        accepted_source_bytes=90,
        shards=[dict(uri="old")],
        source_sha256="unchanged",
    )
    result = accepted_receipt(receipt, [], [])
    assert result["pairs"] == result["accepted_source_bytes"] == 0
    assert result["shards"] == []
    assert receipt["pairs"] == 5


@pytest.mark.parametrize(
    "name", ["../000001.npz", "/000001.npz", "sub/000001.npz", "bad.npz"]
)
def test_unsafe_member_names_rejected(name):
    with pytest.raises(ValueError):
        safe_member(name, 10, True)


def test_acquisition_requires_exact_state_and_nonduplicate_hashes():
    manifest = dict(sha256="m", bucket="b", prefix="raw", sources=[dict(source_id="s")])
    state = dict(
        manifest_sha256="m", status="complete", tasks={"s": dict(state="COMPLETED")}
    )
    obj = dict(
        bucket="b",
        name="raw/s/rgb.tif",
        source_id="s",
        generation="1",
        size=10,
        sha256="a" * 64,
        duplicate=False,
    )
    qa = dict(status="complete", objects=[obj])
    validate_acquisition(manifest, state, qa)
    with pytest.raises(ValueError):
        validate_acquisition(
            manifest, state, {**qa, "objects": [{**obj, "duplicate": True}]}
        )
    with pytest.raises(ValueError):
        validate_acquisition(manifest, {**state, "manifest_sha256": "other"}, qa)


def test_parent_payload_and_generation_mismatch_blocked():
    from scripts.prepare_rgb_expansion_cloud import (
        validate_parent,
        validate_receipt_source,
    )
    from src.satellite_training_data import seal

    parent = seal(dict(sources=[]))
    validate_parent(parent, parent["sha256"])
    with pytest.raises(ValueError, match="parent"):
        validate_parent(parent, "wrong")
    with pytest.raises(ValueError, match="parent"):
        validate_parent({**parent, "sources": [1]}, parent["sha256"])
    receipt = dict(
        source_id="s",
        source_sha256="hash",
        source_object=dict(uri="gs://b/a", generation="2", bytes=3),
    )
    with pytest.raises(ValueError, match="source"):
        validate_receipt_source(
            receipt,
            {"gs://b/a": dict(source_id="s", sha256="hash", generation="1", size=3)},
        )


def test_merge_preserves_all_parent_rows_and_heldouts():
    from scripts.prepare_marine_addition import merge_dataset
    from src.satellite_training_data import seal

    sources = [
        dict(
            source_id=s,
            source_sha256=s,
            source_object=dict(uri=s),
            split=s,
            group_id=s,
            pairs=1,
            accepted_source_bytes=3,
            shards=[],
        )
        for s in ("train", "validation", "test")
    ]
    parent = seal(dict(sources=sources))
    added = dict(
        source_id="new",
        source_sha256="new",
        source_object=dict(uri="new"),
        split="train",
        group_id="new",
        pairs=1,
        accepted_source_bytes=9,
        shards=[{}],
    )
    result = merge_dataset(parent, [added])
    assert result["sources"][:3] == sources
    assert result["totals"]["train"] == dict(pairs=2, accepted_source_bytes=12)
    with pytest.raises(ValueError, match="heldouts"):
        merge_dataset(parent, [{**added, "group_id": "test"}])


def test_every_pair_degradation_is_checked_including_second_member(tmp_path):
    import io
    import json
    import shutil
    import tarfile
    from types import SimpleNamespace
    import numpy as np
    from scripts.prepare_rgb_expansion_cloud import verified_members
    from src.skysat_prepare import degrade_rgb, digest

    original = tmp_path / "original.tar"
    receipt = dict(
        source_id="s",
        group_id="g",
        split="train",
        source_object=dict(uri="gs://b/raw", generation="1", bytes=9),
        manifest_sha256="m",
        recipe="rgb-v2",
        pairs=2,
        accepted_source_bytes=6,
    )
    row = dict(asset_id="asset", group_id="g", grid=dict(crsCode="EPSG:32644"))
    hr = np.full((3, 512, 512), 0.5, dtype=np.float32)
    lr, mask = degrade_rgb(hr, np.ones((512, 512), bool))
    with tarfile.open(original, "w") as tar:
        for i in range(2):
            info = {
                **{
                    k: receipt[k]
                    for k in (
                        "source_id",
                        "group_id",
                        "split",
                        "source_object",
                        "manifest_sha256",
                        "recipe",
                    )
                },
                "asset_id": "asset",
                "crs": "EPSG:32644",
                "cell": [i, 0],
                "native_covered_bytes": 3,
                "transform": [2.5, 0, i * 1280, 0, -2.5, 1280],
                "lr_transform": [10, 0, i * 1280, 0, -10, 1280],
            }
            buffer = io.BytesIO()
            np.savez_compressed(
                buffer,
                hr=hr,
                lr=lr if i == 0 else lr + np.float32(0.1),
                mask=mask,
                metadata=json.dumps(info),
            )
            member = tarfile.TarInfo(f"{i:06d}.npz")
            member.size = len(buffer.getvalue())
            tar.addfile(member, io.BytesIO(buffer.getvalue()))
    receipt["shards"] = [dict(uri="gs://b/shard", sha256=digest(original))]
    store = SimpleNamespace(
        fetch=lambda uri, path: shutil.copyfile(original, path),
        blob=lambda uri: SimpleNamespace(
            reload=lambda: None, generation=1, size=original.stat().st_size
        ),
    )
    iterator = verified_members(store, receipt, row, tmp_path, set())
    assert next(iterator)[0] == "000000.npz"
    with pytest.raises(ValueError, match="degradation"):
        next(iterator)
