"""Cloud-only rgb-v2 expansion preparation, exhaustive review and dataset staging."""

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
from types import SimpleNamespace

from scripts.expand_rgb_cloud import require_cloud_vm, validate_reviewed_manifest
from scripts.prepare_marine_addition import merge_dataset, validate_pair_arrays
from src.satellite_training_data import digest, seal, stage_dataset


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def safe_member(name, size, is_file):
    if (
        not is_file
        or not re.fullmatch(r"[0-9]{6}\.npz", name)
        or not 0 < size < 8_000_000
    ):
        raise ValueError("Unsafe pair member")


def validate_decisions(index, decisions):
    if (
        decisions.get("schema") != 1
        or decisions.get("review_index_sha256") != index["sha256"]
    ):
        raise ValueError("Review index identity mismatch")
    expected = {p["pair_id"] for p in index["pairs"]}
    rows = decisions.get("decisions", [])
    ids = [row.get("pair_id") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Decisions must cover every candidate exactly once")
    if any(
        row.get("decision") not in ("accept", "reject")
        or not isinstance(row.get("reason"), str)
        or not row["reason"].strip()
        for row in rows
    ):
        raise ValueError("Every decision requires accept/reject and a reason")
    return {row["pair_id"] for row in rows if row["decision"] == "accept"}


def accepted_receipt(receipt, pairs, shards):
    if bool(pairs) != bool(shards):
        raise ValueError("Accepted pair/shard accounting mismatch")
    return {
        **receipt,
        "pairs": len(pairs),
        "accepted_source_bytes": sum(p["native_covered_bytes"] for p in pairs),
        "shards": shards,
    }


def validate_parent(parent, expected_sha):
    if (
        seal(parent)["sha256"] != parent.get("sha256")
        or parent["sha256"] != expected_sha
    ):
        raise ValueError("Exact parent manifest mismatch")


def validate_receipt_source(receipt, objects):
    source = receipt["source_object"]
    obj = objects.get(source["uri"])
    if (
        obj is None
        or receipt["source_sha256"] != obj["sha256"]
        or source["generation"] != str(obj["generation"])
        or source["bytes"] != int(obj["size"])
        or receipt["source_id"] != obj["source_id"]
    ):
        raise ValueError("Preparation receipt source differs from acquisition")


def validate_acquisition(manifest, state, qa):
    ids = {r["source_id"] for r in manifest["sources"]}
    if (
        state.get("manifest_sha256") != manifest["sha256"]
        or state.get("status") != "complete"
        or qa.get("status") != "complete"
    ):
        raise ValueError("Acquisition is incomplete or mismatched")
    if set(state["tasks"]) != ids or any(
        t["state"] != "COMPLETED" for t in state["tasks"].values()
    ):
        raise ValueError("Every approved export must complete")
    objects = qa.get("objects", [])
    if {obj["source_id"] for obj in objects} != ids:
        raise ValueError("QA missing source objects")
    identities, hashes = set(), set()
    for obj in objects:
        identity = (obj["bucket"], obj["name"], str(obj["generation"]))
        if (
            obj.get("duplicate") is not False
            or identity in identities
            or obj["sha256"] in hashes
        ):
            raise ValueError("Duplicate source object/hash")
        if (
            obj["bucket"] != manifest["bucket"]
            or not obj["name"].startswith(f"{manifest['prefix']}/{obj['source_id']}/")
            or not str(obj["generation"]).isdigit()
            or int(obj["size"]) <= 0
            or not re.fullmatch(r"[a-f0-9]{64}", obj["sha256"])
        ):
            raise ValueError("Invalid pinned acquisition object")
        identities.add(identity)
        hashes.add(obj["sha256"])


class Store:
    """GCS SDK transport; immutable publication and generation-pinned reads."""

    def __init__(self, client, work):
        self.client, self.work = client, work

    def blob(self, uri):
        match = re.fullmatch(r"gs://([^/]+)/([^#]+)(?:#([0-9]+))?", uri)
        if not match or ".." in Path(match[2]).parts:
            raise ValueError("Invalid GCS URI")
        return self.client.bucket(match[1]).blob(
            match[2], generation=int(match[3]) if match[3] else None
        )

    def fetch(self, uri, path):
        blob = self.blob(uri)
        blob.reload()
        blob.download_to_filename(
            str(path), if_generation_match=int(blob.generation), checksum="crc32c"
        )

    def json(self, uri):
        blob = self.blob(uri)
        blob.reload()
        return json.loads(
            blob.download_as_text(if_generation_match=int(blob.generation))
        )

    def put(self, path, uri):
        from google.api_core.exceptions import PreconditionFailed

        path = Path(path)
        sha = digest(path)
        blob = self.blob(uri)
        blob.metadata = {"sha256": sha}
        try:
            blob.upload_from_filename(
                str(path), if_generation_match=0, checksum="crc32c"
            )
        except PreconditionFailed:
            check = self.work / "immutable-readback"
            self.fetch(uri, check)
            try:
                if digest(check) != sha:
                    raise ValueError("Immutable publication changed: " + uri)
            finally:
                check.unlink(missing_ok=True)
        blob.reload()
        if int(blob.size) != path.stat().st_size or not blob.crc32c:
            raise ValueError("Published object metadata mismatch")
        return dict(
            uri=uri,
            generation=str(blob.generation),
            bytes=int(blob.size),
            sha256=sha,
            crc32c=blob.crc32c,
        )

    def put_json(self, value, uri):
        path = self.work / "publication.json"
        save(path, value)
        result = self.put(path, uri)
        if self.json(uri + "#" + result["generation"]) != value:
            raise ValueError("Published JSON readback mismatch")
        return result


@contextmanager
def sdk_prepare_transport(store):
    """Adapt the established preparation transport without changing its recipe."""
    from google.api_core.exceptions import NotFound
    from src import skysat_prepare as core

    old_cloud, old_publish = core.cloud, core.publish

    def cloud(args, *command, check=True):
        if len(command) != 3 or command[0] != "cp":
            raise ValueError("Unexpected preparation transport command")
        try:
            store.fetch(str(command[1]), Path(command[2]))
        except NotFound:
            if check:
                raise
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    core.cloud = cloud
    core.publish = lambda args, path, destination: store.put(path, destination)
    try:
        yield
    finally:
        core.cloud, core.publish = old_cloud, old_publish


def inputs(store, config, work):
    prefix = f"gs://{config['bucket']}/{config['acquisition_prefix']}"
    manifest = store.json(prefix + "/acquisition-manifest.json")
    state = store.json(prefix + "/state.json")
    qa = store.json(prefix + "/qa.json")
    validate_acquisition(manifest, state, qa)
    parent = store.json(config["parent_manifest_uri"])
    validate_parent(parent, config["parent_payload_sha256"])
    inventory = read(config["inventory"])
    assets = inventory["assets"] if isinstance(inventory, dict) else inventory
    priors = [read(p) for p in config["prior_manifests"]]
    if not priors:
        raise ValueError("All prior manifests required")
    for prior in priors:
        if seal(prior)["sha256"] != prior["sha256"]:
            raise ValueError("Prior manifest checksum mismatch")
        if any(
            ref["sha256"] not in {p["sha256"] for p in priors}
            for ref in prior.get("prior_manifests", [])
        ):
            raise ValueError("Missing ancestor manifest")
    validate_reviewed_manifest(
        manifest,
        {"bucket": config["bucket"], "output_prefix": config["acquisition_prefix"]},
        assets,
        priors,
    )
    if not {s["source_id"] for s in parent["sources"]} <= {
        s["source_id"] for p in priors for s in p["sources"]
    }:
        raise ValueError("Prior manifests do not cover complete parent")
    parent_hashes = {s["source_sha256"] for s in parent["sources"]}
    for obj in qa["objects"]:
        if obj["sha256"] in parent_hashes:
            raise ValueError("Source hash already in parent")
        uri = f"gs://{obj['bucket']}/{obj['name']}"
        current = store.blob(uri)
        current.reload()
        if (
            str(current.generation) != str(obj["generation"])
            or int(current.size) != int(obj["size"])
            or current.crc32c != obj["crc32c"]
        ):
            raise ValueError("Acquisition object generation or checksum changed")
    for name, value in (
        ("manifest.json", manifest),
        ("state.json", state),
        ("objects.json", qa["objects"]),
    ):
        save(work / name, value)
    return manifest, qa, parent


def pair_id(info):
    return hashlib.sha256(
        json.dumps(
            [info["source_object"], info["asset_id"], info["crs"], info["cell"]],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:24]


def verified_members(store, receipt, row, work, seen):
    import numpy as np
    from src.skysat_prepare import degrade_rgb

    count = covered = 0
    for shard in receipt["shards"]:
        path = work / "pair-shard.tar"
        uri = shard["uri"] + (
            "#" + shard["generation"] if "generation" in shard else ""
        )
        store.fetch(uri, path)
        if digest(path) != shard["sha256"]:
            raise ValueError("Pair shard SHA mismatch")
        blob = store.blob(uri)
        blob.reload()
        shard.update(generation=str(blob.generation), bytes=int(blob.size))
        with tarfile.open(path) as archive:
            names = set()
            for member in archive:
                safe_member(member.name, member.size, member.isfile())
                if member.name in names:
                    raise ValueError("Duplicate archive member")
                names.add(member.name)
                content = archive.extractfile(member).read()
                with np.load(io.BytesIO(content), allow_pickle=False) as pair:
                    info = json.loads(str(pair["metadata"]))
                    hr, lr, mask = pair["hr"], pair["lr"], pair["mask"]
                cell = validate_pair_arrays(hr, lr, mask, info, row, receipt)
                if cell in seen:
                    raise ValueError("Duplicate observation cell")
                seen.add(cell)
                expected_lr, expected_mask = degrade_rgb(hr, np.ones((512, 512), bool))
                if not np.allclose(
                    lr, expected_lr, atol=1e-7, rtol=1e-6
                ) or not np.array_equal(mask, expected_mask):
                    raise ValueError("Pair degradation mismatch")
                count += 1
                covered += info["native_covered_bytes"]
                yield member.name, content, info, hr
        path.unlink()
    if count != receipt["pairs"] or covered != receipt["accepted_source_bytes"]:
        raise ValueError("Candidate receipt accounting mismatch")


def prepare_phase(store, config, work, manifest, qa):
    import numpy as np
    from PIL import Image, ImageDraw
    from src.skysat_prepare import prepare

    base = f"gs://{config['bucket']}/{config['output_prefix']}"
    save(work / "status.json", dict(stage="verifying_native_source_hashes"))
    # Verify every native object SHA before generating any pairs.
    for obj in qa["objects"]:
        raw = work / "raw-verify.tif"
        store.fetch(f"gs://{obj['bucket']}/{obj['name']}#{obj['generation']}", raw)
        if digest(raw) != obj["sha256"]:
            raise ValueError("Acquisition native SHA mismatch")
        raw.unlink()
    args = SimpleNamespace(
        manifest=str(work / "manifest.json"),
        task_state=str(work / "state.json"),
        objects=str(work / "objects.json"),
        scratch=str(work),
        max_sources=len(manifest["sources"]),
        max_pairs_per_source=1000000,
        output_prefix=base + "/candidates",
        progress_file=str(work / "prepare-progress.json"),
        summary_file=str(work / "prepare-summary.json"),
    )
    with sdk_prepare_transport(store):
        receipts = prepare(args)
    save(work / "status.json", dict(stage="verifying_pairs_and_contact_sheets"))
    qa_objects = {f"gs://{o['bucket']}/{o['name']}": o for o in qa["objects"]}
    if {r["source_object"]["uri"] for r in receipts} != set(qa_objects) or len(
        receipts
    ) != len(qa_objects):
        raise ValueError("Candidate receipts omit or repeat source objects")
    rows = {r["source_id"]: r for r in manifest["sources"]}
    seen, pairs, sheets, tiles = set(), [], [], []

    def flush_sheet():
        if not tiles:
            return
        sheet = Image.new("RGB", (6 * 256, 4 * 290), "white")
        draw = ImageDraw.Draw(sheet)
        for n, (thumbnail, label) in enumerate(tiles):
            x, y = (n % 6) * 256, (n // 6) * 290
            sheet.paste(thumbnail, (x, y))
            draw.text((x + 2, y + 257), label, fill="black")
        path = work / "contact-sheet.jpg"
        sheet.save(path, quality=95)
        sheets.append(
            store.put(path, base + f"/review/contact-sheet-{len(sheets):04d}.jpg")
        )
        save(
            work / "status.json",
            dict(
                stage="verifying_pairs_and_contact_sheets",
                contact_sheets=len(sheets),
                candidate_pairs=len(pairs),
            ),
        )
        tiles.clear()

    for receipt in receipts:
        validate_receipt_source(receipt, qa_objects)
        row = rows[receipt["source_id"]]
        for name, content, info, hr in verified_members(
            store, receipt, row, work, seen
        ):
            identity = pair_id(info)
            gray = hr.mean(axis=0)
            pairs.append(
                dict(
                    pair_id=identity,
                    source_id=info["source_id"],
                    source_object=info["source_object"],
                    asset_id=info["asset_id"],
                    cell=info["cell"],
                    crs=info["crs"],
                    category=row["category"],
                    member=name,
                    npz_sha256=hashlib.sha256(content).hexdigest(),
                    native_covered_bytes=info["native_covered_bytes"],
                    contact_sheet=f"contact-sheet-{len(sheets):04d}.jpg",
                    contact_position=len(tiles),
                    metrics=dict(
                        mean=float(hr.mean()),
                        std=float(hr.std()),
                        edge_mean=float(
                            (
                                np.abs(np.diff(gray, axis=0)).mean()
                                + np.abs(np.diff(gray, axis=1)).mean()
                            )
                            / 2
                        ),
                        bright_fraction=float((gray > 0.9).mean()),
                    ),
                )
            )
            thumb = Image.fromarray(
                np.clip(hr.transpose(1, 2, 0) * 255, 0, 255).astype("uint8")
            ).resize((256, 256), Image.Resampling.LANCZOS)
            tiles.append((thumb, identity + "\n" + str(info["cell"])))
            if len(tiles) == 24:
                flush_sheet()
    flush_sheet()
    index = seal(
        dict(
            schema=1,
            acquisition_manifest_sha256=manifest["sha256"],
            parent_payload_sha256=config["parent_payload_sha256"],
            pairs=pairs,
            contact_sheets=sheets,
            status="pending_review",
        )
    )
    store.put_json(
        seal(
            dict(
                schema=1,
                receipts=receipts,
                acquisition_manifest_sha256=manifest["sha256"],
            )
        ),
        base + "/review/candidate-receipts.json",
    )
    store.put_json(index, base + "/review/review-index.json")
    save(work / "review-index.json", index)
    return dict(
        status="pending_review",
        candidates=len(pairs),
        review_index_sha256=index["sha256"],
        review_index_uri=base + "/review/review-index.json",
    )


def finalize_phase(store, config, work, manifest, qa, parent):
    base = f"gs://{config['bucket']}/{config['output_prefix']}"
    index = store.json(base + "/review/review-index.json")
    candidate = store.json(base + "/review/candidate-receipts.json")
    for value in (index, candidate):
        if (
            seal(value)["sha256"] != value["sha256"]
            or value["acquisition_manifest_sha256"] != manifest["sha256"]
        ):
            raise ValueError("Candidate review identity mismatch")
    if index["parent_payload_sha256"] != parent["sha256"]:
        raise ValueError("Review parent differs")
    decisions = store.json(base + "/review/review-decisions.json")
    accepted = validate_decisions(index, decisions)
    if not accepted:
        raise ValueError("No accepted pairs; dataset expansion cannot be finalized")
    save(
        work / "status.json",
        dict(stage="repacking_reviewed_pairs", accepted_pairs=len(accepted)),
    )
    store.put_json(decisions, base + "/review/review-decisions-used.json")
    lookup = {p["pair_id"]: p for p in index["pairs"]}
    rows = {r["source_id"]: r for r in manifest["sources"]}
    qa_objects = {f"gs://{o['bucket']}/{o['name']}": o for o in qa["objects"]}
    if {r["source_object"]["uri"] for r in candidate["receipts"]} != set(
        qa_objects
    ) or len(candidate["receipts"]) != len(qa_objects):
        raise ValueError("Candidate receipts omit or repeat source objects")
    additions, seen, encountered = [], set(), set()
    for receipt in candidate["receipts"]:
        validate_receipt_source(receipt, qa_objects)
        chosen, shards, batch = [], [], []
        object_id = hashlib.sha256(
            json.dumps(receipt["source_object"], sort_keys=True).encode()
        ).hexdigest()[:24]

        def flush():
            if not batch:
                return
            path = work / "accepted-shard.tar"
            with tarfile.open(path, "w") as archive:
                for name, content in batch:
                    member = tarfile.TarInfo(name)
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
            shards.append(
                store.put(
                    path, base + f"/accepted/{object_id}/pairs-{len(shards):05d}.tar"
                )
            )
            path.unlink()
            batch.clear()

        for name, content, info, _ in verified_members(
            store, receipt, rows[receipt["source_id"]], work, seen
        ):
            identity = pair_id(info)
            recorded = lookup.get(identity)
            if (
                identity in encountered
                or recorded is None
                or recorded["npz_sha256"] != hashlib.sha256(content).hexdigest()
                or recorded["native_covered_bytes"] != info["native_covered_bytes"]
            ):
                raise ValueError("Pair differs from reviewed identity")
            encountered.add(identity)
            if identity in accepted:
                chosen.append(recorded)
                batch.append((name, content))
                if len(batch) == 32:
                    flush()
        flush()
        addition = accepted_receipt(receipt, chosen, shards)
        additions.append(addition)
        store.put_json(addition, base + f"/accepted/{object_id}/receipt.json")
    if encountered != set(lookup):
        raise ValueError("Review index not fully represented")
    merged = merge_dataset(parent, additions)
    merged["marine_content_review"] = dict(
        status="all_new_cells_reviewed",
        review_index_sha256=index["sha256"],
        decisions_sha256=seal(decisions)["sha256"],
        accepted_pairs=len(accepted),
        rejected_pairs=len(lookup) - len(accepted),
        note="Per-cell visual maritime relevance/quality review; existing parent review status unchanged",
    )
    merged = seal(merged)
    if merged["sources"][: len(parent["sources"])] != parent["sources"]:
        raise ValueError("Parent rows changed")
    staged = work / "staged"
    save(
        work / "status.json",
        dict(stage="staging_complete_merged_dataset", manifest_sha256=merged["sha256"]),
    )
    stage_dataset(merged, staged, store.fetch, workers=4)
    save(work / "status.json", dict(stage="verifying_training_loaders"))
    from src.train import PairCoverage, load_pair, read_pairs

    verified = {
        split: read_pairs(staged / "index.json", split)
        for split in ("train", "validation", "test")
    }
    coverage = PairCoverage(staged / "index.json", verified["train"])
    staged_rows = read(staged / "index.json")
    new_ids = set(rows)
    new_paths = [staged / r["path"] for r in staged_rows if r["source_id"] in new_ids]
    for path in new_paths + verified["validation"][:1] + verified["test"][:1]:
        load_pair(path, "cpu", channels=3)
    if len(new_paths) != len(accepted):
        raise ValueError("Staged accepted pair count mismatch")
    evidence = dict(
        status="passed",
        split_pairs={k: len(v) for k, v in verified.items()},
        new_pairs_loaded=len(new_paths),
        heldout_pairs_loaded=2,
        coverage=coverage.status(0),
        training_started=False,
    )
    archive = work / "dataset.tar"
    save(
        work / "status.json",
        dict(stage="publishing_staged_dataset", staged_pairs=len(staged_rows)),
    )
    with tarfile.open(archive, "w") as tar:
        for path in sorted(staged.rglob("*")):
            if path.is_file():
                tar.add(
                    path, arcname=path.relative_to(staged).as_posix(), recursive=False
                )
    artifacts = dict(
        dataset=store.put(archive, base + "/ready/dataset.tar"),
        index=store.put(staged / "index.json", base + "/ready/index.json"),
        staging_receipt=store.put(
            staged / "staging-receipt.json", base + "/ready/staging-receipt.json"
        ),
        loader_evidence=store.put_json(evidence, base + "/ready/loader-evidence.json"),
        manifest=store.put_json(merged, base + "/training-dataset.json"),
    )
    if (
        store.json(
            artifacts["manifest"]["uri"] + "#" + artifacts["manifest"]["generation"]
        )["sha256"]
        != merged["sha256"]
    ):
        raise ValueError("Published merged manifest readback mismatch")
    completion = dict(
        status="ready",
        training_started=False,
        manifest_payload_sha256=merged["sha256"],
        parent_payload_sha256=parent["sha256"],
        added_pairs=len(accepted),
        added_native_bytes=sum(r["accepted_source_bytes"] for r in additions),
        totals=merged["totals"],
        artifacts=artifacts,
    )
    store.put_json(completion, base + "/ready/completion.json")
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "finalize"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    require_cloud_vm()
    from google.cloud import storage

    config = read(args.config)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    store = Store(storage.Client(project=config.get("project")), args.work_dir)
    try:
        manifest, qa, parent = inputs(store, config, args.work_dir)
        if args.phase == "prepare":
            result = prepare_phase(store, config, args.work_dir, manifest, qa)
        else:
            result = finalize_phase(store, config, args.work_dir, manifest, qa, parent)
        save(args.work_dir / (args.phase + "-result.json"), result)
        save(args.work_dir / "status.json", result)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        failure = dict(
            status="failed", phase=args.phase, error=str(exc), training_started=False
        )
        save(args.work_dir / (args.phase + "-failure.json"), failure)
        blob = store.blob(
            f"gs://{config['bucket']}/{config['output_prefix']}/{args.phase}-failure.json"
        )
        blob.upload_from_string(json.dumps(failure), content_type="application/json")
        raise


if __name__ == "__main__":
    main()
