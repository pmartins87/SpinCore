from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from spincore.production_transaction_checkpoint import (
    ProductionTransactionIdentity,
    load_current_production_transaction,
    publish_production_transaction,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(iteration: int = 3, *, seed: int = 12345) -> ProductionTransactionIdentity:
    return ProductionTransactionIdentity(
        profile_id="spinprofile-v3:test",
        domain="THREE_HANDED",
        algorithm_seed=seed,
        completed_iteration=iteration,
        roots_per_iteration=64,
    )


def _components(tmp_path: Path, identity: ProductionTransactionIdentity):
    tag = f"i{identity.completed_iteration}-s{identity.algorithm_seed}"
    stream = tmp_path / f"stream-{tag}.pt"
    torch.save({
        "schema": "SPINCORE_R7_CHECKPOINT_V2",
        "domain": identity.domain,
        "seed": identity.algorithm_seed,
        "progress": {"iteration": identity.completed_iteration},
    }, stream)

    scheduler = tmp_path / f"scheduler-{tag}.json"
    scheduler.write_text(json.dumps({
        "schema": "SPINCORE_R8_SCHEDULER_DURABLE_CHECKPOINT_V1",
        "scheduler": {
            "schema": "SPINCORE_R8_INDEPENDENT_STREAM_SCHEDULER_V2",
            "lease_counter": identity.completed_iteration,
            "streams": [{
                "profile_id": identity.profile_id,
                "domain": identity.domain,
                "algorithm_seed": identity.algorithm_seed,
                "stream_id": "test-stream-id-not-consumed-by-transaction-validator",
                "total_iterations": 10,
                "next_iteration": identity.completed_iteration + 1,
                "active_lease_id": None,
                "failed_attempts_for_next_iteration": 0,
                "last_checkpoint_sha256": None if identity.completed_iteration == 0 else _sha(stream),
                "last_checkpoint_locator": None if identity.completed_iteration == 0 else str(stream),
                "last_checkpoint_size_bytes": None if identity.completed_iteration == 0 else stream.stat().st_size,
                "last_receipt_id": None if identity.completed_iteration == 0 else "spinreceipt-v1:test",
            }],
            "ready_for_tables": False,
        },
        "ready_for_tables": False,
    }, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    algorithm_r = tmp_path / f"algorithm-r-{tag}.pt"
    expected_root = identity.completed_iteration * identity.roots_per_iteration
    torch.save({
        "schema": "SPINCORE_R8_CENTRAL_ALGORITHM_R_V2",
        "profile_id": identity.profile_id,
        "domain": identity.domain,
        "algorithm_seed": identity.algorithm_seed,
        "roots_per_iteration": identity.roots_per_iteration,
        "next_global_root": expected_root,
        "committed_roots": expected_root,
        "pending": [],
        "ready_for_tables": False,
    }, algorithm_r)
    return stream, scheduler, algorithm_r


def _publish(root: Path, tmp_path: Path, identity: ProductionTransactionIdentity):
    stream, scheduler, algorithm_r = _components(tmp_path, identity)
    gid = publish_production_transaction(
        root,
        identity=identity,
        stream_checkpoint=stream,
        scheduler_checkpoint=scheduler,
        algorithm_r_checkpoint=algorithm_r,
    )
    return gid, stream, scheduler, algorithm_r


def test_integrated_generation_round_trip(tmp_path):
    root = tmp_path / "txn"
    identity = _identity()
    gid, _, _, _ = _publish(root, tmp_path, identity)
    loaded = load_current_production_transaction(root)
    assert loaded.generation_id == gid
    assert loaded.identity == identity
    assert loaded.manifest["semantic_consistency_validated"] is True
    assert loaded.manifest["ready_for_tables"] is False


def test_new_generation_atomically_becomes_current_without_destroying_old(tmp_path):
    root = tmp_path / "txn"
    gid1, _, _, _ = _publish(root, tmp_path, _identity(1))
    gid2, _, _, _ = _publish(root, tmp_path, _identity(2))
    assert gid1 != gid2
    assert (root / "generations" / gid1 / "manifest.json").is_file()
    assert (root / "generations" / gid2 / "manifest.json").is_file()
    assert load_current_production_transaction(root).generation_id == gid2


def test_component_tampering_fails_closed(tmp_path):
    root = tmp_path / "txn"
    _publish(root, tmp_path, _identity())
    loaded = load_current_production_transaction(root)
    loaded.component_paths["algorithm_r"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="component (size|hash) mismatch"):
        load_current_production_transaction(root)


def test_cross_seed_component_mix_is_rejected_before_publish(tmp_path):
    identity = _identity(seed=12345)
    stream, scheduler, _ = _components(tmp_path, identity)
    _, _, wrong_algorithm_r = _components(tmp_path, _identity(seed=99999))
    with pytest.raises(ValueError, match="Algorithm-R algorithm-seed mismatch"):
        publish_production_transaction(
            tmp_path / "txn",
            identity=identity,
            stream_checkpoint=stream,
            scheduler_checkpoint=scheduler,
            algorithm_r_checkpoint=wrong_algorithm_r,
        )


def test_scheduler_must_acknowledge_exact_stream_bytes_and_iteration(tmp_path):
    identity = _identity(2)
    stream, scheduler, algorithm_r = _components(tmp_path, identity)
    wrapper = json.loads(scheduler.read_text(encoding="utf-8"))
    wrapper["scheduler"]["streams"][0]["last_checkpoint_sha256"] = "0" * 64
    scheduler.write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(ValueError, match="scheduler stream checkpoint SHA"):
        publish_production_transaction(
            tmp_path / "txn",
            identity=identity,
            stream_checkpoint=stream,
            scheduler_checkpoint=scheduler,
            algorithm_r_checkpoint=algorithm_r,
        )


def test_algorithm_r_root_position_must_match_completed_iteration(tmp_path):
    identity = _identity(3)
    stream, scheduler, algorithm_r = _components(tmp_path, identity)
    payload = torch.load(algorithm_r, map_location="cpu", weights_only=False)
    payload["next_global_root"] -= 1
    torch.save(payload, algorithm_r)
    with pytest.raises(ValueError, match="global-root/iteration mismatch"):
        publish_production_transaction(
            tmp_path / "txn",
            identity=identity,
            stream_checkpoint=stream,
            scheduler_checkpoint=scheduler,
            algorithm_r_checkpoint=algorithm_r,
        )


def test_pointer_to_incomplete_or_wrong_manifest_fails_closed(tmp_path):
    root = tmp_path / "txn"
    root.mkdir()
    (root / "CURRENT.json").write_text(json.dumps({
        "schema": "SPINCORE_R8_PRODUCTION_TRANSACTION_POINTER_V1",
        "generation_id": "missing",
        "manifest_sha256": "0" * 64,
        "ready_for_tables": False,
    }), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_current_production_transaction(root)


def test_missing_or_empty_component_is_rejected_before_publish(tmp_path):
    identity = _identity()
    _, scheduler, algorithm_r = _components(tmp_path, identity)
    with pytest.raises(FileNotFoundError):
        publish_production_transaction(
            tmp_path / "txn",
            identity=identity,
            stream_checkpoint=tmp_path / "missing.pt",
            scheduler_checkpoint=scheduler,
            algorithm_r_checkpoint=algorithm_r,
        )

    empty = tmp_path / "empty.pt"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty stream checkpoint"):
        publish_production_transaction(
            tmp_path / "txn2",
            identity=identity,
            stream_checkpoint=empty,
            scheduler_checkpoint=scheduler,
            algorithm_r_checkpoint=algorithm_r,
        )
