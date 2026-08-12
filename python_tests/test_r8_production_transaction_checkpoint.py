from __future__ import annotations

import json
from pathlib import Path

import pytest

from spincore.production_transaction_checkpoint import (
    ProductionTransactionIdentity,
    load_current_production_transaction,
    publish_production_transaction,
)


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _identity(iteration: int = 3) -> ProductionTransactionIdentity:
    return ProductionTransactionIdentity(
        profile_id="spinprofile-v3:test",
        domain="THREE_HANDED",
        algorithm_seed=12345,
        completed_iteration=iteration,
        roots_per_iteration=64,
    )


def test_integrated_generation_round_trip(tmp_path):
    root = tmp_path / "txn"
    gid = publish_production_transaction(
        root,
        identity=_identity(),
        stream_checkpoint=_write(tmp_path / "stream.pt", b"stream-state"),
        scheduler_checkpoint=_write(tmp_path / "scheduler.json", b"scheduler-state"),
        algorithm_r_checkpoint=_write(tmp_path / "algorithm_r.pt", b"algorithm-r-state"),
    )
    loaded = load_current_production_transaction(root)
    assert loaded.generation_id == gid
    assert loaded.identity == _identity()
    assert loaded.component_paths["stream"].read_bytes() == b"stream-state"
    assert loaded.component_paths["scheduler"].read_bytes() == b"scheduler-state"
    assert loaded.component_paths["algorithm_r"].read_bytes() == b"algorithm-r-state"
    assert loaded.manifest["ready_for_tables"] is False


def test_new_generation_atomically_becomes_current_without_destroying_old(tmp_path):
    root = tmp_path / "txn"
    stream = _write(tmp_path / "stream.pt", b"stream-i1")
    scheduler = _write(tmp_path / "scheduler.json", b"scheduler-i1")
    algorithm_r = _write(tmp_path / "algorithm_r.pt", b"algorithm-r-i1")
    gid1 = publish_production_transaction(
        root,
        identity=_identity(1),
        stream_checkpoint=stream,
        scheduler_checkpoint=scheduler,
        algorithm_r_checkpoint=algorithm_r,
    )

    stream.write_bytes(b"stream-i2")
    scheduler.write_bytes(b"scheduler-i2")
    algorithm_r.write_bytes(b"algorithm-r-i2")
    gid2 = publish_production_transaction(
        root,
        identity=_identity(2),
        stream_checkpoint=stream,
        scheduler_checkpoint=scheduler,
        algorithm_r_checkpoint=algorithm_r,
    )

    assert gid1 != gid2
    assert (root / "generations" / gid1 / "manifest.json").is_file()
    assert (root / "generations" / gid2 / "manifest.json").is_file()
    assert load_current_production_transaction(root).generation_id == gid2


def test_component_tampering_fails_closed(tmp_path):
    root = tmp_path / "txn"
    publish_production_transaction(
        root,
        identity=_identity(),
        stream_checkpoint=_write(tmp_path / "stream.pt", b"stream-state"),
        scheduler_checkpoint=_write(tmp_path / "scheduler.json", b"scheduler-state"),
        algorithm_r_checkpoint=_write(tmp_path / "algorithm_r.pt", b"algorithm-r-state"),
    )
    loaded = load_current_production_transaction(root)
    loaded.component_paths["algorithm_r"].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="component (size|hash) mismatch"):
        load_current_production_transaction(root)


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
    with pytest.raises(FileNotFoundError):
        publish_production_transaction(
            tmp_path / "txn",
            identity=_identity(),
            stream_checkpoint=tmp_path / "missing.pt",
            scheduler_checkpoint=_write(tmp_path / "scheduler.json", b"scheduler"),
            algorithm_r_checkpoint=_write(tmp_path / "algorithm_r.pt", b"algorithm-r"),
        )

    empty = _write(tmp_path / "empty.pt", b"")
    with pytest.raises(ValueError, match="empty stream checkpoint"):
        publish_production_transaction(
            tmp_path / "txn2",
            identity=_identity(),
            stream_checkpoint=empty,
            scheduler_checkpoint=_write(tmp_path / "scheduler2.json", b"scheduler"),
            algorithm_r_checkpoint=_write(tmp_path / "algorithm_r2.pt", b"algorithm-r"),
        )
