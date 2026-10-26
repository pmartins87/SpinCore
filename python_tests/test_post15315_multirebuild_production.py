from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import post15315_member_cache as cache
import post15315_collection as collection
import run_post15315_multirebuild_production as production


def sample(tag: bytes, iteration: int = 15315):
    return collection.distill.ActionStrategySample(
        observation=tag,
        legal=(True, True, False, True, False, True, False, True, True, True),
        target=(0.1, 0.2, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.1, 0.3),
        weight=1.0,
        iteration=iteration,
    )


def cache_payload():
    seeds = {
        "member": 1,
        "base_collection_seed": 153151001,
        "augment_collection_seed": 153152001,
        "specialist_collection_seed": 153153001,
        "tail_fit_seed": 153154001,
        "fullpool_fit_seed": 153155001,
        "specialist_fit_seed": 153156001,
    }
    rows = {
        "ordinary": ([sample(b"a")], [sample(b"b")]),
        "augmentation": ([sample(b"c")], [sample(b"d")]),
        "specialist": ([sample(b"e")], [sample(b"f")]),
    }
    payload = {
        "schema": cache.SCHEMA,
        "member": 1,
        "sample_iteration": 15315,
        "teacher_sha256": production.EXPECTED_TEACHER_SHA,
        "protocol_sha256": "proto",
        "member_seeds": seeds,
        "collection_threads": 2,
    }
    for key, (train, hold) in rows.items():
        payload[key] = {
            "stats": {"sample_iteration": 15315},
            "train_digest": collection.sample_digest(train),
            "holdout_digest": collection.sample_digest(hold),
            "train": train,
            "holdout": hold,
        }
    return payload, seeds


def test_collection_cache_roundtrip_revalidates_identity_and_order(tmp_path):
    payload, seeds = cache_payload()
    path = tmp_path / "cache.pt"
    torch.save(payload, path)
    loaded = cache.load_member_cache(
        path,
        expected_member=1,
        expected_teacher_sha256=production.EXPECTED_TEACHER_SHA,
        expected_protocol_sha256="proto",
        expected_member_seeds=seeds,
        expected_collection_threads=2,
    )
    assert loaded["ordinary"]["train_digest"] == payload["ordinary"]["train_digest"]


def test_collection_cache_rejects_unvalidated_thread_profile(tmp_path):
    payload, seeds = cache_payload()
    path = tmp_path / "cache.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="thread-profile"):
        cache.load_member_cache(
            path,
            expected_member=1,
            expected_teacher_sha256=production.EXPECTED_TEACHER_SHA,
            expected_protocol_sha256="proto",
            expected_member_seeds=seeds,
            expected_collection_threads=4,
        )


def test_collection_cache_rejects_tampered_sample_order_or_content(tmp_path):
    payload, seeds = cache_payload()
    payload["ordinary"]["train"] = [sample(b"tampered")]
    path = tmp_path / "cache.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="train digest mismatch"):
        cache.load_member_cache(
            path,
            expected_member=1,
            expected_teacher_sha256=production.EXPECTED_TEACHER_SHA,
            expected_protocol_sha256="proto",
            expected_member_seeds=seeds,
            expected_collection_threads=2,
        )


def perf_payload(protocol_sha: str):
    return {
        "schema": production.PERF_SCHEMA,
        "status": "PASS",
        "teacher_iteration": 15315,
        "teacher_sha256": production.EXPECTED_TEACHER_SHA,
        "protocol_sha256": protocol_sha,
        "criteria": {"a": True, "b": True},
        "production_profile_if_pass": {
            "collection_workers": 4,
            "collection_threads_per_worker": 2,
            "fit_threads": 8,
            "checkpoint_fanout_to_collection_workers": False,
            "member_seed_set": "FROZEN_MEMBERS_1_TO_8",
        },
    }


def test_production_runner_requires_exact_performance_artifact_sha_and_profile(tmp_path):
    path = tmp_path / "perf.json"
    path.write_text(json.dumps(perf_payload("proto"), sort_keys=True) + "\n", encoding="utf-8")
    digest = production.sha256(path)
    p = production.validate_performance_evidence(path, digest, "proto")
    assert p["production_profile_if_pass"]["collection_workers"] == 4

    with pytest.raises(RuntimeError, match="SHA mismatch"):
        production.validate_performance_evidence(path, "0" * 64, "proto")

    bad = perf_payload("proto")
    bad["production_profile_if_pass"]["fit_threads"] = 4
    path.write_text(json.dumps(bad, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="profile drift"):
        production.validate_performance_evidence(path, production.sha256(path), "proto")


def test_production_source_keeps_collection_parallel_and_fit_nonfanned():
    source = (ROOT / "tools" / "run_post15315_multirebuild_production.py").read_text(
        encoding="utf-8"
    )
    assert "max_workers=4" in source
    assert '"--collection-threads", "2"' in source
    assert '"--fit-threads", "8"' in source
    assert "evaluate_post15315_rebuild_stability" in source
    assert "stability_evaluation_performed" in source
    assert "strength_benchmark_performed" in source
