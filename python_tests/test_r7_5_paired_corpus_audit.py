from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import r7_5_audit_paired_corpus as audit
from spincore.r7_5_paired_corpus import BottomHashCorpus, PairedSample


def _v1(index: int) -> bytes:
    payload = bytearray(126)
    payload[:8] = b"SPNNIV1\x00"
    payload[8] = index % 52
    payload[87:93] = bytes((1, 1, 0, 0, 0, 0))
    return bytes(payload)


def _v2(index: int, street: int) -> bytes:
    payload = bytearray(830)
    payload[:8] = b"SPNNIV2\x00"
    payload[8] = index % 169
    payload[111 + 1] = street
    payload[111 + 10] = index % 4
    payload[111 + 17] = index % 5
    payload[111 + 18] = index % 4
    payload[111 + 19] = index % 3
    payload[111 + 21] = index % 3
    payload[111 + 27] = index % 6
    payload[111 + 28] = index % 5
    payload[111 + 32] = 1 if index % 2 == 0 else 0
    payload[111 + 37] = 1 if index % 3 == 0 else 0
    payload[111 + 38] = 1 if index % 5 == 0 else 0
    payload[111 + 44] = min(4, street + 1)
    payload[111 + 52] = 1 if index % 7 == 0 else 0
    payload[183:189] = bytes((1, 1, 0, 0, 0, 0))
    return bytes(payload)


def _samples(kind: str, domain: str, seed: int, count: int) -> list[PairedSample]:
    out = []
    for index in range(count):
        target = (
            (1.0, -1.0, 0.0, 0.0, 0.0, 0.0)
            if kind == "advantage"
            else (0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
        )
        out.append(
            PairedSample(
                kind=kind,
                domain=domain,
                corpus_seed=seed,
                observation_v1=_v1(index),
                observation_v2=_v2(index, index % 4),
                legal=(1, 1, 0, 0, 0, 0),
                target=target,
                weight=1.0,
                iteration=2,
            )
        )
    return out


def _producer_corpus(samples: list[PairedSample], capacity: int) -> BottomHashCorpus[PairedSample]:
    corpus = BottomHashCorpus[PairedSample](capacity)
    for sample in samples:
        corpus.add(sample)
    return corpus


def _save_rows(path: Path, samples: list[PairedSample]) -> None:
    torch.save(
        [
            {
                "kind": sample.kind,
                "domain": sample.domain,
                "corpus_seed": sample.corpus_seed,
                "observation_v1": sample.observation_v1,
                "observation_v2": sample.observation_v2,
                "legal": sample.legal,
                "target": sample.target,
                "weight": sample.weight,
                "iteration": sample.iteration,
            }
            for sample in samples
        ],
        path,
    )


def test_auditor_reconstructs_producer_hash_after_save_load(tmp_path: Path) -> None:
    samples = _samples("advantage", "TRUE_HEADS_UP", audit.EXPECTED_SEEDS[0], 12)
    corpus = _producer_corpus(samples, capacity=5)
    saved = corpus.items
    path = tmp_path / "pairs.pt"
    _save_rows(path, saved)
    loaded = audit._load_samples(path, "advantage", "TRUE_HEADS_UP", audit.EXPECTED_SEEDS[0])
    assert len(loaded) == 5
    assert audit._identity_digest(loaded) == corpus.state_summary()["ordered_identity_sha256"]


def test_audit_validates_full_domain_seed_matrix_and_semantic_counts(tmp_path: Path) -> None:
    old_adv = audit.MIN_ADVANTAGE
    old_strategy = audit.MIN_STRATEGY
    audit.MIN_ADVANTAGE = 4
    audit.MIN_STRATEGY = 3
    try:
        directories = []
        for domain in audit.EXPECTED_DOMAINS:
            for seed in audit.EXPECTED_SEEDS:
                directory = tmp_path / f"{domain}_{seed}"
                directory.mkdir()

                adv_corpus = _producer_corpus(_samples("advantage", domain, seed, 8), capacity=4)
                strategy_corpus = _producer_corpus(_samples("strategy", domain, seed, 7), capacity=3)
                advantage = adv_corpus.items
                strategy = strategy_corpus.items

                adv_path = directory / "advantage_pairs.pt"
                pol_path = directory / "strategy_pairs.pt"
                _save_rows(adv_path, advantage)
                _save_rows(pol_path, strategy)
                report = {
                    "schema": audit.CORPUS_SCHEMA,
                    "domain": domain,
                    "corpus_seed": seed,
                    "paired_roots": 64,
                    "scenario_counts_paired_phase": [1, 1],
                    "all_scenarios_exercised_paired_phase": True,
                    "coverage_pass": True,
                    "candidate_inference_used": False,
                    "behavior_observation_wire": "SPNNIV1",
                    "paired_secondary_wire": "SPNNIV2",
                    "ready_for_tables": False,
                    "advantage": adv_corpus.state_summary(),
                    "strategy": strategy_corpus.state_summary(),
                }
                (directory / "report.json").write_text(
                    json.dumps(report, sort_keys=True), encoding="utf-8"
                )
                directories.append(directory)

        payload = audit.audit_corpus_dirs(directories)
        assert payload["coverage_pass"] is True
        assert len(payload["matrix_entries"]) == 4
        assert payload["global_street_counts"]["advantage/street:0"] > 0
        assert payload["global_semantic_tag_counts"]["advantage/flush_draw"] > 0
        assert payload["identity_digest_order"] == "BOTTOM_HASH_RETENTION_KEY_CANONICAL_ORDER"
        assert payload["candidate_inference_used"] is False
        assert payload["ready_for_tables"] is False
    finally:
        audit.MIN_ADVANTAGE = old_adv
        audit.MIN_STRATEGY = old_strategy
