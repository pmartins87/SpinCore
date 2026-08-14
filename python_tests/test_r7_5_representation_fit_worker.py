from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from spincore.r7_5_paired_corpus import PairedSample, is_train_sample

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools" / "r7_5_fit_representation_candidate.py"
SPLIT_SEED = 1925930899


def _wire_v1(index: int) -> bytes:
    payload = bytearray(126)
    payload[:8] = b"SPNNIV1\x00"
    # Keep card/categorical IDs valid while giving the corpus stable variation.
    payload[8] = index % 52
    payload[87:93] = bytes((1, 1, 0, 0, 0, 0))
    payload[93] = 0
    return bytes(payload)


def _wire_v2(index: int) -> bytes:
    payload = bytearray(830)
    payload[:8] = b"SPNNIV2\x00"
    payload[8] = index % 169
    payload[183:189] = bytes((1, 1, 0, 0, 0, 0))
    payload[189] = 0
    return bytes(payload)


def _samples(kind: str) -> list[PairedSample]:
    out: list[PairedSample] = []
    index = 0
    # Generate enough deterministic identities to guarantee both hash partitions
    # rather than relying on probability inside the test.
    while True:
        if kind == "advantage":
            target = (1.0 + 0.01 * index, -1.0 - 0.01 * index, 0.0, 0.0, 0.0, 0.0)
        else:
            target = (0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
        sample = PairedSample(
            kind=kind,
            domain="TRUE_HEADS_UP",
            corpus_seed=1202035427 + (index % 2),
            observation_v1=_wire_v1(index),
            observation_v2=_wire_v2(index),
            legal=(1, 1, 0, 0, 0, 0),
            target=target,
            weight=1.0 + float(index % 3),
            iteration=2,
        )
        out.append(sample)
        index += 1
        train = sum(is_train_sample(item, SPLIT_SEED) for item in out)
        heldout = len(out) - train
        if len(out) >= 40 and train >= 8 and heldout >= 8:
            return out
        if index > 500:
            raise AssertionError("failed to construct deterministic train/heldout fixture")


def _save(path: Path, samples: list[PairedSample]) -> None:
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


def _run_candidate(tmp_path: Path, candidate: str) -> dict:
    advantage = _samples("advantage")
    strategy = _samples("strategy")
    adv_path = tmp_path / f"{candidate}_adv.pt"
    pol_path = tmp_path / f"{candidate}_pol.pt"
    out_path = tmp_path / f"{candidate}.pt"
    _save(adv_path, advantage)
    _save(pol_path, strategy)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "python")
    env["SPINCORE_TORCH_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(WORKER),
            "--candidate",
            candidate,
            "--domain",
            "TRUE_HEADS_UP",
            "--fit-seed",
            "2102028507",
            "--split-seed",
            str(SPLIT_SEED),
            "--advantage",
            str(adv_path),
            "--strategy",
            str(pol_path),
            "--out",
            str(out_path),
            "--advantage-steps",
            "1",
            "--policy-steps",
            "1",
            "--batch-size",
            "8",
            "--min-tag-samples",
            "1",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    report_path = out_path.with_suffix(".json")
    assert report_path.exists()
    return json.loads(report_path.read_text(encoding="utf-8"))


def test_fit_worker_executes_frozen_v1_path(tmp_path: Path) -> None:
    report = _run_candidate(tmp_path, "C0_V1_FROZEN_CONTROL")
    assert report["parameter_count"] == 152438
    assert report["fit_contract"]["sample_multiplicity_preserved"] is True
    assert report["ready_for_tables"] is False


def test_fit_worker_executes_semantic_v2_no_flop_path(tmp_path: Path) -> None:
    report = _run_candidate(tmp_path, "C1_V2_NO_FLOP_TOKEN")
    assert report["parameter_count"] == 153350
    assert report["flop_candidate"] == "NONE"
    assert report["ready_for_tables"] is False
