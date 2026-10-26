#!/usr/bin/env python3
from __future__ import annotations

"""Replay only VALID-040 flagged 3H states and expose raw teacher Advantages.

PROJECT_CONTRACT_IDS:
DC-002,SAFE-001,SAFE-011,VALID-025,VALID-031,VALID-040,RNG-001,RNG-002,RNG-003

This is a development-only root-cause diagnostic. It does not select a model,
change policy behavior, or authorize promotion. It replays the already-frozen
VALID-040 scenarios/deals and records the eight member raw Advantage vectors,
their ensemble mean, and whether canonical lean regret matching used positive
regret matching or the all-nonpositive softmax fallback.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import audit_3h_fresh_semantic_strategy_distill_10105 as distill
import evaluate_deepcrusher_dc1 as base

from spincore.deepcrusher_benchmark import (
    SPINCORE_POLICY_ID,
    ExternalExactAction,
    OfflineHeadToHeadEngine,
)
from spincore.deepcrusher_policy import DeepCrusherR8Policy
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import lean_legal_actions, resolve_lean_exact
from spincore.r7_5_action_cfr import legal_mask
from spincore.r7_5_action_contract import NAME_BY_SLOT
from spincore.solver import SolverLibrary
from spincore_nn.action_models import collate_action_observations

ENSEMBLE_SCHEMA = "SPINCORE_3H_SEMANTIC_LONG_ENSEMBLE_V1"
DOMAIN = "THREE_HANDED"
ITERATION = 17315
SPNNIV2_MAGIC = b"SPNNIV2" + bytes((0,))

_SOLVER = None
_POLICY = None
_DC = None
_ENGINE_SEED = None


def street_from_state(state) -> int:
    payload = state.neural_bytes_v2()
    if len(payload) != 830 or not payload.startswith(SPNNIV2_MAGIC):
        raise RuntimeError("invalid SPNNIV2 state")
    street = int(payload[112])
    if street not in (0, 1, 2, 3):
        raise RuntimeError("invalid street")
    return street


def raw_sigma(models, obs: bytes, legal: tuple[int, ...]):
    batch = collate_action_observations(
        distill.REPRESENTATION,
        [obs],
        [legal_mask(legal)],
        device="cpu",
    )
    class S:
        pass
    state = S()
    state.observation = obs
    semantic = distill.semantic_vector(state)
    sb = dict(batch)
    sb["semantic"] = torch.tensor(
        np.asarray([semantic], dtype=np.float32),
        dtype=torch.float32,
    )
    member_raw = []
    with torch.no_grad():
        for model in models:
            member_raw.append(model(sb)[0].detach().cpu().tolist())
    mean_raw = np.asarray(member_raw, dtype=np.float64).mean(axis=0).tolist()
    positive_total = sum(max(0.0, float(mean_raw[a])) for a in legal)
    branch = "POSITIVE_REGRET_MATCHING" if positive_total > 0.0 else "SOFTMAX_FALLBACK"
    sigma = lean_regret_matching_policy(mean_raw, legal)
    return member_raw, mean_raw, sigma, branch, float(positive_total)


class RawDiagnosticTeacherPolicy:
    policy_id = SPINCORE_POLICY_ID

    def __init__(self, models):
        self.models = [m.eval() for m in models]
        self._last: dict[int, dict[str, Any]] = {}

    def choose_exact(self, state, *, seat: int, rng: random.Random):
        street = street_from_state(state)
        active = FIRST_RELEASE_ACTION_SPEC.active_mask(street)
        legal = tuple(int(x) for x in lean_legal_actions(state, active))
        if not legal:
            raise RuntimeError("nonterminal state has no legal action")
        member_raw, mean_raw, sigma, branch, positive_total = raw_sigma(
            self.models,
            state.neural_bytes(),
            legal,
        )
        u = rng.random()
        cumulative = 0.0
        slot = int(legal[-1])
        for action in legal:
            cumulative += float(sigma[action])
            if u < cumulative:
                slot = int(action)
                break
        action_type, amount_to = resolve_lean_exact(state, active, slot)
        self._last[int(seat)] = {
            "selected_slot": int(slot),
            "selected_slot_name": str(NAME_BY_SLOT.get(slot, f"SLOT_{slot}")),
            "selected_probability": float(sigma[slot]),
            "sample_u": float(u),
            "branch": branch,
            "positive_total": positive_total,
            "legal_slots": [int(a) for a in legal],
            "legal_names": [str(NAME_BY_SLOT.get(int(a), f"SLOT_{int(a)}")) for a in legal],
            "mean_raw_advantages": {
                str(NAME_BY_SLOT.get(int(a), f"SLOT_{int(a)}")): float(mean_raw[a])
                for a in legal
            },
            "member_raw_advantages": [
                {
                    str(NAME_BY_SLOT.get(int(a), f"SLOT_{int(a)}")): float(raw[a])
                    for a in legal
                }
                for raw in member_raw
            ],
            "sigma": {
                str(NAME_BY_SLOT.get(int(a), f"SLOT_{int(a)}")): float(sigma[a])
                for a in legal
            },
        }
        return ExternalExactAction(int(action_type), int(amount_to))

    def decision_metadata(self, *, seat: int):
        row = self._last.get(int(seat))
        return None if row is None else dict(row)


def init_worker(solver_path: str, ensemble_path: str, root_path: str, seed: int) -> None:
    global _SOLVER, _POLICY, _DC, _ENGINE_SEED
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["SPINCORE_TORCH_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    _SOLVER = SolverLibrary(solver_path)
    payload = torch.load(ensemble_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != ENSEMBLE_SCHEMA:
        raise RuntimeError(f"wrong ensemble schema: {payload.get('schema')!r}")
    if int(payload.get("completed_iteration", -1)) != ITERATION:
        raise RuntimeError("teacher iteration mismatch")
    states = list(payload.get("members") or [])
    if len(states) != 8:
        raise RuntimeError("expected 8 members")
    _POLICY = RawDiagnosticTeacherPolicy(
        [distill.load_semantic_advantage(state) for state in states]
    )
    _DC = DeepCrusherR8Policy.from_repository(root_path)
    _ENGINE_SEED = int(seed)


def worker(task):
    index, episode, deal_seed, target_keys, max_decisions = task
    traces = []
    engine = OfflineHeadToHeadEngine(
        _SOLVER,
        spincore_policy=_POLICY,
        deepcrusher_policy=_DC,
        master_seed=int(_ENGINE_SEED),
        max_decisions=int(max_decisions),
        decision_sink=traces.append,
    )
    engine.play_balanced_block(
        episode,
        deal_seed=int(deal_seed),
        scenario_index=int(index),
    )
    wanted = {(int(a), int(b)) for a, b in target_keys}
    out = []
    for trace in traces:
        if trace.policy_id != SPINCORE_POLICY_ID:
            continue
        key = (int(trace.lineup_index), int(trace.decision_index))
        if key not in wanted:
            continue
        detail = dict(trace.policy_detail or {})
        out.append(
            {
                "scenario": int(index),
                "lineup_index": int(trace.lineup_index),
                "decision_index": int(trace.decision_index),
                "actor": int(trace.actor),
                "hole": list(trace.hole_cards or ()),
                "board": list(trace.board or ()),
                "pot": int(trace.pot),
                "to_call": int(trace.to_call),
                "effective_stack_bb": float(
                    min(trace.stack_before, trace.opponent_stack_before)
                    / float(trace.big_blind)
                ) if trace.big_blind else None,
                "policy_detail": detail,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--sanity-report", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--max-decisions", type=int, default=300)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    sanity = json.loads(args.sanity_report.read_text(encoding="utf-8"))
    if sanity.get("schema") != "SPINCORE_3H_TEACHER_SANITY_17315_V1":
        raise SystemExit("wrong sanity report schema")
    if int(sanity.get("seed", -1)) != int(args.seed):
        raise SystemExit("seed mismatch")
    flagged = list(sanity.get("flagged_decisions") or [])
    targets_by_scenario: dict[int, list[tuple[int, int]]] = {}
    expected: dict[tuple[int, int, int], dict[str, Any]] = {}
    for item in flagged:
        scenario = int(item["scenario"])
        lineup = int(item["lineup_index"])
        decision = int(item["decision_index"])
        targets_by_scenario.setdefault(scenario, []).append((lineup, decision))
        expected[(scenario, lineup, decision)] = item

    sampler = LegacyScenarioSampler(
        seed=int(args.seed) ^ 0x5CE0A710,
        config=LegacyScenarioConfig(),
    )
    max_scenario = max(targets_by_scenario) if targets_by_scenario else -1
    episodes = {}
    for index in range(max_scenario + 1):
        episode = sampler.sample_episode(force_domain=DOMAIN)
        if index in targets_by_scenario:
            episodes[index] = episode

    tasks = [
        (
            scenario,
            episodes[scenario],
            int(base._mix64(args.seed, scenario, 0xD34A1)),
            targets_by_scenario[scenario],
            int(args.max_decisions),
        )
        for scenario in sorted(targets_by_scenario)
    ]

    replayed = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers), max(1, len(tasks))),
        mp_context=ctx,
        initializer=init_worker,
        initargs=(
            str(args.solver.resolve()),
            str(args.teacher.resolve()),
            str(ROOT.resolve()),
            int(args.seed),
        ),
    ) as pool:
        for chunk in pool.map(worker, tasks, chunksize=1):
            replayed.extend(chunk)

    replayed.sort(key=lambda r: (r["scenario"], r["lineup_index"], r["decision_index"]))
    matched = []
    mismatches = []
    for row in replayed:
        key = (row["scenario"], row["lineup_index"], row["decision_index"])
        exp = expected.get(key)
        if exp is None:
            continue
        got_slot = row["policy_detail"].get("selected_slot_name")
        exp_slot = exp.get("selected_slot_name")
        if got_slot != exp_slot:
            mismatches.append({"key": key, "expected": exp_slot, "got": got_slot})
        flag_code = str(exp["flag"]["code"])
        detail = row["policy_detail"]
        matched.append(
            {
                **row,
                "flag_code": flag_code,
                "expected_selected_slot_name": exp_slot,
                "replay_selected_slot_matches": got_slot == exp_slot,
            }
        )

    expected_count = len(expected)
    missing = sorted(
        [list(k) for k in expected if k not in {
            (r["scenario"], r["lineup_index"], r["decision_index"]) for r in replayed
        }]
    )

    branch_counts = {}
    branch_by_flag = {}
    for row in matched:
        branch = str(row["policy_detail"].get("branch"))
        code = str(row["flag_code"])
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
        branch_by_flag.setdefault(code, {})
        branch_by_flag[code][branch] = branch_by_flag[code].get(branch, 0) + 1

    report = {
        "schema": "SPINCORE_3H_TEACHER_RAW_ADVANTAGE_DIAGNOSTIC_17315_V1",
        "status": "PASS" if len(matched) == expected_count and not mismatches else "REPLAY_MISMATCH",
        "teacher_iteration": ITERATION,
        "seed": int(args.seed),
        "source_flagged_count": len(flagged),
        "unique_target_count": expected_count,
        "matched_count": len(matched),
        "missing_keys": missing,
        "selection_mismatches": mismatches,
        "branch_counts": branch_counts,
        "branch_by_flag": branch_by_flag,
        "rows": matched,
        "interpretation": {
            "SOFTMAX_FALLBACK": "Every legal ensemble-mean raw Advantage was non-positive; canonical lean policy therefore used softmax over raw values.",
            "POSITIVE_REGRET_MATCHING": "At least one legal ensemble-mean raw Advantage was positive; canonical lean regret matching normalized positive Advantages and zeroed non-positive ones.",
            "scope": "Root-cause diagnostic only. No model selection, policy mutation, or promotion claim.",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== SpinCore teacher 17315 raw-Advantage replay ===")
    print(f"status={report['status']}")
    print(f"matched={len(matched)}/{expected_count}")
    print("branch_counts=" + json.dumps(branch_counts, sort_keys=True))
    print("branch_by_flag=" + json.dumps(branch_by_flag, sort_keys=True))
    print(f"selection_mismatches={len(mismatches)} missing={len(missing)}")
    print(f"report={args.report.resolve()}")
    print("SPINCORE_3H_TEACHER_RAW_ADVANTAGE_17315_COMPLETE")
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
