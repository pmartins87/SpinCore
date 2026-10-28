#!/usr/bin/env python3
from __future__ import annotations

"""Counterfactual checkpoint trend on VALID-040 flagged 3H states.

PROJECT_CONTRACT_IDS:
DC-002,SAFE-001,SAFE-011,VALID-025,VALID-031,VALID-040,RNG-001,RNG-002,RNG-003

Replays the frozen teacher-17315 VALID-040 trajectories, but at every flagged
information state evaluates raw Advantage and the canonical lean policy under
teachers 10315, 14815, 15315, and 17315. The actual replay action is sampled
only from teacher 17315, preserving the frozen source trajectory.

This is diagnostic evidence only: no model selection, mutation, promotion, or
canonical DeepCrusher claim.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
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
from spincore.decision_sanity import effective_stack_bb
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import lean_legal_actions, resolve_lean_exact
from spincore.r7_5_action_cfr import legal_mask
from spincore.r7_5_action_contract import NAME_BY_SLOT
from spincore.solver import SolverLibrary
from spincore_nn.action_models import collate_action_observations

DOMAIN = "THREE_HANDED"
SPNNIV2_MAGIC = b"SPNNIV2" + bytes((0,))
CHECKPOINTS = (10315, 14815, 15315, 17315)

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


def eval_models(models, obs: bytes, legal: tuple[int, ...]):
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
    return {
        "branch": branch,
        "positive_total": float(positive_total),
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
        "_sigma_tuple": sigma,
    }


class TrendTeacherPolicy:
    policy_id = SPINCORE_POLICY_ID

    def __init__(self, models_by_checkpoint):
        self.models_by_checkpoint = models_by_checkpoint
        self._last: dict[int, dict[str, Any]] = {}

    def choose_exact(self, state, *, seat: int, rng: random.Random):
        street = street_from_state(state)
        active = FIRST_RELEASE_ACTION_SPEC.active_mask(street)
        legal = tuple(int(x) for x in lean_legal_actions(state, active))
        if not legal:
            raise RuntimeError("nonterminal state has no legal action")

        obs = state.neural_bytes()
        evaluations = {
            str(cp): eval_models(self.models_by_checkpoint[cp], obs, legal)
            for cp in CHECKPOINTS
        }
        current = evaluations["17315"]
        sigma = current["_sigma_tuple"]

        u = rng.random()
        cumulative = 0.0
        slot = int(legal[-1])
        for action in legal:
            cumulative += float(sigma[action])
            if u < cumulative:
                slot = int(action)
                break
        action_type, amount_to = resolve_lean_exact(state, active, slot)

        clean = {}
        for cp, ev in evaluations.items():
            clean[cp] = {k: v for k, v in ev.items() if k != "_sigma_tuple"}
        self._last[int(seat)] = {
            "selected_slot": int(slot),
            "selected_slot_name": str(NAME_BY_SLOT.get(slot, f"SLOT_{slot}")),
            "selected_probability_17315": float(sigma[slot]),
            "sample_u": float(u),
            "checkpoints": clean,
        }
        return ExternalExactAction(int(action_type), int(amount_to))

    def decision_metadata(self, *, seat: int):
        row = self._last.get(int(seat))
        return None if row is None else dict(row)


def load_ensemble(path: str, expected_iteration: int):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("completed_iteration", -1)) != int(expected_iteration):
        raise RuntimeError(
            f"teacher iteration mismatch for {path}: "
            f"{payload.get('completed_iteration')} != {expected_iteration}"
        )
    states = list(payload.get("members") or [])
    if len(states) != 8:
        raise RuntimeError(f"expected 8 members in {path}")
    return [distill.load_semantic_advantage(state) for state in states]


def init_worker(
    solver_path: str,
    teacher_10315: str,
    teacher_14815: str,
    teacher_15315: str,
    teacher_17315: str,
    root_path: str,
    seed: int,
) -> None:
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
    models = {
        10315: load_ensemble(teacher_10315, 10315),
        14815: load_ensemble(teacher_14815, 14815),
        15315: load_ensemble(teacher_15315, 15315),
        17315: load_ensemble(teacher_17315, 17315),
    }
    _POLICY = TrendTeacherPolicy(models)
    _DC = DeepCrusherR8Policy.from_repository(root_path)
    _ENGINE_SEED = int(seed)


def worker(task):
    index, episode, deal_seed, targets, max_decisions = task
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
    target_map = {
        (int(row["lineup_index"]), int(row["decision_index"])): row
        for row in targets
    }
    out = []
    for trace in traces:
        if trace.policy_id != SPINCORE_POLICY_ID:
            continue
        key = (int(trace.lineup_index), int(trace.decision_index))
        source = target_map.get(key)
        if source is None:
            continue
        out.append(
            {
                "scenario": int(index),
                "lineup_index": int(trace.lineup_index),
                "decision_index": int(trace.decision_index),
                "flag_code": str(source["flag"]["code"]),
                "source_selected_slot_name": str(source["selected_slot_name"]),
                "actor": int(trace.actor),
                "hole": list(trace.hole_cards or ()),
                "board": list(trace.board or ()),
                "street": int(trace.street),
                "pot": int(trace.pot),
                "to_call": int(trace.to_call),
                "effective_stack_bb": float(effective_stack_bb(trace)),
                "has_immediate_draw": source["flag"]["context"].get(
                    "has_immediate_straight_or_flush_draw"
                ),
                "policy_detail": dict(trace.policy_detail or {}),
            }
        )
    return out


def max_uniform_deviation(sigma: dict[str, float]) -> float:
    if not sigma:
        return 0.0
    u = 1.0 / float(len(sigma))
    return max(abs(float(v) - u) for v in sigma.values())


def summarize(rows):
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["flag_code"]), []).append(row)
    no_draw = [
        row for row in rows
        if row["flag_code"] == "POSTFLOP_DEEP_HIGH_CARD_JAM"
        and row.get("has_immediate_draw") is False
    ]
    groups["POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW"] = no_draw

    summary = {}
    for name, items in groups.items():
        cps = {}
        for cp in CHECKPOINTS:
            cp_key = str(cp)
            action_probs = []
            allin_probs = []
            uniformish = 0
            positive = 0
            softmax = 0
            for row in items:
                ev = row["policy_detail"]["checkpoints"][cp_key]
                sigma = ev["sigma"]
                action = row["source_selected_slot_name"]
                action_probs.append(float(sigma.get(action, 0.0)))
                allin_probs.append(float(sigma.get("ALL_IN", 0.0)))
                if max_uniform_deviation(sigma) < 0.01:
                    uniformish += 1
                if ev["branch"] == "POSITIVE_REGRET_MATCHING":
                    positive += 1
                else:
                    softmax += 1
            cps[cp_key] = {
                "count": len(items),
                "mean_flagged_action_probability": (
                    None if not action_probs else float(np.mean(action_probs))
                ),
                "median_flagged_action_probability": (
                    None if not action_probs else float(np.median(action_probs))
                ),
                "mean_allin_probability": (
                    None if not allin_probs else float(np.mean(allin_probs))
                ),
                "median_allin_probability": (
                    None if not allin_probs else float(np.median(allin_probs))
                ),
                "near_uniform_under_1pp_count": int(uniformish),
                "positive_regret_matching_count": int(positive),
                "softmax_fallback_count": int(softmax),
            }
        summary[name] = cps
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--teacher-10315", type=Path, required=True)
    ap.add_argument("--teacher-14815", type=Path, required=True)
    ap.add_argument("--teacher-15315", type=Path, required=True)
    ap.add_argument("--teacher-17315", type=Path, required=True)
    ap.add_argument("--sanity-report", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
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
    targets_by_scenario: dict[int, list[dict[str, Any]]] = {}
    expected = {}
    for item in flagged:
        scenario = int(item["scenario"])
        targets_by_scenario.setdefault(scenario, []).append(item)
        expected[
            (scenario, int(item["lineup_index"]), int(item["decision_index"]))
        ] = item

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

    rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers), max(1, len(tasks))),
        mp_context=ctx,
        initializer=init_worker,
        initargs=(
            str(args.solver.resolve()),
            str(args.teacher_10315.resolve()),
            str(args.teacher_14815.resolve()),
            str(args.teacher_15315.resolve()),
            str(args.teacher_17315.resolve()),
            str(ROOT.resolve()),
            int(args.seed),
        ),
    ) as pool:
        for chunk in pool.map(worker, tasks, chunksize=1):
            rows.extend(chunk)

    rows.sort(key=lambda r: (r["scenario"], r["lineup_index"], r["decision_index"]))
    got_keys = {
        (r["scenario"], r["lineup_index"], r["decision_index"])
        for r in rows
    }
    missing = [list(k) for k in sorted(set(expected) - got_keys)]
    selection_mismatches = []
    for row in rows:
        selected = row["policy_detail"].get("selected_slot_name")
        if selected != row["source_selected_slot_name"]:
            selection_mismatches.append(
                {
                    "key": [
                        row["scenario"],
                        row["lineup_index"],
                        row["decision_index"],
                    ],
                    "source": row["source_selected_slot_name"],
                    "replay": selected,
                }
            )

    report = {
        "schema": "SPINCORE_3H_TEACHER_FLAGGED_CHECKPOINT_TREND_V1",
        "status": (
            "PASS"
            if len(rows) == len(expected) and not missing and not selection_mismatches
            else "REPLAY_MISMATCH"
        ),
        "seed": int(args.seed),
        "checkpoints": list(CHECKPOINTS),
        "source_flagged_count": len(flagged),
        "matched_count": len(rows),
        "missing_keys": missing,
        "selection_mismatches": selection_mismatches,
        "summary": summarize(rows),
        "rows": rows,
        "interpretation": {
            "comparison": (
                "All checkpoints are evaluated counterfactually on the exact same "
                "teacher-17315 information state and legal action set."
            ),
            "trajectory": (
                "A monotone or persistent increase in suspicious action probability "
                "is evidence against assuming more of the same training will repair it. "
                "Oscillation around the zero-Advantage boundary is materially weaker "
                "evidence of entrenchment."
            ),
            "scope": (
                "Development-only root-cause/training-efficiency diagnostic; no model "
                "selection or promotion authorization."
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== SpinCore flagged checkpoint trend ===")
    print(f"status={report['status']}")
    print(f"matched={len(rows)}/{len(expected)}")
    for group, cp_rows in report["summary"].items():
        print("TREND " + group + " " + json.dumps(cp_rows, sort_keys=True))
    print(f"selection_mismatches={len(selection_mismatches)} missing={len(missing)}")
    print(f"report={args.report.resolve()}")
    print("SPINCORE_3H_TEACHER_FLAGGED_TREND_COMPLETE")
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
