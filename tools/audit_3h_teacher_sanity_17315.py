#!/usr/bin/env python3
from __future__ import annotations

"""Fresh 3H raw-teacher strategic sanity audit for semantic iteration 17315.

PROJECT_CONTRACT_IDS:
DC-002,SAFE-001,SAFE-011,VALID-025,VALID-031,VALID-040,RNG-001,RNG-002,RNG-003

Development-only strategic sanity diagnostic. It evaluates the raw semantic
Advantage teacher directly, avoiding deployable-rebuild variance. It does not
authorize promotion or a canonical DeepCrusher strength claim while real
OpenHoldem DC0 parity is pending.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import random
import sys
from typing import Any

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
from spincore.decision_sanity import (
    card_rank,
    effective_stack_bb,
    evaluate_visible_hand,
    flag_to_dict,
    preflop_class,
    sanity_flags,
)
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import lean_legal_actions, resolve_lean_exact
from spincore.r7_5_action_contract import NAME_BY_SLOT
from spincore.solver import SolverLibrary

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


class SemanticAdvantageTeacherPolicy:
    policy_id = SPINCORE_POLICY_ID

    def __init__(self, models, label: str):
        self.models = [m.eval() for m in models]
        self.label = str(label)
        self._last: dict[int, dict[str, Any]] = {}

    def distribution(self, state):
        street = street_from_state(state)
        active = FIRST_RELEASE_ACTION_SPEC.active_mask(street)
        legal = tuple(int(x) for x in lean_legal_actions(state, active))
        if not legal:
            raise RuntimeError("nonterminal state has no legal action")
        sigma = distill.semantic_sigma(self.models, state.neural_bytes(), legal)
        return int(active), legal, tuple(float(x) for x in sigma)

    def choose_exact(self, state, *, seat: int, rng: random.Random):
        active, legal, probs = self.distribution(state)
        x = rng.random()
        cumulative = 0.0
        slot = int(legal[-1])
        for action in legal:
            cumulative += float(probs[action])
            if x < cumulative:
                slot = int(action)
                break
        action_type, amount_to = resolve_lean_exact(state, active, slot)
        self._last[int(seat)] = {
            "oracle": "SemanticAdvantageTeacherPolicy",
            "teacher": self.label,
            "sample_u": float(x),
            "selected_slot": int(slot),
            "selected_slot_name": str(NAME_BY_SLOT.get(slot, f"SLOT_{slot}")),
            "selected_probability": float(probs[slot]),
            "legal_actions": [
                {
                    "slot": int(action),
                    "name": str(NAME_BY_SLOT.get(int(action), f"SLOT_{int(action)}")),
                    "probability": float(probs[action]),
                }
                for action in legal
            ],
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
    if not _SOLVER.explicit_deal_available:
        raise RuntimeError("explicit-deal solver ABI required")

    payload = torch.load(ensemble_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != ENSEMBLE_SCHEMA:
        raise RuntimeError(f"wrong ensemble schema: {payload.get('schema')!r}")
    if int(payload.get("completed_iteration", -1)) != ITERATION:
        raise RuntimeError("teacher iteration mismatch")
    states = list(payload.get("members") or [])
    if len(states) != 8:
        raise RuntimeError("expected 8 semantic teacher members")
    _POLICY = SemanticAdvantageTeacherPolicy(
        [distill.load_semantic_advantage(state) for state in states],
        str(ITERATION),
    )
    _DC = DeepCrusherR8Policy.from_repository(root_path)
    _ENGINE_SEED = int(seed)


def _legal_slot_names(trace) -> set[str]:
    rows = (trace.policy_detail or {}).get("legal_actions") or []
    return {str(row.get("name")) for row in rows if isinstance(row, dict)}


def _selected_slot_name(trace) -> str | None:
    value = (trace.policy_detail or {}).get("selected_slot_name")
    return None if value is None else str(value)


def _opportunity_codes(trace) -> list[str]:
    if trace.policy_id != SPINCORE_POLICY_ID or trace.hole_cards is None:
        return []
    legal = _legal_slot_names(trace)
    hole = trace.hole_cards
    out: list[str] = []

    if trace.visible_board_count == 0:
        cls = preflop_class(hole)
        if cls == "AA" and "FOLD" in legal:
            out.append("PREFLOP_AA_FOLD")
        if cls == "72o" and effective_stack_bb(trace) >= 10.0 and "ALL_IN" in legal:
            out.append("PREFLOP_DEEP_72O_JAM")
        return out

    category, _ = evaluate_visible_hand(hole, trace.board)
    board_ranks = [card_rank(x) for x in trace.board]
    hole_ranks = [card_rank(x) for x in hole]
    just_top_pair = (
        category == 1
        and bool(board_ranks)
        and max(board_ranks) in hole_ranks
        and board_ranks.count(max(board_ranks)) == 1
    )
    if "FOLD" in legal:
        if category >= 6:
            out.append("POSTFLOP_MONSTER_FOLD")
        if category >= 3:
            out.append("POSTFLOP_TRIPS_PLUS_FOLD")
        if just_top_pair:
            out.append("POSTFLOP_TOP_PAIR_FOLD")
    if category == 0 and effective_stack_bb(trace) >= 10.0 and "ALL_IN" in legal:
        out.append("POSTFLOP_DEEP_HIGH_CARD_JAM")
    return out



def _target_action_for_opportunity(code: str) -> str:
    if code in {
        "PREFLOP_AA_FOLD",
        "POSTFLOP_MONSTER_FOLD",
        "POSTFLOP_TRIPS_PLUS_FOLD",
        "POSTFLOP_TOP_PAIR_FOLD",
    }:
        return "FOLD"
    if code in {
        "PREFLOP_DEEP_72O_JAM",
        "POSTFLOP_DEEP_HIGH_CARD_JAM",
    }:
        return "ALL_IN"
    raise RuntimeError(f"unknown sanity opportunity code: {code}")


def _policy_probability(trace, action_name: str) -> float:
    rows = (trace.policy_detail or {}).get("legal_actions") or []
    for row in rows:
        if str(row.get("name")) == str(action_name):
            return float(row.get("probability", 0.0))
    raise RuntimeError(
        f"target action {action_name} absent from declared legal actions"
    )


def _opportunity_policy_row(trace, code: str, probability: float) -> dict[str, Any]:
    return {
        "code": str(code),
        "target_action": _target_action_for_opportunity(code),
        "target_probability": float(probability),
        "lineup_index": int(trace.lineup_index),
        "decision_index": int(trace.decision_index),
        "street": int(trace.street),
        "hole": list(trace.hole_cards or ()),
        "board": list(trace.board or ()),
        "pot": int(trace.pot),
        "to_call": int(trace.to_call),
        "effective_stack_bb": float(effective_stack_bb(trace)),
        "selected_slot_name": _selected_slot_name(trace),
    }


def _merge_opportunity_policy(rows) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        for code, data in row["opportunity_policy"].items():
            dst = merged.setdefault(
                str(code),
                {
                    "opportunities": 0,
                    "probability_sum": 0.0,
                    "probability_max": 0.0,
                    "ge_50pct": 0,
                    "ge_80pct": 0,
                    "ge_95pct": 0,
                    "top_probability_rows": [],
                },
            )
            dst["opportunities"] += int(data["opportunities"])
            dst["probability_sum"] += float(data["probability_sum"])
            dst["probability_max"] = max(
                float(dst["probability_max"]),
                float(data["probability_max"]),
            )
            dst["ge_50pct"] += int(data["ge_50pct"])
            dst["ge_80pct"] += int(data["ge_80pct"])
            dst["ge_95pct"] += int(data["ge_95pct"])
            dst["top_probability_rows"].extend(
                list(data.get("top_probability_rows") or [])
            )
            dst["top_probability_rows"].sort(
                key=lambda x: float(x["target_probability"]),
                reverse=True,
            )
            dst["top_probability_rows"] = dst["top_probability_rows"][:20]

    out: dict[str, Any] = {}
    for code, data in sorted(merged.items()):
        n = int(data["opportunities"])
        out[code] = {
            "opportunities": n,
            "mean_target_probability": (
                None if n == 0 else float(data["probability_sum"]) / float(n)
            ),
            "max_target_probability": float(data["probability_max"]),
            "ge_50pct_count": int(data["ge_50pct"]),
            "ge_80pct_count": int(data["ge_80pct"]),
            "ge_95pct_count": int(data["ge_95pct"]),
            "ge_50pct_fraction": (
                None if n == 0 else int(data["ge_50pct"]) / float(n)
            ),
            "ge_80pct_fraction": (
                None if n == 0 else int(data["ge_80pct"]) / float(n)
            ),
            "ge_95pct_fraction": (
                None if n == 0 else int(data["ge_95pct"]) / float(n)
            ),
            "top_probability_rows": list(data["top_probability_rows"]),
        }
    return out

def worker(task):
    index, episode, deal_seed, max_decisions = task
    traces = []
    engine = OfflineHeadToHeadEngine(
        _SOLVER,
        spincore_policy=_POLICY,
        deepcrusher_policy=_DC,
        master_seed=int(_ENGINE_SEED),
        max_decisions=int(max_decisions),
        decision_sink=traces.append,
    )
    observations = engine.play_balanced_block(
        episode,
        deal_seed=int(deal_seed),
        scenario_index=int(index),
    )
    summary = base._scenario_summary(observations)

    flag_counts: dict[str, int] = {}
    deliberate_flag_counts: dict[str, int] = {}
    opportunities: dict[str, int] = {}
    opportunity_policy: dict[str, dict[str, Any]] = {}
    flagged: list[dict[str, Any]] = []
    spin_decisions = 0

    for trace in traces:
        if trace.policy_id != SPINCORE_POLICY_ID:
            continue
        spin_decisions += 1

        for code in _opportunity_codes(trace):
            opportunities[code] = opportunities.get(code, 0) + 1
            action_name = _target_action_for_opportunity(code)
            probability = _policy_probability(trace, action_name)
            stats = opportunity_policy.setdefault(
                code,
                {
                    "opportunities": 0,
                    "probability_sum": 0.0,
                    "probability_max": 0.0,
                    "ge_50pct": 0,
                    "ge_80pct": 0,
                    "ge_95pct": 0,
                    "top_probability_rows": [],
                },
            )
            stats["opportunities"] += 1
            stats["probability_sum"] += float(probability)
            stats["probability_max"] = max(
                float(stats["probability_max"]), float(probability)
            )
            stats["ge_50pct"] += int(probability >= 0.50)
            stats["ge_80pct"] += int(probability >= 0.80)
            stats["ge_95pct"] += int(probability >= 0.95)
            stats["top_probability_rows"].append(
                _opportunity_policy_row(trace, code, probability)
            )
            stats["top_probability_rows"].sort(
                key=lambda x: float(x["target_probability"]),
                reverse=True,
            )
            stats["top_probability_rows"] = stats["top_probability_rows"][:5]

        for flag in sanity_flags(trace):
            code = str(flag.code)
            flag_counts[code] = flag_counts.get(code, 0) + 1
            if _selected_slot_name(trace) == "ALL_IN":
                deliberate_flag_counts[code] = deliberate_flag_counts.get(code, 0) + 1
            flagged.append(
                {
                    "scenario": int(index),
                    "deal_seed": int(deal_seed),
                    "lineup_index": int(trace.lineup_index),
                    "decision_index": int(trace.decision_index),
                    "selected_slot_name": _selected_slot_name(trace),
                    "flag": flag_to_dict(flag),
                }
            )

    return {
        "scenario": int(index),
        "deal_seed": int(deal_seed),
        "spin_decisions": int(spin_decisions),
        "summary": summary,
        "flag_counts": flag_counts,
        "deliberate_flag_counts": deliberate_flag_counts,
        "opportunities": opportunities,
        "opportunity_policy": opportunity_policy,
        "flagged": flagged,
    }


def _merge_counts(rows, key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        for code, value in row[key].items():
            out[str(code)] = out.get(str(code), 0) + int(value)
    return dict(sorted(out.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--scenarios", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--max-decisions", type=int, default=300)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    if args.scenarios <= 0 or args.workers <= 0 or args.max_decisions <= 0:
        raise SystemExit("scenarios/workers/max-decisions must be positive")
    for path in (args.solver, args.teacher, base.DC_SOURCE, base.LIB1, base.LIB2):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    sampler = LegacyScenarioSampler(
        seed=int(args.seed) ^ 0x5CE0A710,
        config=LegacyScenarioConfig(),
    )
    tasks = []
    for index in range(int(args.scenarios)):
        episode = sampler.sample_episode(force_domain=DOMAIN)
        tasks.append(
            (
                int(index),
                episode,
                int(base._mix64(args.seed, index, 0xD34A1)),
                int(args.max_decisions),
            )
        )

    rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers), int(args.scenarios)),
        mp_context=ctx,
        initializer=init_worker,
        initargs=(
            str(args.solver.resolve()),
            str(args.teacher.resolve()),
            str(ROOT.resolve()),
            int(args.seed),
        ),
    ) as pool:
        for row in pool.map(worker, tasks, chunksize=1):
            rows.append(row)
    rows.sort(key=lambda row: int(row["scenario"]))

    flag_counts = _merge_counts(rows, "flag_counts")
    deliberate_flag_counts = _merge_counts(rows, "deliberate_flag_counts")
    opportunities = _merge_counts(rows, "opportunities")
    opportunity_policy = _merge_opportunity_policy(rows)
    flagged = [item for row in rows for item in row["flagged"]]
    spin_decisions = sum(int(row["spin_decisions"]) for row in rows)

    critical_codes = {"PREFLOP_AA_FOLD", "POSTFLOP_MONSTER_FOLD"}
    high_review_codes = {"PREFLOP_DEEP_72O_JAM", "POSTFLOP_TRIPS_PLUS_FOLD"}
    critical_count = sum(flag_counts.get(code, 0) for code in critical_codes)
    high_review_count = sum(flag_counts.get(code, 0) for code in high_review_codes)

    if critical_count:
        status = "CRITICAL_ABSURDITY_PRESENT"
    elif high_review_count:
        status = "HIGH_REVIEW_REQUIRED"
    else:
        status = "NO_GROSS_ABSURDITY_DETECTED_ON_PREDECLARED_SURFACES"

    rates = {}
    for code in sorted(set(opportunities) | set(flag_counts)):
        opp = int(opportunities.get(code, 0))
        count = int(flag_counts.get(code, 0))
        rates[code] = {
            "flags": count,
            "opportunities": opp,
            "flag_rate_per_opportunity": None if opp == 0 else count / float(opp),
            "flags_per_100k_spin_decisions": (
                None if spin_decisions == 0
                else 100000.0 * count / float(spin_decisions)
            ),
        }

    report = {
        "schema": "SPINCORE_3H_TEACHER_SANITY_17315_V1",
        "status": status,
        "canonical_quality_claim_authorized": False,
        "dc0_status": "PENDING_REAL_OPENHOLDEM_PARITY",
        "teacher_iteration": ITERATION,
        "teacher_path": str(args.teacher.resolve()),
        "seed": int(args.seed),
        "scenarios": int(args.scenarios),
        "balanced_games": sum(int(row["summary"]["games"]) for row in rows),
        "spin_decisions": int(spin_decisions),
        "domain": DOMAIN,
        "pairing": (
            "fresh empirical 3H scenarios; exact deal seed; balanced AAB/ABB "
            "lineups against the frozen DeepCrusher development opponent"
        ),
        "flag_counts": flag_counts,
        "deliberate_allin_slot_flag_counts": deliberate_flag_counts,
        "opportunity_counts": opportunities,
        "opportunity_policy": opportunity_policy,
        "rates": rates,
        "flagged_decisions": flagged,
        "interpretation": {
            "critical": (
                "Any AA preflop fold or full-house-plus postflop fold is a gross-error "
                "candidate requiring immediate trace inspection."
            ),
            "high_review": (
                "Deep 72o jams and trips-plus folds require trace inspection; the "
                "heuristic flag alone is not a poker-theory verdict."
            ),
            "review": (
                "Top-pair folds and deep high-card jams are context-sensitive review "
                "surfaces and are not automatically errors."
            ),
            "policy_surface": (
                "opportunity_policy measures the teacher's target-action probability "
                "on every eligible state, independent of which action RNG sampled. "
                "This distinguishes rare sampled anomalies from systematic high-probability "
                "pathology."
            ),
            "scope": (
                "Development-only raw-teacher sanity evidence. This does not measure "
                "deployable rebuild stability and does not replace DC0/DC1 promotion gates."
            ),
        },
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== SpinCore 3H teacher 17315 strategic sanity audit ===")
    print(f"status={status}")
    print(f"scenarios={args.scenarios} spin_decisions={spin_decisions}")
    print("flag_counts=" + json.dumps(flag_counts, sort_keys=True))
    print("opportunities=" + json.dumps(opportunities, sort_keys=True))
    print("rates=" + json.dumps(rates, sort_keys=True))
    print("opportunity_policy=" + json.dumps(opportunity_policy, sort_keys=True))
    important = [
        item for item in flagged
        if item["flag"]["code"] in (critical_codes | high_review_codes)
    ]
    print(f"important_flagged_decisions={len(important)}")
    for item in important[:50]:
        print("SANITY_REVIEW " + json.dumps(item, sort_keys=True))
    print(f"report={args.report.resolve()}")
    print("SPINCORE_3H_TEACHER_SANITY_17315_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
