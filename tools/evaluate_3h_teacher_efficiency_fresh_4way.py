#!/usr/bin/env python3
from __future__ import annotations

"""Fresh 4-way 3H teacher efficiency benchmark for long-run go/no-go.

PROJECT_CONTRACT_IDS:
DC-002,SAFE-001,SAFE-011,VALID-025,VALID-031,VALID-034,RNG-001,RNG-002,RNG-003

Exactly the same fresh scenario/deal stream is replayed for teachers 10315,
14815, 15315, and 17315 against the frozen DeepCrusher development opponent.
This is development-only evidence while DC0 real-OpenHoldem parity is pending.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import statistics
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
from spincore.decision_sanity import (
    effective_stack_bb,
    evaluate_visible_hand,
    immediate_straight_flush_draw_outs,
    preflop_class,
    sanity_flags,
)
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
CHECKPOINTS = (10315, 14815, 15315, 17315)
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


def semantic_eval(models, obs: bytes, legal: tuple[int, ...]) -> tuple[tuple[float, ...], str]:
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

    raws = []
    with torch.no_grad():
        for model in models:
            raws.append(model(sb)[0].detach().cpu().numpy().astype(np.float64))
    mean_raw = np.mean(np.stack(raws, axis=0), axis=0)
    positive_total = sum(max(0.0, float(mean_raw[a])) for a in legal)
    branch = "POSITIVE_REGRET_MATCHING" if positive_total > 0.0 else "SOFTMAX_FALLBACK"
    sigma = lean_regret_matching_policy(mean_raw.tolist(), legal)
    return tuple(float(x) for x in sigma), branch


class DiagnosticTeacherPolicy:
    policy_id = SPINCORE_POLICY_ID

    def __init__(self, models, label: str):
        self.models = [m.eval() for m in models]
        self.label = str(label)
        self._last: dict[int, dict[str, Any]] = {}

    def choose_exact(self, state, *, seat: int, rng: random.Random):
        street = street_from_state(state)
        active = FIRST_RELEASE_ACTION_SPEC.active_mask(street)
        legal = tuple(int(x) for x in lean_legal_actions(state, active))
        if not legal:
            raise RuntimeError("nonterminal state has no legal action")

        sigma, branch = semantic_eval(self.models, state.neural_bytes(), legal)
        u = rng.random()
        cumulative = 0.0
        slot = int(legal[-1])
        for action in legal:
            cumulative += float(sigma[action])
            if u < cumulative:
                slot = int(action)
                break

        action_type, amount_to = resolve_lean_exact(state, active, slot)
        legal_probs = [float(sigma[a]) for a in legal]
        uniform = 1.0 / float(len(legal))
        near_uniform = max(abs(p - uniform) for p in legal_probs) < 0.01

        self._last[int(seat)] = {
            "teacher": self.label,
            "branch": branch,
            "near_uniform_under_1pp": bool(near_uniform),
            "selected_slot_name": str(NAME_BY_SLOT.get(slot, f"SLOT_{slot}")),
            "legal_names": [
                str(NAME_BY_SLOT.get(int(a), f"SLOT_{int(a)}")) for a in legal
            ],
            "selected_probability": float(sigma[slot]),
        }
        return ExternalExactAction(int(action_type), int(amount_to))

    def decision_metadata(self, *, seat: int):
        row = self._last.get(int(seat))
        return None if row is None else dict(row)


def load_models(path: str, expected_iteration: int):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != ENSEMBLE_SCHEMA:
        raise RuntimeError(f"wrong ensemble schema for {path}: {payload.get('schema')!r}")
    if int(payload.get("completed_iteration", -1)) != int(expected_iteration):
        raise RuntimeError(
            f"teacher iteration mismatch for {path}: "
            f"{payload.get('completed_iteration')} != {expected_iteration}"
        )
    states = list(payload.get("members") or [])
    if len(states) != 8:
        raise RuntimeError(f"expected 8 semantic members in {path}")
    return [distill.load_semantic_advantage(state) for state in states]


def init_worker(solver_path: str, ensemble_path: str, expected_iteration: int, root_path: str, seed: int):
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
    _POLICY = DiagnosticTeacherPolicy(
        load_models(ensemble_path, expected_iteration),
        str(expected_iteration),
    )
    _DC = DeepCrusherR8Policy.from_repository(root_path)
    _ENGINE_SEED = int(seed)


def opportunity_codes(trace) -> list[str]:
    if trace.policy_id != SPINCORE_POLICY_ID or trace.hole_cards is None:
        return []

    detail = trace.policy_detail or {}
    legal = set(detail.get("legal_names") or [])
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
    if "FOLD" in legal:
        if category >= 6:
            out.append("POSTFLOP_MONSTER_FOLD")
        if category >= 3:
            out.append("POSTFLOP_TRIPS_PLUS_FOLD")

    if category == 0 and effective_stack_bb(trace) >= 10.0 and "ALL_IN" in legal:
        out.append("POSTFLOP_DEEP_HIGH_CARD_JAM")
        straight_outs, flush_outs = immediate_straight_flush_draw_outs(hole, trace.board)
        if not straight_outs and not flush_outs:
            out.append("POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW")
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

    flags: dict[str, int] = {}
    opportunities: dict[str, int] = {}
    spin_decisions = 0
    fallback = 0
    near_uniform = 0

    for tr in traces:
        if tr.policy_id != SPINCORE_POLICY_ID:
            continue
        spin_decisions += 1
        detail = tr.policy_detail or {}
        if detail.get("branch") == "SOFTMAX_FALLBACK":
            fallback += 1
        if bool(detail.get("near_uniform_under_1pp")):
            near_uniform += 1

        for code in opportunity_codes(tr):
            opportunities[code] = opportunities.get(code, 0) + 1
        for flag in sanity_flags(tr):
            flags[flag.code] = flags.get(flag.code, 0) + 1
            if (
                flag.code == "POSTFLOP_DEEP_HIGH_CARD_JAM"
                and not bool(flag.context.get("has_immediate_straight_or_flush_draw"))
            ):
                flags["POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW"] = (
                    flags.get("POSTFLOP_DEEP_HIGH_CARD_JAM_NO_DRAW", 0) + 1
                )

    return {
        **summary,
        "scenario": int(index),
        "blind": f"{int(episode.small_blind)}/{int(episode.big_blind)}",
        "dealer_id": int(episode.dealer_id),
        "deal_seed": int(deal_seed),
        "spin_decisions": int(spin_decisions),
        "fallback_decisions": int(fallback),
        "near_uniform_decisions": int(near_uniform),
        "sanity_flags": flags,
        "opportunities": opportunities,
    }


def mean_ci(values):
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "sem": None, "ci95_low": None, "ci95_high": None}
    mean = float(statistics.fmean(values))
    sem = 0.0 if n == 1 else float(statistics.stdev(values) / math.sqrt(n))
    h = 1.96 * sem
    return {
        "n": n,
        "mean": mean,
        "sem": sem,
        "ci95_low": mean - h,
        "ci95_high": mean + h,
    }


def summarize_arm(rows):
    flags: dict[str, int] = {}
    opportunities: dict[str, int] = {}
    spin_decisions = sum(int(r["spin_decisions"]) for r in rows)
    fallback = sum(int(r["fallback_decisions"]) for r in rows)
    near_uniform = sum(int(r["near_uniform_decisions"]) for r in rows)

    for r in rows:
        for code, value in (r.get("sanity_flags") or {}).items():
            flags[code] = flags.get(code, 0) + int(value)
        for code, value in (r.get("opportunities") or {}).items():
            opportunities[code] = opportunities.get(code, 0) + int(value)

    rates = {}
    for code in sorted(set(flags) | set(opportunities)):
        opp = int(opportunities.get(code, 0))
        count = int(flags.get(code, 0))
        rates[code] = {
            "flags": count,
            "opportunities": opp,
            "flag_rate_per_opportunity": None if opp == 0 else count / float(opp),
        }

    return {
        "scenario_clusters": len(rows),
        "balanced_games": sum(int(r["games"]) for r in rows),
        "spin_decisions": int(spin_decisions),
        "spincore_chips_per_policy_seat_hand": mean_ci(
            [float(r["spincore_chips_per_exposure"]) for r in rows]
        ),
        "paired_spincore_minus_deepcrusher": mean_ci(
            [float(r["paired_spincore_minus_deepcrusher_chips_per_exposure"]) for r in rows]
        ),
        "softmax_fallback": {
            "count": int(fallback),
            "rate": None if spin_decisions == 0 else fallback / float(spin_decisions),
        },
        "near_uniform_under_1pp": {
            "count": int(near_uniform),
            "rate": None if spin_decisions == 0 else near_uniform / float(spin_decisions),
        },
        "sanity_rates": rates,
    }


def run_arm(*, solver, ensemble, iteration, tasks, workers, seed):
    rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=ctx,
        initializer=init_worker,
        initargs=(str(solver), str(ensemble), int(iteration), str(ROOT.resolve()), int(seed)),
    ) as pool:
        for row in pool.map(worker, tasks, chunksize=1):
            rows.append(row)
    rows.sort(key=lambda x: int(x["scenario"]))
    return rows


def paired_delta(old, new):
    values = [
        float(b["spincore_chips_per_exposure"]) - float(a["spincore_chips_per_exposure"])
        for a, b in zip(old, new)
    ]
    return mean_ci(values)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solver", type=Path, required=True)
    for cp in CHECKPOINTS:
        ap.add_argument(f"--teacher-{cp}", type=Path, required=True)
    ap.add_argument("--scenarios", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--max-decisions", type=int, default=300)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    teachers = {cp: getattr(args, f"teacher_{cp}") for cp in CHECKPOINTS}
    for path in [args.solver, *teachers.values(), base.DC_SOURCE, base.LIB1, base.LIB2]:
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

    arms = {}
    for cp in CHECKPOINTS:
        print(f"ARM_START iteration={cp}", flush=True)
        arms[cp] = run_arm(
            solver=args.solver.resolve(),
            ensemble=teachers[cp].resolve(),
            iteration=cp,
            tasks=tasks,
            workers=min(int(args.workers), int(args.scenarios)),
            seed=int(args.seed),
        )
        print(f"ARM_PASS iteration={cp}", flush=True)

    base_identity = [
        (r["scenario"], r["deal_seed"], r["blind"], r["dealer_id"]) for r in arms[10315]
    ]
    for cp in CHECKPOINTS[1:]:
        ident = [
            (r["scenario"], r["deal_seed"], r["blind"], r["dealer_id"]) for r in arms[cp]
        ]
        if ident != base_identity:
            raise RuntimeError(f"paired scenario identity drift at {cp}")

    summaries = {str(cp): summarize_arm(arms[cp]) for cp in CHECKPOINTS}
    deltas = {
        "14815_minus_10315": paired_delta(arms[10315], arms[14815]),
        "15315_minus_14815": paired_delta(arms[14815], arms[15315]),
        "17315_minus_15315": paired_delta(arms[15315], arms[17315]),
        "17315_minus_10315": paired_delta(arms[10315], arms[17315]),
    }

    primary = deltas["17315_minus_15315"]
    f15315 = summaries["15315"]["softmax_fallback"]["rate"]
    f17315 = summaries["17315"]["softmax_fallback"]["rate"]
    n15315 = summaries["15315"]["near_uniform_under_1pp"]["rate"]
    n17315 = summaries["17315"]["near_uniform_under_1pp"]["rate"]
    fallback_delta = None if f15315 is None or f17315 is None else f17315 - f15315
    near_uniform_delta = None if n15315 is None or n17315 is None else n17315 - n15315

    if (
        primary["ci95_low"] is not None
        and primary["ci95_low"] > 0.0
        and fallback_delta is not None
        and fallback_delta <= 0.03
        and near_uniform_delta is not None
        and near_uniform_delta <= 0.03
    ):
        go_status = "GREEN_SAME_RECIPE_LONG_RUN_SUPPORTED"
    elif (
        primary["ci95_high"] is not None
        and primary["ci95_high"] < 0.0
    ) or (
        fallback_delta is not None and fallback_delta > 0.05
    ) or (
        near_uniform_delta is not None and near_uniform_delta > 0.05
    ):
        go_status = "RED_DO_NOT_RUN_SAME_RECIPE_FOR_THREE_MONTHS"
    else:
        go_status = "YELLOW_ONLY_AUTOGATED_SHORT_MACROBLOCKS"

    report = {
        "schema": "SPINCORE_3H_TEACHER_EFFICIENCY_FRESH_4WAY_V1",
        "scope": "FRESH_DEVELOPMENT_ONLY_LONG_RUN_GO_NO_GO",
        "canonical_quality_claim_authorized": False,
        "dc0_status": "PENDING_REAL_OPENHOLDEM_PARITY",
        "seed": int(args.seed),
        "scenarios": int(args.scenarios),
        "workers": int(min(args.workers, args.scenarios)),
        "domain": DOMAIN,
        "pairing": (
            "same fresh empirical scenario, exact deal seed, balanced AAB/ABB lineups, "
            "same master RNG seed across all four raw-teacher arms"
        ),
        "arms": summaries,
        "paired_deltas": deltas,
        "health_delta_17315_minus_15315": {
            "softmax_fallback_rate": fallback_delta,
            "near_uniform_under_1pp_rate": near_uniform_delta,
        },
        "go_no_go_status": go_status,
        "frozen_decision_rule": {
            "GREEN": (
                "Primary 17315-15315 paired EV CI95 low > 0, fallback-rate delta <= +3pp, "
                "and near-uniform delta <= +3pp."
            ),
            "RED": (
                "Primary paired EV CI95 high < 0 OR fallback-rate delta > +5pp OR "
                "near-uniform delta > +5pp."
            ),
            "YELLOW": (
                "Anything else: do not authorize a blind 3-month same-recipe run; "
                "only autonomous short macroblocks with repeated gates."
            ),
            "note": (
                "Sanity rates are reported for interpretation but high-card jam alone "
                "is not declared erroneous because pure bluffs can be strategically valid."
            ),
        },
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== SpinCore fresh 4-way teacher efficiency ===")
    for cp in CHECKPOINTS:
        print(f"ARM_{cp}=" + json.dumps(summaries[str(cp)], sort_keys=True))
    print("PAIRED_DELTAS=" + json.dumps(deltas, sort_keys=True))
    print(
        "HEALTH_DELTA_17315_MINUS_15315="
        + json.dumps(report["health_delta_17315_minus_15315"], sort_keys=True)
    )
    print("GO_NO_GO_STATUS=" + go_status)
    print(f"report={args.report.resolve()}")
    print("SPINCORE_3H_TEACHER_EFFICIENCY_FRESH_4WAY_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
