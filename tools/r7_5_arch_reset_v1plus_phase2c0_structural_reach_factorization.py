from __future__ import annotations

"""Phase2C0: structural preflop reach-factorization feasibility audit.

This is not a target estimator. It verifies whether the frozen public-action
likelihood at preflop continuation infosets factorizes exactly into seat-local
private-hand likelihood tables, coupled only by card removal. PASS permits a
separately precommitted solver-level range/reach prototype; FAIL selects the
certified stable V1 fallback.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Sequence

import numpy as np

import r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition as b10
import r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training as b13
import r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance as b15
import r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance_runtimefix as b15fix
import r7_5_arch_reset_v1plus_phase2b16_exact_rejection_posterior_continuation as b16

from spincore.r7_5_action_cfr import validate_policy
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_representation_v3_referee_states import effective_pf0
from spincore.r7_5_representation_v3_stage_contract import TRAINING_SEEDS, EVALUATION_SEEDS
from spincore.solver import DealSnapshot
from spincore.solver_v3 import neural_bytes_v3

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2C0_STRUCTURAL_REACH_FACTORIZATION_V1"
DOMAIN = b15.DOMAIN
REGIONS = b15.REGIONS
ANCHORS_PER_REGION_PER_EVAL = 2
MAX_WORKERS = 16
FACTORIZATION_CHECKS = 128
FILLER_CHECKS_PER_SEAT = 32
TOL = 1e-12
B16_RESULT_SHA256 = "3b5e71c3cc92ed530589877f6790333b1f94b579bb39e7c687082787693d958c"
B16_STATUS = "EXACT_POSTERIOR_STILL_TOO_UNSTABLE_CLOSE_ESTIMATOR_REPAIR_PATH"
MASK64 = (1 << 64) - 1


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _summary(values: Sequence[float]) -> dict:
    arr = np.asarray([float(x) for x in values], dtype=np.float64)
    if not arr.size:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50, method="linear")),
        "p95": float(np.quantile(arr, 0.95, method="linear")),
        "max": float(arr.max()),
    }


def _mix64(*parts: int) -> int:
    x = 0x2C00FACADE000001
    for raw in parts:
        y = int(raw) & MASK64
        x ^= (y + 0x9E3779B97F4A7C15 + ((x << 6) & MASK64) + (x >> 2)) & MASK64
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & MASK64
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & MASK64
        x ^= x >> 31
    return x & MASK64


def _ordered_hands(actor_cards: Sequence[int]) -> list[tuple[int, int]]:
    blocked = {int(x) for x in actor_cards}
    cards = [c for c in range(52) if c not in blocked]
    hands = [(a, b) for a in cards for b in cards if b != a]
    if len(hands) != 2450:
        raise RuntimeError(f"Phase2C0 ordered-hand count drift: {len(hands)}")
    return hands


def _deal_for_single_hand(snapshot: DealSnapshot, actor: int, target_seat: int,
                          hand: tuple[int, int], filler_variant: int = 0) -> DealSnapshot:
    actor_cards = tuple(int(x) for x in snapshot.holes[int(actor)])
    used = set(actor_cards) | {int(hand[0]), int(hand[1])}
    if len(used) != 4:
        raise ValueError("Phase2C0 candidate hand collides with actor cards")
    other = next(seat for seat in range(3) if seat not in (int(actor), int(target_seat)))
    remaining = [c for c in range(52) if c not in used]
    if int(filler_variant) == 0:
        filler = remaining[:2]
        board = remaining[2:7]
    else:
        filler = remaining[-2:]
        filler_set = set(filler)
        board = [c for c in reversed(remaining) if c not in filler_set][:5]
    holes = [[-1, -1] for _ in range(3)]
    holes[int(actor)] = [actor_cards[0], actor_cards[1]]
    holes[int(target_seat)] = [int(hand[0]), int(hand[1])]
    holes[int(other)] = [int(filler[0]), int(filler[1])]
    return DealSnapshot(
        holes=tuple(tuple(int(x) for x in row) for row in holes),
        board=tuple(int(x) for x in board),
        visible_board_count=0,
    )


def _deal_for_joint(snapshot: DealSnapshot, actor: int, seat_a: int, hand_a: tuple[int, int],
                    seat_b: int, hand_b: tuple[int, int]) -> DealSnapshot:
    actor_cards = tuple(int(x) for x in snapshot.holes[int(actor)])
    flat = [*actor_cards, *map(int, hand_a), *map(int, hand_b)]
    if len(flat) != len(set(flat)):
        raise ValueError("Phase2C0 joint opponent hands collide")
    board = tuple(c for c in range(52) if c not in set(flat))[:5]
    holes = [[-1, -1] for _ in range(3)]
    holes[int(actor)] = [actor_cards[0], actor_cards[1]]
    holes[int(seat_a)] = [int(hand_a[0]), int(hand_a[1])]
    holes[int(seat_b)] = [int(hand_b[0]), int(hand_b[1])]
    return DealSnapshot(tuple(tuple(row) for row in holes), board, 0)


def _seat_component_log(task: dict, deal: DealSnapshot, target_seat: int) -> float:
    if b10._WORKER_SOLVER is None or b10._WORKER_COLLECTOR is None or b10._WORKER_ACTION_SPEC is None:
        raise RuntimeError("Phase2C0 worker not initialized")
    episode = action_scenario_cycle(DOMAIN)[int(task["scenario_index"])]
    state = b10._WORKER_SOLVER.create_with_deal(episode, deal.holes, deal.board)
    collector = b10._WORKER_COLLECTOR
    logp = 0.0
    try:
        for action in task["action_path"]:
            if state.terminal:
                raise RuntimeError("Phase2C0 replay reaches terminal early")
            acting = int(state.actor)
            observation = neural_bytes_v3(state)
            active_mask, legal, _exact = effective_pf0(state, b10._WORKER_ACTION_SPEC)
            if int(action) not in legal:
                raise RuntimeError("Phase2C0 observed action became illegal")
            probs = validate_policy(collector.policy(state, observation, legal), legal)
            if acting == int(target_seat):
                p = float(probs[int(action)])
                if p <= 0.0:
                    logp = -math.inf
                elif math.isfinite(logp):
                    logp += math.log(p)
            state.apply_universal(active_mask, int(action))
        if state.terminal:
            raise RuntimeError("Phase2C0 target continuation unexpectedly terminal")
        observation = neural_bytes_v3(state)
        active_mask, legal, _exact = effective_pf0(state, b10._WORKER_ACTION_SPEC)
        if observation != bytes(task["observation"]):
            raise RuntimeError("Phase2C0 replay changed target SPNNIV3")
        if int(state.actor) != int(task["actor"]):
            raise RuntimeError("Phase2C0 replay changed target actor")
        if int(active_mask) != int(task["active_mask"]) or tuple(legal) != tuple(task["legal_slots"]):
            raise RuntimeError("Phase2C0 replay changed target legal identity")
    finally:
        state.close()
    return float(logp)


def _prob(logp: float) -> float:
    return 0.0 if not math.isfinite(float(logp)) else float(math.exp(float(logp)))


def _joint_stats(hands: Sequence[tuple[int, int]], wa: Sequence[float], wb: Sequence[float]) -> dict:
    if len(hands) != 2450 or len(wa) != 2450 or len(wb) != 2450:
        raise ValueError("Phase2C0 joint stats require two 2450-hand tables")
    b0 = np.asarray([h[0] for h in hands], dtype=np.int16)
    b1 = np.asarray([h[1] for h in hands], dtype=np.int16)
    wbv = np.asarray(wb, dtype=np.float64)
    wb2 = wbv * wbv
    z = 0.0
    s2 = 0.0
    positive_pairs = 0
    for i, (a0, a1) in enumerate(hands):
        mask = (b0 != a0) & (b1 != a0) & (b0 != a1) & (b1 != a1)
        valid_w = wbv[mask]
        w = float(wa[i])
        z += w * float(valid_w.sum())
        s2 += (w * w) * float(wb2[mask].sum())
        if w > 0.0:
            positive_pairs += int(np.count_nonzero(valid_w > 0.0))
    if not math.isfinite(z) or z <= 0.0 or not math.isfinite(s2) or s2 <= 0.0:
        return {"normalizer": float(z), "effective_joint_support": 0.0, "positive_joint_assignments": int(positive_pairs)}
    return {
        "normalizer": float(z),
        "effective_joint_support": float((z * z) / s2),
        "positive_joint_assignments": int(positive_pairs),
    }


def _worker_task(task: dict) -> dict:
    snapshot = b15fix._canonical_snapshot(task)
    actor = int(task["actor"])
    opponents = [s for s in range(3) if s != actor]
    canonical = DealSnapshot(snapshot.holes, snapshot.board, snapshot.visible_board_count)
    actor_log = _seat_component_log(task, canonical, actor)
    actor_prob = _prob(actor_log)
    hands = _ordered_hands(snapshot.holes[actor])

    tables: dict[int, list[float]] = {}
    table_evals = 1
    filler_error = 0.0
    for seat in opponents:
        weights = []
        for hand in hands:
            deal = _deal_for_single_hand(snapshot, actor, seat, hand, 0)
            weights.append(_prob(_seat_component_log(task, deal, seat)))
        tables[int(seat)] = weights
        table_evals += len(hands)

        rng = random.Random(_mix64(int(task["behavior_seed"]), int(task["evaluation_seed"]),
                                   int(task["state_index"]), int(seat), 0xF111))
        indices = rng.sample(range(len(hands)), FILLER_CHECKS_PER_SEAT)
        for idx in indices:
            alt = _deal_for_single_hand(snapshot, actor, seat, hands[idx], 1)
            altp = _prob(_seat_component_log(task, alt, seat))
            filler_error = max(filler_error, abs(float(weights[idx]) - altp))

    seat_a, seat_b = opponents
    wa, wb = tables[seat_a], tables[seat_b]
    rng = random.Random(_mix64(int(task["behavior_seed"]), int(task["evaluation_seed"]),
                               int(task["state_index"]), 0xFAC70))
    factor_error = 0.0
    checks = 0
    attempts = 0
    while checks < FACTORIZATION_CHECKS and attempts < FACTORIZATION_CHECKS * 100:
        ia = rng.randrange(len(hands))
        ib = rng.randrange(len(hands))
        attempts += 1
        ha, hb = hands[ia], hands[ib]
        if len({ha[0], ha[1], hb[0], hb[1]}) != 4:
            continue
        joint = _deal_for_joint(snapshot, actor, seat_a, ha, seat_b, hb)
        full = _prob(b16._log_likelihood(task, joint))
        factored = float(actor_prob * wa[ia] * wb[ib])
        factor_error = max(factor_error, abs(full - factored))
        checks += 1
    if checks != FACTORIZATION_CHECKS:
        raise RuntimeError("Phase2C0 could not complete deterministic factorization checks")

    stats = _joint_stats(hands, wa, wb)
    return {
        "behavior_seed": int(task["behavior_seed"]),
        "evaluation_seed": int(task["evaluation_seed"]),
        "state_index": int(task["state_index"]),
        "scenario_index": int(task["scenario_index"]),
        "region": str(task["region"]),
        "actor": actor,
        "action_path_length": len(task["action_path"]),
        "opponent_seats": opponents,
        "ordered_hands_per_opponent": len(hands),
        "table_policy_evaluations": int(table_evals),
        "validation_replays": int(FACTORIZATION_CHECKS + 2 * FILLER_CHECKS_PER_SEAT),
        "max_factorization_abs_error": float(factor_error),
        "max_filler_independence_abs_error": float(filler_error),
        "actor_component_probability": float(actor_prob),
        **stats,
    }


def _select_c0_anchors(heldout_root: Path, b14_result: dict) -> tuple[list[dict], list[dict]]:
    all_anchors, heldout_identity = b15._select_anchors(heldout_root, b14_result)
    grouped = defaultdict(list)
    for row in all_anchors:
        grouped[(int(row["evaluation_seed"]), str(row["region"]))].append(row)
    out = []
    for evaluation_seed in map(int, EVALUATION_SEEDS):
        for region in REGIONS:
            rows = sorted(grouped[(evaluation_seed, region)], key=lambda r: int(r["state_index"]))
            if len(rows) < ANCHORS_PER_REGION_PER_EVAL:
                raise RuntimeError("Phase2C0 insufficient source anchors")
            out.extend(rows[:ANCHORS_PER_REGION_PER_EVAL])
    if len(out) != 8:
        raise RuntimeError(f"Phase2C0 anchor count drift: {len(out)}")
    return out, heldout_identity


def run(args) -> dict:
    repo_root = Path(args.repo_root).resolve()
    solver_path = Path(args.solver).resolve()
    heldout_root = Path(args.heldout_root).resolve()
    b13_root = Path(args.phase2b13_root).resolve()
    b13_result = Path(args.phase2b13_result).resolve()
    b14_result_path = Path(args.phase2b14_result).resolve()
    b16_result_path = Path(args.phase2b16_result).resolve()
    if _sha256(b16_result_path) != B16_RESULT_SHA256:
        raise RuntimeError("Phase2C0 Phase2B16 result SHA drift")
    j16 = json.loads(b16_result_path.read_text(encoding="utf-8"))
    if j16.get("schema") != b16.SCHEMA or j16.get("status") != B16_STATUS:
        raise RuntimeError("Phase2C0 requires exact failed Phase2B16 result")
    if bool((j16.get("decision") or {}).get("screen_pass")):
        raise RuntimeError("Phase2C0 cannot follow a passing Phase2B16")

    _j13, j14 = b15._validate_source_results(b13_result, b14_result_path)
    anchors, heldout_identity = _select_c0_anchors(heldout_root, j14)
    tasks = []
    behavior_identity = []
    states_by_seed = {}
    for behavior_seed in map(int, TRAINING_SEEDS):
        checkpoint = b13_root / b13.CANDIDATE_ARM / f"seed_{behavior_seed}" / "resume_checkpoint.pt"
        states, identity = b15._load_behavior_states(checkpoint, behavior_seed)
        states_by_seed[int(behavior_seed)] = states
        behavior_identity.append(identity)
        for anchor in anchors:
            task = dict(anchor)
            task["behavior_seed"] = int(behavior_seed)
            tasks.append(task)

    rows = []
    for behavior_seed in map(int, TRAINING_SEEDS):
        seed_tasks = [dict(t) for t in tasks if int(t["behavior_seed"]) == behavior_seed]
        with ProcessPoolExecutor(
            max_workers=min(int(args.workers), len(seed_tasks)),
            initializer=b10._worker_init,
            initargs=(str(repo_root), str(solver_path), int(behavior_seed), states_by_seed[behavior_seed]),
        ) as pool:
            fmap = {pool.submit(_worker_task, t): t for t in seed_tasks}
            for future in as_completed(fmap):
                row = future.result()
                rows.append(row)
                print(
                    f"[Phase2C0 task] behavior={behavior_seed} eval={row['evaluation_seed']} "
                    f"state={row['state_index']} {row['region']} factor_err={row['max_factorization_abs_error']:.3e} "
                    f"filler_err={row['max_filler_independence_abs_error']:.3e} "
                    f"effective_support={row['effective_joint_support']:.1f}",
                    flush=True,
                )

    rows.sort(key=lambda r: (int(r["behavior_seed"]), int(r["evaluation_seed"]), str(r["region"]), int(r["state_index"])))
    complete = len(rows) == 16
    max_factor = max(float(r["max_factorization_abs_error"]) for r in rows) if rows else math.inf
    max_filler = max(float(r["max_filler_independence_abs_error"]) for r in rows) if rows else math.inf
    normalizers_ok = bool(rows and all(math.isfinite(float(r["normalizer"])) and float(r["normalizer"]) > 0.0 for r in rows))
    eval_budget_ok = bool(rows and all(int(r["table_policy_evaluations"]) <= 4901 for r in rows))
    pass_gate = bool(complete and max_factor <= TOL and max_filler <= TOL and normalizers_ok and eval_budget_ok)
    decision = {
        "complete_16_tasks": complete,
        "max_factorization_abs_error": float(max_factor),
        "max_filler_independence_abs_error": float(max_filler),
        "positive_finite_normalizers": normalizers_ok,
        "table_policy_evaluation_budget_pass": eval_budget_ok,
        "screen_pass": pass_gate,
        "next_route": (
            "PRECOMMIT_PHASE2C1_EXACT_RANGE_REACH_SOLVER_PROTOTYPE"
            if pass_gate else
            "SELECT_CERTIFIED_STABLE_V1_FALLBACK_AND_CLOSE_V1PLUS_ARCHITECTURE_RESET"
        ),
        "training_authorized": False,
        "architecture_winner_selected": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    return {
        "schema": SCHEMA,
        "status": "STRUCTURAL_REACH_FACTORIZATION_FEASIBLE" if pass_gate else "STRUCTURAL_REACH_FACTORIZATION_NOT_FEASIBLE",
        "execution_sha": str(args.execution_sha),
        "domain": DOMAIN,
        "representation": b15.REPRESENTATION,
        "source_phase2b16_result_sha256": B16_RESULT_SHA256,
        "contract": {
            "anchors": len(anchors),
            "behavior_seeds": list(map(int, TRAINING_SEEDS)),
            "ordered_hands_per_opponent": 2450,
            "ordered_joint_prior_assignments": 5527200,
            "factorization_checks_per_task": FACTORIZATION_CHECKS,
            "filler_checks_per_seat_per_task": FILLER_CHECKS_PER_SEAT,
            "tolerance": TOL,
        },
        "frozen_inputs": {"heldout": heldout_identity, "behavior_checkpoints": behavior_identity},
        "rows": rows,
        "effective_joint_support": _summary([float(r["effective_joint_support"]) for r in rows]),
        "positive_joint_assignments": _summary([float(r["positive_joint_assignments"]) for r in rows]),
        "decision": decision,
        "training_authorized": False,
        "architecture_winner_selected": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase2C0 structural reach-factorization feasibility audit")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--phase2b13-root", type=Path, required=True)
    parser.add_argument("--phase2b13-result", type=Path, required=True)
    parser.add_argument("--phase2b14-result", type=Path, required=True)
    parser.add_argument("--phase2b16-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    if int(args.workers) < 1 or int(args.workers) > MAX_WORKERS:
        raise RuntimeError("Phase2C0 workers outside frozen range")
    result = run(args)
    out = Path(args.output_root).resolve() / "R7_5_ARCH_RESET_V1PLUS_PHASE2C0_STRUCTURAL_REACH_FACTORIZATION.json"
    _atomic_json(result, out)
    print(json.dumps({
        "status": result["status"],
        "max_factorization_abs_error": result["decision"]["max_factorization_abs_error"],
        "max_filler_independence_abs_error": result["decision"]["max_filler_independence_abs_error"],
        "effective_joint_support_p50": result["effective_joint_support"]["p50"],
        "screen_pass": result["decision"]["screen_pass"],
        "next_route": result["decision"]["next_route"],
        "result": str(out),
        "result_sha256": _sha256(out),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
