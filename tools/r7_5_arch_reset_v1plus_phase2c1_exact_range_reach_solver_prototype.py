from __future__ import annotations

"""Phase2C1: exact range/reach public-action transition prototype.

This is a structural solver prototype, not another target estimator.  It carries
two explicit 2,450-entry opponent reach vectors through the observed public
preflop action path and proves parity against the direct full-history
factorization already accepted in Phase2C0.
"""

import argparse
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
import r7_5_arch_reset_v1plus_phase2c0_structural_reach_factorization as c0

from spincore.r7_5_action_cfr import validate_policy
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_representation_v3_referee_states import effective_pf0
from spincore.r7_5_representation_v3_stage_contract import TRAINING_SEEDS
from spincore.solver import DealSnapshot
from spincore.solver_v3 import neural_bytes_v3

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2C1_EXACT_RANGE_REACH_SOLVER_PROTOTYPE_V1"
DOMAIN = c0.DOMAIN
REPRESENTATION = b15.REPRESENTATION
MAX_WORKERS = 16
DIRECT_CHECKS_PER_OPPONENT = 128
TOL = 1e-12
MAX_TABLE_POLICY_EVALUATIONS = 4902
MAX_RAW_REACH_BYTES = 39200
C0_RESULT_SHA256 = "55e83be4fd8776e0fcdb63e7d4400ed05aff8c48213898ad8f1abe3713a35876"
C0_STATUS = "STRUCTURAL_REACH_FACTORIZATION_FEASIBLE"
MASK64 = (1 << 64) - 1


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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
    x = 0x2C010A11CE000001
    for raw in parts:
        y = int(raw) & MASK64
        x ^= (y + 0x9E3779B97F4A7C15 + ((x << 6) & MASK64) + (x >> 2)) & MASK64
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & MASK64
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & MASK64
        x ^= x >> 31
    return x & MASK64


def _relative_error(actual: float, expected: float) -> float:
    a = float(actual)
    e = float(expected)
    if not math.isfinite(a) or not math.isfinite(e):
        return math.inf
    return abs(a - e) / max(abs(e), 1e-300)


def _reach_storage_bytes(opponent_count: int, hand_count: int) -> int:
    return int(opponent_count) * int(hand_count) * 8


def _table_sha256(values: Sequence[float]) -> str:
    arr = np.asarray([float(x) for x in values], dtype="<f8")
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _canonical_state_and_snapshot(task: dict):
    if b10._WORKER_SOLVER is None or b10._WORKER_ACTION_SPEC is None:
        raise RuntimeError("Phase2C1 worker not initialized")
    snapshot = b15fix._canonical_snapshot(task)
    episode = action_scenario_cycle(DOMAIN)[int(task["scenario_index"])]
    state = b10._WORKER_SOLVER.create_with_deal(episode, snapshot.holes, snapshot.board)
    return state, snapshot


def _event_probability(
    task: dict,
    snapshot: DealSnapshot,
    *,
    target_seat: int,
    hand: tuple[int, int],
    event_index: int,
) -> float:
    """Probability of the stored action at one public prefix for one private hand."""
    if b10._WORKER_SOLVER is None or b10._WORKER_COLLECTOR is None or b10._WORKER_ACTION_SPEC is None:
        raise RuntimeError("Phase2C1 worker not initialized")
    actor = int(task["actor"])
    deal = c0._deal_for_single_hand(snapshot, actor, int(target_seat), hand, 0)
    episode = action_scenario_cycle(DOMAIN)[int(task["scenario_index"])]
    state = b10._WORKER_SOLVER.create_with_deal(episode, deal.holes, deal.board)
    try:
        for index, action in enumerate(task["action_path"]):
            if state.terminal:
                raise RuntimeError("Phase2C1 candidate prefix reached terminal early")
            active_mask, legal, _exact = effective_pf0(state, b10._WORKER_ACTION_SPEC)
            if int(action) not in legal:
                raise RuntimeError("Phase2C1 stored public action became illegal")
            if index == int(event_index):
                if int(state.actor) != int(target_seat):
                    raise RuntimeError("Phase2C1 event target-seat actor drift")
                observation = neural_bytes_v3(state)
                probabilities = validate_policy(
                    b10._WORKER_COLLECTOR.policy(state, observation, legal), legal
                )
                return float(probabilities[int(action)])
            state.apply_universal(active_mask, int(action))
    finally:
        state.close()
    raise RuntimeError("Phase2C1 event index outside public path")


def _known_actor_event_probability(state, action: int) -> float:
    if b10._WORKER_COLLECTOR is None or b10._WORKER_ACTION_SPEC is None:
        raise RuntimeError("Phase2C1 worker not initialized")
    observation = neural_bytes_v3(state)
    active_mask, legal, _exact = effective_pf0(state, b10._WORKER_ACTION_SPEC)
    if int(action) not in legal:
        raise RuntimeError("Phase2C1 known-actor stored action became illegal")
    probabilities = validate_policy(
        b10._WORKER_COLLECTOR.policy(state, observation, legal), legal
    )
    return float(probabilities[int(action)])


def _reference_row(c0_result: dict, task: dict) -> dict:
    matches = [
        row for row in c0_result.get("rows", [])
        if int(row["behavior_seed"]) == int(task["behavior_seed"])
        and int(row["evaluation_seed"]) == int(task["evaluation_seed"])
        and int(row["state_index"]) == int(task["state_index"])
        and str(row["region"]) == str(task["region"])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Phase2C1 expected one Phase2C0 reference row, got {len(matches)}")
    return matches[0]


def _worker_task(task: dict) -> dict:
    c0_result = task.pop("_c0_result")
    reference = _reference_row(c0_result, task)
    canonical_state, snapshot = _canonical_state_and_snapshot(task)
    actor = int(task["actor"])
    opponents = [seat for seat in range(3) if seat != actor]
    hands = c0._ordered_hands(snapshot.holes[actor])
    if len(hands) != 2450:
        raise RuntimeError("Phase2C1 ordered-hand support drift")

    reaches = {seat: np.ones(len(hands), dtype=np.float64) for seat in opponents}
    actor_scalar = 1.0
    table_policy_evaluations = 0
    actor_sequence: list[int] = []

    try:
        for event_index, action in enumerate(task["action_path"]):
            if canonical_state.terminal:
                raise RuntimeError("Phase2C1 canonical public path reached terminal early")
            acting = int(canonical_state.actor)
            actor_sequence.append(acting)
            active_mask, legal, _exact = effective_pf0(canonical_state, b10._WORKER_ACTION_SPEC)
            if int(action) not in legal:
                raise RuntimeError("Phase2C1 canonical stored action became illegal")

            if acting == actor:
                actor_scalar *= _known_actor_event_probability(canonical_state, int(action))
                table_policy_evaluations += 1
            else:
                if acting not in reaches:
                    raise RuntimeError("Phase2C1 public actor outside opponent reach state")
                updated = reaches[acting]
                for hand_index, hand in enumerate(hands):
                    p = _event_probability(
                        task,
                        snapshot,
                        target_seat=acting,
                        hand=hand,
                        event_index=event_index,
                    )
                    if not math.isfinite(p) or p < 0.0 or p > 1.0 + 1e-12:
                        raise RuntimeError("Phase2C1 invalid frozen behavior probability")
                    updated[hand_index] *= float(p)
                table_policy_evaluations += len(hands)

            canonical_state.apply_universal(active_mask, int(action))

        if canonical_state.terminal:
            raise RuntimeError("Phase2C1 target continuation unexpectedly terminal")
        observation = neural_bytes_v3(canonical_state)
        active_mask, legal, _exact = effective_pf0(canonical_state, b10._WORKER_ACTION_SPEC)
        final_identity_exact = bool(
            observation == bytes(task["observation"])
            and int(canonical_state.actor) == actor
            and int(active_mask) == int(task["active_mask"])
            and tuple(legal) == tuple(task["legal_slots"])
        )
        if not final_identity_exact:
            raise RuntimeError("Phase2C1 final target identity drift")
    finally:
        canonical_state.close()

    rng = random.Random(
        _mix64(int(task["behavior_seed"]), int(task["evaluation_seed"]), int(task["state_index"]), 0xD1EC7)
    )
    direct_indices = rng.sample(range(len(hands)), DIRECT_CHECKS_PER_OPPONENT)
    max_direct_error = 0.0
    for seat in opponents:
        table = reaches[seat]
        for index in direct_indices:
            deal = c0._deal_for_single_hand(snapshot, actor, seat, hands[index], 1)
            direct = c0._prob(c0._seat_component_log(task, deal, seat))
            max_direct_error = max(max_direct_error, abs(float(table[index]) - float(direct)))

    canonical = DealSnapshot(snapshot.holes, snapshot.board, snapshot.visible_board_count)
    direct_actor = c0._prob(c0._seat_component_log(task, canonical, actor))
    actor_scalar_error = abs(float(actor_scalar) - float(direct_actor))

    seat_a, seat_b = opponents
    joint = c0._joint_stats(hands, reaches[seat_a], reaches[seat_b])
    normalizer_rel_error = _relative_error(joint["normalizer"], reference["normalizer"])
    support_rel_error = _relative_error(
        joint["effective_joint_support"], reference["effective_joint_support"]
    )
    positive_assignments_match = bool(
        int(joint["positive_joint_assignments"]) == int(reference["positive_joint_assignments"])
    )
    storage_bytes = _reach_storage_bytes(len(opponents), len(hands))

    return {
        "behavior_seed": int(task["behavior_seed"]),
        "evaluation_seed": int(task["evaluation_seed"]),
        "state_index": int(task["state_index"]),
        "scenario_index": int(task["scenario_index"]),
        "region": str(task["region"]),
        "actor": actor,
        "action_path_length": len(task["action_path"]),
        "actor_sequence": actor_sequence,
        "opponent_seats": opponents,
        "ordered_hands_per_opponent": len(hands),
        "raw_reach_storage_bytes": int(storage_bytes),
        "table_policy_evaluations": int(table_policy_evaluations),
        "direct_checks_per_opponent": DIRECT_CHECKS_PER_OPPONENT,
        "max_incremental_vs_direct_abs_error": float(max_direct_error),
        "actor_scalar_abs_error": float(actor_scalar_error),
        "joint_normalizer": float(joint["normalizer"]),
        "joint_normalizer_relative_error_vs_c0": float(normalizer_rel_error),
        "effective_joint_support": float(joint["effective_joint_support"]),
        "effective_support_relative_error_vs_c0": float(support_rel_error),
        "positive_joint_assignments": int(joint["positive_joint_assignments"]),
        "positive_joint_assignments_match_c0": positive_assignments_match,
        "final_target_identity_exact": True,
        "reach_table_sha256": {
            str(seat_a): _table_sha256(reaches[seat_a]),
            str(seat_b): _table_sha256(reaches[seat_b]),
        },
    }


def run(args) -> dict:
    repo_root = Path(args.repo_root).resolve()
    solver_path = Path(args.solver).resolve()
    heldout_root = Path(args.heldout_root).resolve()
    b13_root = Path(args.phase2b13_root).resolve()
    b13_result_path = Path(args.phase2b13_result).resolve()
    b14_result_path = Path(args.phase2b14_result).resolve()
    c0_result_path = Path(args.phase2c0_result).resolve()

    if _sha256(c0_result_path) != C0_RESULT_SHA256:
        raise RuntimeError("Phase2C1 Phase2C0 result SHA drift")
    j0 = json.loads(c0_result_path.read_text(encoding="utf-8"))
    if j0.get("schema") != c0.SCHEMA or j0.get("status") != C0_STATUS:
        raise RuntimeError("Phase2C1 requires exact successful Phase2C0 result")
    if not bool((j0.get("decision") or {}).get("screen_pass")):
        raise RuntimeError("Phase2C1 cannot follow a failed Phase2C0")
    if (j0.get("decision") or {}).get("next_route") != "PRECOMMIT_PHASE2C1_EXACT_RANGE_REACH_SOLVER_PROTOTYPE":
        raise RuntimeError("Phase2C1 Phase2C0 next-route drift")

    _j13, j14 = b15._validate_source_results(b13_result_path, b14_result_path)
    anchors, heldout_identity = c0._select_c0_anchors(heldout_root, j14)
    if len(anchors) != 8:
        raise RuntimeError("Phase2C1 source anchor count drift")

    rows = []
    behavior_identity = []
    for behavior_seed in map(int, TRAINING_SEEDS):
        checkpoint = b13_root / b13.CANDIDATE_ARM / f"seed_{behavior_seed}" / "resume_checkpoint.pt"
        states, identity = b15._load_behavior_states(checkpoint, behavior_seed)
        behavior_identity.append(identity)
        seed_tasks = []
        for anchor in anchors:
            task = dict(anchor)
            task["behavior_seed"] = int(behavior_seed)
            task["_c0_result"] = j0
            seed_tasks.append(task)

        with ProcessPoolExecutor(
            max_workers=min(int(args.workers), len(seed_tasks)),
            initializer=b10._worker_init,
            initargs=(str(repo_root), str(solver_path), int(behavior_seed), states),
        ) as pool:
            fmap = {pool.submit(_worker_task, task): task for task in seed_tasks}
            for future in as_completed(fmap):
                row = future.result()
                rows.append(row)
                print(
                    f"[Phase2C1 task] behavior={behavior_seed} eval={row['evaluation_seed']} "
                    f"state={row['state_index']} {row['region']} "
                    f"direct_err={row['max_incremental_vs_direct_abs_error']:.3e} "
                    f"norm_rel={row['joint_normalizer_relative_error_vs_c0']:.3e} "
                    f"evals={row['table_policy_evaluations']}",
                    flush=True,
                )

    rows.sort(
        key=lambda r: (
            int(r["behavior_seed"]), int(r["evaluation_seed"]), str(r["region"]), int(r["state_index"])
        )
    )
    complete = len(rows) == 16
    identity_ok = bool(rows and all(bool(r["final_target_identity_exact"]) for r in rows))
    max_direct = max((float(r["max_incremental_vs_direct_abs_error"]) for r in rows), default=math.inf)
    max_actor = max((float(r["actor_scalar_abs_error"]) for r in rows), default=math.inf)
    max_norm_rel = max((float(r["joint_normalizer_relative_error_vs_c0"]) for r in rows), default=math.inf)
    max_support_rel = max((float(r["effective_support_relative_error_vs_c0"]) for r in rows), default=math.inf)
    positive_ok = bool(rows and all(bool(r["positive_joint_assignments_match_c0"]) for r in rows))
    max_storage = max((int(r["raw_reach_storage_bytes"]) for r in rows), default=10**18)
    max_evals = max((int(r["table_policy_evaluations"]) for r in rows), default=10**18)

    pass_gate = bool(
        complete
        and identity_ok
        and max_direct <= TOL
        and max_actor <= TOL
        and max_norm_rel <= TOL
        and max_support_rel <= TOL
        and positive_ok
        and max_storage <= MAX_RAW_REACH_BYTES
        and max_evals <= MAX_TABLE_POLICY_EVALUATIONS
    )
    decision = {
        "complete_16_tasks": complete,
        "final_target_identity_exact_all": identity_ok,
        "max_incremental_vs_direct_abs_error": float(max_direct),
        "max_actor_scalar_abs_error": float(max_actor),
        "max_joint_normalizer_relative_error_vs_c0": float(max_norm_rel),
        "max_effective_support_relative_error_vs_c0": float(max_support_rel),
        "positive_joint_assignments_match_all": positive_ok,
        "max_raw_reach_storage_bytes": int(max_storage),
        "max_table_policy_evaluations": int(max_evals),
        "screen_pass": pass_gate,
        "next_route": (
            "PRECOMMIT_SINGLE_BOUNDED_RANGE_REACH_TARGET_KERNEL_CAUSAL_PILOT"
            if pass_gate
            else "SELECT_CERTIFIED_STABLE_V1_FALLBACK_AND_CLOSE_V1PLUS_ARCHITECTURE_RESET"
        ),
        "training_authorized": False,
        "full_x4_confirmation_authorized": False,
        "architecture_winner_selected": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    return {
        "schema": SCHEMA,
        "status": "EXACT_RANGE_REACH_TRANSITION_PROTOTYPE_FEASIBLE" if pass_gate else "EXACT_RANGE_REACH_TRANSITION_PROTOTYPE_NOT_FEASIBLE",
        "execution_sha": str(args.execution_sha),
        "domain": DOMAIN,
        "representation": REPRESENTATION,
        "source_phase2c0_result_sha256": C0_RESULT_SHA256,
        "contract": {
            "anchors": len(anchors),
            "behavior_seeds": list(map(int, TRAINING_SEEDS)),
            "ordered_hands_per_opponent": 2450,
            "direct_checks_per_opponent": DIRECT_CHECKS_PER_OPPONENT,
            "tolerance": TOL,
            "max_table_policy_evaluations": MAX_TABLE_POLICY_EVALUATIONS,
            "max_raw_reach_storage_bytes": MAX_RAW_REACH_BYTES,
            "target_estimator": None,
            "training": False,
        },
        "frozen_inputs": {"heldout": heldout_identity, "behavior_checkpoints": behavior_identity},
        "rows": rows,
        "incremental_vs_direct_error": _summary([r["max_incremental_vs_direct_abs_error"] for r in rows]),
        "joint_normalizer_relative_error_vs_c0": _summary([r["joint_normalizer_relative_error_vs_c0"] for r in rows]),
        "effective_support_relative_error_vs_c0": _summary([r["effective_support_relative_error_vs_c0"] for r in rows]),
        "table_policy_evaluations": _summary([r["table_policy_evaluations"] for r in rows]),
        "decision": decision,
        "training_authorized": False,
        "full_x4_confirmation_authorized": False,
        "architecture_winner_selected": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase2C1 exact range/reach solver prototype")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--phase2b13-root", type=Path, required=True)
    parser.add_argument("--phase2b13-result", type=Path, required=True)
    parser.add_argument("--phase2b14-result", type=Path, required=True)
    parser.add_argument("--phase2c0-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    if int(args.workers) < 1 or int(args.workers) > MAX_WORKERS:
        raise RuntimeError("Phase2C1 workers outside frozen range")
    result = run(args)
    out = Path(args.output_root).resolve() / "R7_5_ARCH_RESET_V1PLUS_PHASE2C1_EXACT_RANGE_REACH_SOLVER_PROTOTYPE.json"
    _atomic_json(result, out)
    print(json.dumps({
        "status": result["status"],
        "max_incremental_vs_direct_abs_error": result["decision"]["max_incremental_vs_direct_abs_error"],
        "max_joint_normalizer_relative_error_vs_c0": result["decision"]["max_joint_normalizer_relative_error_vs_c0"],
        "max_effective_support_relative_error_vs_c0": result["decision"]["max_effective_support_relative_error_vs_c0"],
        "max_table_policy_evaluations": result["decision"]["max_table_policy_evaluations"],
        "screen_pass": result["decision"]["screen_pass"],
        "next_route": result["decision"]["next_route"],
        "result": str(out),
        "result_sha256": _sha256(out),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
