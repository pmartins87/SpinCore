from __future__ import annotations

"""Read-only Phase2B2 common-chance / behavior-feedback decomposition.

Uses the exact Phase2B1 collision groups and deck seeds.  Both frozen Phase2A
behavior ensembles are evaluated on identical deals, either with common or
independent traversal RNG.  No optimizer step, model fit, reservoir insertion,
or checkpoint mutation occurs.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np
import torch

from spincore.deep_cfr import icm_delta_utility
from spincore.r7_5_action_cfr import legal_mask, regret_matching_policy
from spincore.r7_5_action_scenarios import action_scenario_cycle, scenario_descriptor
from spincore.r7_5_representation_v3 import H2_FINAL, UniversalPartialExactCollectorV3
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    ENSEMBLE_SIZE,
    EPSILON_CAP,
    EPSILON_SCALE,
    EXACT_OPPONENT_LEVELS,
    PAYOUT,
    TRAINING_SEEDS,
    validate_phase2_v3_contract,
)
from spincore.solver import SolverLibrary
from spincore.solver_v3 import neural_bytes_v3

from r7_5_arch_reset_v1plus_phase2b1_target_variance import _load_behavior

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B2_COMMON_CHANCE_FEEDBACK_V1"
DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL
SOURCE_EXECUTION_SHA = "4bfa55d69029cd69536fa6dbfcadd162719cb887"
PHASE2B1_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B1_TARGET_VARIANCE_V1"
PHASE2B1_STATUS = "PHASE2B1_K4_SCREEN_FAIL_NO_GENERIC_K4_TRAINING_PILOT"
REFERENCE_CHANCE_K1_TV = 0.5153716032136447
FEEDBACK_DOMINANT_FRACTION = 0.80
SHARED_ABSOLUTE_IMPROVEMENT_MIN = 0.10
SHARED_RELATIVE_IMPROVEMENT_MIN = 0.30
SHARED_K1_TV_MAX = 0.35
COMMON_VS_INDEPENDENT_TOLERANCE = 0.05
TARGET_ITERATION = 3
REPLICATES = 16
K_VALUES = (1, 2, 4, 8, 16)
ARMS = ("COMMON_TRAVERSAL_RNG", "INDEPENDENT_TRAVERSAL_RNG")
MAX_WORKERS = 12
MASK64 = (1 << 64) - 1

_WORKER_SOLVER = None
_WORKER_COLLECTORS = None


class _Sink:
    def __init__(self):
        self.items = []

    def add(self, item) -> None:
        self.items.append(item)


def _summary(values: Sequence[float]) -> dict:
    rows = [float(value) for value in values]
    if not rows:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    arr = np.asarray(rows, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50, method="linear")),
        "p95": float(np.quantile(arr, 0.95, method="linear")),
        "max": float(arr.max()),
    }


def _policy_tv(left: Sequence[float], right: Sequence[float]) -> float:
    return float(0.5 * sum(abs(float(a) - float(b)) for a, b in zip(left, right)))


def _mean_targets(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not rows or any(len(row) != 10 for row in rows):
        raise ValueError("Phase2B2 requires nonempty ten-slot target rows")
    return tuple(
        float(sum(float(row[slot]) for row in rows) / len(rows))
        for slot in range(10)
    )


def paired_k_metrics(
    targets_a: Sequence[Sequence[float]],
    targets_b: Sequence[Sequence[float]],
    legal_mask_row: Sequence[int],
    k: int,
) -> list[dict]:
    a = [tuple(float(value) for value in row) for row in targets_a]
    b = [tuple(float(value) for value in row) for row in targets_b]
    if len(a) != REPLICATES or len(b) != REPLICATES:
        raise ValueError(f"Phase2B2 requires exactly {REPLICATES} paired target replicates")
    if int(k) not in K_VALUES or REPLICATES % int(k) != 0:
        raise ValueError("invalid frozen K")
    legal = tuple(index for index, enabled in enumerate(legal_mask_row) if int(enabled))
    if not legal:
        raise ValueError("empty legal mask")
    out = []
    for start in range(0, REPLICATES, int(k)):
        mean_a = _mean_targets(a[start : start + int(k)])
        mean_b = _mean_targets(b[start : start + int(k)])
        mad = float(sum(abs(mean_a[slot] - mean_b[slot]) for slot in legal) / len(legal))
        sign = float(sum((mean_a[slot] > 0.0) != (mean_b[slot] > 0.0) for slot in legal) / len(legal))
        policy_a = regret_matching_policy(mean_a, legal)
        policy_b = regret_matching_policy(mean_b, legal)
        dom_a = max(legal, key=lambda slot: float(policy_a[slot]))
        dom_b = max(legal, key=lambda slot: float(policy_b[slot]))
        out.append({
            "k": int(k),
            "block_start": int(start),
            "target_mean_abs_diff": mad,
            "legal_sign_disagreement_fraction": sign,
            "regret_matching_policy_tv": _policy_tv(policy_a, policy_b),
            "dominant_legal_action_mismatch": int(dom_a != dom_b),
        })
    return out


def _traversal_seed(scenario_index: int, replicate: int, namespace: int) -> int:
    return (
        0xB2C04D4F4E000001
        ^ (int(scenario_index) * 0x9E3779B97F4A7C15)
        ^ (int(replicate) * 0xBF58476D1CE4E5B9)
        ^ (int(namespace) * 0x94D049BB133111EB)
    ) & MASK64


def _worker_init(repo_root: str, solver_path: str, input_root: str, source_sha: str) -> None:
    global _WORKER_SOLVER, _WORKER_COLLECTORS
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError("Phase2B2 worker torch-thread contract drift")
    root = Path(repo_root)
    contract = validate_phase2_v3_contract(
        root,
        representation=REPRESENTATION,
        domain=DOMAIN,
        training_seed=int(TRAINING_SEEDS[0]),
    )
    action_spec = contract["action_spec"]
    _WORKER_SOLVER = SolverLibrary(solver_path)
    collectors = {}
    for seed in map(int, TRAINING_SEEDS):
        checkpoint = Path(input_root) / f"seed_{seed}" / "resume_checkpoint.pt"
        loaded_seed, behavior = _load_behavior(checkpoint, source_sha)
        if loaded_seed != seed:
            raise RuntimeError("Phase2B2 worker source behavior seed mismatch")
        collectors[seed] = UniversalPartialExactCollectorV3(
            action_spec=action_spec,
            policy=behavior,
            terminal_utility=icm_delta_utility(PAYOUT),
            rng=random.Random(0),
            advantage_memory=_Sink(),
            strategy_memory=_Sink(),
        )
    _WORKER_COLLECTORS = collectors


def _root_identity(root, action_spec) -> tuple[bytes, int, tuple[int, ...], tuple[int, ...]]:
    observation = neural_bytes_v3(root)
    actor = int(root.actor)
    payload_v2 = root.neural_bytes_v2()
    if len(payload_v2) != 830 or not payload_v2.startswith(b"SPNNIV2\x00"):
        raise RuntimeError("Phase2B2 requires authoritative SPNNIV2 street metadata")
    street = int(payload_v2[112])
    active_mask = int(action_spec.active_mask(street))
    legal = tuple(int(value) for value in root.universal_legal_actions(active_mask))
    if not legal:
        raise RuntimeError("Phase2B2 root has no legal universal actions")
    return observation, actor, legal, legal_mask(legal)


def _one_target(
    collector,
    episode,
    *,
    deck_seed: int,
    traversal_seed: int,
    expected_sha: str,
    expected_actor: int,
    expected_legal: tuple[int, ...],
    expected_mask: tuple[int, ...],
) -> tuple[tuple[float, ...], int]:
    if _WORKER_SOLVER is None:
        raise RuntimeError("Phase2B2 worker solver not initialized")
    root = _WORKER_SOLVER.create(episode, int(deck_seed))
    sink = _Sink()
    collector.advantage_memory = sink
    collector.strategy_memory = _Sink()
    collector.rng = random.Random(int(traversal_seed))
    try:
        observation, actor, legal, mask = _root_identity(root, collector.action_spec)
        if hashlib.sha256(observation).hexdigest() != str(expected_sha):
            raise RuntimeError("Phase2B2 exact root observation hash drift")
        if actor != int(expected_actor) or legal != tuple(expected_legal) or mask != tuple(expected_mask):
            raise RuntimeError("Phase2B2 root actor/legal identity drift")
        result = collector.collect_advantage_partial_exact(
            root,
            traverser=actor,
            iteration=TARGET_ITERATION,
            exact_opponent_levels=EXACT_OPPONENT_LEVELS,
        )
    finally:
        root.close()
    matches = [
        sample for sample in sink.items
        if hashlib.sha256(sample.observation).hexdigest() == str(expected_sha)
        and tuple(sample.legal) == tuple(expected_mask)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Phase2B2 expected exactly one root target sample, got {len(matches)}")
    return tuple(float(value) for value in matches[0].target), int(result.nodes)


def _worker_task(task: dict) -> dict:
    if _WORKER_COLLECTORS is None:
        raise RuntimeError("Phase2B2 worker collectors not initialized")
    scenario_index = int(task["scenario_index"])
    episode = action_scenario_cycle(DOMAIN)[scenario_index]
    if scenario_descriptor(episode) != dict(task["scenario"]):
        raise RuntimeError("Phase2B2 scenario descriptor drift")
    arm = str(task["arm"])
    expected_sha = str(task["observation_sha256"])
    actor = int(task["actor"])
    legal = tuple(int(value) for value in task["legal"])
    mask = tuple(int(value) for value in task["legal_mask"])
    deck_seeds = [int(value) for value in task["deck_seeds"]]
    if len(deck_seeds) != REPLICATES or len(set(deck_seeds)) != REPLICATES:
        raise RuntimeError("Phase2B2 requires 16 distinct stored Phase2B1 deck seeds")

    seed_a, seed_b = map(int, TRAINING_SEEDS)
    targets_a = []
    targets_b = []
    nodes_a = []
    nodes_b = []
    started = time.perf_counter()
    for replicate, deck_seed in enumerate(deck_seeds):
        base = _traversal_seed(scenario_index, replicate, 0)
        if arm == "COMMON_TRAVERSAL_RNG":
            rng_a = rng_b = base
        elif arm == "INDEPENDENT_TRAVERSAL_RNG":
            rng_a = _traversal_seed(scenario_index, replicate, 1)
            rng_b = _traversal_seed(scenario_index, replicate, 2)
        else:
            raise RuntimeError(f"unknown Phase2B2 arm {arm}")
        target_a, count_a = _one_target(
            _WORKER_COLLECTORS[seed_a], episode,
            deck_seed=deck_seed, traversal_seed=rng_a,
            expected_sha=expected_sha, expected_actor=actor,
            expected_legal=legal, expected_mask=mask,
        )
        target_b, count_b = _one_target(
            _WORKER_COLLECTORS[seed_b], episode,
            deck_seed=deck_seed, traversal_seed=rng_b,
            expected_sha=expected_sha, expected_actor=actor,
            expected_legal=legal, expected_mask=mask,
        )
        targets_a.append([float(value) for value in target_a])
        targets_b.append([float(value) for value in target_b])
        nodes_a.append(int(count_a))
        nodes_b.append(int(count_b))
    return {
        "scenario_index": scenario_index,
        "arm": arm,
        "targets_a": targets_a,
        "targets_b": targets_b,
        "legal_mask": list(mask),
        "nodes_a": nodes_a,
        "nodes_b": nodes_b,
        "seconds": float(time.perf_counter() - started),
    }


def _summarize_pair_rows(rows: Sequence[dict]) -> dict:
    return {
        "pair_count": len(rows),
        "target_mean_abs_diff": _summary([row["target_mean_abs_diff"] for row in rows]),
        "legal_sign_disagreement_fraction": _summary([row["legal_sign_disagreement_fraction"] for row in rows]),
        "regret_matching_policy_tv": _summary([row["regret_matching_policy_tv"] for row in rows]),
        "dominant_legal_action_mismatch_rate": (
            float(sum(int(row["dominant_legal_action_mismatch"]) for row in rows) / len(rows)) if rows else None
        ),
    }


def _metrics(task_rows: Sequence[dict]) -> tuple[list[dict], dict]:
    pair_rows = []
    for task in task_rows:
        for k in K_VALUES:
            for row in paired_k_metrics(task["targets_a"], task["targets_b"], task["legal_mask"], int(k)):
                pair_rows.append({
                    "scenario_index": int(task["scenario_index"]),
                    "arm": str(task["arm"]),
                    **row,
                })
    pooled = {}
    for arm in ARMS:
        pooled[arm] = {}
        for k in K_VALUES:
            rows = [row for row in pair_rows if row["arm"] == arm and int(row["k"]) == int(k)]
            pooled[arm][f"K{k}"] = _summarize_pair_rows(rows)
    return pair_rows, pooled


def decision_from_pooled(pooled: dict, reference_tv: float = REFERENCE_CHANCE_K1_TV) -> dict:
    independent = float(pooled["INDEPENDENT_TRAVERSAL_RNG"]["K1"]["regret_matching_policy_tv"]["mean"])
    common = float(pooled["COMMON_TRAVERSAL_RNG"]["K1"]["regret_matching_policy_tv"]["mean"])
    abs_improvement = float(reference_tv - independent)
    rel_improvement = float(abs_improvement / reference_tv) if reference_tv > 0.0 else 0.0
    shared_pass = bool(
        abs_improvement >= SHARED_ABSOLUTE_IMPROVEMENT_MIN
        and rel_improvement >= SHARED_RELATIVE_IMPROVEMENT_MIN
        and independent <= SHARED_K1_TV_MAX
        and common <= independent + COMMON_VS_INDEPENDENT_TOLERANCE
    )
    feedback_threshold = float(reference_tv * FEEDBACK_DOMINANT_FRACTION)
    if shared_pass:
        classification = "COMMON_CHANCE_SUPPORT_MATERIALLY_SUPPORTED"
        route = "PRECOMMIT_SMALL_SHARED_CHANCE_SUPPORT_TRAINING_PILOT"
        pilot_allowed = True
    elif independent >= feedback_threshold:
        classification = "BEHAVIOR_FEEDBACK_REMAINS_DOMINANT_ON_COMMON_CHANCE"
        route = "DIAGNOSE_BEHAVIOR_FEEDBACK_STABILIZATION_BEFORE_TRAINING"
        pilot_allowed = False
    else:
        classification = "MIXED_CHANCE_SUPPORT_AND_FEEDBACK"
        route = "LOCALIZE_REMAINING_FEEDBACK_BEFORE_TRAINING"
        pilot_allowed = False
    return {
        "reference_phase2b1_chance_only_k1_tv": float(reference_tv),
        "common_traversal_rng_k1_cross_behavior_tv": common,
        "independent_traversal_rng_k1_cross_behavior_tv": independent,
        "absolute_reduction_vs_phase2b1_chance_reference": abs_improvement,
        "relative_reduction_vs_phase2b1_chance_reference": rel_improvement,
        "feedback_dominant_threshold": feedback_threshold,
        "shared_support_gate_pass": shared_pass,
        "classification": classification,
        "next_route": route,
        "small_training_pilot_precommit_allowed": pilot_allowed,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V1+ Phase2B2 common-chance / feedback decomposition")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--phase2b1-result", type=Path, required=True)
    parser.add_argument("--source-execution-sha", default=SOURCE_EXECUTION_SHA)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if str(args.source_execution_sha) != SOURCE_EXECUTION_SHA:
        raise SystemExit("Phase2B2 source execution SHA drift")
    workers = max(1, min(int(args.workers), MAX_WORKERS, os.cpu_count() or MAX_WORKERS))
    torch.set_num_threads(1)

    repo_root = args.repo_root.resolve()
    solver_path = args.solver.resolve()
    input_root = args.input_root.resolve()
    phase2b1_path = args.phase2b1_result.resolve()
    output = args.out.resolve()
    phase2b1_raw = phase2b1_path.read_bytes()
    phase2b1 = json.loads(phase2b1_raw)
    if phase2b1.get("schema") != PHASE2B1_SCHEMA or phase2b1.get("status") != PHASE2B1_STATUS:
        raise RuntimeError("Phase2B2 requires completed frozen Phase2B1 result")
    decision = dict(phase2b1.get("decision") or {})
    if decision.get("source_classification") != "CHANCE_DOMINANT" or bool(decision.get("small_causal_pilot_precommit_allowed")):
        raise RuntimeError("Phase2B2 requires Phase2B1 CHANCE_DOMINANT / no-generic-K4 route")
    observed_reference = float(decision["k1_source_policy_tv"]["CHANCE_ONLY"])
    if abs(observed_reference - REFERENCE_CHANCE_K1_TV) > 1e-15:
        raise RuntimeError("Phase2B2 Phase2B1 reference-TV drift")

    groups = list(phase2b1.get("collision_groups") or [])
    if len(groups) != 15:
        raise RuntimeError("Phase2B2 requires exactly 15 Phase2B1 collision groups")
    groups.sort(key=lambda row: int(row["scenario_index"]))
    if [int(row["scenario_index"]) for row in groups] != list(range(15)):
        raise RuntimeError("Phase2B2 collision-group scenario coverage drift")

    tasks = []
    for group in groups:
        for arm in ARMS:
            tasks.append({**group, "arm": arm})

    started = time.perf_counter()
    task_rows = []
    print(f"[Phase2B2] running {len(tasks)} tasks with {workers} workers; 960 total root traversals...", flush=True)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(str(repo_root), str(solver_path), str(input_root), str(args.source_execution_sha)),
    ) as pool:
        future_map = {pool.submit(_worker_task, task): (int(task["scenario_index"]), str(task["arm"])) for task in tasks}
        for future in as_completed(future_map):
            scenario_index, arm = future_map[future]
            row = future.result()
            task_rows.append(row)
            print(
                f"[Phase2B2 target] scenario={scenario_index:02d} arm={arm} seconds={row['seconds']:.2f}",
                flush=True,
            )
    task_rows.sort(key=lambda row: (int(row["scenario_index"]), str(row["arm"])))

    pair_rows, pooled = _metrics(task_rows)
    final_decision = decision_from_pooled(pooled)

    task_audit = []
    for row in task_rows:
        task_audit.append({
            "scenario_index": int(row["scenario_index"]),
            "arm": str(row["arm"]),
            "replicates": REPLICATES,
            "nodes_behavior_1342191342": _summary([float(value) for value in row["nodes_a"]]),
            "nodes_behavior_1801739323": _summary([float(value) for value in row["nodes_b"]]),
            "seconds": float(row["seconds"]),
            "targets_a_sha256": hashlib.sha256(json.dumps(row["targets_a"], separators=(",", ":")).encode("utf-8")).hexdigest(),
            "targets_b_sha256": hashlib.sha256(json.dumps(row["targets_b"], separators=(",", ":")).encode("utf-8")).hexdigest(),
        })

    result = {
        "schema": SCHEMA,
        "status": final_decision["classification"],
        "governance_scope": "Post-R7.5.3 V1+ architecture-reset read-only diagnostic; R7.5.3 remains closed.",
        "source_execution_sha": str(args.source_execution_sha),
        "phase2b1_result_sha256": hashlib.sha256(phase2b1_raw).hexdigest(),
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "source_behavior_seeds": [int(value) for value in TRAINING_SEEDS],
        "action_candidate": ACTION_CANDIDATE,
        "exact_opponent_levels": EXACT_OPPONENT_LEVELS,
        "target_iteration": TARGET_ITERATION,
        "arms": list(ARMS),
        "replicates_per_scenario": REPLICATES,
        "k_values": list(K_VALUES),
        "worker_processes": workers,
        "torch_threads_per_worker": 1,
        "pooled": pooled,
        "decision": final_decision,
        "pair_metric_row_count": len(pair_rows),
        "task_audit": task_audit,
        "runtime_seconds_total": float(time.perf_counter() - started),
        "interpretation_guardrails": [
            "Both frozen source behavior ensembles were evaluated on exactly the same stored Phase2B1 deck seeds and exact root observation identities.",
            "COMMON_TRAVERSAL_RNG is a common-random-numbers diagnostic and does not force identical sampled actions when behavior probabilities differ.",
            "INDEPENDENT_TRAVERSAL_RNG is the primary gate so a favorable RNG coupling cannot masquerade as a chance-support remedy.",
            "K aggregation here characterizes convergence of cross-behavior targets on paired common chance support and does not revive the rejected generic K4 training candidate.",
            "No optimizer step, model fit, reservoir insertion, or checkpoint mutation occurred.",
            "A shared-support PASS may authorize only freezing one small causal pilot with independent chance-block validation later; production training remains unauthorized.",
        ],
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, output)
    print(json.dumps({
        "status": result["status"],
        "decision": final_decision,
        "runtime_seconds_total": result["runtime_seconds_total"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
