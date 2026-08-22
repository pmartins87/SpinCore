from __future__ import annotations

"""Read-only Phase2B4 localization of downstream behavior feedback by street."""

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
from spincore.r7_5_action_cfr import legal_mask
from spincore.r7_5_action_scenarios import action_scenario_cycle, scenario_descriptor
from spincore.r7_5_representation_v3 import H2_FINAL, UniversalPartialExactCollectorV3
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    EXACT_OPPONENT_LEVELS,
    PAYOUT,
    TRAINING_SEEDS,
    validate_phase2_v3_contract,
)
from spincore.solver import SolverLibrary
from spincore.solver_v3 import neural_bytes_v3

from r7_5_arch_reset_v1plus_phase2b1_target_variance import _load_behavior
from r7_5_arch_reset_v1plus_phase2b2_common_chance_feedback import _traversal_seed
from r7_5_arch_reset_v1plus_phase2b3_root_feedback_decomposition import (
    _mean_policy,
    _target,
    _target_pair_metrics,
)

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B4_DOWNSTREAM_STREET_LOCALIZATION_V1"
DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL
SOURCE_EXECUTION_SHA = "4bfa55d69029cd69536fa6dbfcadd162719cb887"
PHASE2B1_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B1_TARGET_VARIANCE_V1"
PHASE2B1_SHA256 = "f95751afeb17fcd5844bfcb2971577b92a400750444e5dabe2f4ddb5718ba6ef"
PHASE2B3_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B3_ROOT_FEEDBACK_V1"
PHASE2B3_SHA256 = "158e450e96027871b5bf44caa5cd0cb9105782e648e91583960b49d3986fe0a7"
REFERENCE_COMMON_ROOT_SIGMA_TV = 0.32770276958712846
TARGET_ITERATION = 3
REPLICATES = 16
MAX_WORKERS = 12
ABS_MATERIALITY = 0.05
REL_MATERIALITY = 0.15
FULL_COMMON_RESIDUAL_MAX = 0.10

ARM_THRESHOLDS = {
    "NATIVE_CONTINUATION": None,
    "COMMON_FROM_RIVER": 3,
    "COMMON_FROM_TURN": 2,
    "COMMON_FROM_FLOP": 1,
    "COMMON_FROM_PREFLOP": 0,
}
ARMS = tuple(ARM_THRESHOLDS)

_WORKER_SOLVER = None
_WORKER_ACTION_SPEC = None
_WORKER_BEHAVIORS = None


class _Sink:
    def add(self, _item) -> None:
        return None


def _summary(values: Sequence[float]) -> dict:
    arr = np.asarray([float(value) for value in values], dtype=np.float64)
    if not arr.size:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50, method="linear")),
        "p95": float(np.quantile(arr, 0.95, method="linear")),
        "max": float(arr.max()),
    }


def _street(state) -> int:
    payload = state.neural_bytes_v2()
    if len(payload) != 830 or not payload.startswith(b"SPNNIV2\x00"):
        raise RuntimeError("Phase2B4 requires authoritative SPNNIV2 street metadata")
    street = int(payload[112])
    if street not in (0, 1, 2, 3):
        raise RuntimeError(f"Phase2B4 invalid street {street}")
    return street


class _HybridPolicy:
    def __init__(self, behavior_a, behavior_b, *, source_side: int, common_from: int | None):
        self.behavior_a = behavior_a
        self.behavior_b = behavior_b
        self.source_side = int(source_side)
        self.common_from = common_from

    def __call__(self, state, observation: bytes, legal: tuple[int, ...]):
        if self.common_from is None or _street(state) < int(self.common_from):
            source = self.behavior_a if self.source_side == 0 else self.behavior_b
            return source(state, observation, legal)
        pa = self.behavior_a(state, observation, legal)
        pb = self.behavior_b(state, observation, legal)
        return _mean_policy(pa, pb, legal)


def _root_identity(root) -> tuple[bytes, int, tuple[int, ...], tuple[int, ...], int]:
    observation = neural_bytes_v3(root)
    actor = int(root.actor)
    street = _street(root)
    active_mask = int(_WORKER_ACTION_SPEC.active_mask(street))
    legal = tuple(int(value) for value in root.universal_legal_actions(active_mask))
    if not legal:
        raise RuntimeError("Phase2B4 root has no legal universal actions")
    return observation, actor, legal, legal_mask(legal), active_mask


def _worker_init(repo_root: str, solver_path: str, input_root: str, source_sha: str) -> None:
    global _WORKER_SOLVER, _WORKER_ACTION_SPEC, _WORKER_BEHAVIORS
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError("Phase2B4 worker torch-thread contract drift")
    root = Path(repo_root)
    contract = validate_phase2_v3_contract(
        root,
        representation=REPRESENTATION,
        domain=DOMAIN,
        training_seed=int(TRAINING_SEEDS[0]),
    )
    _WORKER_ACTION_SPEC = contract["action_spec"]
    _WORKER_SOLVER = SolverLibrary(solver_path)
    behaviors = []
    for seed in map(int, TRAINING_SEEDS):
        checkpoint = Path(input_root) / f"seed_{seed}" / "resume_checkpoint.pt"
        loaded_seed, behavior = _load_behavior(checkpoint, source_sha)
        if loaded_seed != seed:
            raise RuntimeError("Phase2B4 source behavior seed mismatch")
        behaviors.append(behavior)
    _WORKER_BEHAVIORS = tuple(behaviors)


def _verify_root(root, *, expected_sha: str, expected_actor: int, expected_legal: tuple[int, ...], expected_mask: tuple[int, ...]):
    observation, actor, legal, mask, active_mask = _root_identity(root)
    if hashlib.sha256(observation).hexdigest() != str(expected_sha):
        raise RuntimeError("Phase2B4 exact root observation hash drift")
    if actor != int(expected_actor) or legal != tuple(expected_legal) or mask != tuple(expected_mask):
        raise RuntimeError("Phase2B4 root actor/legal identity drift")
    return observation, actor, legal, active_mask


def _root_sigma_bar(episode, *, deck_seed: int, expected_sha: str, expected_actor: int, expected_legal: tuple[int, ...], expected_mask: tuple[int, ...]):
    root = _WORKER_SOLVER.create(episode, int(deck_seed))
    try:
        observation, _actor, legal, _active_mask = _verify_root(
            root,
            expected_sha=expected_sha,
            expected_actor=expected_actor,
            expected_legal=expected_legal,
            expected_mask=expected_mask,
        )
        pa = _WORKER_BEHAVIORS[0](root, observation, legal)
        pb = _WORKER_BEHAVIORS[1](root, observation, legal)
        return _mean_policy(pa, pb, legal)
    finally:
        root.close()


def _root_values(
    episode,
    *,
    deck_seed: int,
    traversal_seed: int,
    expected_sha: str,
    expected_actor: int,
    expected_legal: tuple[int, ...],
    expected_mask: tuple[int, ...],
    source_side: int,
    common_from: int | None,
) -> tuple[tuple[float, ...], int]:
    behavior_a, behavior_b = _WORKER_BEHAVIORS
    policy = _HybridPolicy(behavior_a, behavior_b, source_side=source_side, common_from=common_from)
    collector = UniversalPartialExactCollectorV3(
        action_spec=_WORKER_ACTION_SPEC,
        policy=policy,
        terminal_utility=icm_delta_utility(PAYOUT),
        rng=random.Random(int(traversal_seed)),
        advantage_memory=_Sink(),
        strategy_memory=_Sink(),
    )
    root = _WORKER_SOLVER.create(episode, int(deck_seed))
    try:
        _observation, actor, legal, active_mask = _verify_root(
            root,
            expected_sha=expected_sha,
            expected_actor=expected_actor,
            expected_legal=expected_legal,
            expected_mask=expected_mask,
        )
        values = [0.0] * 10
        nodes = 1
        for action in legal:
            child = root.child_universal(active_mask, action)
            try:
                value, child_nodes, _added = collector._adv_partial(
                    child,
                    actor,
                    TARGET_ITERATION,
                    EXACT_OPPONENT_LEVELS,
                    1.0,
                )
            finally:
                child.close()
            values[action] = float(value)
            nodes += int(child_nodes)
        return tuple(values), int(nodes)
    finally:
        root.close()


def _worker_task(task: dict) -> dict:
    if _WORKER_SOLVER is None or _WORKER_BEHAVIORS is None:
        raise RuntimeError("Phase2B4 worker not initialized")
    scenario_index = int(task["scenario_index"])
    episode = action_scenario_cycle(DOMAIN)[scenario_index]
    if scenario_descriptor(episode) != dict(task["scenario"]):
        raise RuntimeError("Phase2B4 scenario descriptor drift")
    expected_sha = str(task["observation_sha256"])
    actor = int(task["actor"])
    legal = tuple(int(value) for value in task["legal"])
    mask = tuple(int(value) for value in task["legal_mask"])
    deck_seeds = [int(value) for value in task["deck_seeds"]]
    if len(deck_seeds) != REPLICATES or len(set(deck_seeds)) != REPLICATES:
        raise RuntimeError("Phase2B4 requires exactly 16 stored deck seeds")

    rows = []
    node_a = {arm: [] for arm in ARMS}
    node_b = {arm: [] for arm in ARMS}
    started = time.perf_counter()
    for replicate, deck_seed in enumerate(deck_seeds):
        sigma_bar = _root_sigma_bar(
            episode,
            deck_seed=deck_seed,
            expected_sha=expected_sha,
            expected_actor=actor,
            expected_legal=legal,
            expected_mask=mask,
        )
        for arm, threshold in ARM_THRESHOLDS.items():
            values_a, nodes_a = _root_values(
                episode,
                deck_seed=deck_seed,
                traversal_seed=_traversal_seed(scenario_index, replicate, 1),
                expected_sha=expected_sha,
                expected_actor=actor,
                expected_legal=legal,
                expected_mask=mask,
                source_side=0,
                common_from=threshold,
            )
            values_b, nodes_b = _root_values(
                episode,
                deck_seed=deck_seed,
                traversal_seed=_traversal_seed(scenario_index, replicate, 2),
                expected_sha=expected_sha,
                expected_actor=actor,
                expected_legal=legal,
                expected_mask=mask,
                source_side=1,
                common_from=threshold,
            )
            target_a = _target(values_a, sigma_bar, legal)
            target_b = _target(values_b, sigma_bar, legal)
            rows.append({
                "scenario_index": scenario_index,
                "replicate": int(replicate),
                "arm": arm,
                "root_action_value_mean_abs_diff": float(sum(abs(values_a[s] - values_b[s]) for s in legal) / len(legal)),
                **_target_pair_metrics(target_a, target_b, legal),
            })
            node_a[arm].append(int(nodes_a))
            node_b[arm].append(int(nodes_b))
    return {
        "scenario_index": scenario_index,
        "rows": rows,
        "nodes_a": node_a,
        "nodes_b": node_b,
        "seconds": float(time.perf_counter() - started),
    }


def _arm_summary(rows: Sequence[dict], arm: str) -> dict:
    selected = [row for row in rows if row["arm"] == arm]
    return {
        "pair_count": len(selected),
        "root_action_value_mean_abs_diff": _summary([row["root_action_value_mean_abs_diff"] for row in selected]),
        "target_mean_abs_diff": _summary([row["target_mean_abs_diff"] for row in selected]),
        "legal_sign_disagreement_fraction": _summary([row["legal_sign_disagreement_fraction"] for row in selected]),
        "regret_matching_policy_tv": _summary([row["regret_matching_policy_tv"] for row in selected]),
        "dominant_legal_action_mismatch_rate": float(sum(int(row["dominant_legal_action_mismatch"]) for row in selected) / len(selected)),
    }


def _material(native: float, residual: float) -> dict:
    absolute = float(native - residual)
    relative = float(absolute / native) if native > 0 else 0.0
    return {
        "residual_tv": float(residual),
        "absolute_reduction": absolute,
        "relative_reduction": relative,
        "material": bool(absolute >= ABS_MATERIALITY or relative >= REL_MATERIALITY),
    }


def decision_from_pooled(pooled: dict) -> dict:
    tv = {arm: float(pooled[arm]["regret_matching_policy_tv"]["mean"]) for arm in ARMS}
    native = tv["NATIVE_CONTINUATION"]
    if abs(native - REFERENCE_COMMON_ROOT_SIGMA_TV) > 1e-12:
        raise RuntimeError(
            f"Phase2B4 failed common-root-sigma reproduction: {native} != {REFERENCE_COMMON_ROOT_SIGMA_TV}"
        )
    cumulative = {arm: _material(native, tv[arm]) for arm in ARMS if arm != "NATIVE_CONTINUATION"}
    postflop = cumulative["COMMON_FROM_FLOP"]
    preflop_increment_abs = float(tv["COMMON_FROM_FLOP"] - tv["COMMON_FROM_PREFLOP"])
    preflop_increment_rel = float(preflop_increment_abs / native) if native > 0 else 0.0
    preflop_increment_material = bool(preflop_increment_abs >= ABS_MATERIALITY or preflop_increment_rel >= REL_MATERIALITY)

    if postflop["material"] and preflop_increment_material:
        classification = "PREFLOP_AND_POSTFLOP_FEEDBACK_MIXED"
        route = "DESIGN_STREET_AWARE_FEEDBACK_STABILIZATION_DIAGNOSTIC"
    elif postflop["material"]:
        classification = "POSTFLOP_FEEDBACK_DOMINANT"
        route = "DESIGN_POSTFLOP_FEEDBACK_STABILIZATION_DIAGNOSTIC"
    elif preflop_increment_material:
        classification = "PREFLOP_DOWNSTREAM_FEEDBACK_DOMINANT"
        route = "DESIGN_PREFLOP_FEEDBACK_STABILIZATION_DIAGNOSTIC"
    else:
        classification = "DEPTH_LOCALIZATION_WEAK_OR_UNRESOLVED"
        route = "REASSESS_NONLINEAR_OR_REPRESENTATION_FRAGMENTATION"

    high_residual = bool(tv["COMMON_FROM_PREFLOP"] > FULL_COMMON_RESIDUAL_MAX)
    if high_residual:
        route = "DIAGNOSE_RESIDUAL_AFTER_FULL_POLICY_COMMONIZATION"

    sequential = {
        "RIVER": float(tv["NATIVE_CONTINUATION"] - tv["COMMON_FROM_RIVER"]),
        "TURN": float(tv["COMMON_FROM_RIVER"] - tv["COMMON_FROM_TURN"]),
        "FLOP": float(tv["COMMON_FROM_TURN"] - tv["COMMON_FROM_FLOP"]),
        "REMAINING_PREFLOP": float(tv["COMMON_FROM_FLOP"] - tv["COMMON_FROM_PREFLOP"]),
    }
    largest = max(sequential, key=lambda key: sequential[key])
    return {
        "native_common_root_sigma_tv_reproduced": native,
        "arm_tv": tv,
        "cumulative_effects": cumulative,
        "sequential_nested_tv_steps": sequential,
        "largest_positive_sequential_step": largest,
        "preflop_increment_absolute_reduction": preflop_increment_abs,
        "preflop_increment_relative_to_native": preflop_increment_rel,
        "preflop_increment_material": preflop_increment_material,
        "high_residual_after_full_policy_commonization": high_residual,
        "classification": classification,
        "next_route": route,
        "training_pilot_precommit_allowed": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V1+ Phase2B4 downstream street localization")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--phase2b1-result", type=Path, required=True)
    parser.add_argument("--phase2b3-result", type=Path, required=True)
    parser.add_argument("--source-execution-sha", default=SOURCE_EXECUTION_SHA)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if str(args.source_execution_sha) != SOURCE_EXECUTION_SHA:
        raise SystemExit("Phase2B4 source execution SHA drift")
    workers = max(1, min(int(args.workers), MAX_WORKERS, os.cpu_count() or MAX_WORKERS))
    torch.set_num_threads(1)

    phase2b1_raw = args.phase2b1_result.resolve().read_bytes()
    phase2b1 = json.loads(phase2b1_raw)
    if hashlib.sha256(phase2b1_raw).hexdigest() != PHASE2B1_SHA256 or phase2b1.get("schema") != PHASE2B1_SCHEMA:
        raise RuntimeError("Phase2B4 Phase2B1 identity drift")
    phase2b3_raw = args.phase2b3_result.resolve().read_bytes()
    phase2b3 = json.loads(phase2b3_raw)
    if hashlib.sha256(phase2b3_raw).hexdigest() != PHASE2B3_SHA256 or phase2b3.get("schema") != PHASE2B3_SCHEMA:
        raise RuntimeError("Phase2B4 Phase2B3 identity drift")
    if phase2b3.get("status") != "MIXED_ROOT_AND_DOWNSTREAM_FEEDBACK" or phase2b3.get("decision", {}).get("next_route") != "LOCALIZE_DOWNSTREAM_STREET_DEPTH_AND_RETAIN_ROOT_BASELINE_CONTROL":
        raise RuntimeError("Phase2B4 requires the frozen Phase2B3 routing decision")

    groups = sorted(list(phase2b1.get("collision_groups") or []), key=lambda row: int(row["scenario_index"]))
    if len(groups) != 15 or [int(row["scenario_index"]) for row in groups] != list(range(15)):
        raise RuntimeError("Phase2B4 requires all 15 frozen collision groups")

    started = time.perf_counter()
    task_rows = []
    print(f"[Phase2B4] running 15 scenario tasks with {workers} workers; 2400 root action-value reconstructions...", flush=True)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(str(args.repo_root.resolve()), str(args.solver.resolve()), str(args.input_root.resolve()), str(args.source_execution_sha)),
    ) as pool:
        future_map = {pool.submit(_worker_task, group): int(group["scenario_index"]) for group in groups}
        for future in as_completed(future_map):
            scenario_index = future_map[future]
            row = future.result()
            task_rows.append(row)
            print(f"[Phase2B4 target] scenario={scenario_index:02d} seconds={row['seconds']:.2f}", flush=True)
    task_rows.sort(key=lambda row: int(row["scenario_index"]))

    pair_rows = []
    task_audit = []
    for task in task_rows:
        pair_rows.extend(task["rows"])
        task_audit.append({
            "scenario_index": int(task["scenario_index"]),
            "seconds": float(task["seconds"]),
            "nodes_behavior_1342191342": {arm: _summary(task["nodes_a"][arm]) for arm in ARMS},
            "nodes_behavior_1801739323": {arm: _summary(task["nodes_b"][arm]) for arm in ARMS},
        })
    pooled = {arm: _arm_summary(pair_rows, arm) for arm in ARMS}
    per_scenario = {}
    for scenario_index in range(15):
        local = [row for row in pair_rows if int(row["scenario_index"]) == scenario_index]
        per_scenario[str(scenario_index)] = {arm: _arm_summary(local, arm) for arm in ARMS}
    decision = decision_from_pooled(pooled)

    result = {
        "schema": SCHEMA,
        "status": decision["classification"],
        "governance_scope": "Post-R7.5.3 V1+ architecture-reset read-only diagnostic; R7.5.3 remains closed.",
        "source_execution_sha": str(args.source_execution_sha),
        "phase2b1_result_sha256": hashlib.sha256(phase2b1_raw).hexdigest(),
        "phase2b3_result_sha256": hashlib.sha256(phase2b3_raw).hexdigest(),
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "source_behavior_seeds": [int(value) for value in TRAINING_SEEDS],
        "action_candidate": ACTION_CANDIDATE,
        "exact_opponent_levels": EXACT_OPPONENT_LEVELS,
        "target_iteration": TARGET_ITERATION,
        "paired_deal_count": 240,
        "arms": list(ARMS),
        "arm_thresholds": ARM_THRESHOLDS,
        "worker_processes": workers,
        "torch_threads_per_worker": 1,
        "pooled": pooled,
        "per_scenario": per_scenario,
        "decision": decision,
        "task_audit": task_audit,
        "runtime_seconds_total": float(time.perf_counter() - started),
        "interpretation_guardrails": [
            "All arms use the same per-deal common root sigma, controlling out the Phase2B3 root-baseline effect.",
            "NATIVE_CONTINUATION must exactly reproduce the Phase2B3 COMMON_ROOT_SIGMA pooled TV before interpretation.",
            "Each commonization arm replaces both source policies with the same pointwise mean policy only from the frozen street threshold onward.",
            "Sequential nested steps are localization diagnostics and are not assumed additive because traversal and regret matching are nonlinear.",
            "No optimizer step, model fit, reservoir insertion, checkpoint mutation, or architecture selection occurred.",
        ],
        "training_pilot_precommit_allowed": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, output)
    print(json.dumps({"status": result["status"], "decision": decision, "runtime_seconds_total": result["runtime_seconds_total"]}, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
