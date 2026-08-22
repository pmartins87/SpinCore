from __future__ import annotations

"""Read-only Phase2B3 decomposition of same-chance behavior feedback.

Reconstructs the root CFR action-value vector and root behavior policy for both
frozen Phase2A behavior ensembles on the exact stored Phase2B1 deals.  It then
removes root-sigma disagreement or downstream action-value disagreement only in
counterfactual arithmetic.  No model fit, optimizer step, reservoir insertion,
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
from r7_5_arch_reset_v1plus_phase2b2_common_chance_feedback import _traversal_seed

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B3_ROOT_FEEDBACK_V1"
DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL
SOURCE_EXECUTION_SHA = "4bfa55d69029cd69536fa6dbfcadd162719cb887"
PHASE2B1_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B1_TARGET_VARIANCE_V1"
PHASE2B1_SHA256 = "f95751afeb17fcd5844bfcb2971577b92a400750444e5dabe2f4ddb5718ba6ef"
PHASE2B2_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B2_COMMON_CHANCE_FEEDBACK_V1"
PHASE2B2_SHA256 = "49cd1bd98ffe30f21a2b4263c50eb0b5c6d3e616b651a1353f136a670453e281"
REFERENCE_NATIVE_TV = 0.38892191351328625
TARGET_ITERATION = 3
REPLICATES = 16
MAX_WORKERS = 12
ABS_MATERIALITY = 0.05
REL_MATERIALITY = 0.15

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


def _target(values: Sequence[float], sigma: Sequence[float], legal: tuple[int, ...]) -> tuple[float, ...]:
    node_value = float(sum(float(sigma[slot]) * float(values[slot]) for slot in legal))
    out = [0.0] * 10
    for slot in legal:
        out[slot] = float(values[slot]) - node_value
    return tuple(out)


def _mean_policy(a: Sequence[float], b: Sequence[float], legal: tuple[int, ...]) -> tuple[float, ...]:
    out = [0.0] * 10
    total = 0.0
    for slot in legal:
        out[slot] = 0.5 * (float(a[slot]) + float(b[slot]))
        total += out[slot]
    if total <= 0.0:
        for slot in legal:
            out[slot] = 1.0 / len(legal)
    else:
        for slot in legal:
            out[slot] /= total
    return tuple(out)


def _mean_values(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(0.5 * (float(x) + float(y)) for x, y in zip(a, b))


def _target_pair_metrics(left: Sequence[float], right: Sequence[float], legal: tuple[int, ...]) -> dict:
    lp = regret_matching_policy(left, legal)
    rp = regret_matching_policy(right, legal)
    return {
        "target_mean_abs_diff": float(sum(abs(float(left[s]) - float(right[s])) for s in legal) / len(legal)),
        "legal_sign_disagreement_fraction": float(sum((float(left[s]) > 0.0) != (float(right[s]) > 0.0) for s in legal) / len(legal)),
        "regret_matching_policy_tv": _policy_tv(lp, rp),
        "dominant_legal_action_mismatch": int(max(legal, key=lambda s: float(lp[s])) != max(legal, key=lambda s: float(rp[s]))),
    }


def _root_identity(root, action_spec) -> tuple[bytes, int, tuple[int, ...], tuple[int, ...], int]:
    observation = neural_bytes_v3(root)
    actor = int(root.actor)
    payload_v2 = root.neural_bytes_v2()
    if len(payload_v2) != 830 or not payload_v2.startswith(b"SPNNIV2\x00"):
        raise RuntimeError("Phase2B3 requires authoritative SPNNIV2 street metadata")
    street = int(payload_v2[112])
    active_mask = int(action_spec.active_mask(street))
    legal = tuple(int(value) for value in root.universal_legal_actions(active_mask))
    if not legal:
        raise RuntimeError("Phase2B3 root has no legal universal actions")
    return observation, actor, legal, legal_mask(legal), active_mask


def _worker_init(repo_root: str, solver_path: str, input_root: str, source_sha: str) -> None:
    global _WORKER_SOLVER, _WORKER_COLLECTORS
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError("Phase2B3 worker torch-thread contract drift")
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
            raise RuntimeError("Phase2B3 worker source behavior seed mismatch")
        collectors[seed] = UniversalPartialExactCollectorV3(
            action_spec=action_spec,
            policy=behavior,
            terminal_utility=icm_delta_utility(PAYOUT),
            rng=random.Random(0),
            advantage_memory=_Sink(),
            strategy_memory=_Sink(),
        )
    _WORKER_COLLECTORS = collectors


def _root_components(
    collector,
    episode,
    *,
    deck_seed: int,
    traversal_seed: int,
    expected_sha: str,
    expected_actor: int,
    expected_legal: tuple[int, ...],
    expected_mask: tuple[int, ...],
) -> dict:
    if _WORKER_SOLVER is None:
        raise RuntimeError("Phase2B3 worker solver not initialized")
    root = _WORKER_SOLVER.create(episode, int(deck_seed))
    collector.advantage_memory = _Sink()
    collector.strategy_memory = _Sink()
    collector.rng = random.Random(int(traversal_seed))
    try:
        observation, actor, legal, mask, active_mask = _root_identity(root, collector.action_spec)
        if hashlib.sha256(observation).hexdigest() != str(expected_sha):
            raise RuntimeError("Phase2B3 exact root observation hash drift")
        if actor != int(expected_actor) or legal != tuple(expected_legal) or mask != tuple(expected_mask):
            raise RuntimeError("Phase2B3 root actor/legal identity drift")
        sigma = tuple(float(x) for x in collector._p(root, observation, legal))
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
        target = _target(values, sigma, legal)
    finally:
        root.close()
    return {
        "sigma": sigma,
        "values": tuple(values),
        "target": target,
        "nodes": int(nodes),
    }


def _worker_task(task: dict) -> dict:
    if _WORKER_COLLECTORS is None:
        raise RuntimeError("Phase2B3 worker collectors not initialized")
    scenario_index = int(task["scenario_index"])
    episode = action_scenario_cycle(DOMAIN)[scenario_index]
    if scenario_descriptor(episode) != dict(task["scenario"]):
        raise RuntimeError("Phase2B3 scenario descriptor drift")
    expected_sha = str(task["observation_sha256"])
    actor = int(task["actor"])
    legal = tuple(int(value) for value in task["legal"])
    mask = tuple(int(value) for value in task["legal_mask"])
    deck_seeds = [int(value) for value in task["deck_seeds"]]
    if len(deck_seeds) != REPLICATES or len(set(deck_seeds)) != REPLICATES:
        raise RuntimeError("Phase2B3 requires 16 distinct stored deck seeds")

    seed_a, seed_b = map(int, TRAINING_SEEDS)
    rows = []
    nodes_a = []
    nodes_b = []
    started = time.perf_counter()
    for replicate, deck_seed in enumerate(deck_seeds):
        comp_a = _root_components(
            _WORKER_COLLECTORS[seed_a], episode,
            deck_seed=deck_seed,
            traversal_seed=_traversal_seed(scenario_index, replicate, 1),
            expected_sha=expected_sha,
            expected_actor=actor,
            expected_legal=legal,
            expected_mask=mask,
        )
        comp_b = _root_components(
            _WORKER_COLLECTORS[seed_b], episode,
            deck_seed=deck_seed,
            traversal_seed=_traversal_seed(scenario_index, replicate, 2),
            expected_sha=expected_sha,
            expected_actor=actor,
            expected_legal=legal,
            expected_mask=mask,
        )
        sigma_a, sigma_b = comp_a["sigma"], comp_b["sigma"]
        values_a, values_b = comp_a["values"], comp_b["values"]
        native_a, native_b = comp_a["target"], comp_b["target"]
        sigma_bar = _mean_policy(sigma_a, sigma_b, legal)
        values_bar = _mean_values(values_a, values_b)

        common_sigma_a = _target(values_a, sigma_bar, legal)
        common_sigma_b = _target(values_b, sigma_bar, legal)
        common_values_a = _target(values_bar, sigma_a, legal)
        common_values_b = _target(values_bar, sigma_b, legal)

        va_sa = native_a
        va_sb = _target(values_a, sigma_b, legal)
        vb_sa = _target(values_b, sigma_a, legal)
        vb_sb = native_b
        root_step = 0.5 * (
            _policy_tv(regret_matching_policy(va_sa, legal), regret_matching_policy(va_sb, legal))
            + _policy_tv(regret_matching_policy(vb_sa, legal), regret_matching_policy(vb_sb, legal))
        )
        downstream_step = 0.5 * (
            _policy_tv(regret_matching_policy(va_sa, legal), regret_matching_policy(vb_sa, legal))
            + _policy_tv(regret_matching_policy(va_sb, legal), regret_matching_policy(vb_sb, legal))
        )

        row = {
            "scenario_index": scenario_index,
            "replicate": int(replicate),
            "root_sigma_tv": _policy_tv(sigma_a, sigma_b),
            "root_action_value_mean_abs_diff": float(sum(abs(values_a[s] - values_b[s]) for s in legal) / len(legal)),
            "NATIVE": _target_pair_metrics(native_a, native_b, legal),
            "COMMON_ROOT_SIGMA": _target_pair_metrics(common_sigma_a, common_sigma_b, legal),
            "COMMON_ACTION_VALUES": _target_pair_metrics(common_values_a, common_values_b, legal),
            "crossed_root_sigma_step_tv": float(root_step),
            "crossed_downstream_value_step_tv": float(downstream_step),
        }
        rows.append(row)
        nodes_a.append(int(comp_a["nodes"]))
        nodes_b.append(int(comp_b["nodes"]))
    return {
        "scenario_index": scenario_index,
        "rows": rows,
        "nodes_a": nodes_a,
        "nodes_b": nodes_b,
        "seconds": float(time.perf_counter() - started),
    }


def _mode_summary(rows: Sequence[dict], mode: str) -> dict:
    return {
        "pair_count": len(rows),
        "target_mean_abs_diff": _summary([row[mode]["target_mean_abs_diff"] for row in rows]),
        "legal_sign_disagreement_fraction": _summary([row[mode]["legal_sign_disagreement_fraction"] for row in rows]),
        "regret_matching_policy_tv": _summary([row[mode]["regret_matching_policy_tv"] for row in rows]),
        "dominant_legal_action_mismatch_rate": float(sum(int(row[mode]["dominant_legal_action_mismatch"]) for row in rows) / len(rows)),
    }


def _aggregate(rows: Sequence[dict]) -> dict:
    return {
        "root_sigma_tv": _summary([row["root_sigma_tv"] for row in rows]),
        "root_action_value_mean_abs_diff": _summary([row["root_action_value_mean_abs_diff"] for row in rows]),
        "NATIVE": _mode_summary(rows, "NATIVE"),
        "COMMON_ROOT_SIGMA": _mode_summary(rows, "COMMON_ROOT_SIGMA"),
        "COMMON_ACTION_VALUES": _mode_summary(rows, "COMMON_ACTION_VALUES"),
        "crossed_root_sigma_step_tv": _summary([row["crossed_root_sigma_step_tv"] for row in rows]),
        "crossed_downstream_value_step_tv": _summary([row["crossed_downstream_value_step_tv"] for row in rows]),
    }


def _effect(native: float, residual: float) -> dict:
    absolute = float(native - residual)
    relative = float(absolute / native) if native > 0.0 else 0.0
    return {
        "residual_tv": float(residual),
        "absolute_reduction": absolute,
        "relative_reduction": relative,
        "material": bool(absolute >= ABS_MATERIALITY or relative >= REL_MATERIALITY),
    }


def decision_from_pooled(pooled: dict) -> dict:
    native = float(pooled["NATIVE"]["regret_matching_policy_tv"]["mean"])
    common_sigma = float(pooled["COMMON_ROOT_SIGMA"]["regret_matching_policy_tv"]["mean"])
    common_values = float(pooled["COMMON_ACTION_VALUES"]["regret_matching_policy_tv"]["mean"])
    if abs(native - REFERENCE_NATIVE_TV) > 1e-12:
        raise RuntimeError(f"Phase2B3 failed native Phase2B2 reproduction: {native} != {REFERENCE_NATIVE_TV}")
    root_effect = _effect(native, common_sigma)
    downstream_effect = _effect(native, common_values)
    if root_effect["material"] and not downstream_effect["material"]:
        classification = "ROOT_BASELINE_DOMINANT"
        route = "SCREEN_ROOT_BASELINE_POLICY_STABILIZATION_BEFORE_TRAINING"
    elif downstream_effect["material"] and not root_effect["material"]:
        classification = "DOWNSTREAM_CONTINUATION_DOMINANT"
        route = "LOCALIZE_DOWNSTREAM_FEEDBACK_BY_STREET_DEPTH_BEFORE_TRAINING"
    elif root_effect["material"] and downstream_effect["material"]:
        classification = "MIXED_ROOT_AND_DOWNSTREAM_FEEDBACK"
        route = "LOCALIZE_DOWNSTREAM_STREET_DEPTH_AND_RETAIN_ROOT_BASELINE_CONTROL"
    else:
        classification = "NONLINEAR_INTERACTION_OR_UNRESOLVED"
        route = "LOCALIZE_FEEDBACK_BY_SCENARIO_ACTION_INTERACTION_BEFORE_TRAINING"
    return {
        "native_k1_tv_reproduced": native,
        "root_sigma_removal_effect": root_effect,
        "downstream_value_removal_effect": downstream_effect,
        "classification": classification,
        "next_route": route,
        "training_pilot_precommit_allowed": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V1+ Phase2B3 root-baseline/downstream feedback decomposition")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--phase2b1-result", type=Path, required=True)
    parser.add_argument("--phase2b2-result", type=Path, required=True)
    parser.add_argument("--source-execution-sha", default=SOURCE_EXECUTION_SHA)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if str(args.source_execution_sha) != SOURCE_EXECUTION_SHA:
        raise SystemExit("Phase2B3 source execution SHA drift")
    workers = max(1, min(int(args.workers), MAX_WORKERS, os.cpu_count() or MAX_WORKERS))
    torch.set_num_threads(1)

    phase2b1_path = args.phase2b1_result.resolve()
    phase2b2_path = args.phase2b2_result.resolve()
    raw_b1 = phase2b1_path.read_bytes()
    raw_b2 = phase2b2_path.read_bytes()
    if hashlib.sha256(raw_b1).hexdigest() != PHASE2B1_SHA256:
        raise RuntimeError("Phase2B3 Phase2B1 SHA drift")
    if hashlib.sha256(raw_b2).hexdigest() != PHASE2B2_SHA256:
        raise RuntimeError("Phase2B3 Phase2B2 SHA drift")
    b1 = json.loads(raw_b1)
    b2 = json.loads(raw_b2)
    if b1.get("schema") != PHASE2B1_SCHEMA or b1.get("decision", {}).get("source_classification") != "CHANCE_DOMINANT":
        raise RuntimeError("Phase2B3 invalid Phase2B1 route")
    if b2.get("schema") != PHASE2B2_SCHEMA or b2.get("status") != "MIXED_CHANCE_SUPPORT_AND_FEEDBACK":
        raise RuntimeError("Phase2B3 requires Phase2B2 mixed chance-feedback result")
    if b2.get("decision", {}).get("next_route") != "LOCALIZE_REMAINING_FEEDBACK_BEFORE_TRAINING":
        raise RuntimeError("Phase2B3 Phase2B2 route mismatch")

    groups = list(b1.get("collision_groups") or [])
    groups.sort(key=lambda row: int(row["scenario_index"]))
    if len(groups) != 15 or [int(row["scenario_index"]) for row in groups] != list(range(15)):
        raise RuntimeError("Phase2B3 collision-group coverage drift")

    repo_root = args.repo_root.resolve()
    solver_path = args.solver.resolve()
    input_root = args.input_root.resolve()
    started = time.perf_counter()
    task_rows = []
    print(f"[Phase2B3] running 15 scenario tasks with {workers} workers; 480 root-component reconstructions...", flush=True)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(str(repo_root), str(solver_path), str(input_root), str(args.source_execution_sha)),
    ) as pool:
        future_map = {pool.submit(_worker_task, group): int(group["scenario_index"]) for group in groups}
        for future in as_completed(future_map):
            scenario_index = future_map[future]
            result = future.result()
            task_rows.append(result)
            print(f"[Phase2B3 target] scenario={scenario_index:02d} seconds={result['seconds']:.2f}", flush=True)
    task_rows.sort(key=lambda row: int(row["scenario_index"]))

    pair_rows = []
    per_scenario = {}
    task_audit = []
    for task in task_rows:
        rows = list(task["rows"])
        rows.sort(key=lambda row: int(row["replicate"]))
        pair_rows.extend(rows)
        per_scenario[str(task["scenario_index"])] = _aggregate(rows)
        task_audit.append({
            "scenario_index": int(task["scenario_index"]),
            "replicates": len(rows),
            "nodes_behavior_1342191342": _summary(task["nodes_a"]),
            "nodes_behavior_1801739323": _summary(task["nodes_b"]),
            "seconds": float(task["seconds"]),
        })
    pair_rows.sort(key=lambda row: (int(row["scenario_index"]), int(row["replicate"])))
    if len(pair_rows) != 15 * REPLICATES:
        raise RuntimeError("Phase2B3 paired-row count drift")

    pooled = _aggregate(pair_rows)
    decision = decision_from_pooled(pooled)
    result = {
        "schema": SCHEMA,
        "status": decision["classification"],
        "governance_scope": "Post-R7.5.3 V1+ architecture-reset read-only diagnostic; R7.5.3 remains closed.",
        "source_execution_sha": str(args.source_execution_sha),
        "phase2b1_result_sha256": PHASE2B1_SHA256,
        "phase2b2_result_sha256": PHASE2B2_SHA256,
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "source_behavior_seeds": [int(value) for value in TRAINING_SEEDS],
        "action_candidate": ACTION_CANDIDATE,
        "exact_opponent_levels": EXACT_OPPONENT_LEVELS,
        "target_iteration": TARGET_ITERATION,
        "paired_deal_count": len(pair_rows),
        "worker_processes": workers,
        "torch_threads_per_worker": 1,
        "pooled": pooled,
        "per_scenario": per_scenario,
        "decision": decision,
        "task_audit": task_audit,
        "runtime_seconds_total": float(time.perf_counter() - started),
        "interpretation_guardrails": [
            "NATIVE must exactly reproduce the Phase2B2 primary independent-traversal K1 target-policy TV before any counterfactual interpretation is allowed.",
            "COMMON_ROOT_SIGMA changes only the root traverser baseline policy used to center each source behavior's own downstream action values.",
            "COMMON_ACTION_VALUES changes only downstream action values by replacing both source value vectors with their arithmetic mean while retaining native root policies.",
            "Counterfactuals are diagnostic arithmetic only; they do not modify CFR training semantics or authorize a training candidate.",
            "Crossed path-step magnitudes are not assumed additive because regret matching is nonlinear.",
            "No optimizer step, model fit, reservoir insertion, or checkpoint mutation occurred.",
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
    print(json.dumps({
        "status": result["status"],
        "decision": decision,
        "root_sigma_tv": pooled["root_sigma_tv"]["mean"],
        "root_action_value_mad": pooled["root_action_value_mean_abs_diff"]["mean"],
        "crossed_root_sigma_step_tv": pooled["crossed_root_sigma_step_tv"]["mean"],
        "crossed_downstream_value_step_tv": pooled["crossed_downstream_value_step_tv"]["mean"],
        "runtime_seconds_total": result["runtime_seconds_total"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
