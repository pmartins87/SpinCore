from __future__ import annotations

"""Read-only Phase2B5 preflop feedback stabilization screen.

Uses the frozen Phase2A behavior ensembles and the exact Phase2B1 stored deals.
The root baseline is commonized and all postflop continuation is commonized in
all arms.  Only the preflop continuation behavior changes.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
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

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B5_PREFLOP_FEEDBACK_STABILIZATION_V1"
DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL
ACTION_CANDIDATE_NAME = "PF0_CONTROL_33_75_AI"
SOURCE_EXECUTION_SHA = "4bfa55d69029cd69536fa6dbfcadd162719cb887"
PHASE2B1_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B1_TARGET_VARIANCE_V1"
PHASE2B1_SHA256 = "f95751afeb17fcd5844bfcb2971577b92a400750444e5dabe2f4ddb5718ba6ef"
PHASE2B4_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B4_DOWNSTREAM_STREET_LOCALIZATION_V1"
PHASE2B4_SHA256 = "6b639b1608a0572c0ae2f6641c038786fa30cb8858bdda62da3fd5e30f49f0aa"
REFERENCE_CONTROL_TV = 0.32010786853721923
REFERENCE_ORACLE_COMMON_PREFLOP_TV = 0.060271017892879135
TARGET_ITERATION = 3
REPLICATES = 16
MAX_WORKERS = 12

CONTROL_ARM = "PREFLOP_NATIVE_POSTFLOP_COMMON"
DEPTH_ARMS = {f"DEPTH_COMMON_GE_{depth}": depth for depth in range(1, 7)}
FLOOR_ARMS = {
    "UNIFORM_FLOOR_010": 0.10,
    "UNIFORM_FLOOR_025": 0.25,
    "UNIFORM_FLOOR_050": 0.50,
    "UNIFORM_FLOOR_075": 0.75,
    "UNIFORM_FLOOR_100": 1.00,
}
ARMS = (CONTROL_ARM, *DEPTH_ARMS.keys(), *FLOOR_ARMS.keys())

PILOT_ABS_REDUCTION = 0.08
PILOT_REL_REDUCTION = 0.25
PILOT_RESIDUAL_MAX = 0.24
PILOT_SCENARIO_IMPROVE_MIN = 12
PILOT_SCENARIO_MAX_DEGRADE = 0.05
PILOT_DOMINANT_MISMATCH_MAX_INCREASE = 0.02

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
        raise RuntimeError("Phase2B5 requires authoritative SPNNIV2 street metadata")
    street = int(payload[112])
    if street not in (0, 1, 2, 3):
        raise RuntimeError(f"Phase2B5 invalid street {street}")
    return street


def _nonforced_preflop_count(observation: bytes) -> int:
    if len(observation) < 120 or not observation.startswith(b"SPNNIV3\x00"):
        raise RuntimeError("Phase2B5 requires authoritative SPNNIV3 bytes")
    history_count = int.from_bytes(observation[116:120], "little", signed=False)
    expected = 120 + 20 * history_count
    if len(observation) != expected:
        raise RuntimeError(
            f"Phase2B5 SPNNIV3 length drift: {len(observation)} != {expected}"
        )
    count = 0
    for index in range(history_count):
        offset = 120 + 20 * index
        street = int(observation[offset + 1])
        forced = int(observation[offset + 3])
        if street == 0 and forced == 0:
            count += 1
    return count


def _mix_uniform(policy: Sequence[float], legal: tuple[int, ...], floor: float) -> tuple[float, ...]:
    floor = float(floor)
    if floor < 0.0 or floor > 1.0:
        raise ValueError("uniform floor must be in [0,1]")
    out = [0.0] * 10
    uniform = 1.0 / len(legal)
    total = 0.0
    for slot in legal:
        value = (1.0 - floor) * float(policy[slot]) + floor * uniform
        if value < 0.0 or not np.isfinite(value):
            raise RuntimeError("Phase2B5 invalid mixed policy")
        out[slot] = value
        total += value
    if total <= 0.0:
        raise RuntimeError("Phase2B5 mixed policy has zero legal mass")
    for slot in legal:
        out[slot] /= total
    return tuple(out)


class _PreflopPolicy:
    def __init__(
        self,
        behavior_a,
        behavior_b,
        *,
        source_side: int,
        arm: str,
        root_nonforced_preflop: int,
    ):
        self.behavior_a = behavior_a
        self.behavior_b = behavior_b
        self.source_side = int(source_side)
        self.arm = str(arm)
        self.root_nonforced_preflop = int(root_nonforced_preflop)
        if self.arm not in ARMS:
            raise ValueError(f"unknown Phase2B5 arm {self.arm}")

    def _source(self, state, observation: bytes, legal: tuple[int, ...]):
        source = self.behavior_a if self.source_side == 0 else self.behavior_b
        return source(state, observation, legal)

    def _common(self, state, observation: bytes, legal: tuple[int, ...]):
        pa = self.behavior_a(state, observation, legal)
        pb = self.behavior_b(state, observation, legal)
        return _mean_policy(pa, pb, legal)

    def __call__(self, state, observation: bytes, legal: tuple[int, ...]):
        street = _street(state)
        if street >= 1:
            return self._common(state, observation, legal)

        current = _nonforced_preflop_count(observation)
        delta = int(current - self.root_nonforced_preflop)
        if delta < 1:
            raise RuntimeError(
                f"Phase2B5 preflop continuation depth must be >=1, got {delta}"
            )

        if self.arm == CONTROL_ARM:
            return self._source(state, observation, legal)
        if self.arm in DEPTH_ARMS:
            if delta >= int(DEPTH_ARMS[self.arm]):
                return self._common(state, observation, legal)
            return self._source(state, observation, legal)
        if self.arm in FLOOR_ARMS:
            native = self._source(state, observation, legal)
            return _mix_uniform(native, legal, FLOOR_ARMS[self.arm])
        raise RuntimeError("unreachable Phase2B5 arm")


def _root_identity(root) -> tuple[bytes, int, tuple[int, ...], tuple[int, ...], int]:
    observation = neural_bytes_v3(root)
    actor = int(root.actor)
    street = _street(root)
    active_mask = int(_WORKER_ACTION_SPEC.active_mask(street))
    legal = tuple(int(value) for value in root.universal_legal_actions(active_mask))
    if not legal:
        raise RuntimeError("Phase2B5 root has no legal universal actions")
    return observation, actor, legal, legal_mask(legal), active_mask


def _worker_init(repo_root: str, solver_path: str, input_root: str, source_sha: str) -> None:
    global _WORKER_SOLVER, _WORKER_ACTION_SPEC, _WORKER_BEHAVIORS
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError("Phase2B5 worker torch-thread contract drift")
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
            raise RuntimeError("Phase2B5 source behavior seed mismatch")
        behaviors.append(behavior)
    _WORKER_BEHAVIORS = tuple(behaviors)


def _verify_root(
    root,
    *,
    expected_sha: str,
    expected_actor: int,
    expected_legal: tuple[int, ...],
    expected_mask: tuple[int, ...],
):
    observation, actor, legal, mask, active_mask = _root_identity(root)
    if hashlib.sha256(observation).hexdigest() != str(expected_sha):
        raise RuntimeError("Phase2B5 exact root observation hash drift")
    if actor != int(expected_actor) or legal != tuple(expected_legal) or mask != tuple(expected_mask):
        raise RuntimeError("Phase2B5 root actor/legal identity drift")
    return observation, actor, legal, active_mask


def _root_sigma_bar(
    episode,
    *,
    deck_seed: int,
    expected_sha: str,
    expected_actor: int,
    expected_legal: tuple[int, ...],
    expected_mask: tuple[int, ...],
):
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
        return _mean_policy(pa, pb, legal), _nonforced_preflop_count(observation)
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
    arm: str,
    root_nonforced_preflop: int,
) -> tuple[tuple[float, ...], int]:
    behavior_a, behavior_b = _WORKER_BEHAVIORS
    policy = _PreflopPolicy(
        behavior_a,
        behavior_b,
        source_side=source_side,
        arm=arm,
        root_nonforced_preflop=root_nonforced_preflop,
    )
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
        raise RuntimeError("Phase2B5 worker not initialized")
    scenario_index = int(task["scenario_index"])
    episode = action_scenario_cycle(DOMAIN)[scenario_index]
    if scenario_descriptor(episode) != dict(task["scenario"]):
        raise RuntimeError("Phase2B5 scenario descriptor drift")
    expected_sha = str(task["observation_sha256"])
    actor = int(task["actor"])
    legal = tuple(int(value) for value in task["legal"])
    mask = tuple(int(value) for value in task["legal_mask"])
    deck_seeds = [int(value) for value in task["deck_seeds"]]
    if len(deck_seeds) != REPLICATES or len(set(deck_seeds)) != REPLICATES:
        raise RuntimeError("Phase2B5 requires exactly 16 stored deck seeds")

    rows = []
    node_a = {arm: [] for arm in ARMS}
    node_b = {arm: [] for arm in ARMS}
    root_history_counts = []
    started = time.perf_counter()
    for replicate, deck_seed in enumerate(deck_seeds):
        sigma_bar, root_count = _root_sigma_bar(
            episode,
            deck_seed=deck_seed,
            expected_sha=expected_sha,
            expected_actor=actor,
            expected_legal=legal,
            expected_mask=mask,
        )
        root_history_counts.append(int(root_count))
        for arm in ARMS:
            values_a, nodes_a = _root_values(
                episode,
                deck_seed=deck_seed,
                traversal_seed=_traversal_seed(scenario_index, replicate, 1),
                expected_sha=expected_sha,
                expected_actor=actor,
                expected_legal=legal,
                expected_mask=mask,
                source_side=0,
                arm=arm,
                root_nonforced_preflop=root_count,
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
                arm=arm,
                root_nonforced_preflop=root_count,
            )
            target_a = _target(values_a, sigma_bar, legal)
            target_b = _target(values_b, sigma_bar, legal)
            rows.append(
                {
                    "scenario_index": scenario_index,
                    "replicate": int(replicate),
                    "arm": arm,
                    "root_action_value_mean_abs_diff": float(
                        sum(abs(values_a[s] - values_b[s]) for s in legal) / len(legal)
                    ),
                    **_target_pair_metrics(target_a, target_b, legal),
                }
            )
            node_a[arm].append(int(nodes_a))
            node_b[arm].append(int(nodes_b))
    return {
        "scenario_index": scenario_index,
        "rows": rows,
        "nodes_a": node_a,
        "nodes_b": node_b,
        "root_nonforced_preflop_counts": root_history_counts,
        "seconds": float(time.perf_counter() - started),
    }


def _arm_summary(rows: Sequence[dict], arm: str) -> dict:
    selected = [row for row in rows if row["arm"] == arm]
    return {
        "pair_count": len(selected),
        "root_action_value_mean_abs_diff": _summary(
            [row["root_action_value_mean_abs_diff"] for row in selected]
        ),
        "target_mean_abs_diff": _summary([row["target_mean_abs_diff"] for row in selected]),
        "legal_sign_disagreement_fraction": _summary(
            [row["legal_sign_disagreement_fraction"] for row in selected]
        ),
        "regret_matching_policy_tv": _summary(
            [row["regret_matching_policy_tv"] for row in selected]
        ),
        "dominant_legal_action_mismatch_rate": float(
            sum(int(row["dominant_legal_action_mismatch"]) for row in selected) / len(selected)
        ),
    }


def _aggregate(rows: Sequence[dict]) -> dict:
    return {arm: _arm_summary(rows, arm) for arm in ARMS}


def _candidate_screen(arm: str, pooled: dict, per_scenario: dict) -> dict:
    control_tv = float(pooled[CONTROL_ARM]["regret_matching_policy_tv"]["mean"])
    residual = float(pooled[arm]["regret_matching_policy_tv"]["mean"])
    absolute = float(control_tv - residual)
    relative = float(absolute / control_tv) if control_tv > 0.0 else 0.0
    control_mismatch = float(pooled[CONTROL_ARM]["dominant_legal_action_mismatch_rate"])
    mismatch = float(pooled[arm]["dominant_legal_action_mismatch_rate"])
    deltas = []
    improved = 0
    for scenario_index in sorted(per_scenario, key=int):
        control_s = float(
            per_scenario[scenario_index][CONTROL_ARM]["regret_matching_policy_tv"]["mean"]
        )
        arm_s = float(per_scenario[scenario_index][arm]["regret_matching_policy_tv"]["mean"])
        delta = float(arm_s - control_s)
        deltas.append(delta)
        if arm_s < control_s:
            improved += 1
    max_degrade = max(deltas) if deltas else 0.0
    mismatch_increase = float(mismatch - control_mismatch)
    passes = bool(
        absolute >= PILOT_ABS_REDUCTION
        and relative >= PILOT_REL_REDUCTION
        and residual <= PILOT_RESIDUAL_MAX
        and improved >= PILOT_SCENARIO_IMPROVE_MIN
        and max_degrade <= PILOT_SCENARIO_MAX_DEGRADE
        and mismatch_increase <= PILOT_DOMINANT_MISMATCH_MAX_INCREASE
    )
    return {
        "floor": float(FLOOR_ARMS[arm]),
        "residual_tv": residual,
        "absolute_reduction": absolute,
        "relative_reduction": relative,
        "scenarios_improved": int(improved),
        "scenario_count": int(len(per_scenario)),
        "max_scenario_degradation": float(max_degrade),
        "dominant_action_mismatch_rate": mismatch,
        "dominant_action_mismatch_increase": mismatch_increase,
        "numerical_screen_pass": passes,
    }


def decision_from_results(pooled: dict, per_scenario: dict) -> dict:
    control = float(pooled[CONTROL_ARM]["regret_matching_policy_tv"]["mean"])
    oracle = float(pooled["DEPTH_COMMON_GE_1"]["regret_matching_policy_tv"]["mean"])
    if abs(control - REFERENCE_CONTROL_TV) > 1e-12:
        raise RuntimeError(
            f"Phase2B5 control reproduction failed: {control} != {REFERENCE_CONTROL_TV}"
        )
    if abs(oracle - REFERENCE_ORACLE_COMMON_PREFLOP_TV) > 1e-12:
        raise RuntimeError(
            "Phase2B5 oracle-common-preflop reproduction failed: "
            f"{oracle} != {REFERENCE_ORACLE_COMMON_PREFLOP_TV}"
        )

    depth_tv = {
        arm: float(pooled[arm]["regret_matching_policy_tv"]["mean"])
        for arm in DEPTH_ARMS
    }
    sequential = {}
    for depth in range(1, 6):
        sequential[f"DELTA{depth}"] = float(
            depth_tv[f"DEPTH_COMMON_GE_{depth + 1}"]
            - depth_tv[f"DEPTH_COMMON_GE_{depth}"]
        )
    sequential["DEEPER_THAN_6"] = float(control - depth_tv["DEPTH_COMMON_GE_6"])
    largest_depth = max(sequential, key=lambda key: sequential[key])

    screens = {arm: _candidate_screen(arm, pooled, per_scenario) for arm in FLOOR_ARMS}
    mild = [
        arm
        for arm in ("UNIFORM_FLOOR_010", "UNIFORM_FLOOR_025")
        if screens[arm]["numerical_screen_pass"]
    ]
    selected = mild[0] if mild else None
    if selected is not None:
        classification = "MILD_PREFLOP_DAMPING_CANDIDATE"
        route = "PRECOMMIT_SMALL_PREFLOP_DAMPING_TRAINING_PILOT"
        pilot = True
    elif screens["UNIFORM_FLOOR_050"]["numerical_screen_pass"]:
        classification = "HEAVY_DAMPING_REQUIRED_NO_PILOT"
        route = "DESIGN_BETTER_PREFLOP_ANCHOR_OR_LAGGED_TARGET_DIAGNOSTIC"
        pilot = False
    elif (
        screens["UNIFORM_FLOOR_075"]["numerical_screen_pass"]
        or screens["UNIFORM_FLOOR_100"]["numerical_screen_pass"]
    ):
        classification = "STRONG_ANCHOR_REQUIRED_NO_PILOT"
        route = "DESIGN_SEED_INDEPENDENT_PREFLOP_ANCHOR_DIAGNOSTIC"
        pilot = False
    else:
        classification = "UNIFORM_DAMPING_INSUFFICIENT"
        route = "DESIGN_LAGGED_TARGET_OR_ALGORITHM_LEVEL_FEEDBACK_CONTROL"
        pilot = False

    return {
        "classification": classification,
        "control_tv_reproduced": control,
        "oracle_common_preflop_tv_reproduced": oracle,
        "depth_arm_tv": depth_tv,
        "depth_sequential_tv_increments": sequential,
        "largest_positive_depth_increment": largest_depth,
        "uniform_floor_screens": screens,
        "selected_mild_candidate": selected,
        "next_route": route,
        "small_training_pilot_precommit_allowed": pilot,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--phase2b1-result", required=True)
    parser.add_argument("--phase2b4-result", required=True)
    parser.add_argument("--source-execution-sha", required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if str(args.source_execution_sha) != SOURCE_EXECUTION_SHA:
        raise RuntimeError("Phase2B5 source execution SHA drift")
    repo_root = Path(args.repo_root)
    validate_phase2_v3_contract(
        repo_root,
        representation=REPRESENTATION,
        domain=DOMAIN,
        training_seed=int(TRAINING_SEEDS[0]),
    )

    phase2b1_raw = Path(args.phase2b1_result).read_bytes()
    phase2b1 = json.loads(phase2b1_raw)
    if hashlib.sha256(phase2b1_raw).hexdigest() != PHASE2B1_SHA256:
        raise RuntimeError("Phase2B5 Phase2B1 SHA drift")
    if phase2b1.get("schema") != PHASE2B1_SCHEMA:
        raise RuntimeError("Phase2B5 Phase2B1 schema drift")

    phase2b4_raw = Path(args.phase2b4_result).read_bytes()
    phase2b4 = json.loads(phase2b4_raw)
    if hashlib.sha256(phase2b4_raw).hexdigest() != PHASE2B4_SHA256:
        raise RuntimeError("Phase2B5 Phase2B4 SHA drift")
    if phase2b4.get("schema") != PHASE2B4_SCHEMA:
        raise RuntimeError("Phase2B5 Phase2B4 schema drift")
    if phase2b4.get("status") != "PREFLOP_DOWNSTREAM_FEEDBACK_DOMINANT":
        raise RuntimeError("Phase2B5 requires frozen Phase2B4 preflop-dominant result")
    if phase2b4.get("decision", {}).get("next_route") != "DESIGN_PREFLOP_FEEDBACK_STABILIZATION_DIAGNOSTIC":
        raise RuntimeError("Phase2B5 Phase2B4 route drift")

    groups = sorted(
        list(phase2b1.get("collision_groups") or []),
        key=lambda row: int(row["scenario_index"]),
    )
    if len(groups) != 15 or [int(row["scenario_index"]) for row in groups] != list(range(15)):
        raise RuntimeError("Phase2B5 requires all 15 frozen collision groups")

    workers = max(1, min(int(args.workers), MAX_WORKERS))
    started = time.perf_counter()
    task_rows = []
    print(
        f"[Phase2B5] running 15 scenario tasks with {workers} workers; "
        f"{15 * REPLICATES * 2 * len(ARMS)} root action-value reconstructions...",
        flush=True,
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(
            str(repo_root),
            str(args.solver),
            str(args.input_root),
            SOURCE_EXECUTION_SHA,
        ),
    ) as pool:
        futures = {pool.submit(_worker_task, dict(group)): int(group["scenario_index"]) for group in groups}
        for future in as_completed(futures):
            result = future.result()
            task_rows.append(result)
            print(
                f"[Phase2B5 target] scenario={int(result['scenario_index']):02d} "
                f"seconds={float(result['seconds']):.2f}",
                flush=True,
            )

    task_rows.sort(key=lambda row: int(row["scenario_index"]))
    all_rows = [row for task in task_rows for row in task["rows"]]
    per_scenario = {}
    for scenario_index in range(15):
        selected = [row for row in all_rows if int(row["scenario_index"]) == scenario_index]
        per_scenario[str(scenario_index)] = _aggregate(selected)
    pooled = _aggregate(all_rows)
    decision = decision_from_results(pooled, per_scenario)

    task_audit = []
    for task in task_rows:
        task_audit.append(
            {
                "scenario_index": int(task["scenario_index"]),
                "seconds": float(task["seconds"]),
                "root_nonforced_preflop_counts": sorted(
                    set(int(value) for value in task["root_nonforced_preflop_counts"])
                ),
                "nodes_behavior_1342191342": {
                    arm: _summary(task["nodes_a"][arm]) for arm in ARMS
                },
                "nodes_behavior_1801739323": {
                    arm: _summary(task["nodes_b"][arm]) for arm in ARMS
                },
            }
        )

    status = str(decision["classification"])
    result = {
        "schema": SCHEMA,
        "status": status,
        "governance_scope": "Post-R7.5.3 V1+ architecture-reset read-only diagnostic; R7.5.3 remains closed.",
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "action_candidate": ACTION_CANDIDATE_NAME,
        "exact_opponent_levels": int(EXACT_OPPONENT_LEVELS),
        "target_iteration": TARGET_ITERATION,
        "source_execution_sha": SOURCE_EXECUTION_SHA,
        "source_behavior_seeds": [int(value) for value in TRAINING_SEEDS],
        "phase2b1_result_sha256": PHASE2B1_SHA256,
        "phase2b4_result_sha256": PHASE2B4_SHA256,
        "paired_deal_count": 15 * REPLICATES,
        "arms": list(ARMS),
        "depth_cutoffs": DEPTH_ARMS,
        "uniform_floors": FLOOR_ARMS,
        "pooled": pooled,
        "per_scenario": per_scenario,
        "decision": decision,
        "task_audit": task_audit,
        "worker_processes": workers,
        "torch_threads_per_worker": 1,
        "runtime_seconds_total": float(time.perf_counter() - started),
        "small_training_pilot_precommit_allowed": bool(
            decision["small_training_pilot_precommit_allowed"]
        ),
        "production_training_authorized": False,
        "ready_for_tables": False,
        "interpretation_guardrails": [
            "Root sigma is commonized in every arm, retaining the Phase2B3 baseline control.",
            "Postflop continuation is commonized in every arm so only preflop continuation feedback varies.",
            "DEPTH_COMMON_GE_1 and the control must exactly reproduce the two frozen Phase2B4 preflop boundary values before interpretation.",
            "Oracle depth arms cannot authorize training because independent learners cannot average policies across training seeds.",
            "Uniform-floor arms are seed-independent screens applied after the frozen source uncertainty damping.",
            "No optimizer step, model fit, reservoir insertion, checkpoint mutation, or architecture selection occurred.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "runtime_seconds_total": result["runtime_seconds_total"], "status": status}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
