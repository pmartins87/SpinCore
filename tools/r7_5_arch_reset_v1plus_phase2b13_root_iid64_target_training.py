from __future__ import annotations

"""Phase2B13: equal-compute root IID64 target training pilot.

The ordinary Deep-CFR trajectory is retained, including the Phase2B6 25%
preflop-continuation behavior floor. For each logical root only the initial
root actor's one Advantage sample is replaced in-place. The equal-compute
control uses IID sample 0 after computing 64 conditional-IID targets; the
candidate uses the arithmetic mean of the same 64 raw targets.

No downstream Advantage sample is averaged. Heldout inference is the learned
AveragePolicy with no behavior floor.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Sequence

import torch

import r7_5_3d_v1plus_phase2a_strategy_capacity as phase2a
import r7_5_arch_reset_v1plus_phase2b1_target_variance as b1
import r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot as b6
import r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition as b10
import r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator as b11
import spincore.r7_5_representation_v3_stage as stage

from spincore.r7_5_action_cfr import ActionAdvantageSample
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_representation_v3 import H2_FINAL
from spincore.r7_5_representation_v3_checkpoint import (
    RepresentationV3Progress,
    load_representation_v3_checkpoint,
    save_representation_v3_checkpoint,
)
from spincore.r7_5_representation_v3_phase2_eval import (
    cross_seed_policy_stability,
    equal_group_stratified_bootstrap_mean_ci,
)
from spincore.r7_5_representation_v3_referee_artifacts import load_heldout_v3_artifact
from spincore.r7_5_representation_v3_stage import frozen_config, new_phase2_v3_runtime
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    ADVANTAGE_NRMSE_MAX,
    BATCH_SIZE,
    CROSS_SEED_MEAN_TV_MAX,
    CROSS_SEED_P95_TV_MAX,
    EVALUATION_SEEDS,
    EXACT_OPPONENT_LEVELS,
    ITERATIONS,
    LEARNING_RATE,
    MODEL_FINGERPRINTS,
    POLICY_TV_MAX,
    RESERVOIR_CAPACITY,
    ROOTS_PER_ITERATION,
    TORCH_THREADS,
    TRAINING_SEEDS,
    deck_seed,
    validate_phase2_v3_contract,
)
from spincore.solver import SolverLibrary
from spincore_nn.models_v3_final import make_h2_final_v3


SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B13_ROOT_IID64_TARGET_TRAINING_V1"
SEED_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B13_SEED_V1"
CHECKPOINT_EXTRA_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B13_RESUME_V1"
POLICY_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B13_POLICY_V1"

DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL
FLOOR = 0.25
K = 64
ARMS = ("IID1_OF_64_EQUAL_COMPUTE_CONTROL", "IID64_MEAN_CANDIDATE")
CONTROL_ARM, CANDIDATE_ARM = ARMS

CHUNKS_PER_ITERATION = 2
ROOTS_PER_CHUNK = ROOTS_PER_ITERATION
ROOTS_PER_ITERATION_EFFECTIVE = CHUNKS_PER_ITERATION * ROOTS_PER_CHUNK
TOTAL_ROOTS = ITERATIONS * ROOTS_PER_ITERATION_EFFECTIVE
POLICY_COUNT = 1024

PHASE2B6_RESULT_SHA256 = "33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a"
PHASE2B12_RESULT_SHA256 = "dbccadae5805381d0188bef41fb62a72b25b42e03e5564ca88f05d9666e6e182"
PHASE2B12_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B12_IID_CHANCE_EXPECTATION_CONVERGENCE_V1"

COMMON_POLICY_INIT_SEED = b6.COMMON_POLICY_INIT_SEED
COMMON_BATCH_SEED = b6.COMMON_BATCH_SEED
BOOTSTRAP_REPLICATES = 2000
CAUSAL_ABS_MIN = 0.02
CAUSAL_REL_MIN = 0.10
COMMON_P95_MAX_DEGRADE = 0.02
NATIVE_MEAN_MAX_DEGRADE = 0.01
MASK64 = (1 << 64) - 1

CHANCE_PRIVATE_NAMESPACE = 0x2B13010100000001
CHANCE_PUBLIC_NAMESPACE = 0x2B13020200000001
TRAVERSAL_NAMESPACE = 0x2B13A7710A5E0001


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


def _atomic_torch(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _mix64(*parts: int) -> int:
    x = 0x9E3779B97F4A7C15
    for raw in parts:
        y = int(raw) & MASK64
        x ^= (y + 0x9E3779B97F4A7C15 + ((x << 6) & MASK64) + (x >> 2)) & MASK64
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & MASK64
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & MASK64
        x ^= x >> 31
    return x & MASK64


def _chance_seeds(training_seed: int, global_root: int, iteration: int, sample_index: int) -> tuple[int, int]:
    key = (int(training_seed), int(global_root), int(iteration), int(sample_index))
    return (
        _mix64(CHANCE_PRIVATE_NAMESPACE, *key),
        _mix64(CHANCE_PUBLIC_NAMESPACE, *key),
    )


def _traversal_seed(training_seed: int, global_root: int, iteration: int) -> int:
    return _mix64(TRAVERSAL_NAMESPACE, int(training_seed), int(global_root), int(iteration))


def _mean_targets(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if len(rows) != K:
        raise ValueError(f"Phase2B13 requires exactly {K} root targets")
    if any(len(row) != 10 for row in rows):
        raise ValueError("Phase2B13 target width drift")
    return tuple(float(sum(float(row[i]) for row in rows) / K) for i in range(10))


class RootReplacingAdvantageMemory:
    """Replace exactly one initial-root sample while preserving reservoir add order."""

    def __init__(
        self,
        delegate,
        *,
        observation: bytes,
        iteration: int,
        replacement_target: Sequence[float],
        expected_legal_mask: Sequence[int],
    ):
        self.delegate = delegate
        self.observation = bytes(observation)
        self.iteration = int(iteration)
        self.replacement_target = tuple(float(x) for x in replacement_target)
        self.expected_legal_mask = tuple(int(x) for x in expected_legal_mask)
        self.replaced = 0
        self.original_target = None
        self.original_weight = None

    def add(self, sample) -> None:
        if bytes(sample.observation) == self.observation and int(sample.iteration) == self.iteration:
            self.replaced += 1
            if self.replaced != 1:
                raise RuntimeError("Phase2B13 attempted to replace multiple initial-root Advantage samples")
            legal = tuple(int(x) for x in sample.legal)
            if legal != self.expected_legal_mask:
                raise RuntimeError("Phase2B13 root replacement legal-mask drift")
            if len(self.replacement_target) != 10:
                raise RuntimeError("Phase2B13 replacement target width drift")
            expected_weight = float(self.iteration)
            if abs(float(sample.weight) - expected_weight) > 1e-12:
                raise RuntimeError(
                    f"Phase2B13 root sample weight drift: {sample.weight!r} != {expected_weight!r}"
                )
            self.original_target = tuple(float(x) for x in sample.target)
            self.original_weight = float(sample.weight)
            self.delegate.add(
                ActionAdvantageSample(
                    observation=bytes(sample.observation),
                    legal=legal,
                    target=self.replacement_target,
                    weight=float(sample.weight),
                    iteration=int(sample.iteration),
                )
            )
            return
        self.delegate.add(sample)


def _worker_init(repo_root: str, solver_path: str, training_seed: int, behavior_states: list[dict]) -> None:
    b10._worker_init(repo_root, solver_path, int(training_seed), behavior_states)


def _root_target_task(task: dict) -> dict:
    if b10._WORKER_SOLVER is None or b10._WORKER_ACTION_SPEC is None:
        raise RuntimeError("Phase2B13 target worker not initialized")
    scenarios = action_scenario_cycle(DOMAIN)
    scenario_index = int(task["scenario_index"])
    global_root = int(task["global_root"])
    iteration = int(task["iteration"])
    training_seed = int(task["training_seed"])
    anchor_deck_seed = int(task["anchor_deck_seed"])
    episode = scenarios[scenario_index]

    anchor = b10._WORKER_SOLVER.create(episode, anchor_deck_seed)
    try:
        observation, actor, legal, legal_mask_row = b1._root_identity(
            anchor, b10._WORKER_ACTION_SPEC
        )
        snapshot = anchor.deal_snapshot()
    finally:
        anchor.close()

    if snapshot.visible_board_count != 0:
        raise RuntimeError("Phase2B13 auxiliary target anchor is not an initial preflop root")

    expected = {
        "observation_sha256": hashlib.sha256(observation).hexdigest(),
        "actor": int(actor),
        "legal": tuple(int(x) for x in legal),
        "legal_mask": tuple(int(x) for x in legal_mask_row),
    }
    traversal_seed = _traversal_seed(training_seed, global_root, iteration)

    targets = []
    nodes = 0
    started = time.perf_counter()
    for sample_index in range(K):
        private_seed, public_seed = _chance_seeds(
            training_seed, global_root, iteration, sample_index
        )
        deal = b11._deal_from_factors(snapshot, actor, private_seed, public_seed)
        target, node_count = b10._one_target(episode, deal, traversal_seed, expected)
        targets.append(tuple(float(x) for x in target))
        nodes += int(node_count)

    return {
        "training_seed": training_seed,
        "scenario_index": scenario_index,
        "global_root": global_root,
        "iteration": iteration,
        "anchor_deck_seed": anchor_deck_seed,
        "actor": int(actor),
        "root_observation": bytes(observation),
        "root_observation_sha256": expected["observation_sha256"],
        "legal_mask": list(expected["legal_mask"]),
        "first_target": list(targets[0]),
        "mean_target": list(_mean_targets(targets)),
        "aux_nodes": int(nodes),
        "aux_traversals": K,
        "seconds": float(time.perf_counter() - started),
    }


def _behavior_states(behavior) -> list[dict]:
    return [model.state_dict() for model in behavior.models]


def _root_target_rows(
    *,
    repo_root: Path,
    solver_path: Path,
    training_seed: int,
    behavior,
    start_global_root: int,
    target_iteration: int,
    chance_workers: int,
) -> list[dict]:
    scenarios = action_scenario_cycle(DOMAIN)
    tasks = []
    for offset in range(ROOTS_PER_CHUNK):
        global_root = int(start_global_root) + offset
        scenario_index = global_root % len(scenarios)
        tasks.append(
            {
                "training_seed": int(training_seed),
                "scenario_index": int(scenario_index),
                "global_root": int(global_root),
                "iteration": int(target_iteration),
                "anchor_deck_seed": int(
                    deck_seed(int(training_seed), int(global_root), int(target_iteration))
                ),
            }
        )
    states = _behavior_states(behavior)
    rows = []
    with ProcessPoolExecutor(
        max_workers=int(chance_workers),
        initializer=_worker_init,
        initargs=(str(repo_root), str(solver_path), int(training_seed), states),
    ) as pool:
        future_map = {pool.submit(_root_target_task, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            row = future.result()
            rows.append(row)
            print(
                f"[Phase2B13 aux] seed={training_seed} i{target_iteration} "
                f"root={task['global_root']} seconds={row['seconds']:.2f}",
                flush=True,
            )
    rows.sort(key=lambda row: int(row["global_root"]))
    expected_roots = list(range(int(start_global_root), int(start_global_root) + ROOTS_PER_CHUNK))
    if [int(row["global_root"]) for row in rows] != expected_roots:
        raise RuntimeError("Phase2B13 auxiliary target row ordering/coverage drift")
    return rows


def _stage_coords(stage_index: int) -> tuple[int, int]:
    total = ITERATIONS * CHUNKS_PER_ITERATION
    if not 1 <= int(stage_index) <= total:
        raise ValueError("Phase2B13 stage index outside frozen range")
    zero = int(stage_index) - 1
    return zero // CHUNKS_PER_ITERATION + 1, zero % CHUNKS_PER_ITERATION + 1


def _stage_path(seed_root: Path, stage_index: int) -> Path:
    iteration, chunk = _stage_coords(stage_index)
    return seed_root / "stages" / f"i{iteration}c{chunk}.json"


def _save_resume(
    path: Path,
    *,
    bundle,
    behavior,
    floor_policy,
    state: dict,
    config,
    execution_sha: str,
    arm: str,
    stage_index: int,
    last_stage_report: dict,
) -> None:
    extra = {
        "schema": CHECKPOINT_EXTRA_SCHEMA,
        "arm": str(arm),
        "k": K,
        "chunks_per_iteration": CHUNKS_PER_ITERATION,
        "stage_config": config.to_dict(),
        "stage_state": dict(state),
        "stage_index": int(stage_index),
        "behavior_model_states": _behavior_states(behavior),
        "behavior_stats": behavior.stats(),
        "floor_policy_stats": floor_policy.stats(),
        "last_stage_report": dict(last_stage_report),
    }
    save_representation_v3_checkpoint(
        path,
        bundle,
        RepresentationV3Progress(
            iteration=int(state["completed_iteration"]),
            global_root=int(state["global_root"]),
            advantage_optimizer_step=int(bundle.counters["adv_optimizer_steps"]),
            policy_optimizer_step=int(bundle.counters["policy_optimizer_steps"]),
            phase="phase2b13_resume",
        ),
        domain=DOMAIN,
        action_candidate=ACTION_CANDIDATE,
        execution_sha=str(execution_sha),
        architecture_fingerprint_sha256=MODEL_FINGERPRINTS[REPRESENTATION],
        extra=extra,
    )


def _load_resume(
    path: Path,
    *,
    repo_root: Path,
    solver,
    training_seed: int,
    config,
    execution_sha: str,
    arm: str,
):
    bundle, progress, spec, extra = load_representation_v3_checkpoint(
        path,
        repo_root=repo_root,
        expected_domain=DOMAIN,
        expected_representation=REPRESENTATION,
        expected_seed=int(training_seed),
        expected_action_candidate=ACTION_CANDIDATE,
        expected_execution_sha=str(execution_sha),
        expected_architecture_fingerprint_sha256=MODEL_FINGERPRINTS[REPRESENTATION],
        device="cpu",
    )
    if progress.phase != "phase2b13_resume":
        raise RuntimeError("Phase2B13 resume checkpoint phase mismatch")
    if extra.get("schema") != CHECKPOINT_EXTRA_SCHEMA:
        raise RuntimeError("Phase2B13 resume checkpoint schema mismatch")
    if extra.get("arm") != str(arm) or int(extra.get("k", -1)) != K:
        raise RuntimeError("Phase2B13 resume arm/K mismatch")
    if int(extra.get("chunks_per_iteration", -1)) != CHUNKS_PER_ITERATION:
        raise RuntimeError("Phase2B13 resume chunk contract drift")
    if dict(extra.get("stage_config") or {}) != config.to_dict():
        raise RuntimeError("Phase2B13 resume stage config drift")
    state = dict(extra.get("stage_state") or {})
    if int(progress.iteration) != int(state.get("completed_iteration", -1)):
        raise RuntimeError("Phase2B13 resume iteration mismatch")
    if int(progress.global_root) != int(state.get("global_root", -1)):
        raise RuntimeError("Phase2B13 resume global-root mismatch")
    behavior = b6._make_behavior_from_states(
        list(extra.get("behavior_model_states") or []), config=config
    )
    behavior.restore_stats(dict(extra.get("behavior_stats") or {}))
    session = stage._make_session(solver, bundle, spec, behavior)
    floor_policy = b6.PreflopContinuationFloorPolicy(behavior, floor=FLOOR)
    floor_policy.restore_stats(dict(extra.get("floor_policy_stats") or {}))
    session.collector.policy = floor_policy
    return (
        bundle,
        session,
        behavior,
        floor_policy,
        state,
        int(extra.get("stage_index", -1)),
        dict(extra.get("last_stage_report") or {}),
    )


def _validate_stage_prefix(seed_root: Path, completed_stages: int, last_report: dict) -> None:
    for index in range(1, int(completed_stages) + 1):
        path = _stage_path(seed_root, index)
        if not path.is_file():
            if index == int(completed_stages) and last_report:
                _atomic_json(last_report, path)
            else:
                raise RuntimeError(f"Phase2B13 completed stage report missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("stage_index", -1)) != index:
            raise RuntimeError("Phase2B13 stage-report identity mismatch")


def _collect_chunk(
    *,
    repo_root: Path,
    solver_path: Path,
    session,
    bundle,
    behavior,
    floor_policy,
    state: dict,
    arm: str,
    target_iteration: int,
    chance_workers: int,
) -> dict:
    scenarios = action_scenario_cycle(DOMAIN)
    scenario_counts = list(state["scenario_counts"])
    global_root = int(state["global_root"])
    start_global_root = global_root

    aux_rows = _root_target_rows(
        repo_root=repo_root,
        solver_path=solver_path,
        training_seed=int(state["training_seed"]),
        behavior=behavior,
        start_global_root=start_global_root,
        target_iteration=int(target_iteration),
        chance_workers=int(chance_workers),
    )
    aux_index = {int(row["global_root"]): row for row in aux_rows}

    session.collector.reset_telemetry()
    roots_before = int(bundle.counters["roots"])
    nodes_before = int(bundle.counters["nodes"])
    adv_before = int(bundle.adv_mem.seen)
    pol_before = int(bundle.pol_mem.seen)
    floor_before = floor_policy.stats()
    replaced = 0
    started = time.perf_counter()

    for _ in range(ROOTS_PER_CHUNK):
        scenario_index = global_root % len(scenarios)
        row = aux_index[global_root]
        if int(row["scenario_index"]) != scenario_index:
            raise RuntimeError("Phase2B13 scenario scheduling drift")
        scenario_counts[scenario_index] += 1

        replacement = (
            row["first_target"] if arm == CONTROL_ARM else row["mean_target"]
        )
        proxy = RootReplacingAdvantageMemory(
            bundle.adv_mem,
            observation=bytes(row["root_observation"]),
            iteration=int(target_iteration),
            replacement_target=replacement,
            expected_legal_mask=row["legal_mask"],
        )
        original_memory = session.collector.advantage_memory
        session.collector.advantage_memory = proxy
        try:
            session.collect_root(
                scenarios[scenario_index],
                iteration=int(target_iteration),
                exact_opponent_levels=EXACT_OPPONENT_LEVELS,
                deck_seed=int(row["anchor_deck_seed"]),
            )
        finally:
            session.collector.advantage_memory = original_memory
        if proxy.replaced != 1:
            raise RuntimeError(
                f"Phase2B13 expected one root replacement at global_root={global_root}, got {proxy.replaced}"
            )
        replaced += 1
        global_root += 1

    seconds = time.perf_counter() - started
    state["global_root"] = global_root
    state["scenario_counts"] = scenario_counts

    aux_nodes = sum(int(row["aux_nodes"]) for row in aux_rows)
    aux_traversals = sum(int(row["aux_traversals"]) for row in aux_rows)
    if aux_traversals != ROOTS_PER_CHUNK * K:
        raise RuntimeError("Phase2B13 auxiliary traversal count drift")

    report = {
        "arm": str(arm),
        "roots": int(bundle.counters["roots"]) - roots_before,
        "nodes": int(bundle.counters["nodes"]) - nodes_before,
        "advantage_seen": int(bundle.adv_mem.seen) - adv_before,
        "strategy_seen": int(bundle.pol_mem.seen) - pol_before,
        "tree_collection_seconds": float(seconds),
        "branch_geometry": session.collector.telemetry_snapshot(),
        "floor_policy_delta": b6._stats_delta(floor_policy.stats(), floor_before),
        "root_replacements": int(replaced),
        "aux_root_target_traversals": int(aux_traversals),
        "aux_root_target_nodes": int(aux_nodes),
        "aux_root_target_seconds_sum": float(sum(float(row["seconds"]) for row in aux_rows)),
        "aux_target_observation_sha256": [
            str(row["root_observation_sha256"]) for row in aux_rows
        ],
    }
    if report["roots"] != ROOTS_PER_CHUNK:
        raise RuntimeError("Phase2B13 chunk root-count drift")
    if report["root_replacements"] != ROOTS_PER_CHUNK:
        raise RuntimeError("Phase2B13 chunk replacement-count drift")
    return report


def _aggregate_chance(chunks: Sequence[dict]) -> dict:
    return {
        "logical_roots": sum(int(row["roots"]) for row in chunks),
        "root_replacements": sum(int(row["root_replacements"]) for row in chunks),
        "aux_root_target_traversals": sum(
            int(row["aux_root_target_traversals"]) for row in chunks
        ),
        "aux_root_target_nodes": sum(int(row["aux_root_target_nodes"]) for row in chunks),
        "aux_root_target_seconds_sum": float(
            sum(float(row["aux_root_target_seconds_sum"]) for row in chunks)
        ),
    }


def _run_arm_seed_trajectory(
    *,
    repo_root: Path,
    solver_path: Path,
    output_root: Path,
    execution_sha: str,
    training_seed: int,
    arm: str,
    chance_workers: int,
):
    validate_phase2_v3_contract(
        repo_root,
        representation=REPRESENTATION,
        domain=DOMAIN,
        training_seed=int(training_seed),
    )
    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("Phase2B13 Torch-thread contract drift")

    solver = SolverLibrary(solver_path)
    if not solver.explicit_deal_available:
        raise RuntimeError("Phase2B13 requires explicit-deal diagnostic solver extension")
    base_config = frozen_config()
    fit_only = replace(base_config, roots_per_iteration=0)
    seed_root = output_root / str(arm) / f"seed_{int(training_seed)}"
    seed_root.mkdir(parents=True, exist_ok=True)
    resume = seed_root / "resume_checkpoint.pt"

    if resume.is_file():
        (
            bundle,
            session,
            behavior,
            floor_policy,
            state,
            completed_stages,
            last_report,
        ) = _load_resume(
            resume,
            repo_root=repo_root,
            solver=solver,
            training_seed=int(training_seed),
            config=base_config,
            execution_sha=str(execution_sha),
            arm=str(arm),
        )
        _validate_stage_prefix(seed_root, completed_stages, last_report)
        print(
            f"[Phase2B13 resume] arm={arm} seed={training_seed} "
            f"completed_stages={completed_stages}/{ITERATIONS * CHUNKS_PER_ITERATION}",
            flush=True,
        )
    else:
        bundle, session, behavior, _spec, state = new_phase2_v3_runtime(
            repo_root,
            solver=solver,
            representation=REPRESENTATION,
            domain=DOMAIN,
            training_seed=int(training_seed),
            config=base_config,
        )
        floor_policy = b6.PreflopContinuationFloorPolicy(behavior, floor=FLOOR)
        session.collector.policy = floor_policy
        state["phase2b13"] = {
            "schema": CHECKPOINT_EXTRA_SCHEMA,
            "arm": str(arm),
            "k": K,
            "chunks_per_iteration": CHUNKS_PER_ITERATION,
            "roots_per_iteration_effective": ROOTS_PER_ITERATION_EFFECTIVE,
            "heldout_floor_applied": False,
        }
        completed_stages = 0

    total_stages = ITERATIONS * CHUNKS_PER_ITERATION
    for stage_index in range(completed_stages + 1, total_stages + 1):
        iteration, chunk = _stage_coords(stage_index)
        if chunk == 1:
            if int(state["completed_iteration"]) != iteration - 1:
                raise RuntimeError("Phase2B13 iteration-start identity drift")
            state["phase2b13_pending_iteration"] = {
                "iteration": int(iteration),
                "roots_before": int(bundle.counters["roots"]),
                "nodes_before": int(bundle.counters["nodes"]),
                "advantage_seen_before": int(bundle.adv_mem.seen),
                "strategy_seen_before": int(bundle.pol_mem.seen),
                "chunks": [],
            }

        pending = dict(state.get("phase2b13_pending_iteration") or {})
        if int(pending.get("iteration", -1)) != iteration:
            raise RuntimeError("Phase2B13 missing pending iteration")
        chunks = list(pending.get("chunks") or [])
        if len(chunks) != chunk - 1:
            raise RuntimeError("Phase2B13 pending chunk history drift")

        print(
            f"[Phase2B13 train] arm={arm} seed={training_seed} i{iteration}c{chunk}",
            flush=True,
        )
        chunk_report = _collect_chunk(
            repo_root=repo_root,
            solver_path=solver_path,
            session=session,
            bundle=bundle,
            behavior=behavior,
            floor_policy=floor_policy,
            state=state,
            arm=str(arm),
            target_iteration=int(iteration),
            chance_workers=int(chance_workers),
        )
        chunks.append(chunk_report)
        pending["chunks"] = chunks
        state["phase2b13_pending_iteration"] = pending

        iteration_report = None
        if chunk == CHUNKS_PER_ITERATION:
            iteration_report = b6._fit_only_iteration(
                bundle=bundle,
                session=session,
                behavior=behavior,
                state=state,
                config=fit_only,
                target_iteration=int(iteration),
            )
            roots_added = int(bundle.counters["roots"]) - int(pending["roots_before"])
            nodes_added = int(bundle.counters["nodes"]) - int(pending["nodes_before"])
            adv_added = int(bundle.adv_mem.seen) - int(pending["advantage_seen_before"])
            pol_added = int(bundle.pol_mem.seen) - int(pending["strategy_seen_before"])
            if roots_added != ROOTS_PER_ITERATION_EFFECTIVE:
                raise RuntimeError("Phase2B13 iteration logical-root total drift")
            patched = dict(iteration_report)
            patched.update(
                {
                    "roots_added": int(roots_added),
                    "nodes_added": int(nodes_added),
                    "advantage_seen_added": int(adv_added),
                    "strategy_seen_added": int(pol_added),
                    "branch_geometry": b6._aggregate_geometry(chunks),
                    "floor_policy_iteration": b6._aggregate_floor(chunks),
                    "root_iid64_iteration": _aggregate_chance(chunks),
                    "chance_chunks": chunks,
                }
            )
            state["iteration_reports"][-1] = patched
            state.pop("phase2b13_pending_iteration", None)
            iteration_report = patched

        stage_report = {
            "schema": CHECKPOINT_EXTRA_SCHEMA,
            "stage_index": int(stage_index),
            "iteration": int(iteration),
            "root_chunk": int(chunk),
            "training_seed": int(training_seed),
            "arm": str(arm),
            "k": K,
            "roots_total": int(bundle.counters["roots"]),
            "chunk_report": chunk_report,
            "iteration_completed": bool(chunk == CHUNKS_PER_ITERATION),
            "iteration_report": iteration_report,
            "execution_sha": str(execution_sha),
        }
        _save_resume(
            resume,
            bundle=bundle,
            behavior=behavior,
            floor_policy=floor_policy,
            state=state,
            config=base_config,
            execution_sha=str(execution_sha),
            arm=str(arm),
            stage_index=int(stage_index),
            last_stage_report=stage_report,
        )
        _atomic_json(stage_report, _stage_path(seed_root, stage_index))
        print(
            f"[Phase2B13 stage complete] arm={arm} seed={training_seed} "
            f"i{iteration}c{chunk} replacements={chunk_report['root_replacements']} "
            f"aux={chunk_report['aux_root_target_traversals']}",
            flush=True,
        )

    if int(bundle.counters["roots"]) != TOTAL_ROOTS:
        raise RuntimeError("Phase2B13 final logical-root count drift")
    if int(state["completed_iteration"]) != ITERATIONS:
        raise RuntimeError("Phase2B13 final iteration count drift")
    return bundle, state, floor_policy


def _policy_paths(seed_root: Path, mode: str) -> tuple[Path, Path]:
    root = seed_root / "policies"
    return root / f"{mode}.pt", root / f"{mode}.json"


def _fit_policies(
    *, seed_root: Path, arm: str, training_seed: int, execution_sha: str, bundle
) -> dict:
    policy_root = seed_root / "policies"
    policy_root.mkdir(parents=True, exist_ok=True)
    native_state = bundle.batch_rng.getstate()
    audit_seed = int(training_seed) ^ 0x71A5BEEF
    rows = {}

    for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
        artifact, meta = _policy_paths(seed_root, mode)
        if artifact.is_file() and meta.is_file():
            saved = json.loads(meta.read_text(encoding="utf-8"))
            if (
                saved.get("schema") == POLICY_SCHEMA
                and saved.get("status") == "POLICY_FIT_COMPLETE"
                and saved.get("arm") == str(arm)
                and int(saved.get("training_seed", -1)) == int(training_seed)
                and saved.get("learner_mode") == mode
                and saved.get("execution_sha") == str(execution_sha)
                and saved.get("artifact_sha256") == _sha256(artifact)
            ):
                rows[mode] = saved
                print(
                    f"[Phase2B13 policy resume] arm={arm} seed={training_seed} {mode}",
                    flush=True,
                )
                continue

        if mode == "COMMON_LEARNER":
            init_seed = COMMON_POLICY_INIT_SEED
            rng = random.Random(COMMON_BATCH_SEED)
        else:
            init_seed = (int(training_seed) ^ 0x5DEECE66D) & 0x7FFFFFFF
            rng = random.Random()
            rng.setstate(native_state)

        print(
            f"[Phase2B13 policy fit] arm={arm} seed={training_seed} {mode}",
            flush=True,
        )
        model, fit = phase2a._fit_policy(
            bundle.pol_mem,
            init_seed=init_seed,
            rng=rng,
            audit_seed=audit_seed,
        )
        payload = {
            "schema": POLICY_SCHEMA,
            "status": "POLICY_FIT_COMPLETE",
            "execution_sha": str(execution_sha),
            "representation": REPRESENTATION,
            "domain": DOMAIN,
            "arm": str(arm),
            "training_seed": int(training_seed),
            "learner_mode": mode,
            "capacity": RESERVOIR_CAPACITY,
            "authoritative_policy_audit_seed": int(audit_seed),
            "floor_training": FLOOR,
            "floor_inference": 0.0,
            "k": K,
            "model_state": model.state_dict(),
            "fit": fit,
        }
        _atomic_torch(payload, artifact)
        saved = {
            "schema": POLICY_SCHEMA,
            "status": "POLICY_FIT_COMPLETE",
            "execution_sha": str(execution_sha),
            "arm": str(arm),
            "training_seed": int(training_seed),
            "learner_mode": mode,
            "capacity": RESERVOIR_CAPACITY,
            "authoritative_policy_audit_seed": int(audit_seed),
            "floor_training": FLOOR,
            "floor_inference": 0.0,
            "k": K,
            "artifact": str(artifact),
            "artifact_sha256": _sha256(artifact),
            "fit": fit,
        }
        _atomic_json(saved, meta)
        rows[mode] = saved
    return rows


def _run_single(args, arm: str, training_seed: int) -> int:
    output_root = Path(args.output_root).resolve()
    seed_root = output_root / str(arm) / f"seed_{int(training_seed)}"
    result_path = seed_root / "seed_result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "SEED_COMPLETE"
            and existing.get("execution_sha") == str(args.execution_sha)
            and existing.get("arm") == str(arm)
        ):
            print(
                f"[Phase2B13 seed resume] arm={arm} seed={training_seed} already complete",
                flush=True,
            )
            return 0

    bundle, state, floor_policy = _run_arm_seed_trajectory(
        repo_root=Path(args.repo_root).resolve(),
        solver_path=Path(args.solver).resolve(),
        output_root=output_root,
        execution_sha=str(args.execution_sha),
        training_seed=int(training_seed),
        arm=str(arm),
        chance_workers=int(args.chance_workers),
    )
    policy_rows = _fit_policies(
        seed_root=seed_root,
        arm=str(arm),
        training_seed=int(training_seed),
        execution_sha=str(args.execution_sha),
        bundle=bundle,
    )

    advantage_rows = []
    for row in list(state.get("iteration_reports") or []):
        nrmse = float(row.get("ensemble_weighted_nrmse", math.inf))
        chance = dict(row.get("root_iid64_iteration") or {})
        advantage_rows.append(
            {
                "iteration": int(row.get("iteration", -1)),
                "ensemble_weighted_nrmse": nrmse,
                "gate_max": ADVANTAGE_NRMSE_MAX,
                "gate_pass": bool(
                    nrmse <= ADVANTAGE_NRMSE_MAX
                    and bool(row.get("ensemble_advantage_gate_pass"))
                ),
                "logical_roots": int(chance.get("logical_roots", -1)),
                "root_replacements": int(chance.get("root_replacements", -1)),
                "aux_root_target_traversals": int(
                    chance.get("aux_root_target_traversals", -1)
                ),
            }
        )

    result = {
        "schema": SEED_SCHEMA,
        "status": "SEED_COMPLETE",
        "execution_sha": str(args.execution_sha),
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "arm": str(arm),
        "training_seed": int(training_seed),
        "roots": int(bundle.counters["roots"]),
        "iterations": int(state["completed_iteration"]),
        "floor_training": FLOOR,
        "floor_inference": 0.0,
        "k": K,
        "advantage_gates": advantage_rows,
        "all_advantage_gates_pass": bool(
            len(advantage_rows) == ITERATIONS
            and all(bool(row["gate_pass"]) for row in advantage_rows)
        ),
        "advantage_memory": {
            "capacity": int(bundle.adv_mem.capacity),
            "seen": int(bundle.adv_mem.seen),
            "retained": len(bundle.adv_mem.items),
        },
        "strategy_memory": {
            "capacity": int(bundle.pol_mem.capacity),
            "seen": int(bundle.pol_mem.seen),
            "retained": len(bundle.pol_mem.items),
        },
        "floor_policy_stats": floor_policy.stats(),
        "policy_fits": policy_rows,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    _atomic_json(result, result_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "arm": str(arm),
                "training_seed": int(training_seed),
                "roots": result["roots"],
                "advantage_pass": result["all_advantage_gates_pass"],
                "strategy_seen": result["strategy_memory"]["seen"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _load_policy(path: Path, *, arm: str, training_seed: int, mode: str, execution_sha: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != POLICY_SCHEMA or payload.get("status") != "POLICY_FIT_COMPLETE":
        raise RuntimeError("Phase2B13 policy schema/status mismatch")
    expected = (
        payload.get("arm"),
        int(payload.get("training_seed", -1)),
        payload.get("learner_mode"),
        payload.get("execution_sha"),
        int(payload.get("k", -1)),
    )
    if expected != (str(arm), int(training_seed), mode, str(execution_sha), K):
        raise RuntimeError("Phase2B13 policy identity mismatch")
    _cfg, model = make_h2_final_v3(device="cpu", seed=0)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def _validate_prerequisites(b6_result: Path, b12_result: Path) -> dict:
    if _sha256(b6_result) != PHASE2B6_RESULT_SHA256:
        raise RuntimeError("Phase2B13 Phase2B6 result SHA drift")
    if _sha256(b12_result) != PHASE2B12_RESULT_SHA256:
        raise RuntimeError("Phase2B13 Phase2B12 result SHA drift")
    j6 = json.loads(b6_result.read_text(encoding="utf-8"))
    j12 = json.loads(b12_result.read_text(encoding="utf-8"))
    if j6.get("status") != "PREFLOP_DAMPING_CAUSAL_EFFECT_SUPPORTED_BUT_STILL_UNSTABLE":
        raise RuntimeError("Phase2B13 requires exact causal Phase2B6 result")
    if j12.get("schema") != PHASE2B12_SCHEMA or j12.get("status") != "IID_CHANCE_EXPECTATION_CONVERGES_MATERIALLY":
        raise RuntimeError("Phase2B13 requires exact Phase2B12 convergence result")
    d12 = dict(j12.get("decision") or {})
    if not bool(d12.get("screen_pass")) or not bool(
        d12.get("small_causal_training_pilot_precommit_allowed")
    ):
        raise RuntimeError("Phase2B13 Phase2B12 route does not permit precommit pilot")
    return {
        "phase2b6_result_sha256": PHASE2B6_RESULT_SHA256,
        "phase2b12_result_sha256": PHASE2B12_RESULT_SHA256,
    }


def _find_heldout(root: Path, evaluation_seed: int) -> Path:
    return b6._find_heldout(root, evaluation_seed)


def _evaluate(args) -> dict:
    output_root = Path(args.output_root).resolve()
    heldout_root = Path(args.heldout_root).resolve()
    prerequisites = _validate_prerequisites(
        Path(args.phase2b6_result).resolve(), Path(args.phase2b12_result).resolve()
    )
    torch.set_num_threads(TORCH_THREADS)

    seed_results = {}
    for arm in ARMS:
        seed_results[arm] = {}
        for seed in map(int, TRAINING_SEEDS):
            path = output_root / arm / f"seed_{seed}" / "seed_result.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("status") != "SEED_COMPLETE"
                or payload.get("execution_sha") != str(args.execution_sha)
                or payload.get("arm") != arm
                or int(payload.get("roots", -1)) != TOTAL_ROOTS
                or int(payload.get("iterations", -1)) != ITERATIONS
            ):
                raise RuntimeError(f"Phase2B13 seed result invalid: {arm}/{seed}")
            seed_results[arm][seed] = payload

    descriptors = {}
    heldout_identity = []
    for evaluation_seed in map(int, EVALUATION_SEEDS):
        heldout = _find_heldout(heldout_root, evaluation_seed)
        rows = load_heldout_v3_artifact(
            heldout,
            expected_domain=DOMAIN,
            expected_evaluation_seed=evaluation_seed,
            expected_count=2048,
        )[:POLICY_COUNT]
        if len(rows) != POLICY_COUNT:
            raise RuntimeError("Phase2B13 heldout policy-count drift")
        descriptors[evaluation_seed] = rows
        heldout_identity.append(
            {
                "evaluation_seed": int(evaluation_seed),
                "path": str(heldout),
                "sha256": _sha256(heldout),
            }
        )

    local_advantage = []
    local_policy = []
    all_local_valid = True
    models = {}
    for arm in ARMS:
        models[arm] = {}
        for seed in map(int, TRAINING_SEEDS):
            sr = seed_results[arm][seed]
            for row in sr["advantage_gates"]:
                local_advantage.append(
                    {"arm": arm, "training_seed": seed, **dict(row)}
                )
            all_local_valid = all_local_valid and bool(sr["all_advantage_gates_pass"])
            if any(
                int(row.get("logical_roots", -1)) != ROOTS_PER_ITERATION_EFFECTIVE
                or int(row.get("root_replacements", -1)) != ROOTS_PER_ITERATION_EFFECTIVE
                or int(row.get("aux_root_target_traversals", -1))
                != ROOTS_PER_ITERATION_EFFECTIVE * K
                for row in sr["advantage_gates"]
            ):
                all_local_valid = False

            models[arm][seed] = {}
            for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
                artifact = output_root / arm / f"seed_{seed}" / "policies" / f"{mode}.pt"
                model, payload = _load_policy(
                    artifact,
                    arm=arm,
                    training_seed=seed,
                    mode=mode,
                    execution_sha=str(args.execution_sha),
                )
                models[arm][seed][mode] = model
                fit = dict(payload.get("fit") or {})
                gate_pass = bool(fit.get("policy_gate_pass")) and float(
                    fit.get("policy_weighted_mean_tv", math.inf)
                ) <= POLICY_TV_MAX
                local_policy.append(
                    {
                        "arm": arm,
                        "training_seed": seed,
                        "learner_mode": mode,
                        "policy_weighted_mean_tv": float(
                            fit.get("policy_weighted_mean_tv", math.inf)
                        ),
                        "gate_max": POLICY_TV_MAX,
                        "gate_pass": bool(gate_pass),
                    }
                )
                all_local_valid = all_local_valid and bool(gate_pass)

    seed_a, seed_b = map(int, TRAINING_SEEDS)
    comparisons = []
    paired_common = {}
    paired_native = {}
    pooled = {}

    for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
        control_means = []
        candidate_means = []
        for evaluation_seed in map(int, EVALUATION_SEEDS):
            desc = descriptors[evaluation_seed]
            control_left = b6._probabilities_fixed(
                models[CONTROL_ARM][seed_a][mode], desc
            )
            control_right = b6._probabilities_fixed(
                models[CONTROL_ARM][seed_b][mode], desc
            )
            candidate_left = b6._probabilities_fixed(
                models[CANDIDATE_ARM][seed_a][mode], desc
            )
            candidate_right = b6._probabilities_fixed(
                models[CANDIDATE_ARM][seed_b][mode], desc
            )
            control_metric = cross_seed_policy_stability(control_left, control_right)
            candidate_metric = cross_seed_policy_stability(candidate_left, candidate_right)
            control_tv = b6._tv_vector(control_left, control_right)
            candidate_tv = b6._tv_vector(candidate_left, candidate_right)
            paired = [float(c - k) for c, k in zip(control_tv, candidate_tv)]
            if mode == "COMMON_LEARNER":
                paired_common[str(evaluation_seed)] = paired
            else:
                paired_native[str(evaluation_seed)] = paired
            control_means.append(float(control_metric["mean"]))
            candidate_means.append(float(candidate_metric["mean"]))
            comparisons.append(
                {
                    "learner_mode": mode,
                    "evaluation_seed": int(evaluation_seed),
                    "control": {
                        "mean": float(control_metric["mean"]),
                        "p95": float(control_metric["p95"]),
                    },
                    "candidate": {
                        "mean": float(candidate_metric["mean"]),
                        "p95": float(candidate_metric["p95"]),
                        "hard_mean_gate_pass": bool(
                            float(candidate_metric["mean"]) <= CROSS_SEED_MEAN_TV_MAX
                        ),
                        "hard_p95_gate_pass": bool(
                            float(candidate_metric["p95"]) <= CROSS_SEED_P95_TV_MAX
                        ),
                    },
                    "mean_improvement_control_minus_candidate": float(
                        control_metric["mean"] - candidate_metric["mean"]
                    ),
                    "p95_change_candidate_minus_control": float(
                        candidate_metric["p95"] - control_metric["p95"]
                    ),
                }
            )
        pooled[mode] = {
            "control_mean_tv": float(sum(control_means) / len(control_means)),
            "candidate_mean_tv": float(sum(candidate_means) / len(candidate_means)),
        }
        pooled[mode]["absolute_improvement"] = float(
            pooled[mode]["control_mean_tv"] - pooled[mode]["candidate_mean_tv"]
        )
        pooled[mode]["relative_improvement"] = float(
            pooled[mode]["absolute_improvement"] / pooled[mode]["control_mean_tv"]
        ) if pooled[mode]["control_mean_tv"] > 0.0 else -math.inf

    common_boot = equal_group_stratified_bootstrap_mean_ci(
        paired_common,
        seed_parts=("R7.5_ARCH_RESET", "PHASE2B13", "COMMON", "CONTROL_MINUS_CANDIDATE"),
        replicates=BOOTSTRAP_REPLICATES,
        confidence_level=0.95,
    )
    native_boot = equal_group_stratified_bootstrap_mean_ci(
        paired_native,
        seed_parts=("R7.5_ARCH_RESET", "PHASE2B13", "NATIVE", "CONTROL_MINUS_CANDIDATE"),
        replicates=BOOTSTRAP_REPLICATES,
        confidence_level=0.95,
    )

    common_rows = [row for row in comparisons if row["learner_mode"] == "COMMON_LEARNER"]
    native_rows = [row for row in comparisons if row["learner_mode"] == "NATIVE_LEARNER"]
    common_material = bool(
        pooled["COMMON_LEARNER"]["absolute_improvement"] >= CAUSAL_ABS_MIN
        or pooled["COMMON_LEARNER"]["relative_improvement"] >= CAUSAL_REL_MIN
    )
    common_ci_positive = bool(float(common_boot["ci_low"]) > 0.0)
    common_both_improve = bool(
        all(float(row["mean_improvement_control_minus_candidate"]) > 0.0 for row in common_rows)
    )
    common_p95_ok = bool(
        all(
            float(row["p95_change_candidate_minus_control"]) <= COMMON_P95_MAX_DEGRADE
            for row in common_rows
        )
    )
    native_noncontradiction = bool(
        pooled["NATIVE_LEARNER"]["absolute_improvement"] >= 0.0
        and all(
            float(row["mean_improvement_control_minus_candidate"])
            >= -NATIVE_MEAN_MAX_DEGRADE
            for row in native_rows
        )
    )
    causal_supported = bool(
        all_local_valid
        and common_material
        and common_ci_positive
        and common_both_improve
        and common_p95_ok
        and native_noncontradiction
    )
    hard_stability = bool(
        common_rows
        and all(
            bool(row["candidate"]["hard_mean_gate_pass"])
            and bool(row["candidate"]["hard_p95_gate_pass"])
            for row in common_rows
        )
    )

    if not all_local_valid:
        status = "PHASE2B13_INVALID_STOP_AUDIT"
        route = "STOP_AND_AUDIT_PHASE2B13_LOCAL_VALIDITY"
    elif not causal_supported:
        status = "ROOT_IID64_TRAINING_EFFECT_NOT_SUPPORTED"
        route = "REASSESS_CONTINUATION_CONDITIONAL_CHANCE_OR_REPRESENTATION_SUPPORT_NO_SCALEUP"
    elif not hard_stability:
        status = "ROOT_IID64_CAUSAL_EFFECT_SUPPORTED_SMALL_PILOT"
        route = "PRECOMMIT_FULL_X4_ROOT_IID64_CONFIRMATION"
    else:
        status = "ROOT_IID64_SMALL_PILOT_HARD_STABILITY_PASS"
        route = "PRECOMMIT_FULL_X4_ROOT_IID64_CONFIRMATION_BEFORE_STRENGTH"

    return {
        "schema": SCHEMA,
        "status": status,
        "execution_sha": str(args.execution_sha),
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "action_candidate": ACTION_CANDIDATE,
        "training_seeds": [seed_a, seed_b],
        "evaluation_seeds": list(map(int, EVALUATION_SEEDS)),
        "training_contract": {
            "arms": list(ARMS),
            "candidate_k": K,
            "control_k_computed": K,
            "control_k_used": 1,
            "continuation_floor": FLOOR,
            "root_floor": 0.0,
            "postflop_floor": 0.0,
            "heldout_inference_floor": 0.0,
            "iterations": ITERATIONS,
            "chunks_per_iteration": CHUNKS_PER_ITERATION,
            "roots_per_chunk": ROOTS_PER_CHUNK,
            "roots_per_iteration_effective": ROOTS_PER_ITERATION_EFFECTIVE,
            "roots_per_arm_seed": TOTAL_ROOTS,
            "exact_opponent_levels": EXACT_OPPONENT_LEVELS,
            "advantage_reservoir_capacity": RESERVOIR_CAPACITY,
            "strategy_reservoir_capacity": RESERVOIR_CAPACITY,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        },
        "frozen_inputs": {
            **prerequisites,
            "heldout": heldout_identity,
        },
        "local_validity": {
            "valid": bool(all_local_valid),
            "advantage_gates": local_advantage,
            "policy_fit_gates": local_policy,
        },
        "heldout_comparisons": comparisons,
        "pooled_mean_tv": pooled,
        "bootstrap": {
            "COMMON_LEARNER_control_minus_candidate": common_boot,
            "NATIVE_LEARNER_control_minus_candidate": native_boot,
        },
        "decision": {
            "common_materiality_pass": common_material,
            "common_bootstrap_ci_strictly_positive": common_ci_positive,
            "both_common_evaluation_seed_means_improve": common_both_improve,
            "common_p95_non_degradation_pass": common_p95_ok,
            "native_noncontradiction_pass": native_noncontradiction,
            "causal_effect_supported": causal_supported,
            "hard_stability_common_pass_both_heldouts": hard_stability,
            "classification": status,
            "next_route": route,
            "full_x4_confirmation_authorized": bool(causal_supported),
            "architecture_winner_selected": False,
            "production_training_authorized": False,
            "ready_for_tables": False,
        },
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def _run_parent(args) -> int:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    entrypoint = str(Path(__file__).resolve())

    jobs = []
    for seed in map(int, TRAINING_SEEDS):
        for arm in ARMS:
            cmd = [
                sys.executable,
                entrypoint,
                "--repo-root",
                str(Path(args.repo_root).resolve()),
                "--solver",
                str(Path(args.solver).resolve()),
                "--heldout-root",
                str(Path(args.heldout_root).resolve()),
                "--phase2b6-result",
                str(Path(args.phase2b6_result).resolve()),
                "--phase2b12-result",
                str(Path(args.phase2b12_result).resolve()),
                "--output-root",
                str(output_root),
                "--execution-sha",
                str(args.execution_sha),
                "--chance-workers",
                str(int(args.chance_workers)),
                "--single-seed",
                str(seed),
                "--arm",
                str(arm),
            ]
            jobs.append((seed, arm, cmd))

    with ThreadPoolExecutor(max_workers=min(int(args.arm_workers), len(jobs))) as pool:
        futures = {
            pool.submit(subprocess.run, cmd, check=False): (seed, arm)
            for seed, arm, cmd in jobs
        }
        for future in as_completed(futures):
            seed, arm = futures[future]
            completed = future.result()
            if int(completed.returncode) != 0:
                raise RuntimeError(
                    f"Phase2B13 worker failed arm={arm} seed={seed} "
                    f"exit={completed.returncode}"
                )

    result = _evaluate(args)
    out = output_root / "R7_5_ARCH_RESET_V1PLUS_PHASE2B13_ROOT_IID64_TARGET_TRAINING.json"
    _atomic_json(result, out)
    print(
        json.dumps(
            {
                "status": result["status"],
                "control_common_mean_tv": result["pooled_mean_tv"]["COMMON_LEARNER"][
                    "control_mean_tv"
                ],
                "candidate_common_mean_tv": result["pooled_mean_tv"]["COMMON_LEARNER"][
                    "candidate_mean_tv"
                ],
                "common_absolute_improvement": result["pooled_mean_tv"]["COMMON_LEARNER"][
                    "absolute_improvement"
                ],
                "common_bootstrap_ci": [
                    result["bootstrap"]["COMMON_LEARNER_control_minus_candidate"]["ci_low"],
                    result["bootstrap"]["COMMON_LEARNER_control_minus_candidate"]["ci_high"],
                ],
                "causal_effect_supported": result["decision"]["causal_effect_supported"],
                "hard_stability_pass": result["decision"][
                    "hard_stability_common_pass_both_heldouts"
                ],
                "next_route": result["decision"]["next_route"],
                "result": str(out),
                "result_sha256": _sha256(out),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="R7.5 architecture-reset Phase2B13 root IID64 target training pilot"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--phase2b6-result", type=Path, required=True)
    parser.add_argument("--phase2b12-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--arm-workers", type=int, default=2)
    parser.add_argument("--chance-workers", type=int, default=14)
    parser.add_argument("--single-seed", type=int, choices=TRAINING_SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()

    if (args.single_seed is None) != (args.arm is None):
        raise RuntimeError("Phase2B13 --single-seed and --arm must be supplied together")

    repo_root = args.repo_root.resolve()
    validate_phase2_v3_contract(
        repo_root,
        representation=REPRESENTATION,
        domain=DOMAIN,
        training_seed=int(TRAINING_SEEDS[0]),
    )
    if K != 64 or ROOTS_PER_CHUNK != 64 or CHUNKS_PER_ITERATION != 2 or TOTAL_ROOTS != 384:
        raise RuntimeError("Phase2B13 frozen pilot budget drift")
    if RESERVOIR_CAPACITY != 100000 or BATCH_SIZE != 256 or LEARNING_RATE != 0.001:
        raise RuntimeError("Phase2B13 learner contract drift")

    _validate_prerequisites(
        Path(args.phase2b6_result).resolve(), Path(args.phase2b12_result).resolve()
    )

    if args.single_seed is not None:
        return _run_single(args, str(args.arm), int(args.single_seed))
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
