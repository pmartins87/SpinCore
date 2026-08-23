from __future__ import annotations

"""Phase2B6: one small causal H2/3H preflop-damping training pilot.

The only training intervention is a 25% uniform floor on the existing H2
uncertainty-damped behavior policy at preflop continuation calls after at least
one non-forced preflop event.  Initial root decisions and all postflop calls stay
native.  Heldout evaluation removes the floor completely and evaluates the
learned AveragePolicy directly.
"""

import argparse
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence

import torch

import r7_5_3d_v1plus_phase2a_strategy_capacity as phase2a
import spincore.r7_5_representation_v3_stage as stage
from r7_5_3c_chance_coverage_x4_domain_worker_runtimefix import _fit_only_iteration
from spincore.r7_5_action_cfr import legal_mask, validate_policy
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
    POLICY_STEPS,
    POLICY_TV_MAX,
    RESERVOIR_CAPACITY,
    ROOTS_PER_ITERATION,
    TORCH_THREADS,
    TRAINING_SEEDS,
    deck_seed,
    validate_phase2_v3_contract,
)
from spincore.solver import SolverLibrary
from spincore.solver_v3 import neural_bytes_v3
from spincore_nn.models_v3_final import collate_v3_observations, make_h2_final_v3

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B6_PREFLOP_DAMPING_TRAINING_PILOT_V1"
SEED_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B6_SEED_V1"
CHECKPOINT_EXTRA_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B6_RESUME_V1"
POLICY_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B6_POLICY_V1"
DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL
FLOOR = 0.25
CHUNKS_PER_ITERATION = 4
ROOTS_PER_CHUNK = ROOTS_PER_ITERATION
ROOTS_PER_ITERATION_EFFECTIVE = CHUNKS_PER_ITERATION * ROOTS_PER_CHUNK
TOTAL_ROOTS = ITERATIONS * ROOTS_PER_ITERATION_EFFECTIVE
POLICY_COUNT = 1024
BASELINE_SOURCE_EXECUTION_SHA = "4bfa55d69029cd69536fa6dbfcadd162719cb887"
BASELINE_RESULT_SHA256 = "65f691e6b9cf7fbbddf88852c5ac6e0dcd2211af45f53cc4bb3e8271dbaa6149"
PHASE2B5_RESULT_SHA256 = "0fb028c02dbbea0c4fa7a323a3edeed5c4e12789145235be2e851452e16ab5b8"
PHASE2B5_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B5_PREFLOP_FEEDBACK_STABILIZATION_V1"
COMMON_POLICY_INIT_SEED = phase2a.COMMON_POLICY_INIT_SEED
COMMON_BATCH_SEED = phase2a.COMMON_BATCH_SEED
CAUSAL_ABS_MIN = 0.02
CAUSAL_REL_MIN = 0.10
COMMON_P95_MAX_DEGRADE = 0.02
NATIVE_MEAN_MAX_DEGRADE = 0.01
BOOTSTRAP_REPLICATES = 2000


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _v3_street_and_nonforced_preflop(observation: bytes) -> tuple[int, int]:
    if len(observation) < 120 or not observation.startswith(b"SPNNIV3\x00"):
        raise RuntimeError("Phase2B6 requires authoritative SPNNIV3 bytes")
    history_count = int.from_bytes(observation[116:120], "little", signed=False)
    expected = 120 + 20 * history_count
    if len(observation) != expected:
        raise RuntimeError(f"Phase2B6 SPNNIV3 length drift: {len(observation)} != {expected}")
    street = int(observation[9])
    if street not in (0, 1, 2, 3):
        raise RuntimeError(f"Phase2B6 invalid SPNNIV3 street {street}")
    count = 0
    for index in range(history_count):
        offset = 120 + 20 * index
        event_street = int(observation[offset + 1])
        forced = int(observation[offset + 3])
        if event_street == 0 and forced == 0:
            count += 1
    return street, count


def _mix_uniform(policy: Sequence[float], legal: tuple[int, ...], floor: float = FLOOR) -> tuple[float, ...]:
    native = validate_policy(policy, legal)
    f = float(floor)
    if not 0.0 <= f <= 1.0:
        raise ValueError("Phase2B6 uniform floor outside [0,1]")
    out = [0.0] * 10
    u = 1.0 / len(legal)
    for action in legal:
        out[action] = (1.0 - f) * float(native[action]) + f * u
    return validate_policy(out, legal)


class PreflopContinuationFloorPolicy:
    """Wrap the live behavior ensemble without modifying its model lifecycle."""

    def __init__(self, native_behavior, *, floor: float = FLOOR):
        self.native_behavior = native_behavior
        self.floor = float(floor)
        self.calls = 0
        self.damped_calls = 0
        self.root_preflop_native_calls = 0
        self.postflop_native_calls = 0

    def stats(self) -> dict[str, int | float]:
        return {
            "floor": float(self.floor),
            "calls": int(self.calls),
            "damped_calls": int(self.damped_calls),
            "root_preflop_native_calls": int(self.root_preflop_native_calls),
            "postflop_native_calls": int(self.postflop_native_calls),
        }

    def restore_stats(self, payload: dict) -> None:
        if float(payload.get("floor", self.floor)) != self.floor:
            raise RuntimeError("Phase2B6 floor telemetry identity drift")
        for name in ("calls", "damped_calls", "root_preflop_native_calls", "postflop_native_calls"):
            setattr(self, name, int(payload.get(name, 0)))

    def __call__(self, state, observation: bytes, legal: tuple[int, ...]):
        self.calls += 1
        native = validate_policy(self.native_behavior(state, observation, legal), legal)
        street, nonforced = _v3_street_and_nonforced_preflop(observation)
        if street == 0 and nonforced >= 1:
            self.damped_calls += 1
            return _mix_uniform(native, legal, self.floor)
        if street == 0:
            if nonforced != 0:
                raise RuntimeError("Phase2B6 impossible negative/root preflop history state")
            self.root_preflop_native_calls += 1
        else:
            self.postflop_native_calls += 1
        return native


def _stats_delta(after: dict, before: dict) -> dict:
    return {
        "calls": int(after["calls"]) - int(before["calls"]),
        "damped_calls": int(after["damped_calls"]) - int(before["damped_calls"]),
        "root_preflop_native_calls": int(after["root_preflop_native_calls"]) - int(before["root_preflop_native_calls"]),
        "postflop_native_calls": int(after["postflop_native_calls"]) - int(before["postflop_native_calls"]),
    }


def _make_behavior_from_states(states: list[dict], *, config):
    behavior = stage.V3UncertaintyDampedPolicyMixture(
        representation=REPRESENTATION,
        device="cpu",
        epsilon_scale=config.epsilon_scale,
        epsilon_cap=config.epsilon_cap,
    )
    models = []
    for index, state_dict in enumerate(states):
        _cfg, model = stage._make_v3_model(REPRESENTATION, 0xB60000 + index)
        model.load_state_dict(state_dict)
        models.append(model)
    behavior.models = models
    return behavior


def _verify_all_roots_start_before_voluntary_action(repo_root: Path, solver_path: Path) -> None:
    solver = SolverLibrary(solver_path)
    scenarios = action_scenario_cycle(DOMAIN)
    for training_seed in map(int, TRAINING_SEEDS):
        for scenario_index, episode in enumerate(scenarios):
            root = solver.create(episode, deck_seed(training_seed, scenario_index, 1))
            try:
                observation = neural_bytes_v3(root)
                street, count = _v3_street_and_nonforced_preflop(observation)
                if street != 0 or count != 0:
                    raise RuntimeError(
                        f"Phase2B6 root boundary drift seed={training_seed} scenario={scenario_index}: street={street} nonforced={count}"
                    )
            finally:
                root.close()


def _save_resume_checkpoint(
    path: Path,
    *,
    bundle,
    behavior,
    floor_policy: PreflopContinuationFloorPolicy,
    state: dict,
    config,
    execution_sha: str,
    stage_index: int,
    last_stage_report: dict,
) -> None:
    extra = {
        "schema": CHECKPOINT_EXTRA_SCHEMA,
        "stage_config": config.to_dict(),
        "stage_state": dict(state),
        "stage_index": int(stage_index),
        "behavior_model_states": [model.state_dict() for model in behavior.models],
        "behavior_stats": behavior.stats(),
        "floor_policy_stats": floor_policy.stats(),
        "intervention": {
            "floor": FLOOR,
            "scope": "PREFLOP_CONTINUATION_AFTER_AT_LEAST_ONE_NONFORCED_PREFLOP_EVENT",
        },
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
            phase="phase2b6_resume",
        ),
        domain=DOMAIN,
        action_candidate=ACTION_CANDIDATE,
        execution_sha=str(execution_sha),
        architecture_fingerprint_sha256=MODEL_FINGERPRINTS[REPRESENTATION],
        extra=extra,
    )


def _load_resume_checkpoint(path: Path, *, repo_root: Path, solver, training_seed: int, config, execution_sha: str):
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
    if progress.phase != "phase2b6_resume":
        raise RuntimeError("Phase2B6 resume checkpoint phase mismatch")
    if extra.get("schema") != CHECKPOINT_EXTRA_SCHEMA:
        raise RuntimeError("Phase2B6 resume checkpoint schema mismatch")
    if dict(extra.get("stage_config") or {}) != config.to_dict():
        raise RuntimeError("Phase2B6 resume config drift")
    intervention = dict(extra.get("intervention") or {})
    if float(intervention.get("floor", -1.0)) != FLOOR:
        raise RuntimeError("Phase2B6 resume intervention floor drift")
    state = dict(extra.get("stage_state") or {})
    if int(progress.iteration) != int(state.get("completed_iteration", -1)):
        raise RuntimeError("Phase2B6 resume iteration mismatch")
    if int(progress.global_root) != int(state.get("global_root", -1)):
        raise RuntimeError("Phase2B6 resume global-root mismatch")
    behavior = _make_behavior_from_states(list(extra.get("behavior_model_states") or []), config=config)
    behavior.restore_stats(dict(extra.get("behavior_stats") or {}))
    session = stage._make_session(solver, bundle, spec, behavior)
    floor_policy = PreflopContinuationFloorPolicy(behavior, floor=FLOOR)
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


def _stage_coords(stage_index: int) -> tuple[int, int]:
    if not 1 <= int(stage_index) <= ITERATIONS * CHUNKS_PER_ITERATION:
        raise ValueError("Phase2B6 stage index outside frozen range")
    zero = int(stage_index) - 1
    return zero // CHUNKS_PER_ITERATION + 1, zero % CHUNKS_PER_ITERATION + 1


def _report_path(seed_root: Path, stage_index: int) -> Path:
    iteration, chunk = _stage_coords(stage_index)
    return seed_root / "stages" / f"i{iteration}c{chunk}.json"


def _validate_stage_prefix(seed_root: Path, completed_stages: int, last_report: dict) -> None:
    for index in range(1, int(completed_stages) + 1):
        rp = _report_path(seed_root, index)
        if not rp.is_file():
            if index == int(completed_stages) and last_report:
                _atomic_json(last_report, rp)
            else:
                raise RuntimeError(f"Phase2B6 completed stage report missing: {rp}")
        report = json.loads(rp.read_text(encoding="utf-8"))
        if int(report.get("stage_index", -1)) != index:
            raise RuntimeError("Phase2B6 stage-report identity mismatch")


def _collect_chunk(*, session, bundle, floor_policy, state: dict, target_iteration: int) -> dict:
    scenarios = action_scenario_cycle(DOMAIN)
    scenario_counts = list(state["scenario_counts"])
    global_root = int(state["global_root"])
    session.collector.reset_telemetry()
    roots_before = int(bundle.counters["roots"])
    nodes_before = int(bundle.counters["nodes"])
    adv_before = int(bundle.adv_mem.seen)
    pol_before = int(bundle.pol_mem.seen)
    floor_before = floor_policy.stats()
    started = time.perf_counter()
    for _ in range(ROOTS_PER_CHUNK):
        scenario_index = global_root % len(scenarios)
        scenario_counts[scenario_index] += 1
        session.collect_root(
            scenarios[scenario_index],
            iteration=int(target_iteration),
            exact_opponent_levels=EXACT_OPPONENT_LEVELS,
            deck_seed=deck_seed(int(state["training_seed"]), global_root, int(target_iteration)),
        )
        global_root += 1
    seconds = time.perf_counter() - started
    state["global_root"] = global_root
    state["scenario_counts"] = scenario_counts
    report = {
        "roots": int(bundle.counters["roots"]) - roots_before,
        "nodes": int(bundle.counters["nodes"]) - nodes_before,
        "advantage_seen": int(bundle.adv_mem.seen) - adv_before,
        "strategy_seen": int(bundle.pol_mem.seen) - pol_before,
        "tree_collection_seconds": float(seconds),
        "branch_geometry": session.collector.telemetry_snapshot(),
        "floor_policy_delta": _stats_delta(floor_policy.stats(), floor_before),
    }
    if report["roots"] != ROOTS_PER_CHUNK:
        raise RuntimeError("Phase2B6 chunk root-count drift")
    return report


def _aggregate_geometry(chunks: list[dict]) -> dict:
    visits = sum(int(row["branch_geometry"]["advantage_decision_visits"]) for row in chunks)
    nominal = sum(int(row["branch_geometry"]["nominal_aggressive_branches"]) for row in chunks)
    effective = sum(int(row["branch_geometry"]["effective_unique_aggressive_branches"]) for row in chunks)
    return {
        "advantage_decision_visits": visits,
        "nominal_aggressive_branches": nominal,
        "effective_unique_aggressive_branches": effective,
        "nominal_aggressive_branches_per_decision": float(nominal / visits) if visits else 0.0,
        "effective_unique_aggressive_branches_per_decision": float(effective / visits) if visits else 0.0,
    }


def _aggregate_floor(chunks: list[dict]) -> dict:
    keys = ("calls", "damped_calls", "root_preflop_native_calls", "postflop_native_calls")
    out = {key: sum(int(row["floor_policy_delta"][key]) for row in chunks) for key in keys}
    out["damped_fraction"] = float(out["damped_calls"] / out["calls"]) if out["calls"] else 0.0
    return out


def _run_seed_trajectory(*, repo_root: Path, solver_path: Path, output_root: Path, execution_sha: str, training_seed: int):
    validate_phase2_v3_contract(
        repo_root,
        representation=REPRESENTATION,
        domain=DOMAIN,
        training_seed=int(training_seed),
    )
    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("Phase2B6 torch-thread contract drift")
    solver = SolverLibrary(solver_path)
    base_config = frozen_config()
    fit_only = replace(base_config, roots_per_iteration=0)
    seed_root = output_root / f"seed_{int(training_seed)}"
    seed_root.mkdir(parents=True, exist_ok=True)
    resume = seed_root / "resume_checkpoint.pt"

    if resume.is_file():
        bundle, session, behavior, floor_policy, state, completed_stages, last_report = _load_resume_checkpoint(
            resume,
            repo_root=repo_root,
            solver=solver,
            training_seed=int(training_seed),
            config=base_config,
            execution_sha=str(execution_sha),
        )
        if not 0 <= completed_stages <= ITERATIONS * CHUNKS_PER_ITERATION:
            raise RuntimeError("Phase2B6 resume stage index invalid")
        _validate_stage_prefix(seed_root, completed_stages, last_report)
        print(f"[Phase2B6 resume] seed={training_seed} completed_stages={completed_stages}/12", flush=True)
    else:
        bundle, session, behavior, _spec, state = new_phase2_v3_runtime(
            repo_root,
            solver=solver,
            representation=REPRESENTATION,
            domain=DOMAIN,
            training_seed=int(training_seed),
            config=base_config,
        )
        floor_policy = PreflopContinuationFloorPolicy(behavior, floor=FLOOR)
        session.collector.policy = floor_policy
        state["phase2b6"] = {
            "schema": CHECKPOINT_EXTRA_SCHEMA,
            "floor": FLOOR,
            "chance_coverage_multiplier": CHUNKS_PER_ITERATION,
            "effective_roots_per_iteration": ROOTS_PER_ITERATION_EFFECTIVE,
            "heldout_floor_applied": False,
        }
        completed_stages = 0

    for stage_index in range(completed_stages + 1, ITERATIONS * CHUNKS_PER_ITERATION + 1):
        iteration, chunk = _stage_coords(stage_index)
        if chunk == 1:
            if int(state["completed_iteration"]) != iteration - 1:
                raise RuntimeError("Phase2B6 iteration-start identity drift")
            state["phase2b6_pending_iteration"] = {
                "iteration": iteration,
                "roots_before": int(bundle.counters["roots"]),
                "nodes_before": int(bundle.counters["nodes"]),
                "advantage_seen_before": int(bundle.adv_mem.seen),
                "strategy_seen_before": int(bundle.pol_mem.seen),
                "chunks": [],
            }
        pending = dict(state.get("phase2b6_pending_iteration") or {})
        if int(pending.get("iteration", -1)) != iteration:
            raise RuntimeError("Phase2B6 missing pending-iteration state")
        chunks = list(pending.get("chunks") or [])
        if len(chunks) != chunk - 1:
            raise RuntimeError("Phase2B6 chunk history length drift")

        print(f"[Phase2B6 train] seed={training_seed} i{iteration}c{chunk}", flush=True)
        chunk_report = _collect_chunk(
            session=session,
            bundle=bundle,
            floor_policy=floor_policy,
            state=state,
            target_iteration=iteration,
        )
        chunks.append(chunk_report)
        pending["chunks"] = chunks
        state["phase2b6_pending_iteration"] = pending

        iteration_report = None
        if chunk == CHUNKS_PER_ITERATION:
            iteration_report = _fit_only_iteration(
                bundle=bundle,
                session=session,
                behavior=behavior,
                state=state,
                config=fit_only,
                target_iteration=iteration,
            )
            roots_added = int(bundle.counters["roots"]) - int(pending["roots_before"])
            nodes_added = int(bundle.counters["nodes"]) - int(pending["nodes_before"])
            adv_added = int(bundle.adv_mem.seen) - int(pending["advantage_seen_before"])
            pol_added = int(bundle.pol_mem.seen) - int(pending["strategy_seen_before"])
            tree_seconds = sum(float(row["tree_collection_seconds"]) for row in chunks)
            if roots_added != ROOTS_PER_ITERATION_EFFECTIVE:
                raise RuntimeError("Phase2B6 x4 iteration root total drift")
            patched = dict(iteration_report)
            patched.update({
                "roots_added": roots_added,
                "nodes_added": nodes_added,
                "advantage_seen_added": adv_added,
                "strategy_seen_added": pol_added,
                "tree_collection_seconds": tree_seconds,
                "nodes_per_root": float(nodes_added / roots_added),
                "advantage_samples_per_root": float(adv_added / roots_added),
                "strategy_samples_per_root": float(pol_added / roots_added),
                "tree_seconds_per_root": float(tree_seconds / roots_added),
                "branch_geometry": _aggregate_geometry(chunks),
                "floor_policy_iteration": _aggregate_floor(chunks),
                "chance_coverage_chunks": chunks,
            })
            state["iteration_reports"][-1] = patched
            state["tree_collection_seconds_total"] = float(state["tree_collection_seconds_total"]) + tree_seconds
            state.pop("phase2b6_pending_iteration", None)
            iteration_report = patched

        stage_report = {
            "schema": CHECKPOINT_EXTRA_SCHEMA,
            "stage_index": stage_index,
            "iteration": iteration,
            "root_chunk": chunk,
            "training_seed": int(training_seed),
            "roots_total": int(bundle.counters["roots"]),
            "chunk_report": chunk_report,
            "iteration_completed": bool(chunk == CHUNKS_PER_ITERATION),
            "iteration_report": iteration_report,
            "execution_sha": str(execution_sha),
            "floor": FLOOR,
        }
        _save_resume_checkpoint(
            resume,
            bundle=bundle,
            behavior=behavior,
            floor_policy=floor_policy,
            state=state,
            config=base_config,
            execution_sha=str(execution_sha),
            stage_index=stage_index,
            last_stage_report=stage_report,
        )
        _atomic_json(stage_report, _report_path(seed_root, stage_index))
        print(
            f"[Phase2B6 stage complete] seed={training_seed} i{iteration}c{chunk} "
            f"damped_calls={chunk_report['floor_policy_delta']['damped_calls']}",
            flush=True,
        )

    if int(bundle.counters["roots"]) != TOTAL_ROOTS:
        raise RuntimeError("Phase2B6 final root count drift")
    if int(state["completed_iteration"]) != ITERATIONS:
        raise RuntimeError("Phase2B6 final iteration count drift")
    return bundle, state, floor_policy


def _valid_policy_meta(meta: Path, artifact: Path, *, training_seed: int, mode: str) -> dict | None:
    if not meta.is_file() or not artifact.is_file():
        return None
    try:
        saved = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return None
    if (
        saved.get("schema") == POLICY_SCHEMA
        and saved.get("status") == "POLICY_FIT_COMPLETE"
        and int(saved.get("training_seed", -1)) == int(training_seed)
        and saved.get("learner_mode") == mode
        and int(saved.get("capacity", -1)) == RESERVOIR_CAPACITY
        and int(saved.get("authoritative_policy_audit_seed", -1)) == (int(training_seed) ^ 0x71A5BEEF)
    ):
        return saved
    return None


def _fit_seed_policies(*, seed_root: Path, training_seed: int, bundle) -> dict:
    policy_root = seed_root / "policies"
    policy_root.mkdir(parents=True, exist_ok=True)
    native_state = bundle.batch_rng.getstate()
    rows = {}
    audit_seed = int(training_seed) ^ 0x71A5BEEF
    for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
        artifact = policy_root / f"{mode}.pt"
        meta = policy_root / f"{mode}.json"
        existing = _valid_policy_meta(meta, artifact, training_seed=int(training_seed), mode=mode)
        if existing is not None:
            rows[mode] = existing
            print(f"[Phase2B6 policy resume] seed={training_seed} {mode}", flush=True)
            continue
        if mode == "COMMON_LEARNER":
            init_seed = COMMON_POLICY_INIT_SEED
            rng = random.Random(COMMON_BATCH_SEED)
        else:
            init_seed = (int(training_seed) ^ 0x5DEECE66D) & 0x7FFFFFFF
            rng = random.Random()
            rng.setstate(native_state)
        print(f"[Phase2B6 policy fit] seed={training_seed} {mode}", flush=True)
        model, fit = phase2a._fit_policy(
            bundle.pol_mem,
            init_seed=init_seed,
            rng=rng,
            audit_seed=audit_seed,
        )
        model_payload = {
            "schema": POLICY_SCHEMA,
            "status": "POLICY_FIT_COMPLETE",
            "representation": REPRESENTATION,
            "domain": DOMAIN,
            "training_seed": int(training_seed),
            "learner_mode": mode,
            "capacity": RESERVOIR_CAPACITY,
            "authoritative_policy_audit_seed": int(audit_seed),
            "floor_training": FLOOR,
            "floor_inference": 0.0,
            "model_state": model.state_dict(),
            "fit": fit,
        }
        _atomic_torch_save(model_payload, artifact)
        saved = {
            "schema": POLICY_SCHEMA,
            "status": "POLICY_FIT_COMPLETE",
            "training_seed": int(training_seed),
            "learner_mode": mode,
            "capacity": RESERVOIR_CAPACITY,
            "authoritative_policy_audit_seed": int(audit_seed),
            "floor_training": FLOOR,
            "floor_inference": 0.0,
            "artifact": str(artifact),
            "artifact_sha256": _sha256(artifact),
            "fit": fit,
        }
        _atomic_json(saved, meta)
        rows[mode] = saved
    return rows


def _run_single_seed(args, training_seed: int) -> int:
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    seed_root = output_root / f"seed_{int(training_seed)}"
    result_path = seed_root / "seed_result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "SEED_COMPLETE" and existing.get("execution_sha") == str(args.execution_sha):
            print(f"[Phase2B6 seed resume] seed={training_seed} already complete", flush=True)
            return 0

    bundle, state, floor_policy = _run_seed_trajectory(
        repo_root=repo_root,
        solver_path=Path(args.solver).resolve(),
        output_root=output_root,
        execution_sha=str(args.execution_sha),
        training_seed=int(training_seed),
    )
    policy_rows = _fit_seed_policies(seed_root=seed_root, training_seed=int(training_seed), bundle=bundle)
    advantage_rows = []
    for row in list(state.get("iteration_reports") or []):
        value = float(row.get("ensemble_weighted_nrmse", math.inf))
        advantage_rows.append({
            "iteration": int(row.get("iteration", -1)),
            "ensemble_weighted_nrmse": value,
            "gate_max": ADVANTAGE_NRMSE_MAX,
            "gate_pass": bool(value <= ADVANTAGE_NRMSE_MAX and bool(row.get("ensemble_advantage_gate_pass"))),
            "floor_policy_iteration": dict(row.get("floor_policy_iteration") or {}),
            "advantage_seen_added": int(row.get("advantage_seen_added", -1)),
            "strategy_seen_added": int(row.get("strategy_seen_added", -1)),
        })
    result = {
        "schema": SEED_SCHEMA,
        "status": "SEED_COMPLETE",
        "execution_sha": str(args.execution_sha),
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "training_seed": int(training_seed),
        "roots": int(bundle.counters["roots"]),
        "iterations": int(state["completed_iteration"]),
        "floor_training": FLOOR,
        "floor_inference": 0.0,
        "advantage_gates": advantage_rows,
        "all_advantage_gates_pass": bool(len(advantage_rows) == ITERATIONS and all(row["gate_pass"] for row in advantage_rows)),
        "advantage_memory": {
            "capacity": int(bundle.adv_mem.capacity),
            "seen": int(bundle.adv_mem.seen),
            "retained": len(bundle.adv_mem.items),
            "retention_fraction": float(len(bundle.adv_mem.items) / bundle.adv_mem.seen) if bundle.adv_mem.seen else 0.0,
        },
        "strategy_memory": {
            "capacity": int(bundle.pol_mem.capacity),
            "seen": int(bundle.pol_mem.seen),
            "retained": len(bundle.pol_mem.items),
            "retention_fraction": float(len(bundle.pol_mem.items) / bundle.pol_mem.seen) if bundle.pol_mem.seen else 0.0,
        },
        "floor_policy_stats": floor_policy.stats(),
        "policy_fits": policy_rows,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    _atomic_json(result, result_path)
    print(json.dumps({
        "status": result["status"],
        "training_seed": int(training_seed),
        "roots": result["roots"],
        "advantage_pass": result["all_advantage_gates_pass"],
        "damped_calls": result["floor_policy_stats"]["damped_calls"],
        "strategy_seen": result["strategy_memory"]["seen"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


def _probabilities_fixed(model, descriptors) -> list[list[float]]:
    if not descriptors:
        return []
    legal_sets = [tuple(int(x) for x in item.legal_slots) for item in descriptors]
    masks = [legal_mask(row) for row in legal_sets]
    if any(len(mask) != 10 for mask in masks):
        raise RuntimeError("Phase2B6 legal-mask width drift")
    batch = collate_v3_observations(
        [item.observation_v3 for item in descriptors],
        masks,
        with_semantics=False,
        device="cpu",
    )
    model.eval()
    with torch.no_grad():
        logits = model(batch).masked_fill(~batch["legal"], -1e9)
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
    return [list(validate_policy(tuple(float(x) for x in raw), legal)) for raw, legal in zip(probs, legal_sets)]


def _tv_vector(left, right) -> list[float]:
    a = torch.tensor(left, dtype=torch.float64)
    b = torch.tensor(right, dtype=torch.float64)
    return [float(x) for x in (0.5 * torch.abs(a - b).sum(dim=1)).tolist()]


def _find_heldout(root: Path, evaluation_seed: int) -> Path:
    matches = []
    for path in root.rglob("states.json.gz"):
        try:
            payload = load_heldout_v3_artifact(
                path,
                expected_domain=DOMAIN,
                expected_evaluation_seed=int(evaluation_seed),
                expected_count=2048,
            )
        except Exception:
            continue
        if payload:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"Phase2B6 heldout identity mismatch for {evaluation_seed}: {matches}")
    return matches[0]


def _load_baseline_policy(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != phase2a.SEED_SCHEMA or payload.get("status") != "POLICY_FIT_COMPLETE":
        raise RuntimeError("Phase2B6 Phase2A baseline policy schema/status mismatch")
    if payload.get("arm") != "S100K_CONTROL" or int(payload.get("capacity", -1)) != RESERVOIR_CAPACITY:
        raise RuntimeError("Phase2B6 Phase2A baseline arm/capacity mismatch")
    _cfg, model = make_h2_final_v3(device="cpu", seed=0)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def _load_pilot_policy(path: Path, *, training_seed: int, mode: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != POLICY_SCHEMA or payload.get("status") != "POLICY_FIT_COMPLETE":
        raise RuntimeError("Phase2B6 pilot policy schema/status mismatch")
    if int(payload.get("training_seed", -1)) != int(training_seed) or payload.get("learner_mode") != mode:
        raise RuntimeError("Phase2B6 pilot policy identity mismatch")
    if float(payload.get("floor_training", -1.0)) != FLOOR or float(payload.get("floor_inference", -1.0)) != 0.0:
        raise RuntimeError("Phase2B6 pilot policy floor identity mismatch")
    _cfg, model = make_h2_final_v3(device="cpu", seed=0)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def _validate_phase2a_baseline(phase2a_root: Path, result_path: Path) -> dict:
    if _sha256(result_path) != BASELINE_RESULT_SHA256:
        raise RuntimeError("Phase2B6 exact Phase2A result SHA drift")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("schema") != phase2a.SCHEMA or result.get("status") != "CAPACITY_EFFECT_NOT_SUPPORTED":
        raise RuntimeError("Phase2B6 Phase2A baseline result schema/status mismatch")
    if result.get("execution_sha") != BASELINE_SOURCE_EXECUTION_SHA:
        raise RuntimeError("Phase2B6 Phase2A source execution SHA drift")
    recovery = dict(result.get("evaluation_recovery") or {})
    inventory = dict(recovery.get("completed_source_inventory") or {})
    rows = list(inventory.get("policy_artifacts") or [])
    expected = {}
    for row in rows:
        if row.get("arm") == "S100K_CONTROL" and row.get("learner_mode") in ("COMMON_LEARNER", "NATIVE_LEARNER"):
            expected[(int(row["training_seed"]), str(row["learner_mode"]))] = str(row["artifact_sha256"])
    if len(expected) != 4:
        raise RuntimeError("Phase2B6 Phase2A baseline inventory missing four S100K policies")
    checked = []
    for seed in map(int, TRAINING_SEEDS):
        for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
            path = phase2a_root / f"seed_{seed}" / "policies" / f"{mode}__S100K_CONTROL.pt"
            if not path.is_file():
                raise RuntimeError(f"Phase2B6 missing Phase2A baseline policy: {path}")
            actual = _sha256(path)
            if actual != expected[(seed, mode)]:
                raise RuntimeError(f"Phase2B6 Phase2A baseline policy SHA drift: {seed}/{mode}")
            checked.append({"training_seed": seed, "learner_mode": mode, "sha256": actual})
    return {"result_sha256": BASELINE_RESULT_SHA256, "policy_artifacts": checked}


def _validate_phase2b5(path: Path) -> dict:
    if _sha256(path) != PHASE2B5_RESULT_SHA256:
        raise RuntimeError("Phase2B6 exact Phase2B5 result SHA drift")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("schema") != PHASE2B5_SCHEMA or result.get("status") != "MILD_PREFLOP_DAMPING_CANDIDATE":
        raise RuntimeError("Phase2B6 Phase2B5 result schema/status mismatch")
    decision = dict(result.get("decision") or {})
    if decision.get("selected_mild_candidate") != "UNIFORM_FLOOR_025" or not bool(decision.get("small_training_pilot_precommit_allowed")):
        raise RuntimeError("Phase2B6 Phase2B5 route does not authorize the frozen small pilot")
    return {"result_sha256": PHASE2B5_RESULT_SHA256, "selected_floor": FLOOR}


def _evaluate_parent(args) -> dict:
    output_root = Path(args.output_root).resolve()
    phase2a_root = Path(args.phase2a_root).resolve()
    heldout_root = Path(args.heldout_root).resolve()
    phase2a_result = Path(args.phase2a_result).resolve()
    phase2b5_result = Path(args.phase2b5_result).resolve()
    torch.set_num_threads(TORCH_THREADS)

    baseline_identity = _validate_phase2a_baseline(phase2a_root, phase2a_result)
    b5_identity = _validate_phase2b5(phase2b5_result)
    seed_results = {}
    for seed in map(int, TRAINING_SEEDS):
        path = output_root / f"seed_{seed}" / "seed_result.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "SEED_COMPLETE" or payload.get("execution_sha") != str(args.execution_sha):
            raise RuntimeError(f"Phase2B6 seed result invalid for {seed}")
        if int(payload.get("roots", -1)) != TOTAL_ROOTS or int(payload.get("iterations", -1)) != ITERATIONS:
            raise RuntimeError(f"Phase2B6 seed root/iteration drift for {seed}")
        seed_results[seed] = payload

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
            raise RuntimeError("Phase2B6 heldout policy-count drift")
        descriptors[evaluation_seed] = rows
        heldout_identity.append({"evaluation_seed": evaluation_seed, "path": str(heldout), "sha256": _sha256(heldout)})

    comparisons = []
    paired_groups_common = {}
    paired_groups_native = {}
    pooled = {}
    seed_a, seed_b = map(int, TRAINING_SEEDS)
    local_fit_gates = []
    for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
        baseline_models = {}
        pilot_models = {}
        for seed in (seed_a, seed_b):
            baseline_path = phase2a_root / f"seed_{seed}" / "policies" / f"{mode}__S100K_CONTROL.pt"
            pilot_path = output_root / f"seed_{seed}" / "policies" / f"{mode}.pt"
            baseline_models[seed], _bp = _load_baseline_policy(baseline_path)
            pilot_models[seed], pp = _load_pilot_policy(pilot_path, training_seed=seed, mode=mode)
            fit = dict(pp.get("fit") or {})
            local_fit_gates.append({
                "training_seed": seed,
                "learner_mode": mode,
                "policy_weighted_mean_tv": float(fit.get("policy_weighted_mean_tv", math.inf)),
                "gate_max": POLICY_TV_MAX,
                "gate_pass": bool(fit.get("policy_gate_pass")) and float(fit.get("policy_weighted_mean_tv", math.inf)) <= POLICY_TV_MAX,
            })

        baseline_means = []
        pilot_means = []
        for evaluation_seed in map(int, EVALUATION_SEEDS):
            desc = descriptors[evaluation_seed]
            base_left = _probabilities_fixed(baseline_models[seed_a], desc)
            base_right = _probabilities_fixed(baseline_models[seed_b], desc)
            pilot_left = _probabilities_fixed(pilot_models[seed_a], desc)
            pilot_right = _probabilities_fixed(pilot_models[seed_b], desc)
            base_metric = cross_seed_policy_stability(base_left, base_right)
            pilot_metric = cross_seed_policy_stability(pilot_left, pilot_right)
            base_tv = _tv_vector(base_left, base_right)
            pilot_tv = _tv_vector(pilot_left, pilot_right)
            paired = [float(a - b) for a, b in zip(base_tv, pilot_tv)]
            if mode == "COMMON_LEARNER":
                paired_groups_common[str(evaluation_seed)] = paired
            else:
                paired_groups_native[str(evaluation_seed)] = paired
            baseline_means.append(float(base_metric["mean"]))
            pilot_means.append(float(pilot_metric["mean"]))
            comparisons.append({
                "learner_mode": mode,
                "evaluation_seed": evaluation_seed,
                "baseline_phase2a_s100k": {
                    "mean": float(base_metric["mean"]),
                    "p95": float(base_metric["p95"]),
                },
                "pilot_phase2b6": {
                    "mean": float(pilot_metric["mean"]),
                    "p95": float(pilot_metric["p95"]),
                    "hard_mean_gate_pass": bool(float(pilot_metric["mean"]) <= CROSS_SEED_MEAN_TV_MAX),
                    "hard_p95_gate_pass": bool(float(pilot_metric["p95"]) <= CROSS_SEED_P95_TV_MAX),
                },
                "mean_improvement": float(base_metric["mean"] - pilot_metric["mean"]),
                "p95_change_pilot_minus_baseline": float(pilot_metric["p95"] - base_metric["p95"]),
            })
        pooled[mode] = {
            "baseline_mean_tv": float(sum(baseline_means) / len(baseline_means)),
            "pilot_mean_tv": float(sum(pilot_means) / len(pilot_means)),
        }
        pooled[mode]["absolute_improvement"] = float(pooled[mode]["baseline_mean_tv"] - pooled[mode]["pilot_mean_tv"])
        pooled[mode]["relative_improvement"] = float(
            pooled[mode]["absolute_improvement"] / pooled[mode]["baseline_mean_tv"]
        ) if pooled[mode]["baseline_mean_tv"] > 0 else -math.inf

    common_boot = equal_group_stratified_bootstrap_mean_ci(
        paired_groups_common,
        seed_parts=("R7.5_ARCH_RESET", "PHASE2B6", "COMMON_LEARNER", "BASELINE_MINUS_PILOT"),
        replicates=BOOTSTRAP_REPLICATES,
        confidence_level=0.95,
    )
    native_boot = equal_group_stratified_bootstrap_mean_ci(
        paired_groups_native,
        seed_parts=("R7.5_ARCH_RESET", "PHASE2B6", "NATIVE_LEARNER", "BASELINE_MINUS_PILOT"),
        replicates=BOOTSTRAP_REPLICATES,
        confidence_level=0.95,
    )

    advantage_gates = []
    for seed in (seed_a, seed_b):
        for row in seed_results[seed]["advantage_gates"]:
            advantage_gates.append({"training_seed": seed, **row})
    all_advantage = bool(advantage_gates and all(bool(row["gate_pass"]) for row in advantage_gates))
    common_fit_rows = [row for row in local_fit_gates if row["learner_mode"] == "COMMON_LEARNER"]
    native_fit_rows = [row for row in local_fit_gates if row["learner_mode"] == "NATIVE_LEARNER"]
    all_common_fits = bool(common_fit_rows and all(bool(row["gate_pass"]) for row in common_fit_rows))
    all_native_fits = bool(native_fit_rows and all(bool(row["gate_pass"]) for row in native_fit_rows))
    local_valid = bool(all_advantage and all_common_fits and all_native_fits)

    common_rows = [row for row in comparisons if row["learner_mode"] == "COMMON_LEARNER"]
    native_rows = [row for row in comparisons if row["learner_mode"] == "NATIVE_LEARNER"]
    common_material = bool(
        pooled["COMMON_LEARNER"]["absolute_improvement"] >= CAUSAL_ABS_MIN
        or pooled["COMMON_LEARNER"]["relative_improvement"] >= CAUSAL_REL_MIN
    )
    common_ci_positive = bool(float(common_boot["ci_low"]) > 0.0)
    both_common_means_improve = bool(all(float(row["mean_improvement"]) > 0.0 for row in common_rows))
    common_p95_ok = bool(all(float(row["p95_change_pilot_minus_baseline"]) <= COMMON_P95_MAX_DEGRADE for row in common_rows))
    native_pooled_nonworse = bool(pooled["NATIVE_LEARNER"]["absolute_improvement"] >= 0.0)
    native_no_eval_bad = bool(all(float(row["mean_improvement"]) >= -NATIVE_MEAN_MAX_DEGRADE for row in native_rows))
    native_noncontradiction = bool(native_pooled_nonworse and native_no_eval_bad)
    causal_supported = bool(
        local_valid
        and common_material
        and common_ci_positive
        and both_common_means_improve
        and common_p95_ok
        and native_noncontradiction
    )
    hard_stability_common = bool(
        common_rows
        and all(
            bool(row["pilot_phase2b6"]["hard_mean_gate_pass"])
            and bool(row["pilot_phase2b6"]["hard_p95_gate_pass"])
            for row in common_rows
        )
    )

    if not local_valid:
        status = "PHASE2B6_INVALID_LOCAL_GATES"
        next_route = "STOP_AND_AUDIT_LOCAL_TRAINING_VALIDITY"
    elif not causal_supported:
        status = "PREFLOP_DAMPING_TRAINING_EFFECT_NOT_SUPPORTED"
        next_route = "PRECOMMIT_ALTERNATIVE_PREFLOP_ANCHOR_OR_LAGGED_TARGET_DIAGNOSTIC"
    elif not hard_stability_common:
        status = "PREFLOP_DAMPING_CAUSAL_EFFECT_SUPPORTED_BUT_STILL_UNSTABLE"
        next_route = "LOCALIZE_RESIDUAL_WITHOUT_ESCALATING_DAMPING_FLOOR"
    else:
        status = "PREFLOP_DAMPING_PILOT_STABILITY_PASS"
        next_route = "PRECOMMIT_STRATEGIC_STRENGTH_COMPARISON_VS_STABLE_V1_CONTROL"

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
            "floor": FLOOR,
            "scope": "PREFLOP_CONTINUATION_AFTER_AT_LEAST_ONE_NONFORCED_PREFLOP_EVENT",
            "root_floor": 0.0,
            "postflop_floor": 0.0,
            "heldout_inference_floor": 0.0,
            "iterations": ITERATIONS,
            "chunks_per_iteration": CHUNKS_PER_ITERATION,
            "roots_per_chunk": ROOTS_PER_CHUNK,
            "roots_per_iteration_effective": ROOTS_PER_ITERATION_EFFECTIVE,
            "roots_per_seed": TOTAL_ROOTS,
            "exact_opponent_levels": EXACT_OPPONENT_LEVELS,
            "advantage_reservoir_capacity": RESERVOIR_CAPACITY,
            "strategy_reservoir_capacity": RESERVOIR_CAPACITY,
            "policy_steps": POLICY_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        },
        "frozen_inputs": {
            "phase2a": baseline_identity,
            "phase2b5": b5_identity,
            "heldout": heldout_identity,
        },
        "seed_results": {
            str(seed): {
                "roots": int(seed_results[seed]["roots"]),
                "iterations": int(seed_results[seed]["iterations"]),
                "advantage_memory": dict(seed_results[seed]["advantage_memory"]),
                "strategy_memory": dict(seed_results[seed]["strategy_memory"]),
                "floor_policy_stats": dict(seed_results[seed]["floor_policy_stats"]),
            }
            for seed in (seed_a, seed_b)
        },
        "local_validity": {
            "all_advantage_gates_pass": all_advantage,
            "all_common_policy_fit_gates_pass": all_common_fits,
            "all_native_policy_fit_gates_pass": all_native_fits,
            "valid": local_valid,
            "advantage_gates": advantage_gates,
            "policy_fit_gates": local_fit_gates,
        },
        "heldout_comparisons": comparisons,
        "pooled_mean_tv": pooled,
        "bootstrap": {
            "COMMON_LEARNER_baseline_minus_pilot": common_boot,
            "NATIVE_LEARNER_baseline_minus_pilot": native_boot,
        },
        "decision": {
            "common_materiality_pass": common_material,
            "common_bootstrap_ci_strictly_positive": common_ci_positive,
            "both_common_evaluation_seed_means_improve": both_common_means_improve,
            "common_p95_non_degradation_pass": common_p95_ok,
            "native_noncontradiction_pass": native_noncontradiction,
            "causal_effect_supported": causal_supported,
            "hard_stability_common_pass_both_heldouts": hard_stability_common,
            "classification": status,
            "next_route": next_route,
            "higher_floor_training_authorized": False,
            "architecture_winner_selected": False,
            "production_training_authorized": False,
            "ready_for_tables": False,
        },
        "governance_scope": "Post-R7.5.3 architecture-reset causal training pilot only; old R7.5.3 remains closed.",
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def _run_parent(args) -> int:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    entrypoint = str(Path(__file__).resolve())
    commands = []
    for seed in map(int, TRAINING_SEEDS):
        cmd = [
            sys.executable,
            entrypoint,
            "--repo-root", str(Path(args.repo_root).resolve()),
            "--solver", str(Path(args.solver).resolve()),
            "--heldout-root", str(Path(args.heldout_root).resolve()),
            "--phase2a-root", str(Path(args.phase2a_root).resolve()),
            "--phase2a-result", str(Path(args.phase2a_result).resolve()),
            "--phase2b5-result", str(Path(args.phase2b5_result).resolve()),
            "--output-root", str(output_root),
            "--execution-sha", str(args.execution_sha),
            "--single-seed", str(seed),
        ]
        commands.append((seed, cmd))
    with ThreadPoolExecutor(max_workers=min(int(args.seed_workers), len(commands))) as pool:
        futures = {pool.submit(subprocess.run, cmd, check=False): seed for seed, cmd in commands}
        for future in as_completed(futures):
            seed = futures[future]
            completed = future.result()
            if int(completed.returncode) != 0:
                raise RuntimeError(f"Phase2B6 seed worker {seed} failed with exit code {completed.returncode}")
    result = _evaluate_parent(args)
    out = output_root / "R7_5_ARCH_RESET_V1PLUS_PHASE2B6_PREFLOP_DAMPING_TRAINING_PILOT.json"
    _atomic_json(result, out)
    print(json.dumps({
        "status": result["status"],
        "common_baseline_mean_tv": result["pooled_mean_tv"]["COMMON_LEARNER"]["baseline_mean_tv"],
        "common_pilot_mean_tv": result["pooled_mean_tv"]["COMMON_LEARNER"]["pilot_mean_tv"],
        "common_absolute_improvement": result["pooled_mean_tv"]["COMMON_LEARNER"]["absolute_improvement"],
        "common_bootstrap_ci": [
            result["bootstrap"]["COMMON_LEARNER_baseline_minus_pilot"]["ci_low"],
            result["bootstrap"]["COMMON_LEARNER_baseline_minus_pilot"]["ci_high"],
        ],
        "causal_effect_supported": result["decision"]["causal_effect_supported"],
        "hard_stability_pass": result["decision"]["hard_stability_common_pass_both_heldouts"],
        "next_route": result["decision"]["next_route"],
        "result": str(out),
        "result_sha256": _sha256(out),
    }, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="R7.5 architecture-reset Phase2B6 preflop damping training pilot")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--phase2a-root", type=Path, required=True)
    parser.add_argument("--phase2a-result", type=Path, required=True)
    parser.add_argument("--phase2b5-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--seed-workers", type=int, default=2)
    parser.add_argument("--single-seed", type=int, choices=TRAINING_SEEDS)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    validate_phase2_v3_contract(
        repo_root,
        representation=REPRESENTATION,
        domain=DOMAIN,
        training_seed=int(TRAINING_SEEDS[0]),
    )
    if ROOTS_PER_CHUNK != 64 or CHUNKS_PER_ITERATION != 4 or TOTAL_ROOTS != 768:
        raise RuntimeError("Phase2B6 x4 training budget drift")
    if RESERVOIR_CAPACITY != 100000 or POLICY_STEPS != 16384 or BATCH_SIZE != 256 or LEARNING_RATE != 0.001:
        raise RuntimeError("Phase2B6 learner contract drift")
    _validate_phase2b5(Path(args.phase2b5_result).resolve())
    _validate_phase2a_baseline(Path(args.phase2a_root).resolve(), Path(args.phase2a_result).resolve())

    if args.single_seed is not None:
        return _run_single_seed(args, int(args.single_seed))

    _verify_all_roots_start_before_voluntary_action(repo_root, Path(args.solver).resolve())
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
