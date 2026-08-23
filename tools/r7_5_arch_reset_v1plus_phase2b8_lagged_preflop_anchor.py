from __future__ import annotations

"""Phase2B8: small causal training screen for a 25% lagged learned preflop anchor.

The candidate is intentionally identical to Phase2B6 through the end of
iteration 2. In iteration 3, preflop-continuation behavior mixes 75% of the
current native H2 behavior with 25% of the previous learned behavior instead
of 25% uniform. Root and postflop behavior remain native. Heldout inference
uses the learned AveragePolicy directly, with no anchor.
"""

import argparse
from dataclasses import replace
import copy
import json
import math
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

import r7_5_3d_v1plus_phase2a_strategy_capacity as phase2a
import r7_5_3c_chance_coverage_x4_domain_worker_runtimefix as runtimefix
import r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot as b6
import r7_5_arch_reset_v1plus_phase2b7_residual_localization as b7
import spincore.r7_5_representation_v3_stage as stage
from spincore.r7_5_action_cfr import validate_policy
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
    validate_phase2_v3_contract,
)
from spincore.r7_5_representation_v3_uncertainty import V3UncertaintyDampedPolicyMixture
from spincore_nn.models_v3_final import make_h2_final_v3

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B8_LAGGED_PREFLOP_ANCHOR_V1"
SEED_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B8_SEED_V1"
CHECKPOINT_EXTRA_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B8_RESUME_V1"
POLICY_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B8_POLICY_V1"
DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL
ANCHOR_WEIGHT = 0.25
CHUNKS_PER_ITERATION = 4
ROOTS_PER_CHUNK = ROOTS_PER_ITERATION
ROOTS_PER_ITERATION_EFFECTIVE = CHUNKS_PER_ITERATION * ROOTS_PER_CHUNK
TOTAL_ROOTS = ITERATIONS * ROOTS_PER_ITERATION_EFFECTIVE
POLICY_COUNT = 1024
PHASE2B6_RESULT_SHA256 = "33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a"
PHASE2B6_EXECUTION_SHA = "4fa96434321c32efc734a55ae75982018ff2d091"
PHASE2B7_RESULT_SHA256 = "ff55a5a047d62952e505b8e4d59d79d4016f30b6696a339318bc696dd6f77fe6"
CAUSAL_ABS_MIN = 0.015
CAUSAL_REL_MIN = 0.08
COMMON_P95_MAX_DEGRADE = 0.02
NATIVE_POOLED_MAX_DEGRADE = 0.01
ROOT_MEAN_MAX_DEGRADE = 0.005
CONTINUATION_MEAN_MAX_DEGRADE = 0.005
REPRO_TOL = 1e-12
BOOTSTRAP_REPLICATES = 2000
B6_COMMON_ROOT_MEAN = 0.25663072380695223
B6_COMMON_CONTINUATION_MEAN = 0.1778058850139139


def _sha256(path: Path) -> str:
    return b6._sha256(path)


def _atomic_json(payload: dict, path: Path) -> None:
    b6._atomic_json(payload, path)


def _atomic_torch_save(payload, path: Path) -> None:
    b6._atomic_torch_save(payload, path)


def _new_behavior() -> V3UncertaintyDampedPolicyMixture:
    cfg = frozen_config()
    return V3UncertaintyDampedPolicyMixture(
        representation=REPRESENTATION,
        device="cpu",
        epsilon_scale=cfg.epsilon_scale,
        epsilon_cap=cfg.epsilon_cap,
    )


def _states(models) -> list[dict]:
    return [copy.deepcopy(model.state_dict()) for model in models]


def _set_behavior_states(behavior, states: list[dict]) -> None:
    models = []
    for index, state_dict in enumerate(states):
        _cfg, model = stage._make_v3_model(REPRESENTATION, 0xB80000 + index)
        model.load_state_dict(state_dict)
        models.append(model)
    behavior.models = models


class LaggedBehaviorAnchorPolicy:
    def __init__(self, current_behavior, lagged_behavior, *, weight: float = ANCHOR_WEIGHT):
        self.current_behavior = current_behavior
        self.lagged_behavior = lagged_behavior
        self.weight = float(weight)
        self.calls = 0
        self.damped_calls = 0
        self.root_preflop_native_calls = 0
        self.postflop_native_calls = 0
        self.lagged_uniform_calls = 0
        self.lagged_learned_calls = 0

    def stats(self) -> dict:
        return {
            "anchor_weight": self.weight,
            "calls": int(self.calls),
            "damped_calls": int(self.damped_calls),
            "root_preflop_native_calls": int(self.root_preflop_native_calls),
            "postflop_native_calls": int(self.postflop_native_calls),
            "lagged_uniform_calls": int(self.lagged_uniform_calls),
            "lagged_learned_calls": int(self.lagged_learned_calls),
        }

    def restore_stats(self, payload: dict) -> None:
        if abs(float(payload.get("anchor_weight", self.weight)) - self.weight) > 1e-15:
            raise RuntimeError("Phase2B8 anchor-weight telemetry drift")
        for name in (
            "calls",
            "damped_calls",
            "root_preflop_native_calls",
            "postflop_native_calls",
            "lagged_uniform_calls",
            "lagged_learned_calls",
        ):
            setattr(self, name, int(payload.get(name, 0)))

    def __call__(self, state, observation: bytes, legal: tuple[int, ...]):
        self.calls += 1
        current = validate_policy(self.current_behavior(state, observation, legal), legal)
        street, nonforced = b6._v3_street_and_nonforced_preflop(observation)
        if street == 0 and nonforced >= 1:
            self.damped_calls += 1
            if not self.lagged_behavior.models:
                self.lagged_uniform_calls += 1
                return b6._mix_uniform(current, legal, self.weight)
            self.lagged_learned_calls += 1
            lagged = validate_policy(self.lagged_behavior(state, observation, legal), legal)
            out = [0.0] * 10
            for action in legal:
                out[action] = (1.0 - self.weight) * float(current[action]) + self.weight * float(lagged[action])
            return validate_policy(out, legal)
        if street == 0:
            self.root_preflop_native_calls += 1
        else:
            self.postflop_native_calls += 1
        return current


def _anchor_stats_delta(after: dict, before: dict) -> dict:
    keys = (
        "calls",
        "damped_calls",
        "root_preflop_native_calls",
        "postflop_native_calls",
        "lagged_uniform_calls",
        "lagged_learned_calls",
    )
    return {key: int(after[key]) - int(before[key]) for key in keys}


def _collect_chunk(*, session, bundle, anchor_policy, state: dict, target_iteration: int) -> dict:
    before = anchor_policy.stats()
    row = b6._collect_chunk(
        session=session,
        bundle=bundle,
        floor_policy=anchor_policy,
        state=state,
        target_iteration=target_iteration,
    )
    row["anchor_policy_delta"] = _anchor_stats_delta(anchor_policy.stats(), before)
    return row


def _aggregate_anchor(chunks: list[dict]) -> dict:
    keys = (
        "calls",
        "damped_calls",
        "root_preflop_native_calls",
        "postflop_native_calls",
        "lagged_uniform_calls",
        "lagged_learned_calls",
    )
    out = {key: sum(int(row["anchor_policy_delta"][key]) for row in chunks) for key in keys}
    out["anchor_weight"] = ANCHOR_WEIGHT
    out["damped_fraction"] = float(out["damped_calls"] / out["calls"]) if out["calls"] else 0.0
    return out


def _save_resume_checkpoint(
    path: Path,
    *,
    bundle,
    current_behavior,
    lagged_behavior,
    anchor_policy,
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
        "current_behavior_model_states": _states(current_behavior.models),
        "lagged_behavior_model_states": _states(lagged_behavior.models),
        "current_behavior_stats": current_behavior.stats(),
        "lagged_behavior_stats": lagged_behavior.stats(),
        "anchor_policy_stats": anchor_policy.stats(),
        "intervention": {
            "anchor_weight": ANCHOR_WEIGHT,
            "scope": "PREFLOP_CONTINUATION_AFTER_AT_LEAST_ONE_NONFORCED_PREFLOP_EVENT",
            "root_anchor": 0.0,
            "postflop_anchor": 0.0,
            "heldout_anchor": 0.0,
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
            phase="phase2b8_resume",
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
    if progress.phase != "phase2b8_resume" or extra.get("schema") != CHECKPOINT_EXTRA_SCHEMA:
        raise RuntimeError("Phase2B8 resume checkpoint identity mismatch")
    if dict(extra.get("stage_config") or {}) != config.to_dict():
        raise RuntimeError("Phase2B8 resume config drift")
    intervention = dict(extra.get("intervention") or {})
    if abs(float(intervention.get("anchor_weight", -1.0)) - ANCHOR_WEIGHT) > 1e-15:
        raise RuntimeError("Phase2B8 resume anchor drift")
    state = dict(extra.get("stage_state") or {})
    if int(progress.iteration) != int(state.get("completed_iteration", -1)):
        raise RuntimeError("Phase2B8 resume iteration mismatch")
    if int(progress.global_root) != int(state.get("global_root", -1)):
        raise RuntimeError("Phase2B8 resume global-root mismatch")

    current_behavior = _new_behavior()
    _set_behavior_states(current_behavior, list(extra.get("current_behavior_model_states") or []))
    current_behavior.restore_stats(dict(extra.get("current_behavior_stats") or {}))
    lagged_behavior = _new_behavior()
    _set_behavior_states(lagged_behavior, list(extra.get("lagged_behavior_model_states") or []))
    lagged_behavior.restore_stats(dict(extra.get("lagged_behavior_stats") or {}))
    anchor_policy = LaggedBehaviorAnchorPolicy(current_behavior, lagged_behavior)
    anchor_policy.restore_stats(dict(extra.get("anchor_policy_stats") or {}))
    session = stage._make_session(solver, bundle, spec, anchor_policy)
    return (
        bundle,
        session,
        current_behavior,
        lagged_behavior,
        anchor_policy,
        state,
        int(extra.get("stage_index", -1)),
        dict(extra.get("last_stage_report") or {}),
    )


def _run_seed_trajectory(*, repo_root: Path, solver_path: Path, output_root: Path, execution_sha: str, training_seed: int):
    validate_phase2_v3_contract(
        repo_root,
        representation=REPRESENTATION,
        domain=DOMAIN,
        training_seed=int(training_seed),
    )
    torch.set_num_threads(TORCH_THREADS)
    solver = b6.SolverLibrary(solver_path)
    base_config = frozen_config()
    fit_only = replace(base_config, roots_per_iteration=0)
    seed_root = output_root / f"seed_{int(training_seed)}"
    seed_root.mkdir(parents=True, exist_ok=True)
    resume = seed_root / "resume_checkpoint.pt"

    if resume.is_file():
        (
            bundle,
            session,
            current_behavior,
            lagged_behavior,
            anchor_policy,
            state,
            completed_stages,
            last_report,
        ) = _load_resume_checkpoint(
            resume,
            repo_root=repo_root,
            solver=solver,
            training_seed=int(training_seed),
            config=base_config,
            execution_sha=str(execution_sha),
        )
        b6._validate_stage_prefix(seed_root, completed_stages, last_report)
        print(f"[Phase2B8 resume] seed={training_seed} completed_stages={completed_stages}/12", flush=True)
    else:
        bundle, session, current_behavior, spec, state = new_phase2_v3_runtime(
            repo_root,
            solver=solver,
            representation=REPRESENTATION,
            domain=DOMAIN,
            training_seed=int(training_seed),
            config=base_config,
        )
        lagged_behavior = _new_behavior()
        anchor_policy = LaggedBehaviorAnchorPolicy(current_behavior, lagged_behavior)
        session.collector.policy = anchor_policy
        state["phase2b8"] = {
            "schema": CHECKPOINT_EXTRA_SCHEMA,
            "anchor_weight": ANCHOR_WEIGHT,
            "candidate": "LAGGED_BEHAVIOR_ANCHOR_025",
        }
        completed_stages = 0

    for stage_index in range(completed_stages + 1, ITERATIONS * CHUNKS_PER_ITERATION + 1):
        iteration, chunk = b6._stage_coords(stage_index)
        if chunk == 1:
            if int(state["completed_iteration"]) != iteration - 1:
                raise RuntimeError("Phase2B8 iteration-start identity drift")
            state["phase2b8_pending_iteration"] = {
                "iteration": iteration,
                "roots_before": int(bundle.counters["roots"]),
                "nodes_before": int(bundle.counters["nodes"]),
                "advantage_seen_before": int(bundle.adv_mem.seen),
                "strategy_seen_before": int(bundle.pol_mem.seen),
                "chunks": [],
            }
        pending = dict(state.get("phase2b8_pending_iteration") or {})
        if int(pending.get("iteration", -1)) != iteration:
            raise RuntimeError("Phase2B8 missing pending iteration")
        chunks = list(pending.get("chunks") or [])
        if len(chunks) != chunk - 1:
            raise RuntimeError("Phase2B8 chunk history drift")

        print(f"[Phase2B8 train] seed={training_seed} i{iteration}c{chunk}", flush=True)
        chunk_report = _collect_chunk(
            session=session,
            bundle=bundle,
            anchor_policy=anchor_policy,
            state=state,
            target_iteration=iteration,
        )
        chunks.append(chunk_report)
        pending["chunks"] = chunks
        state["phase2b8_pending_iteration"] = pending

        iteration_report = None
        if chunk == CHUNKS_PER_ITERATION:
            old_current_states = _states(current_behavior.models)
            iteration_report = runtimefix._fit_only_iteration(
                bundle=bundle,
                session=session,
                behavior=current_behavior,
                state=state,
                config=fit_only,
                target_iteration=iteration,
            )
            _set_behavior_states(lagged_behavior, old_current_states)
            roots_added = int(bundle.counters["roots"]) - int(pending["roots_before"])
            nodes_added = int(bundle.counters["nodes"]) - int(pending["nodes_before"])
            adv_added = int(bundle.adv_mem.seen) - int(pending["advantage_seen_before"])
            pol_added = int(bundle.pol_mem.seen) - int(pending["strategy_seen_before"])
            tree_seconds = sum(float(row["tree_collection_seconds"]) for row in chunks)
            if roots_added != ROOTS_PER_ITERATION_EFFECTIVE:
                raise RuntimeError("Phase2B8 iteration root total drift")
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
                "branch_geometry": b6._aggregate_geometry(chunks),
                "anchor_policy_iteration": _aggregate_anchor(chunks),
                "lagged_source_after_fit": (
                    "UNIFORM_INITIAL" if not old_current_states else f"LEARNED_ITERATION_{iteration - 1}"
                ),
            })
            state["iteration_reports"][-1] = patched
            state["tree_collection_seconds_total"] = float(state["tree_collection_seconds_total"]) + tree_seconds
            state.pop("phase2b8_pending_iteration", None)
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
            "anchor_weight": ANCHOR_WEIGHT,
        }
        _save_resume_checkpoint(
            resume,
            bundle=bundle,
            current_behavior=current_behavior,
            lagged_behavior=lagged_behavior,
            anchor_policy=anchor_policy,
            state=state,
            config=base_config,
            execution_sha=str(execution_sha),
            stage_index=stage_index,
            last_stage_report=stage_report,
        )
        _atomic_json(stage_report, b6._report_path(seed_root, stage_index))
        print(
            f"[Phase2B8 stage complete] seed={training_seed} i{iteration}c{chunk} "
            f"anchored_calls={chunk_report['anchor_policy_delta']['damped_calls']}",
            flush=True,
        )

    if int(bundle.counters["roots"]) != TOTAL_ROOTS or int(state["completed_iteration"]) != ITERATIONS:
        raise RuntimeError("Phase2B8 final training budget drift")
    return bundle, state, anchor_policy


def _fit_seed_policies(*, seed_root: Path, training_seed: int, bundle) -> dict:
    policy_root = seed_root / "policies"
    policy_root.mkdir(parents=True, exist_ok=True)
    native_state = bundle.batch_rng.getstate()
    rows = {}
    audit_seed = int(training_seed) ^ 0x71A5BEEF
    for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
        artifact = policy_root / f"{mode}.pt"
        meta = policy_root / f"{mode}.json"
        if meta.is_file() and artifact.is_file():
            saved = json.loads(meta.read_text(encoding="utf-8"))
            if (
                saved.get("schema") == POLICY_SCHEMA
                and saved.get("status") == "POLICY_FIT_COMPLETE"
                and int(saved.get("training_seed", -1)) == int(training_seed)
                and saved.get("learner_mode") == mode
                and int(saved.get("capacity", -1)) == RESERVOIR_CAPACITY
                and int(saved.get("authoritative_policy_audit_seed", -1)) == audit_seed
                and saved.get("candidate") == "LAGGED_BEHAVIOR_ANCHOR_025"
                and float(saved.get("anchor_training", -1.0)) == ANCHOR_WEIGHT
                and float(saved.get("anchor_inference", -1.0)) == 0.0
                and saved.get("artifact_sha256") == _sha256(artifact)
            ):
                rows[mode] = saved
                continue
        if mode == "COMMON_LEARNER":
            init_seed = phase2a.COMMON_POLICY_INIT_SEED
            rng = random.Random(phase2a.COMMON_BATCH_SEED)
        else:
            init_seed = (int(training_seed) ^ 0x5DEECE66D) & 0x7FFFFFFF
            rng = random.Random()
            rng.setstate(native_state)
        print(f"[Phase2B8 policy fit] seed={training_seed} {mode}", flush=True)
        model, fit = phase2a._fit_policy(
            bundle.pol_mem,
            init_seed=init_seed,
            rng=rng,
            audit_seed=audit_seed,
        )
        payload = {
            "schema": POLICY_SCHEMA,
            "status": "POLICY_FIT_COMPLETE",
            "representation": REPRESENTATION,
            "domain": DOMAIN,
            "training_seed": int(training_seed),
            "learner_mode": mode,
            "capacity": RESERVOIR_CAPACITY,
            "authoritative_policy_audit_seed": audit_seed,
            "candidate": "LAGGED_BEHAVIOR_ANCHOR_025",
            "anchor_training": ANCHOR_WEIGHT,
            "anchor_inference": 0.0,
            "model_state": model.state_dict(),
            "fit": fit,
        }
        _atomic_torch_save(payload, artifact)
        saved = {
            "schema": POLICY_SCHEMA,
            "status": "POLICY_FIT_COMPLETE",
            "training_seed": int(training_seed),
            "learner_mode": mode,
            "capacity": RESERVOIR_CAPACITY,
            "authoritative_policy_audit_seed": audit_seed,
            "candidate": "LAGGED_BEHAVIOR_ANCHOR_025",
            "anchor_training": ANCHOR_WEIGHT,
            "anchor_inference": 0.0,
            "artifact": str(artifact),
            "artifact_sha256": _sha256(artifact),
            "fit": fit,
        }
        _atomic_json(saved, meta)
        rows[mode] = saved
    return rows


def _run_single_seed(args, training_seed: int) -> int:
    output_root = Path(args.output_root).resolve()
    seed_root = output_root / f"seed_{training_seed}"
    result_path = seed_root / "seed_result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "SEED_COMPLETE"
            and existing.get("execution_sha") == str(args.execution_sha)
            and existing.get("candidate") == "LAGGED_BEHAVIOR_ANCHOR_025"
            and float(existing.get("anchor_weight", -1.0)) == ANCHOR_WEIGHT
            and float(existing.get("anchor_inference", -1.0)) == 0.0
        ):
            print(f"[Phase2B8 seed resume] seed={training_seed} already complete", flush=True)
            return 0

    bundle, state, anchor_policy = _run_seed_trajectory(
        repo_root=Path(args.repo_root).resolve(),
        solver_path=Path(args.solver).resolve(),
        output_root=output_root,
        execution_sha=str(args.execution_sha),
        training_seed=int(training_seed),
    )
    policy_rows = _fit_seed_policies(seed_root=seed_root, training_seed=training_seed, bundle=bundle)
    advantage_rows = []
    for row in list(state.get("iteration_reports") or []):
        value = float(row.get("ensemble_weighted_nrmse", math.inf))
        advantage_rows.append({
            "iteration": int(row["iteration"]),
            "ensemble_weighted_nrmse": value,
            "gate_max": ADVANTAGE_NRMSE_MAX,
            "gate_pass": bool(value <= ADVANTAGE_NRMSE_MAX and row.get("ensemble_advantage_gate_pass")),
            "anchor_policy_iteration": dict(row.get("anchor_policy_iteration") or {}),
            "roots_added": int(row.get("roots_added", -1)),
            "advantage_seen_added": int(row.get("advantage_seen_added", -1)),
            "strategy_seen_added": int(row.get("strategy_seen_added", -1)),
            "lagged_source_after_fit": row.get("lagged_source_after_fit"),
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
        "candidate": "LAGGED_BEHAVIOR_ANCHOR_025",
        "anchor_weight": ANCHOR_WEIGHT,
        "anchor_inference": 0.0,
        "advantage_gates": advantage_rows,
        "all_advantage_gates_pass": bool(len(advantage_rows) == ITERATIONS and all(x["gate_pass"] for x in advantage_rows)),
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
        "anchor_policy_stats": anchor_policy.stats(),
        "policy_fits": policy_rows,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    _atomic_json(result, result_path)
    print(json.dumps({
        "status": "SEED_COMPLETE",
        "training_seed": training_seed,
        "roots": result["roots"],
        "advantage_pass": result["all_advantage_gates_pass"],
        "anchored_calls": result["anchor_policy_stats"]["damped_calls"],
        "lagged_learned_calls": result["anchor_policy_stats"]["lagged_learned_calls"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


def _load_candidate_policy(path: Path, *, training_seed: int, mode: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != POLICY_SCHEMA or payload.get("status") != "POLICY_FIT_COMPLETE":
        raise RuntimeError("Phase2B8 candidate policy schema/status mismatch")
    if int(payload.get("training_seed", -1)) != int(training_seed) or payload.get("learner_mode") != mode:
        raise RuntimeError("Phase2B8 candidate policy identity mismatch")
    if float(payload.get("anchor_training", -1.0)) != ANCHOR_WEIGHT or float(payload.get("anchor_inference", -1.0)) != 0.0:
        raise RuntimeError("Phase2B8 candidate anchor identity mismatch")
    _cfg, model = make_h2_final_v3(device="cpu", seed=0)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def _validate_b6_b7(b6_result_path: Path, b7_result_path: Path) -> tuple[dict, dict]:
    if _sha256(b6_result_path) != PHASE2B6_RESULT_SHA256:
        raise RuntimeError("Phase2B8 exact Phase2B6 result SHA drift")
    if _sha256(b7_result_path) != PHASE2B7_RESULT_SHA256:
        raise RuntimeError("Phase2B8 exact Phase2B7 result SHA drift")
    r6 = json.loads(b6_result_path.read_text(encoding="utf-8"))
    r7 = json.loads(b7_result_path.read_text(encoding="utf-8"))
    if r6.get("status") != "PREFLOP_DAMPING_CAUSAL_EFFECT_SUPPORTED_BUT_STILL_UNSTABLE":
        raise RuntimeError("Phase2B8 wrong Phase2B6 control status")
    if r6.get("execution_sha") != PHASE2B6_EXECUTION_SHA:
        raise RuntimeError("Phase2B8 wrong Phase2B6 execution SHA")
    if r7.get("status") != "PREFLOP_CONTINUATION_DOMINANT":
        raise RuntimeError("Phase2B8 wrong Phase2B7 localization status")
    if r7.get("decision", {}).get("next_route") != "PRECOMMIT_EARLY_PREFLOP_LAGGED_TARGET_OR_ANCHOR_SCREEN":
        raise RuntimeError("Phase2B8 route not authorized by Phase2B7")
    if r7.get("representation") != REPRESENTATION or r7.get("domain") != DOMAIN:
        raise RuntimeError("Phase2B8 Phase2B7 representation/domain drift")
    if list(map(int, r7.get("training_seeds") or [])) != list(map(int, TRAINING_SEEDS)):
        raise RuntimeError("Phase2B8 Phase2B7 training-seed drift")
    if list(map(int, r7.get("evaluation_seeds") or [])) != list(map(int, EVALUATION_SEEDS)):
        raise RuntimeError("Phase2B8 Phase2B7 evaluation-seed drift")
    if int(r7.get("policy_count_per_evaluation_seed", -1)) != POLICY_COUNT:
        raise RuntimeError("Phase2B8 Phase2B7 policy-count drift")
    return r6, r7


def _equivalence_before_divergence(candidate_seed_results: dict[int, dict], b6_root: Path) -> dict:
    rows = []
    passed = True
    for seed in map(int, TRAINING_SEEDS):
        control = json.loads((b6_root / f"seed_{seed}" / "seed_result.json").read_text(encoding="utf-8"))
        if control.get("execution_sha") != PHASE2B6_EXECUTION_SHA:
            raise RuntimeError(f"Phase2B8 wrong Phase2B6 local seed artifact {seed}")
        cand_by_i = {int(x["iteration"]): x for x in candidate_seed_results[seed]["advantage_gates"]}
        ctrl_by_i = {int(x["iteration"]): x for x in control["advantage_gates"]}
        for iteration in (1, 2):
            c = cand_by_i[iteration]
            b = ctrl_by_i[iteration]
            control_stage = json.loads(
                (b6_root / f"seed_{seed}" / "stages" / f"i{iteration}c4.json").read_text(encoding="utf-8")
            )
            control_roots = int((control_stage.get("iteration_report") or {}).get("roots_added", -1))
            checks = {
                "roots_added": int(c["roots_added"]) == control_roots == ROOTS_PER_ITERATION_EFFECTIVE,
                "advantage_seen_added": int(c["advantage_seen_added"]) == int(b["advantage_seen_added"]),
                "strategy_seen_added": int(c["strategy_seen_added"]) == int(b["strategy_seen_added"]),
                "nrmse": abs(float(c["ensemble_weighted_nrmse"]) - float(b["ensemble_weighted_nrmse"])) <= REPRO_TOL,
            }
            ca = c["anchor_policy_iteration"]
            ba = b["floor_policy_iteration"]
            for key in ("calls", "damped_calls", "root_preflop_native_calls", "postflop_native_calls"):
                checks[f"policy_{key}"] = int(ca[key]) == int(ba[key])
            row_pass = all(checks.values())
            passed = passed and row_pass
            rows.append({
                "training_seed": seed,
                "iteration": iteration,
                "pass": row_pass,
                "checks": checks,
                "candidate_nrmse": float(c["ensemble_weighted_nrmse"]),
                "control_nrmse": float(b["ensemble_weighted_nrmse"]),
            })
    return {"pass": passed, "rows": rows, "tolerance": REPRO_TOL}


def _region_summary(common_state_rows: list[dict]) -> dict:
    groups = {}
    for region in (
        "PREFLOP_ROOT",
        "PREFLOP_CONTINUATION_1",
        "PREFLOP_CONTINUATION_2PLUS",
        "FLOP",
        "TURN",
        "RIVER",
    ):
        rows = [x for x in common_state_rows if x["region"] == region]
        if not rows:
            continue
        groups[region] = {
            "count": len(rows),
            "control_mean_tv": float(sum(x["control_tv"] for x in rows) / len(rows)),
            "candidate_mean_tv": float(sum(x["candidate_tv"] for x in rows) / len(rows)),
            "control_tail_gt_035": sum(x["control_tv"] > 0.35 for x in rows),
            "candidate_tail_gt_035": sum(x["candidate_tv"] > 0.35 for x in rows),
        }
    cont = [x for x in common_state_rows if x["region"] in ("PREFLOP_CONTINUATION_1", "PREFLOP_CONTINUATION_2PLUS")]
    groups["PREFLOP_CONTINUATION_COMBINED"] = {
        "count": len(cont),
        "control_mean_tv": float(sum(x["control_tv"] for x in cont) / len(cont)),
        "candidate_mean_tv": float(sum(x["candidate_tv"] for x in cont) / len(cont)),
    }
    return groups


def _evaluate_parent(args) -> dict:
    output_root = Path(args.output_root).resolve()
    b6_root = Path(args.phase2b6_root).resolve()
    heldout_root = Path(args.heldout_root).resolve()
    _r6, r7 = _validate_b6_b7(Path(args.phase2b6_result).resolve(), Path(args.phase2b7_result).resolve())
    torch.set_num_threads(TORCH_THREADS)

    seed_results = {}
    for seed in map(int, TRAINING_SEEDS):
        payload = json.loads((output_root / f"seed_{seed}" / "seed_result.json").read_text(encoding="utf-8"))
        if (
            payload.get("status") != "SEED_COMPLETE"
            or payload.get("execution_sha") != str(args.execution_sha)
            or payload.get("candidate") != "LAGGED_BEHAVIOR_ANCHOR_025"
            or float(payload.get("anchor_weight", -1.0)) != ANCHOR_WEIGHT
            or float(payload.get("anchor_inference", -1.0)) != 0.0
            or int(payload.get("roots", -1)) != TOTAL_ROOTS
            or int(payload.get("iterations", -1)) != ITERATIONS
        ):
            raise RuntimeError(f"Phase2B8 incomplete/invalid seed {seed}")
        seed_results[seed] = payload

    equivalence = _equivalence_before_divergence(seed_results, b6_root)
    if not equivalence["pass"]:
        raise RuntimeError("Phase2B8 equivalence-before-divergence check FAILED")

    expected_b6_policy_hashes = {
        (int(row["training_seed"]), str(row["learner_mode"])): str(row["sha256"])
        for row in (
            r7.get("frozen_inputs", {})
            .get("phase2b6_policies", {})
            .get("policy_artifacts", [])
        )
    }
    if len(expected_b6_policy_hashes) != 4:
        raise RuntimeError("Phase2B8 Phase2B7 missing exact four Phase2B6 control policy hashes")
    for seed in map(int, TRAINING_SEEDS):
        for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
            control_path = b6_root / f"seed_{seed}" / "policies" / f"{mode}.pt"
            if _sha256(control_path) != expected_b6_policy_hashes[(seed, mode)]:
                raise RuntimeError(f"Phase2B8 exact Phase2B6 control policy hash drift: {seed}/{mode}")

    expected_heldout_hashes = {
        int(row["evaluation_seed"]): str(row["sha256"])
        for row in r7.get("frozen_inputs", {}).get("heldout", [])
    }
    if set(expected_heldout_hashes) != set(map(int, EVALUATION_SEEDS)):
        raise RuntimeError("Phase2B8 Phase2B7 heldout identity inventory drift")

    descriptors = {}
    heldout_identity = []
    for evaluation_seed in map(int, EVALUATION_SEEDS):
        heldout = b6._find_heldout(heldout_root, evaluation_seed)
        rows = load_heldout_v3_artifact(
            heldout,
            expected_domain=DOMAIN,
            expected_evaluation_seed=evaluation_seed,
            expected_count=2048,
        )[:POLICY_COUNT]
        actual_heldout_hash = _sha256(heldout)
        if actual_heldout_hash != expected_heldout_hashes[evaluation_seed]:
            raise RuntimeError(f"Phase2B8 frozen heldout hash drift: {evaluation_seed}")
        descriptors[evaluation_seed] = rows
        heldout_identity.append({"evaluation_seed": evaluation_seed, "path": str(heldout), "sha256": actual_heldout_hash})

    comparisons = []
    paired = {"COMMON_LEARNER": {}, "NATIVE_LEARNER": {}}
    pooled = {}
    local_fit_gates = []
    common_state_rows = []
    seed_a, seed_b = map(int, TRAINING_SEEDS)

    for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
        control_models = {}
        candidate_models = {}
        for seed in (seed_a, seed_b):
            control_models[seed], _ = b6._load_pilot_policy(
                b6_root / f"seed_{seed}" / "policies" / f"{mode}.pt",
                training_seed=seed,
                mode=mode,
            )
            candidate_models[seed], cp = _load_candidate_policy(
                output_root / f"seed_{seed}" / "policies" / f"{mode}.pt",
                training_seed=seed,
                mode=mode,
            )
            fit = dict(cp.get("fit") or {})
            local_fit_gates.append({
                "training_seed": seed,
                "learner_mode": mode,
                "policy_weighted_mean_tv": float(fit.get("policy_weighted_mean_tv", math.inf)),
                "gate_max": POLICY_TV_MAX,
                "gate_pass": bool(fit.get("policy_gate_pass")) and float(fit.get("policy_weighted_mean_tv", math.inf)) <= POLICY_TV_MAX,
            })

        control_means = []
        candidate_means = []
        for evaluation_seed in map(int, EVALUATION_SEEDS):
            desc = descriptors[evaluation_seed]
            ctl_left = b6._probabilities_fixed(control_models[seed_a], desc)
            ctl_right = b6._probabilities_fixed(control_models[seed_b], desc)
            can_left = b6._probabilities_fixed(candidate_models[seed_a], desc)
            can_right = b6._probabilities_fixed(candidate_models[seed_b], desc)
            ctl_metric = cross_seed_policy_stability(ctl_left, ctl_right)
            can_metric = cross_seed_policy_stability(can_left, can_right)
            ctl_tv = b6._tv_vector(ctl_left, ctl_right)
            can_tv = b6._tv_vector(can_left, can_right)
            paired[mode][str(evaluation_seed)] = [float(a - b) for a, b in zip(ctl_tv, can_tv)]
            control_means.append(float(ctl_metric["mean"]))
            candidate_means.append(float(can_metric["mean"]))
            comparisons.append({
                "learner_mode": mode,
                "evaluation_seed": evaluation_seed,
                "control_phase2b6": {"mean": float(ctl_metric["mean"]), "p95": float(ctl_metric["p95"])},
                "candidate_phase2b8": {
                    "mean": float(can_metric["mean"]),
                    "p95": float(can_metric["p95"]),
                    "hard_mean_gate_pass": bool(float(can_metric["mean"]) <= CROSS_SEED_MEAN_TV_MAX),
                    "hard_p95_gate_pass": bool(float(can_metric["p95"]) <= CROSS_SEED_P95_TV_MAX),
                },
                "mean_improvement": float(ctl_metric["mean"] - can_metric["mean"]),
                "p95_change_candidate_minus_control": float(can_metric["p95"] - ctl_metric["p95"]),
            })
            if mode == "COMMON_LEARNER":
                for item, cv, nv in zip(desc, ctl_tv, can_tv):
                    meta = b7._decode_observation(item.observation_v3)
                    common_state_rows.append({
                        "evaluation_seed": evaluation_seed,
                        "state_index": int(item.state_index),
                        "region": meta["region"],
                        "control_tv": float(cv),
                        "candidate_tv": float(nv),
                    })
        pooled[mode] = {
            "control_mean_tv": float(sum(control_means) / len(control_means)),
            "candidate_mean_tv": float(sum(candidate_means) / len(candidate_means)),
        }
        pooled[mode]["absolute_improvement"] = float(pooled[mode]["control_mean_tv"] - pooled[mode]["candidate_mean_tv"])
        pooled[mode]["relative_improvement"] = float(
            pooled[mode]["absolute_improvement"] / pooled[mode]["control_mean_tv"]
        )

    boot_common = equal_group_stratified_bootstrap_mean_ci(
        paired["COMMON_LEARNER"],
        seed_parts=("R7.5_ARCH_RESET", "PHASE2B8", "COMMON_LEARNER", "B6_MINUS_B8"),
        replicates=BOOTSTRAP_REPLICATES,
        confidence_level=0.95,
    )
    boot_native = equal_group_stratified_bootstrap_mean_ci(
        paired["NATIVE_LEARNER"],
        seed_parts=("R7.5_ARCH_RESET", "PHASE2B8", "NATIVE_LEARNER", "B6_MINUS_B8"),
        replicates=BOOTSTRAP_REPLICATES,
        confidence_level=0.95,
    )

    regions = _region_summary(common_state_rows)
    if abs(regions["PREFLOP_ROOT"]["control_mean_tv"] - B6_COMMON_ROOT_MEAN) > REPRO_TOL:
        raise RuntimeError("Phase2B8 Phase2B7 root control reproduction drift")
    if abs(regions["PREFLOP_CONTINUATION_COMBINED"]["control_mean_tv"] - B6_COMMON_CONTINUATION_MEAN) > REPRO_TOL:
        raise RuntimeError("Phase2B8 Phase2B7 continuation control reproduction drift")

    advantage_gates = []
    for seed in (seed_a, seed_b):
        advantage_gates.extend({"training_seed": seed, **x} for x in seed_results[seed]["advantage_gates"])
    all_advantage = bool(advantage_gates and all(x["gate_pass"] for x in advantage_gates))
    common_fit = [x for x in local_fit_gates if x["learner_mode"] == "COMMON_LEARNER"]
    native_fit = [x for x in local_fit_gates if x["learner_mode"] == "NATIVE_LEARNER"]
    all_common_fit = bool(common_fit and all(x["gate_pass"] for x in common_fit))
    all_native_fit = bool(native_fit and all(x["gate_pass"] for x in native_fit))

    common_rows = [x for x in comparisons if x["learner_mode"] == "COMMON_LEARNER"]
    both_common_improve = all(x["mean_improvement"] > 0.0 for x in common_rows)
    material = (
        pooled["COMMON_LEARNER"]["absolute_improvement"] >= CAUSAL_ABS_MIN
        or pooled["COMMON_LEARNER"]["relative_improvement"] >= CAUSAL_REL_MIN
    )
    ci_positive = float(boot_common["ci_low"]) > 0.0
    p95_ok = all(x["p95_change_candidate_minus_control"] <= COMMON_P95_MAX_DEGRADE for x in common_rows)
    native_ok = pooled["NATIVE_LEARNER"]["absolute_improvement"] >= -NATIVE_POOLED_MAX_DEGRADE
    root_ok = regions["PREFLOP_ROOT"]["candidate_mean_tv"] <= B6_COMMON_ROOT_MEAN + ROOT_MEAN_MAX_DEGRADE
    continuation_ok = (
        regions["PREFLOP_CONTINUATION_COMBINED"]["candidate_mean_tv"]
        <= B6_COMMON_CONTINUATION_MEAN + CONTINUATION_MEAN_MAX_DEGRADE
    )
    local_valid = all_advantage and all_common_fit and all_native_fit
    causal_supported = all([
        local_valid,
        equivalence["pass"],
        both_common_improve,
        material,
        ci_positive,
        p95_ok,
        native_ok,
        root_ok,
        continuation_ok,
    ])
    hard_stability = bool(common_rows and all(
        x["candidate_phase2b8"]["hard_mean_gate_pass"] and x["candidate_phase2b8"]["hard_p95_gate_pass"]
        for x in common_rows
    ))

    if not local_valid:
        status = "PHASE2B8_INVALID_LOCAL_GATES"
        next_route = "STOP_AND_AUDIT_LOCAL_TRAINING_VALIDITY"
    elif not causal_supported:
        status = "LAGGED_ANCHOR_EFFECT_NOT_SUPPORTED"
        next_route = "REASSESS_ROOT_ANCHOR_OR_DIRECT_TARGET_VARIANCE_REDUCTION"
    elif hard_stability:
        status = "LAGGED_ANCHOR_STABILITY_ELIGIBLE_PENDING_STRENGTH"
        next_route = "PRECOMMIT_STRATEGIC_STRENGTH_COMPARISON_VS_STABLE_V1_CONTROL"
    else:
        status = "LAGGED_ANCHOR_EFFECT_SUPPORTED_BUT_STILL_UNSTABLE"
        next_route = "LOCALIZE_RESIDUAL_AFTER_LAGGED_ANCHOR"

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
            "candidate": "LAGGED_BEHAVIOR_ANCHOR_025",
            "anchor_weight": ANCHOR_WEIGHT,
            "anchor_scope": "PREFLOP_CONTINUATION_AFTER_AT_LEAST_ONE_NONFORCED_PREFLOP_EVENT",
            "root_anchor": 0.0,
            "postflop_anchor": 0.0,
            "heldout_inference_anchor": 0.0,
            "iterations": ITERATIONS,
            "chunks_per_iteration": CHUNKS_PER_ITERATION,
            "roots_per_chunk": ROOTS_PER_CHUNK,
            "roots_per_seed": TOTAL_ROOTS,
            "exact_opponent_levels": EXACT_OPPONENT_LEVELS,
            "advantage_reservoir_capacity": RESERVOIR_CAPACITY,
            "strategy_reservoir_capacity": RESERVOIR_CAPACITY,
            "policy_steps": POLICY_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        },
        "frozen_inputs": {
            "phase2b6_result_sha256": PHASE2B6_RESULT_SHA256,
            "phase2b7_result_sha256": PHASE2B7_RESULT_SHA256,
            "heldout": heldout_identity,
        },
        "equivalence_before_divergence": equivalence,
        "seed_results": {
            str(seed): {
                "roots": seed_results[seed]["roots"],
                "advantage_memory": seed_results[seed]["advantage_memory"],
                "strategy_memory": seed_results[seed]["strategy_memory"],
                "anchor_policy_stats": seed_results[seed]["anchor_policy_stats"],
            }
            for seed in (seed_a, seed_b)
        },
        "local_validity": {
            "all_advantage_gates_pass": all_advantage,
            "all_common_policy_fit_gates_pass": all_common_fit,
            "all_native_policy_fit_gates_pass": all_native_fit,
            "valid": local_valid,
            "advantage_gates": advantage_gates,
            "policy_fit_gates": local_fit_gates,
        },
        "heldout_comparisons": comparisons,
        "pooled_mean_tv": pooled,
        "bootstrap": {
            "COMMON_LEARNER_phase2b6_minus_phase2b8": boot_common,
            "NATIVE_LEARNER_phase2b6_minus_phase2b8": boot_native,
        },
        "common_region_comparison": regions,
        "decision": {
            "equivalence_before_divergence_pass": equivalence["pass"],
            "both_common_evaluation_seed_means_improve": both_common_improve,
            "common_materiality_pass": material,
            "common_bootstrap_ci_strictly_positive": ci_positive,
            "common_p95_non_degradation_pass": p95_ok,
            "native_pooled_noncontradiction_pass": native_ok,
            "root_non_degradation_pass": root_ok,
            "continuation_non_degradation_pass": continuation_ok,
            "causal_effect_supported": causal_supported,
            "hard_stability_common_pass_both_heldouts": hard_stability,
            "classification": status,
            "next_route": next_route,
            "architecture_winner_selected": False,
            "production_training_authorized": False,
            "ready_for_tables": False,
        },
        "governance_scope": "Post-R7.5.3 architecture-reset small causal training screen only; old R7.5.3 remains closed.",
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
            "--phase2b6-root", str(Path(args.phase2b6_root).resolve()),
            "--phase2b6-result", str(Path(args.phase2b6_result).resolve()),
            "--phase2b7-result", str(Path(args.phase2b7_result).resolve()),
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
            if completed.returncode != 0:
                raise RuntimeError(f"Phase2B8 seed worker {seed} failed with exit code {completed.returncode}")
    result = _evaluate_parent(args)
    out = output_root / "R7_5_ARCH_RESET_V1PLUS_PHASE2B8_LAGGED_PREFLOP_ANCHOR.json"
    _atomic_json(result, out)
    print(json.dumps({
        "status": result["status"],
        "common_control_mean_tv": result["pooled_mean_tv"]["COMMON_LEARNER"]["control_mean_tv"],
        "common_candidate_mean_tv": result["pooled_mean_tv"]["COMMON_LEARNER"]["candidate_mean_tv"],
        "common_absolute_improvement": result["pooled_mean_tv"]["COMMON_LEARNER"]["absolute_improvement"],
        "common_bootstrap_ci": [
            result["bootstrap"]["COMMON_LEARNER_phase2b6_minus_phase2b8"]["ci_low"],
            result["bootstrap"]["COMMON_LEARNER_phase2b6_minus_phase2b8"]["ci_high"],
        ],
        "root_candidate_mean_tv": result["common_region_comparison"]["PREFLOP_ROOT"]["candidate_mean_tv"],
        "continuation_candidate_mean_tv": result["common_region_comparison"]["PREFLOP_CONTINUATION_COMBINED"]["candidate_mean_tv"],
        "causal_effect_supported": result["decision"]["causal_effect_supported"],
        "hard_stability_pass": result["decision"]["hard_stability_common_pass_both_heldouts"],
        "next_route": result["decision"]["next_route"],
        "result": str(out),
        "result_sha256": _sha256(out),
    }, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="R7.5 architecture-reset Phase2B8 lagged preflop anchor screen")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--phase2b6-root", type=Path, required=True)
    parser.add_argument("--phase2b6-result", type=Path, required=True)
    parser.add_argument("--phase2b7-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--seed-workers", type=int, default=2)
    parser.add_argument("--single-seed", type=int, choices=TRAINING_SEEDS)
    args = parser.parse_args()

    validate_phase2_v3_contract(
        args.repo_root.resolve(),
        representation=REPRESENTATION,
        domain=DOMAIN,
        training_seed=int(TRAINING_SEEDS[0]),
    )
    if ROOTS_PER_CHUNK != 64 or CHUNKS_PER_ITERATION != 4 or TOTAL_ROOTS != 768:
        raise RuntimeError("Phase2B8 training budget drift")
    if RESERVOIR_CAPACITY != 100000 or POLICY_STEPS != 16384 or BATCH_SIZE != 256 or LEARNING_RATE != 0.001:
        raise RuntimeError("Phase2B8 learner contract drift")
    _validate_b6_b7(Path(args.phase2b6_result).resolve(), Path(args.phase2b7_result).resolve())

    if args.single_seed is not None:
        return _run_single_seed(args, int(args.single_seed))

    b6._verify_all_roots_start_before_voluntary_action(args.repo_root.resolve(), args.solver.resolve())
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
