from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

import spincore.r7_5_representation_v3_stage as stage
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_representation_v3 import H2_FINAL
from spincore.r7_5_representation_v3_checkpoint import (
    RepresentationV3Progress,
    load_representation_v3_checkpoint,
    save_representation_v3_checkpoint,
)
from spincore.r7_5_representation_v3_fit import audit_v3_policy_model
from spincore.r7_5_representation_v3_phase2_eval import (
    cross_seed_policy_stability,
    equal_group_stratified_bootstrap_mean_ci,
)
from spincore.r7_5_representation_v3_referee_artifacts import load_heldout_v3_artifact
from spincore.r7_5_representation_v3_stage import frozen_config, new_phase2_v3_runtime, run_one_phase2_v3_iteration
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    ADVANTAGE_NRMSE_MAX,
    AUDIT_SIZE,
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
    ROOTS_PER_ITERATION,
    TORCH_THREADS,
    TRAINING_SEEDS,
    deck_seed,
    validate_phase2_v3_contract,
)
from spincore.r7_5_representation_v3_uncertainty import V3UncertaintyDampedPolicyMixture
from spincore.solver import SolverLibrary
from spincore_nn.models_v3_final import collate_v3_observations, make_h2_final_v3
from spincore_nn.reservoir import UniformReservoir
from spincore_nn.training import train_step

SCHEMA = "SPINCORE_R7_5_3D_V1PLUS_PHASE2A_STRATEGY_CAPACITY_RESULT_V1"
SEED_SCHEMA = "SPINCORE_R7_5_3D_V1PLUS_PHASE2A_SEED_RESULT_V1"
CHECKPOINT_EXTRA_SCHEMA = "SPINCORE_R7_5_3D_V1PLUS_PHASE2A_RESUME_V1"
DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL
CHUNKS_PER_ITERATION = 4
ROOTS_PER_CHUNK = ROOTS_PER_ITERATION
ROOTS_PER_ITERATION_EFFECTIVE = CHUNKS_PER_ITERATION * ROOTS_PER_CHUNK
TOTAL_ROOTS = ITERATIONS * ROOTS_PER_ITERATION_EFFECTIVE
CAPACITIES = {
    "S100K_CONTROL": 100_000,
    "S400K": 400_000,
    "S800K": 800_000,
}
COMMON_POLICY_INIT_SEED = 0x13579BDF
COMMON_BATCH_SEED = 0x2468ACE013579BDF
SHADOW_XOR = {
    "S400K": 0x40040040,
    "S800K": 0x80080080,
}
POLICY_COUNT = 1024


class _StrategyCapture:
    def __init__(self, control: UniformReservoir):
        self.control = control
        self.items = []

    def add(self, item) -> None:
        self.control.add(item)
        self.items.append(item)


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


def _make_behavior_from_states(states: list[dict], *, config) -> V3UncertaintyDampedPolicyMixture:
    behavior = V3UncertaintyDampedPolicyMixture(
        representation=REPRESENTATION,
        device="cpu",
        epsilon_scale=config.epsilon_scale,
        epsilon_cap=config.epsilon_cap,
    )
    models = []
    for index, state_dict in enumerate(states):
        _cfg, model = stage._make_v3_model(REPRESENTATION, 0x620000 + index)
        model.load_state_dict(state_dict)
        models.append(model)
    behavior.models = models
    return behavior


def _save_resume_checkpoint(
    path: Path,
    *,
    bundle,
    behavior,
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
            phase="phase2a_resume",
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
    if progress.phase != "phase2a_resume":
        raise RuntimeError("Phase2A resume checkpoint phase mismatch")
    if extra.get("schema") != CHECKPOINT_EXTRA_SCHEMA:
        raise RuntimeError("Phase2A resume checkpoint schema mismatch")
    if dict(extra.get("stage_config") or {}) != config.to_dict():
        raise RuntimeError("Phase2A resume stage-config drift")
    state = dict(extra.get("stage_state") or {})
    if int(progress.iteration) != int(state.get("completed_iteration", -1)):
        raise RuntimeError("Phase2A resume iteration mismatch")
    if int(progress.global_root) != int(state.get("global_root", -1)):
        raise RuntimeError("Phase2A resume global-root mismatch")
    behavior = _make_behavior_from_states(list(extra.get("behavior_model_states") or []), config=config)
    behavior.restore_stats(dict(extra.get("behavior_stats") or {}))
    session = stage._make_session(solver, bundle, spec, behavior)
    return bundle, session, behavior, state, int(extra.get("stage_index", -1)), dict(extra.get("last_stage_report") or {})


def _stage_coords(stage_index: int) -> tuple[int, int]:
    if not 1 <= int(stage_index) <= ITERATIONS * CHUNKS_PER_ITERATION:
        raise ValueError("Phase2A stage index outside frozen range")
    zero = int(stage_index) - 1
    return zero // CHUNKS_PER_ITERATION + 1, zero % CHUNKS_PER_ITERATION + 1


def _stream_path(seed_root: Path, stage_index: int) -> Path:
    iteration, chunk = _stage_coords(stage_index)
    return seed_root / "streams" / f"i{iteration}c{chunk}_strategy_stream.pt"


def _report_path(seed_root: Path, stage_index: int) -> Path:
    iteration, chunk = _stage_coords(stage_index)
    return seed_root / "stages" / f"i{iteration}c{chunk}.json"


def _validate_stream_prefix(seed_root: Path, stage_index: int) -> None:
    for index in range(1, int(stage_index) + 1):
        sp = _stream_path(seed_root, index)
        rp = _report_path(seed_root, index)
        if not sp.is_file() or not rp.is_file():
            raise RuntimeError(f"Phase2A completed-stage artifact missing at stage {index}: {sp} / {rp}")
        report = json.loads(rp.read_text(encoding="utf-8"))
        if int(report.get("stage_index", -1)) != index:
            raise RuntimeError("Phase2A stage-report identity mismatch")
        items = torch.load(sp, map_location="cpu", weights_only=False)
        if len(items) != int(report.get("strategy_stream_count", -1)):
            raise RuntimeError("Phase2A Strategy stream count mismatch")


def _collect_chunk(*, session, bundle, state: dict, target_iteration: int) -> tuple[dict, list]:
    scenarios = action_scenario_cycle(DOMAIN)
    scenario_counts = list(state["scenario_counts"])
    global_root = int(state["global_root"])
    session.collector.reset_telemetry()
    capture = _StrategyCapture(bundle.pol_mem)
    session.collector.strategy_memory = capture
    roots_before = int(bundle.counters["roots"])
    nodes_before = int(bundle.counters["nodes"])
    adv_before = int(bundle.adv_mem.seen)
    pol_before = int(bundle.pol_mem.seen)
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
    session.collector.strategy_memory = bundle.pol_mem
    state["global_root"] = global_root
    state["scenario_counts"] = scenario_counts
    report = {
        "roots": int(bundle.counters["roots"]) - roots_before,
        "nodes": int(bundle.counters["nodes"]) - nodes_before,
        "advantage_seen": int(bundle.adv_mem.seen) - adv_before,
        "strategy_seen": int(bundle.pol_mem.seen) - pol_before,
        "tree_collection_seconds": float(seconds),
        "branch_geometry": session.collector.telemetry_snapshot(),
    }
    if report["roots"] != ROOTS_PER_CHUNK:
        raise RuntimeError("Phase2A chunk did not collect exactly 64 roots")
    if report["strategy_seen"] != len(capture.items):
        raise RuntimeError("Phase2A captured Strategy stream is not identical to control add count")
    return report, capture.items


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


def _run_seed_trajectory(*, repo_root: Path, solver_path: Path, output_root: Path, execution_sha: str, training_seed: int) -> tuple[object, dict]:
    validate_phase2_v3_contract(
        repo_root,
        representation=REPRESENTATION,
        domain=DOMAIN,
        training_seed=int(training_seed),
    )
    torch.set_num_threads(TORCH_THREADS)
    if torch.get_num_threads() != TORCH_THREADS:
        raise RuntimeError("Phase2A torch-thread contract drift")
    solver = SolverLibrary(solver_path)
    base_config = frozen_config()
    fit_only = replace(base_config, roots_per_iteration=0)
    seed_root = output_root / f"seed_{int(training_seed)}"
    seed_root.mkdir(parents=True, exist_ok=True)
    resume = seed_root / "resume_checkpoint.pt"

    if resume.is_file():
        bundle, session, behavior, state, completed_stages, last_report = _load_resume_checkpoint(
            resume,
            repo_root=repo_root,
            solver=solver,
            training_seed=int(training_seed),
            config=base_config,
            execution_sha=str(execution_sha),
        )
        if not 0 <= completed_stages <= ITERATIONS * CHUNKS_PER_ITERATION:
            raise RuntimeError("Phase2A resume stage index invalid")
        _validate_stream_prefix(seed_root, completed_stages)
        if completed_stages and not _report_path(seed_root, completed_stages).is_file() and last_report:
            _atomic_json(last_report, _report_path(seed_root, completed_stages))
        print(f"[Phase2A resume] seed={training_seed} completed_stages={completed_stages}/12", flush=True)
    else:
        bundle, session, behavior, _spec, state = new_phase2_v3_runtime(
            repo_root,
            solver=solver,
            representation=REPRESENTATION,
            domain=DOMAIN,
            training_seed=int(training_seed),
            config=base_config,
        )
        state["phase2a"] = {
            "schema": CHECKPOINT_EXTRA_SCHEMA,
            "chance_coverage_multiplier": 4,
            "effective_roots_per_iteration": ROOTS_PER_ITERATION_EFFECTIVE,
            "strategy_stream_capture": True,
        }
        completed_stages = 0

    for stage_index in range(completed_stages + 1, ITERATIONS * CHUNKS_PER_ITERATION + 1):
        iteration, chunk = _stage_coords(stage_index)
        if chunk == 1:
            if int(state["completed_iteration"]) != iteration - 1:
                raise RuntimeError("Phase2A iteration-start identity drift")
            state["phase2a_pending_iteration"] = {
                "iteration": iteration,
                "roots_before": int(bundle.counters["roots"]),
                "nodes_before": int(bundle.counters["nodes"]),
                "advantage_seen_before": int(bundle.adv_mem.seen),
                "strategy_seen_before": int(bundle.pol_mem.seen),
                "chunks": [],
            }
        pending = dict(state.get("phase2a_pending_iteration") or {})
        if int(pending.get("iteration", -1)) != iteration:
            raise RuntimeError("Phase2A missing pending-iteration state")
        chunks = list(pending.get("chunks") or [])
        if len(chunks) != chunk - 1:
            raise RuntimeError("Phase2A chunk history length drift")

        print(f"[Phase2A run] seed={training_seed} i{iteration}c{chunk}", flush=True)
        chunk_report, stream_items = _collect_chunk(
            session=session,
            bundle=bundle,
            state=state,
            target_iteration=iteration,
        )
        chunks.append(chunk_report)
        pending["chunks"] = chunks
        state["phase2a_pending_iteration"] = pending

        iteration_report = None
        if chunk == CHUNKS_PER_ITERATION:
            iteration_report = run_one_phase2_v3_iteration(
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
                "chance_coverage_chunks": chunks,
            })
            if roots_added != ROOTS_PER_ITERATION_EFFECTIVE:
                raise RuntimeError("Phase2A x4 iteration root total drift")
            state["iteration_reports"][-1] = patched
            state["tree_collection_seconds_total"] = float(state["tree_collection_seconds_total"]) + tree_seconds
            state.pop("phase2a_pending_iteration", None)
            iteration_report = patched

        stream_path = _stream_path(seed_root, stage_index)
        _atomic_torch_save(stream_items, stream_path)
        stage_report = {
            "schema": CHECKPOINT_EXTRA_SCHEMA,
            "stage_index": stage_index,
            "iteration": iteration,
            "root_chunk": chunk,
            "training_seed": int(training_seed),
            "roots_total": int(bundle.counters["roots"]),
            "strategy_stream_count": len(stream_items),
            "chunk_report": chunk_report,
            "iteration_completed": bool(chunk == CHUNKS_PER_ITERATION),
            "iteration_report": iteration_report,
            "execution_sha": str(execution_sha),
        }
        _save_resume_checkpoint(
            resume,
            bundle=bundle,
            behavior=behavior,
            state=state,
            config=base_config,
            execution_sha=str(execution_sha),
            stage_index=stage_index,
            last_stage_report=stage_report,
        )
        _atomic_json(stage_report, _report_path(seed_root, stage_index))
        print(f"[Phase2A stage complete] seed={training_seed} i{iteration}c{chunk} strategy={len(stream_items)}", flush=True)

    if int(bundle.counters["roots"]) != TOTAL_ROOTS:
        raise RuntimeError("Phase2A final root count drift")
    if int(state["completed_iteration"]) != ITERATIONS:
        raise RuntimeError("Phase2A final iteration count drift")
    return bundle, state


def _replay_strategy_arms(*, seed_root: Path, training_seed: int, control: UniformReservoir):
    replay_control = UniformReservoir(CAPACITIES["S100K_CONTROL"], int(training_seed) ^ 0x5A5A5A5A)
    shadow400 = UniformReservoir(CAPACITIES["S400K"], int(training_seed) ^ SHADOW_XOR["S400K"])
    shadow800 = UniformReservoir(CAPACITIES["S800K"], int(training_seed) ^ SHADOW_XOR["S800K"])
    total = 0
    for stage_index in range(1, ITERATIONS * CHUNKS_PER_ITERATION + 1):
        items = torch.load(_stream_path(seed_root, stage_index), map_location="cpu", weights_only=False)
        for item in items:
            replay_control.add(item)
            shadow400.add(item)
            shadow800.add(item)
        total += len(items)
        del items
        print(f"[Phase2A replay] seed={training_seed} stage={stage_index}/12 seen={total}", flush=True)
    a = replay_control.state_dict()
    b = control.state_dict()
    if int(a["seen"]) != int(b["seen"]) or int(a["capacity"]) != int(b["capacity"]):
        raise RuntimeError("Phase2A control replay count/capacity mismatch")
    if a["rng_state"] != b["rng_state"]:
        raise RuntimeError("Phase2A control replay RNG mismatch")
    if a["items"] != b["items"]:
        raise RuntimeError("Phase2A control replay retained-item mismatch")
    if int(total) != int(control.seen):
        raise RuntimeError("Phase2A captured stream total differs from authoritative control seen")
    return {
        "S100K_CONTROL": control,
        "S400K": shadow400,
        "S800K": shadow800,
    }, {
        "stream_samples": int(total),
        "control_replay_exact": True,
    }


def _fit_policy(memory: UniformReservoir, *, init_seed: int, rng: random.Random, audit_seed: int):
    _cfg, model = make_h2_final_v3(device="cpu", seed=int(init_seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses = []
    count = min(BATCH_SIZE, len(memory.items))
    if count <= 0:
        raise RuntimeError("Phase2A cannot fit empty Strategy memory")
    started = time.perf_counter()
    for _ in range(POLICY_STEPS):
        samples = rng.sample(memory.items, count)
        batch = collate_v3_observations(
            [sample.observation for sample in samples],
            [sample.legal for sample in samples],
            with_semantics=False,
            device="cpu",
        )
        target = torch.tensor([sample.target for sample in samples], dtype=torch.float32)
        weights = torch.tensor([sample.weight for sample in samples], dtype=torch.float32)
        losses.append(train_step(model, optimizer, batch, target, weights, "strategy"))
    seconds = time.perf_counter() - started
    audit = audit_v3_policy_model(
        model,
        memory.items,
        representation=REPRESENTATION,
        sample_size=AUDIT_SIZE,
        seed=int(audit_seed),
    )
    return model, {
        "init_seed": int(init_seed),
        "steps": POLICY_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "fit_seconds": float(seconds),
        "mean_loss": float(sum(losses) / len(losses)),
        "final_loss": float(losses[-1]),
        "policy_weighted_mean_tv": float(audit),
        "policy_tv_max": POLICY_TV_MAX,
        "policy_gate_pass": bool(float(audit) <= POLICY_TV_MAX),
        "memory_capacity": int(memory.capacity),
        "memory_seen": int(memory.seen),
        "memory_retained": len(memory.items),
        "retention_fraction": float(len(memory.items) / memory.seen) if memory.seen else 0.0,
    }


def _fit_seed_policies(*, seed_root: Path, training_seed: int, bundle, arms: dict[str, UniformReservoir]):
    policy_root = seed_root / "policies"
    policy_root.mkdir(parents=True, exist_ok=True)
    native_state = bundle.batch_rng.getstate()
    rows = {}
    for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
        for arm_name in CAPACITIES:
            key = f"{mode}__{arm_name}"
            artifact = policy_root / f"{key}.pt"
            meta = policy_root / f"{key}.json"
            if artifact.is_file() and meta.is_file():
                saved = json.loads(meta.read_text(encoding="utf-8"))
                if saved.get("status") == "POLICY_FIT_COMPLETE" and int(saved.get("training_seed", -1)) == int(training_seed):
                    rows[key] = saved
                    print(f"[Phase2A policy resume] seed={training_seed} {key}", flush=True)
                    continue
            if mode == "COMMON_LEARNER":
                init_seed = COMMON_POLICY_INIT_SEED
                rng = random.Random(COMMON_BATCH_SEED)
            else:
                init_seed = (int(training_seed) ^ 0x5DEECE66D) & 0x7FFFFFFF
                rng = random.Random()
                rng.setstate(native_state)
            audit_seed = (int(training_seed) ^ 0x0A11D17 ^ CAPACITIES[arm_name] ^ (0 if mode == "COMMON_LEARNER" else 0x13579)) & ((1 << 64) - 1)
            print(f"[Phase2A policy fit] seed={training_seed} {key}", flush=True)
            model, fit = _fit_policy(arms[arm_name], init_seed=init_seed, rng=rng, audit_seed=audit_seed)
            payload = {
                "schema": SEED_SCHEMA,
                "status": "POLICY_FIT_COMPLETE",
                "representation": REPRESENTATION,
                "domain": DOMAIN,
                "training_seed": int(training_seed),
                "learner_mode": mode,
                "arm": arm_name,
                "capacity": CAPACITIES[arm_name],
                "model_state": model.state_dict(),
                "fit": fit,
            }
            _atomic_torch_save(payload, artifact)
            saved = {
                "schema": SEED_SCHEMA,
                "status": "POLICY_FIT_COMPLETE",
                "training_seed": int(training_seed),
                "learner_mode": mode,
                "arm": arm_name,
                "capacity": CAPACITIES[arm_name],
                "artifact": str(artifact),
                "fit": fit,
            }
            _atomic_json(saved, meta)
            rows[key] = saved
    return rows


def _run_single_seed(args, training_seed: int) -> int:
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    seed_root = output_root / f"seed_{int(training_seed)}"
    result_path = seed_root / "seed_result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "SEED_COMPLETE" and existing.get("execution_sha") == str(args.execution_sha):
            print(f"[Phase2A seed resume] seed={training_seed} already complete", flush=True)
            return 0
    bundle, state = _run_seed_trajectory(
        repo_root=repo_root,
        solver_path=Path(args.solver).resolve(),
        output_root=output_root,
        execution_sha=str(args.execution_sha),
        training_seed=int(training_seed),
    )
    arms, replay = _replay_strategy_arms(seed_root=seed_root, training_seed=int(training_seed), control=bundle.pol_mem)
    policy_rows = _fit_seed_policies(seed_root=seed_root, training_seed=int(training_seed), bundle=bundle, arms=arms)
    advantage_rows = []
    for row in list(state.get("iteration_reports") or []):
        value = float(row.get("ensemble_weighted_nrmse", math.inf))
        advantage_rows.append({
            "iteration": int(row.get("iteration", -1)),
            "ensemble_weighted_nrmse": value,
            "gate_max": ADVANTAGE_NRMSE_MAX,
            "gate_pass": bool(value <= ADVANTAGE_NRMSE_MAX and bool(row.get("ensemble_advantage_gate_pass"))),
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
        "advantage_gates": advantage_rows,
        "all_advantage_gates_pass": bool(len(advantage_rows) == ITERATIONS and all(row["gate_pass"] for row in advantage_rows)),
        "strategy_replay": replay,
        "strategy_memories": {
            name: {
                "capacity": int(mem.capacity),
                "seen": int(mem.seen),
                "retained": len(mem.items),
                "retention_fraction": float(len(mem.items) / mem.seen) if mem.seen else 0.0,
            }
            for name, mem in arms.items()
        },
        "policy_fits": policy_rows,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    _atomic_json(result, result_path)
    print(json.dumps({"status": result["status"], "training_seed": int(training_seed), "strategy_seen": int(bundle.pol_mem.seen), "advantage_pass": result["all_advantage_gates_pass"]}, indent=2), flush=True)
    return 0


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
        raise RuntimeError(f"Phase2A heldout identity mismatch for {evaluation_seed}: {matches}")
    return matches[0]


def _load_policy(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SEED_SCHEMA or payload.get("status") != "POLICY_FIT_COMPLETE":
        raise RuntimeError("Phase2A policy artifact schema/status mismatch")
    _cfg, model = make_h2_final_v3(device="cpu", seed=0)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def _probabilities(model, descriptors) -> list[list[float]]:
    batch = collate_v3_observations(
        [item.observation_v3 for item in descriptors],
        [item.legal_slots for item in descriptors],
        with_semantics=False,
        device="cpu",
    )
    with torch.no_grad():
        logits = model(batch).masked_fill(~batch["legal"], -1e9)
        probs = torch.softmax(logits, dim=-1).cpu()
    return [[float(x) for x in row] for row in probs.tolist()]


def _tv_vector(left, right) -> list[float]:
    a = torch.tensor(left, dtype=torch.float64)
    b = torch.tensor(right, dtype=torch.float64)
    return [float(x) for x in (0.5 * torch.abs(a - b).sum(dim=1)).tolist()]


def _curve_coherent(pooled: dict, boot: dict) -> tuple[bool, dict]:
    a = float(pooled["S100K_CONTROL"])
    b = float(pooled["S400K"])
    c = float(pooled["S800K"])
    ideal = a >= b >= c
    details = {"ideal_monotone": ideal, "reversals": []}
    if ideal:
        return True, details
    coherent = True
    if b > a:
        reversal = b - a
        ci = boot["S100K_CONTROL_TO_S400K"]
        ok = reversal <= 0.005 and float(ci["ci_low"]) <= 0.0 <= float(ci["ci_high"])
        details["reversals"].append({"adjacent": "100K_400K", "absolute": reversal, "ci_includes_zero": float(ci["ci_low"]) <= 0.0 <= float(ci["ci_high"]), "accepted": ok})
        coherent = coherent and ok
    if c > b:
        reversal = c - b
        ci = boot["S400K_TO_S800K"]
        ok = reversal <= 0.005 and float(ci["ci_low"]) <= 0.0 <= float(ci["ci_high"])
        details["reversals"].append({"adjacent": "400K_800K", "absolute": reversal, "ci_includes_zero": float(ci["ci_low"]) <= 0.0 <= float(ci["ci_high"]), "accepted": ok})
        coherent = coherent and ok
    return bool(coherent), details


def _evaluate_parent(args) -> dict:
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    heldout_root = Path(args.heldout_root).resolve()
    torch.set_num_threads(TORCH_THREADS)
    seed_a, seed_b = map(int, TRAINING_SEEDS)
    seed_results = {}
    for seed in (seed_a, seed_b):
        path = output_root / f"seed_{seed}" / "seed_result.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "SEED_COMPLETE" or payload.get("execution_sha") != str(args.execution_sha):
            raise RuntimeError(f"Phase2A seed result invalid for {seed}")
        seed_results[seed] = payload

    descriptors = {}
    for evaluation_seed in EVALUATION_SEEDS:
        heldout = _find_heldout(heldout_root, int(evaluation_seed))
        descriptors[int(evaluation_seed)] = load_heldout_v3_artifact(
            heldout,
            expected_domain=DOMAIN,
            expected_evaluation_seed=int(evaluation_seed),
            expected_count=2048,
        )[:POLICY_COUNT]

    metrics = []
    tv_by_key = {}
    fit_gates = []
    for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
        for arm in CAPACITIES:
            models = {}
            for seed in (seed_a, seed_b):
                artifact = output_root / f"seed_{seed}" / "policies" / f"{mode}__{arm}.pt"
                model, payload = _load_policy(artifact)
                models[seed] = model
                fit = dict(payload["fit"])
                fit_gates.append({
                    "training_seed": seed,
                    "learner_mode": mode,
                    "arm": arm,
                    "policy_weighted_mean_tv": float(fit["policy_weighted_mean_tv"]),
                    "policy_tv_max": POLICY_TV_MAX,
                    "gate_pass": bool(fit["policy_gate_pass"]),
                })
            for evaluation_seed in EVALUATION_SEEDS:
                desc = descriptors[int(evaluation_seed)]
                left = _probabilities(models[seed_a], desc)
                right = _probabilities(models[seed_b], desc)
                metric = cross_seed_policy_stability(left, right)
                tv = _tv_vector(left, right)
                key = (mode, arm, int(evaluation_seed))
                tv_by_key[key] = tv
                metrics.append({
                    "learner_mode": mode,
                    "arm": arm,
                    "capacity": CAPACITIES[arm],
                    "evaluation_seed": int(evaluation_seed),
                    "training_seed_pair": [seed_a, seed_b],
                    "metric": metric,
                    "state_tv": tv,
                })

    pooled = {}
    for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
        pooled[mode] = {}
        for arm in CAPACITIES:
            means = []
            for evaluation_seed in EVALUATION_SEEDS:
                row = next(r for r in metrics if r["learner_mode"] == mode and r["arm"] == arm and r["evaluation_seed"] == int(evaluation_seed))
                means.append(float(row["metric"]["mean"]))
            pooled[mode][arm] = float(sum(means) / len(means))

    boot = {}
    contrasts = [
        ("COMMON_LEARNER", "S100K_CONTROL", "S400K", "S100K_CONTROL_TO_S400K"),
        ("COMMON_LEARNER", "S400K", "S800K", "S400K_TO_S800K"),
        ("COMMON_LEARNER", "S100K_CONTROL", "S800K", "S100K_CONTROL_TO_S800K"),
        ("NATIVE_LEARNER", "S100K_CONTROL", "S800K", "NATIVE_S100K_CONTROL_TO_S800K"),
    ]
    for mode, smaller, larger, name in contrasts:
        groups = {}
        for evaluation_seed in EVALUATION_SEEDS:
            left = tv_by_key[(mode, smaller, int(evaluation_seed))]
            right = tv_by_key[(mode, larger, int(evaluation_seed))]
            groups[str(int(evaluation_seed))] = [float(a - b) for a, b in zip(left, right)]
        boot[name] = equal_group_stratified_bootstrap_mean_ci(
            groups,
            seed_parts=("R7.5.3D", "PHASE2A", mode, smaller, larger),
            replicates=2000,
            confidence_level=0.95,
        )

    all_advantage = all(bool(seed_results[seed]["all_advantage_gates_pass"]) for seed in (seed_a, seed_b))
    common_fit_rows = [row for row in fit_gates if row["learner_mode"] == "COMMON_LEARNER"]
    all_common_fits = bool(common_fit_rows and all(bool(row["gate_pass"]) for row in common_fit_rows))
    common_100 = pooled["COMMON_LEARNER"]["S100K_CONTROL"]
    common_400 = pooled["COMMON_LEARNER"]["S400K"]
    common_800 = pooled["COMMON_LEARNER"]["S800K"]
    native_100 = pooled["NATIVE_LEARNER"]["S100K_CONTROL"]
    native_800 = pooled["NATIVE_LEARNER"]["S800K"]
    common_improvement = common_100 - common_800
    relative_improvement = common_improvement / common_100 if common_100 > 0.0 else -math.inf
    ci_positive = float(boot["S100K_CONTROL_TO_S800K"]["ci_low"]) > 0.0
    same_direction = True
    individual = []
    for evaluation_seed in EVALUATION_SEEDS:
        m100 = next(float(r["metric"]["mean"]) for r in metrics if r["learner_mode"] == "COMMON_LEARNER" and r["arm"] == "S100K_CONTROL" and r["evaluation_seed"] == int(evaluation_seed))
        m800 = next(float(r["metric"]["mean"]) for r in metrics if r["learner_mode"] == "COMMON_LEARNER" and r["arm"] == "S800K" and r["evaluation_seed"] == int(evaluation_seed))
        delta = m100 - m800
        row = {"evaluation_seed": int(evaluation_seed), "mean_100k": m100, "mean_800k": m800, "improvement": delta, "degradation": max(0.0, -delta)}
        individual.append(row)
        same_direction = same_direction and (m800 <= m100) and (row["degradation"] <= 0.01)
    curve_ok, curve_detail = _curve_coherent(pooled["COMMON_LEARNER"], boot)
    native_direction = (native_100 - native_800) >= 0.0
    statistically_supported = bool(ci_positive and same_direction and curve_ok and native_direction)
    materially_large = bool(common_improvement >= 0.02 or relative_improvement >= 0.10)

    if not (all_advantage and all_common_fits):
        status = "PHASE2A_INVALID_LOCAL_GATES"
    elif not statistically_supported:
        status = "CAPACITY_EFFECT_NOT_SUPPORTED"
    elif not materially_large:
        status = "CAPACITY_EFFECT_REAL_BUT_INSUFFICIENT"
    else:
        status = "CAPACITY_EFFECT_MATERIALLY_SUPPORTED"

    result = {
        "schema": SCHEMA,
        "status": status,
        "purpose": "Causal H2/3H x4 Strategy-memory-capacity ablation; no architecture selection.",
        "execution_sha": str(args.execution_sha),
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "training_seeds": [seed_a, seed_b],
        "evaluation_seeds": list(map(int, EVALUATION_SEEDS)),
        "chance_coverage": {"multiplier": 4, "roots_per_iteration": ROOTS_PER_ITERATION_EFFECTIVE, "iterations": ITERATIONS, "roots_per_seed": TOTAL_ROOTS},
        "capacities": CAPACITIES,
        "local_advantage_gates_pass": all_advantage,
        "common_policy_fit_gates_pass": all_common_fits,
        "policy_fit_gates": fit_gates,
        "cross_seed_rows": metrics,
        "pooled_mean_tv": pooled,
        "paired_bootstrap": boot,
        "decision": {
            "common_100k_mean_tv": common_100,
            "common_400k_mean_tv": common_400,
            "common_800k_mean_tv": common_800,
            "common_100k_to_800k_absolute_improvement": common_improvement,
            "common_100k_to_800k_relative_improvement": relative_improvement,
            "bootstrap_ci_strictly_positive": ci_positive,
            "individual_eval_direction": individual,
            "curve_coherent": curve_ok,
            "curve_detail": curve_detail,
            "native_100k_mean_tv": native_100,
            "native_800k_mean_tv": native_800,
            "native_direction_nonnegative": native_direction,
            "statistically_supported": statistically_supported,
            "practical_materiality": materially_large,
            "hard_gate_mean_tv_max": CROSS_SEED_MEAN_TV_MAX,
            "hard_gate_p95_tv_max": CROSS_SEED_P95_TV_MAX,
        },
        "seed_results": {str(seed): seed_results[seed] for seed in (seed_a, seed_b)},
        "representation_winner": None,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    return result


def _run_parent(args) -> int:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    commands = []
    for seed in TRAINING_SEEDS:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--repo-root", str(Path(args.repo_root).resolve()),
            "--solver", str(Path(args.solver).resolve()),
            "--heldout-root", str(Path(args.heldout_root).resolve()),
            "--output-root", str(output_root),
            "--execution-sha", str(args.execution_sha),
            "--single-seed", str(int(seed)),
        ]
        commands.append((int(seed), cmd))
    with ThreadPoolExecutor(max_workers=min(int(args.seed_workers), len(commands))) as pool:
        futures = {pool.submit(subprocess.run, cmd, check=False): seed for seed, cmd in commands}
        for future in as_completed(futures):
            seed = futures[future]
            completed = future.result()
            if int(completed.returncode) != 0:
                raise RuntimeError(f"Phase2A seed worker {seed} failed with exit code {completed.returncode}")
    result = _evaluate_parent(args)
    out = output_root / "R7_5_3D_V1PLUS_PHASE2A_RESULT.json"
    _atomic_json(result, out)
    print(json.dumps({
        "status": result["status"],
        "common_mean_tv": result["pooled_mean_tv"]["COMMON_LEARNER"],
        "native_mean_tv": result["pooled_mean_tv"]["NATIVE_LEARNER"],
        "absolute_improvement_100k_to_800k": result["decision"]["common_100k_to_800k_absolute_improvement"],
        "relative_improvement_100k_to_800k": result["decision"]["common_100k_to_800k_relative_improvement"],
        "result": str(out),
    }, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="R7.5.3D V1+ Phase2A Strategy-memory capacity causal ablation")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--single-seed", type=int, choices=TRAINING_SEEDS)
    parser.add_argument("--seed-workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if not str(args.execution_sha).strip():
        raise SystemExit("--execution-sha is required")
    if args.single_seed is not None:
        return _run_single_seed(args, int(args.single_seed))
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
