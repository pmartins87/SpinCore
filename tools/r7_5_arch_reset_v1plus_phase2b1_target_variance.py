from __future__ import annotations

"""Phase2B1 read-only decomposition of H2/3H Advantage target variance.

This diagnostic performs fresh solver traversals under already-trained Phase2A
behavior ensembles, but it never trains a model and never writes to a training
reservoir.  Exact root SPNNIV3 collisions are used to hold the acting player's
observable infoset fixed while hidden/future chance is varied.
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
import time
from typing import Sequence

import numpy as np
import torch

from spincore.deep_cfr import icm_delta_utility
from spincore.r7_5_action_cfr import legal_mask, regret_matching_policy
from spincore.r7_5_action_scenarios import action_scenario_cycle, scenario_descriptor
from spincore.r7_5_representation_v3 import H2_FINAL, UniversalPartialExactCollectorV3
from spincore.r7_5_representation_v3_checkpoint import SCHEMA as CHECKPOINT_SCHEMA
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    ENSEMBLE_SIZE,
    EPSILON_CAP,
    EPSILON_SCALE,
    EXACT_OPPONENT_LEVELS,
    MODEL_FINGERPRINTS,
    PAYOUT,
    TRAINING_SEEDS,
    validate_phase2_v3_contract,
)
from spincore.r7_5_representation_v3_uncertainty import V3UncertaintyDampedPolicyMixture
from spincore.solver import SolverLibrary
from spincore.solver_v3 import neural_bytes_v3
from spincore_nn.models_v3_final import make_h2_final_v3

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B1_TARGET_VARIANCE_V1"
DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL
SOURCE_EXECUTION_SHA = "4bfa55d69029cd69536fa6dbfcadd162719cb887"
PHASE2A_EXTRA_SCHEMA = "SPINCORE_R7_5_3D_V1PLUS_PHASE2A_RESUME_V1"
EXPECTED_ROOTS = 768
EXPECTED_STAGE_INDEX = 12
REPLICATES = 16
K_VALUES = (1, 2, 4, 8)
ARMS = ("TRAVERSAL_ONLY", "CHANCE_ONLY", "COMBINED")
COLLISION_SEARCH_BUDGET = 50_000
TARGET_ITERATION = 3
MAX_WORKERS = 12
MASK64 = (1 << 64) - 1

_WORKER_SOLVER = None
_WORKER_COLLECTOR = None


class _Sink:
    def __init__(self):
        self.items = []

    def add(self, item) -> None:
        self.items.append(item)


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), float(q), method="linear"))


def _summary(values: Sequence[float]) -> dict:
    rows = [float(value) for value in values]
    if not rows:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    arr = np.asarray(rows, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": _quantile(rows, 0.50),
        "p95": _quantile(rows, 0.95),
        "max": float(arr.max()),
    }


def _policy_tv(left: Sequence[float], right: Sequence[float]) -> float:
    return float(0.5 * sum(abs(float(a) - float(b)) for a, b in zip(left, right)))


def _mean_targets(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not rows:
        raise ValueError("cannot average empty target set")
    if any(len(row) != 10 for row in rows):
        raise ValueError("Phase2B1 requires ten-slot targets")
    return tuple(float(sum(float(row[slot]) for row in rows) / len(rows)) for slot in range(10))


def k_pair_metrics(targets: Sequence[Sequence[float]], legal_mask_row: Sequence[int], k: int) -> list[dict]:
    """Deterministic non-overlapping left/right K-aggregate comparisons."""
    rows = [tuple(float(value) for value in row) for row in targets]
    if len(rows) != REPLICATES:
        raise ValueError(f"Phase2B1 requires exactly {REPLICATES} target replicates")
    if int(k) not in K_VALUES or REPLICATES % (2 * int(k)) != 0:
        raise ValueError("invalid frozen K")
    legal = tuple(index for index, enabled in enumerate(legal_mask_row) if int(enabled))
    if not legal:
        raise ValueError("empty legal mask")
    out = []
    width = 2 * int(k)
    for start in range(0, REPLICATES, width):
        left = _mean_targets(rows[start : start + int(k)])
        right = _mean_targets(rows[start + int(k) : start + width])
        target_mad = float(sum(abs(left[slot] - right[slot]) for slot in legal) / len(legal))
        sign_disagreement = float(
            sum((left[slot] > 0.0) != (right[slot] > 0.0) for slot in legal) / len(legal)
        )
        left_policy = regret_matching_policy(left, legal)
        right_policy = regret_matching_policy(right, legal)
        left_dom = max(legal, key=lambda slot: float(left_policy[slot]))
        right_dom = max(legal, key=lambda slot: float(right_policy[slot]))
        out.append({
            "k": int(k),
            "pair_start": int(start),
            "target_mean_abs_diff": target_mad,
            "legal_sign_disagreement_fraction": sign_disagreement,
            "regret_matching_policy_tv": _policy_tv(left_policy, right_policy),
            "dominant_legal_action_mismatch": int(left_dom != right_dom),
        })
    return out


def _deck_seed(scenario_index: int, attempt: int) -> int:
    # Diagnostic-only namespace, independent of frozen training/evaluation seeds.
    return (
        0xB1C0111D5EED0001
        + int(scenario_index) * 0x9E3779B97F4A7C15
        + int(attempt) * 0xD1B54A32D192ED03
    ) & MASK64


def _traversal_seed(scenario_index: int, replicate: int) -> int:
    return (
        0x7A4E125A4D500001
        ^ (int(scenario_index) * 0x94D049BB133111EB)
        ^ (int(replicate) * 0xBF58476D1CE4E5B9)
    ) & MASK64


def _root_identity(root, action_spec) -> tuple[bytes, int, tuple[int, ...], tuple[int, ...]]:
    observation = neural_bytes_v3(root)
    actor = int(root.actor)
    payload_v2 = root.neural_bytes_v2()
    if len(payload_v2) != 830 or not payload_v2.startswith(b"SPNNIV2\x00"):
        raise RuntimeError("Phase2B1 requires authoritative SPNNIV2 street metadata")
    street = int(payload_v2[112])
    active_mask = int(action_spec.active_mask(street))
    legal = tuple(int(value) for value in root.universal_legal_actions(active_mask))
    if not legal:
        raise RuntimeError("Phase2B1 root has no legal universal actions")
    return observation, actor, legal, legal_mask(legal)


def discover_collision_groups(repo_root: Path, solver_path: Path) -> list[dict]:
    contract = validate_phase2_v3_contract(
        repo_root,
        representation=REPRESENTATION,
        domain=DOMAIN,
        training_seed=int(TRAINING_SEEDS[0]),
    )
    action_spec = contract["action_spec"]
    solver = SolverLibrary(solver_path)
    groups = []
    scenarios = action_scenario_cycle(DOMAIN)
    for scenario_index, episode in enumerate(scenarios):
        buckets: dict[bytes, dict] = {}
        selected = None
        for attempt in range(COLLISION_SEARCH_BUDGET):
            seed = _deck_seed(scenario_index, attempt)
            root = solver.create(episode, seed)
            try:
                observation, actor, legal, mask = _root_identity(root, action_spec)
            finally:
                root.close()
            digest = hashlib.sha256(observation).digest()
            bucket = buckets.get(digest)
            if bucket is None:
                bucket = {
                    "observation": observation,
                    "actor": actor,
                    "legal": legal,
                    "legal_mask": mask,
                    "deck_seeds": [],
                    "first_attempt": attempt,
                }
                buckets[digest] = bucket
            elif (
                bucket["observation"] != observation
                or int(bucket["actor"]) != actor
                or tuple(bucket["legal"]) != legal
            ):
                raise RuntimeError("SHA-256 root-identity collision or legal/actor inconsistency")
            if seed not in bucket["deck_seeds"]:
                bucket["deck_seeds"].append(int(seed))
            if len(bucket["deck_seeds"]) >= REPLICATES:
                selected = bucket
                selected["attempts_used"] = attempt + 1
                break
        if selected is None:
            raise RuntimeError(
                f"Phase2B1 scenario {scenario_index} failed to find a {REPLICATES}-deck exact-root collision "
                f"within {COLLISION_SEARCH_BUDGET} attempts"
            )
        groups.append({
            "scenario_index": int(scenario_index),
            "scenario": scenario_descriptor(episode),
            "actor": int(selected["actor"]),
            "legal": [int(value) for value in selected["legal"]],
            "legal_mask": [int(value) for value in selected["legal_mask"]],
            "observation_sha256": hashlib.sha256(selected["observation"]).hexdigest(),
            "observation": selected["observation"],
            "deck_seeds": [int(value) for value in selected["deck_seeds"][:REPLICATES]],
            "attempts_used": int(selected["attempts_used"]),
        })
    if len(groups) != 15:
        raise RuntimeError(f"Phase2B1 expected 15 THREE_HANDED scenarios, got {len(groups)}")
    return groups


def _load_behavior(checkpoint: Path, source_sha: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    seed = int(payload.get("seed", -1))
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "action_candidate": ACTION_CANDIDATE,
        "execution_sha": str(source_sha),
        "architecture_fingerprint_sha256": MODEL_FINGERPRINTS[REPRESENTATION],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Phase2B1 checkpoint identity mismatch {seed}/{key}")
    if seed not in tuple(map(int, TRAINING_SEEDS)):
        raise RuntimeError("Phase2B1 checkpoint seed is not frozen")
    progress = dict(payload.get("progress") or {})
    extra = dict(payload.get("extra") or {})
    if progress.get("phase") != "phase2a_resume" or int(progress.get("global_root", -1)) != EXPECTED_ROOTS:
        raise RuntimeError(f"Phase2B1 incomplete Phase2A checkpoint for seed {seed}")
    if extra.get("schema") != PHASE2A_EXTRA_SCHEMA or int(extra.get("stage_index", -1)) != EXPECTED_STAGE_INDEX:
        raise RuntimeError(f"Phase2B1 checkpoint extra mismatch for seed {seed}")
    states = list(extra.get("behavior_model_states") or [])
    if len(states) != ENSEMBLE_SIZE:
        raise RuntimeError(f"Phase2B1 expected four behavior model states for seed {seed}")
    behavior = V3UncertaintyDampedPolicyMixture(
        representation=REPRESENTATION,
        device="cpu",
        epsilon_scale=EPSILON_SCALE,
        epsilon_cap=EPSILON_CAP,
    )
    models = []
    for index, state in enumerate(states):
        _cfg, model = make_h2_final_v3(device="cpu", seed=0x2B1000 + index)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    behavior.models = models
    return seed, behavior


def _worker_init(repo_root: str, solver_path: str, checkpoint_path: str, source_sha: str) -> None:
    global _WORKER_SOLVER, _WORKER_COLLECTOR
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError("Phase2B1 worker torch-thread contract drift")
    root = Path(repo_root)
    contract = validate_phase2_v3_contract(
        root,
        representation=REPRESENTATION,
        domain=DOMAIN,
        training_seed=int(torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("seed")),
    )
    _seed, behavior = _load_behavior(Path(checkpoint_path), source_sha)
    _WORKER_SOLVER = SolverLibrary(solver_path)
    _WORKER_COLLECTOR = UniversalPartialExactCollectorV3(
        action_spec=contract["action_spec"],
        policy=behavior,
        terminal_utility=icm_delta_utility(PAYOUT),
        rng=random.Random(0),
        advantage_memory=_Sink(),
        strategy_memory=_Sink(),
    )


def _one_target(episode, *, deck_seed: int, traversal_seed: int, expected: dict) -> tuple[tuple[float, ...], int]:
    if _WORKER_SOLVER is None or _WORKER_COLLECTOR is None:
        raise RuntimeError("Phase2B1 worker not initialized")
    root = _WORKER_SOLVER.create(episode, int(deck_seed))
    sink = _Sink()
    _WORKER_COLLECTOR.advantage_memory = sink
    _WORKER_COLLECTOR.strategy_memory = _Sink()
    _WORKER_COLLECTOR.rng = random.Random(int(traversal_seed))
    try:
        observation, actor, legal, mask = _root_identity(root, _WORKER_COLLECTOR.action_spec)
        if observation != expected["observation"]:
            raise RuntimeError("Phase2B1 exact root observation drift")
        if actor != int(expected["actor"]) or legal != tuple(expected["legal"]) or mask != tuple(expected["legal_mask"]):
            raise RuntimeError("Phase2B1 root actor/legal identity drift")
        result = _WORKER_COLLECTOR.collect_advantage_partial_exact(
            root,
            traverser=actor,
            iteration=TARGET_ITERATION,
            exact_opponent_levels=EXACT_OPPONENT_LEVELS,
        )
    finally:
        root.close()
    matches = [
        sample for sample in sink.items
        if sample.observation == expected["observation"]
        and tuple(sample.legal) == tuple(expected["legal_mask"])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Phase2B1 expected exactly one root target sample, got {len(matches)}")
    return tuple(float(value) for value in matches[0].target), int(result.nodes)


def _worker_task(task: dict) -> dict:
    scenarios = action_scenario_cycle(DOMAIN)
    scenario_index = int(task["scenario_index"])
    episode = scenarios[scenario_index]
    arm = str(task["arm"])
    expected = {
        "observation": task["observation"],
        "actor": int(task["actor"]),
        "legal": tuple(int(value) for value in task["legal"]),
        "legal_mask": tuple(int(value) for value in task["legal_mask"]),
    }
    deck_seeds = [int(value) for value in task["deck_seeds"]]
    traversal_seeds = [int(value) for value in task["traversal_seeds"]]
    targets = []
    node_counts = []
    started = time.perf_counter()
    for replicate in range(REPLICATES):
        if arm == "TRAVERSAL_ONLY":
            deck = deck_seeds[0]
            traversal = traversal_seeds[replicate]
        elif arm == "CHANCE_ONLY":
            deck = deck_seeds[replicate]
            traversal = traversal_seeds[0]
        elif arm == "COMBINED":
            deck = deck_seeds[replicate]
            traversal = traversal_seeds[replicate]
        else:
            raise RuntimeError(f"unknown Phase2B1 arm {arm}")
        target, nodes = _one_target(
            episode,
            deck_seed=deck,
            traversal_seed=traversal,
            expected=expected,
        )
        targets.append([float(value) for value in target])
        node_counts.append(int(nodes))
    return {
        "scenario_index": scenario_index,
        "arm": arm,
        "targets": targets,
        "legal_mask": [int(value) for value in expected["legal_mask"]],
        "nodes": node_counts,
        "seconds": float(time.perf_counter() - started),
    }


def _task_rows_for_groups(groups: Sequence[dict]) -> list[dict]:
    tasks = []
    for group in groups:
        traversal_seeds = [_traversal_seed(int(group["scenario_index"]), i) for i in range(REPLICATES)]
        for arm in ARMS:
            tasks.append({
                "scenario_index": int(group["scenario_index"]),
                "arm": arm,
                "observation": group["observation"],
                "actor": int(group["actor"]),
                "legal": tuple(int(value) for value in group["legal"]),
                "legal_mask": tuple(int(value) for value in group["legal_mask"]),
                "deck_seeds": tuple(int(value) for value in group["deck_seeds"]),
                "traversal_seeds": tuple(int(value) for value in traversal_seeds),
            })
    return tasks


def _run_behavior_seed(
    *,
    repo_root: Path,
    solver_path: Path,
    checkpoint: Path,
    source_sha: str,
    groups: Sequence[dict],
    workers: int,
) -> tuple[int, list[dict]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    behavior_seed = int(payload.get("seed", -1))
    tasks = _task_rows_for_groups(groups)
    results = []
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_worker_init,
        initargs=(str(repo_root), str(solver_path), str(checkpoint), str(source_sha)),
    ) as pool:
        future_map = {pool.submit(_worker_task, task): (task["scenario_index"], task["arm"]) for task in tasks}
        for future in as_completed(future_map):
            scenario_index, arm = future_map[future]
            row = future.result()
            row["source_behavior_seed"] = behavior_seed
            results.append(row)
            print(
                f"[Phase2B1 target] behavior={behavior_seed} scenario={scenario_index:02d} arm={arm} "
                f"seconds={row['seconds']:.2f}",
                flush=True,
            )
    results.sort(key=lambda row: (int(row["source_behavior_seed"]), int(row["scenario_index"]), str(row["arm"])))
    return behavior_seed, results


def _metrics_from_task_rows(task_rows: Sequence[dict]) -> tuple[list[dict], dict, dict]:
    pair_rows = []
    for task in task_rows:
        for k in K_VALUES:
            for metric in k_pair_metrics(task["targets"], task["legal_mask"], int(k)):
                pair_rows.append({
                    "source_behavior_seed": int(task["source_behavior_seed"]),
                    "scenario_index": int(task["scenario_index"]),
                    "arm": str(task["arm"]),
                    **metric,
                })

    def summarize(rows: Sequence[dict]) -> dict:
        return {
            "pair_count": len(rows),
            "target_mean_abs_diff": _summary([row["target_mean_abs_diff"] for row in rows]),
            "legal_sign_disagreement_fraction": _summary([row["legal_sign_disagreement_fraction"] for row in rows]),
            "regret_matching_policy_tv": _summary([row["regret_matching_policy_tv"] for row in rows]),
            "dominant_legal_action_mismatch_rate": (
                float(sum(int(row["dominant_legal_action_mismatch"]) for row in rows) / len(rows)) if rows else None
            ),
        }

    by_seed: dict[str, dict] = {}
    for seed in map(int, TRAINING_SEEDS):
        by_seed[str(seed)] = {}
        for arm in ARMS:
            by_seed[str(seed)][arm] = {}
            for k in K_VALUES:
                rows = [
                    row for row in pair_rows
                    if int(row["source_behavior_seed"]) == seed
                    and row["arm"] == arm
                    and int(row["k"]) == int(k)
                ]
                by_seed[str(seed)][arm][f"K{k}"] = summarize(rows)

    pooled: dict[str, dict] = {}
    for arm in ARMS:
        pooled[arm] = {}
        for k in K_VALUES:
            rows = [row for row in pair_rows if row["arm"] == arm and int(row["k"]) == int(k)]
            pooled[arm][f"K{k}"] = summarize(rows)
    return pair_rows, by_seed, pooled


def _k4_rule(summary: dict) -> dict:
    k1 = summary["K1"]
    k2 = summary["K2"]
    k4 = summary["K4"]
    k8 = summary["K8"]
    tv1 = float(k1["regret_matching_policy_tv"]["mean"])
    tv2 = float(k2["regret_matching_policy_tv"]["mean"])
    tv4 = float(k4["regret_matching_policy_tv"]["mean"])
    tv8 = float(k8["regret_matching_policy_tv"]["mean"])
    sign1 = float(k1["legal_sign_disagreement_fraction"]["mean"])
    sign4 = float(k4["legal_sign_disagreement_fraction"]["mean"])
    tv_abs = tv1 - tv4
    tv_rel = tv_abs / tv1 if tv1 > 0.0 else 0.0
    sign_abs = sign1 - sign4
    sign_rel = sign_abs / sign1 if sign1 > 0.0 else 0.0
    return {
        "k1_policy_tv": tv1,
        "k2_policy_tv": tv2,
        "k4_policy_tv": tv4,
        "k8_policy_tv": tv8,
        "k1_sign_disagreement": sign1,
        "k4_sign_disagreement": sign4,
        "policy_tv_absolute_improvement_k1_to_k4": tv_abs,
        "policy_tv_relative_improvement_k1_to_k4": tv_rel,
        "sign_absolute_improvement_k1_to_k4": sign_abs,
        "sign_relative_improvement_k1_to_k4": sign_rel,
        "tv_materiality_pass": bool(tv_abs >= 0.05 or tv_rel >= 0.20),
        "sign_materiality_pass": bool(sign_abs >= 0.05 or sign_rel >= 0.15),
        "curve_nonreversal_pass": bool(tv4 <= tv2 + 0.01 and tv8 <= tv4 + 0.01),
    }


def _decision(by_seed: dict, pooled: dict) -> dict:
    traversal_tv = float(pooled["TRAVERSAL_ONLY"]["K1"]["regret_matching_policy_tv"]["mean"])
    chance_tv = float(pooled["CHANCE_ONLY"]["K1"]["regret_matching_policy_tv"]["mean"])
    if traversal_tv >= chance_tv * 1.20 and traversal_tv > 0.0:
        source = "TRAVERSAL_DOMINANT"
    elif chance_tv >= traversal_tv * 1.20 and chance_tv > 0.0:
        source = "CHANCE_DOMINANT"
    else:
        source = "MIXED_OR_UNRESOLVED"

    rules = {arm: _k4_rule(pooled[arm]) for arm in ARMS}
    combined = rules["COMBINED"]
    both_seed_directional = all(
        float(by_seed[str(seed)]["COMBINED"]["K4"]["regret_matching_policy_tv"]["mean"])
        < float(by_seed[str(seed)]["COMBINED"]["K1"]["regret_matching_policy_tv"]["mean"])
        for seed in map(int, TRAINING_SEEDS)
    )
    combined_pass = bool(
        combined["tv_materiality_pass"]
        and combined["sign_materiality_pass"]
        and combined["curve_nonreversal_pass"]
        and both_seed_directional
    )
    if not combined_pass:
        route = "NO_K4_TRAINING_PILOT"
    elif source == "TRAVERSAL_DOMINANT" and rules["TRAVERSAL_ONLY"]["tv_materiality_pass"]:
        route = "PRECOMMIT_SMALL_SAME_DEAL_MULTI_TRAVERSAL_K4_PILOT"
    elif source == "CHANCE_DOMINANT":
        route = "PRECOMMIT_CONDITIONAL_CHANCE_OR_STRATIFIED_CHANCE_PILOT_NOT_SAME_DEAL_ONLY"
    else:
        route = "PRECOMMIT_SMALL_COMBINED_VARIANCE_REDUCTION_PILOT_WITH_CAUSAL_CONTROLS"
    return {
        "source_classification": source,
        "k1_source_policy_tv": {
            "TRAVERSAL_ONLY": traversal_tv,
            "CHANCE_ONLY": chance_tv,
            "COMBINED": float(pooled["COMBINED"]["K1"]["regret_matching_policy_tv"]["mean"]),
        },
        "k4_rules": rules,
        "combined_both_source_behavior_seeds_directionally_improve": bool(both_seed_directional),
        "combined_k4_screen_pass": combined_pass,
        "small_causal_pilot_precommit_allowed": combined_pass,
        "next_route": route,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V1+ Phase2B1 Advantage target-variance decomposition")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--phase2b0-evidence", type=Path, required=True)
    parser.add_argument("--source-execution-sha", default=SOURCE_EXECUTION_SHA)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if str(args.source_execution_sha) != SOURCE_EXECUTION_SHA:
        raise SystemExit("Phase2B1 source execution SHA drift")
    workers = max(1, min(int(args.workers), MAX_WORKERS, os.cpu_count() or MAX_WORKERS))
    torch.set_num_threads(1)

    repo_root = args.repo_root.resolve()
    solver_path = args.solver.resolve()
    input_root = args.input_root.resolve()
    output = args.out.resolve()
    evidence = json.loads(args.phase2b0_evidence.resolve().read_text(encoding="utf-8"))
    if evidence.get("status") != "FAIL_DO_NOT_TRAIN_CANDIDATE" or evidence.get("decision", {}).get("next_frontier") != "PHASE2B1_ADVANTAGE_TARGET_VARIANCE_DECOMPOSITION":
        raise RuntimeError("Phase2B1 requires frozen failed Phase2B0 evidence")

    checkpoints = {
        int(seed): input_root / f"seed_{int(seed)}" / "resume_checkpoint.pt"
        for seed in TRAINING_SEEDS
    }
    for seed, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            raise RuntimeError(f"missing Phase2A checkpoint for seed {seed}: {checkpoint}")
        loaded_seed, _behavior = _load_behavior(checkpoint, str(args.source_execution_sha))
        if loaded_seed != int(seed):
            raise RuntimeError("Phase2B1 parent checkpoint seed mismatch")

    started = time.perf_counter()
    print("[Phase2B1] discovering exact-root SPNNIV3 deck-collision groups...", flush=True)
    groups = discover_collision_groups(repo_root, solver_path)
    for group in groups:
        print(
            f"[Phase2B1 collision] scenario={group['scenario_index']:02d} actor={group['actor']} "
            f"legal={group['legal']} attempts={group['attempts_used']} obs={group['observation_sha256'][:12]}",
            flush=True,
        )

    task_rows = []
    behavior_seconds = {}
    for seed in map(int, TRAINING_SEEDS):
        local_started = time.perf_counter()
        print(f"[Phase2B1] source behavior seed {seed}: running 45 diagnostic tasks with {workers} workers...", flush=True)
        loaded_seed, rows = _run_behavior_seed(
            repo_root=repo_root,
            solver_path=solver_path,
            checkpoint=checkpoints[seed],
            source_sha=str(args.source_execution_sha),
            groups=groups,
            workers=workers,
        )
        if loaded_seed != seed:
            raise RuntimeError("Phase2B1 behavior seed drift")
        task_rows.extend(rows)
        behavior_seconds[str(seed)] = float(time.perf_counter() - local_started)

    pair_rows, by_seed, pooled = _metrics_from_task_rows(task_rows)
    decision = _decision(by_seed, pooled)

    public_groups = []
    for group in groups:
        public_groups.append({key: value for key, value in group.items() if key != "observation"})
    task_audit = []
    for row in task_rows:
        task_audit.append({
            "source_behavior_seed": int(row["source_behavior_seed"]),
            "scenario_index": int(row["scenario_index"]),
            "arm": str(row["arm"]),
            "replicates": REPLICATES,
            "nodes": _summary([float(value) for value in row["nodes"]]),
            "seconds": float(row["seconds"]),
            "target_sha256": hashlib.sha256(
                json.dumps(row["targets"], sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        })

    result = {
        "schema": SCHEMA,
        "status": (
            "PHASE2B1_K4_SCREEN_PASS_SMALL_CAUSAL_PILOT_PRECOMMIT_ALLOWED"
            if decision["combined_k4_screen_pass"]
            else "PHASE2B1_K4_SCREEN_FAIL_NO_GENERIC_K4_TRAINING_PILOT"
        ),
        "governance_scope": "Post-R7.5.3 V1+ architecture-reset diagnostic; R7.5.3 remains closed.",
        "source_execution_sha": str(args.source_execution_sha),
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "source_behavior_seeds": [int(value) for value in TRAINING_SEEDS],
        "action_candidate": ACTION_CANDIDATE,
        "exact_opponent_levels": EXACT_OPPONENT_LEVELS,
        "target_iteration": TARGET_ITERATION,
        "collision_search_budget_per_scenario": COLLISION_SEARCH_BUDGET,
        "replicates_per_arm": REPLICATES,
        "arms": list(ARMS),
        "k_values": list(K_VALUES),
        "worker_processes": workers,
        "torch_threads_per_worker": 1,
        "collision_groups": public_groups,
        "by_source_behavior_seed": by_seed,
        "pooled": pooled,
        "decision": decision,
        "task_audit": task_audit,
        "pair_metric_row_count": len(pair_rows),
        "runtime_seconds_by_source_behavior_seed": behavior_seconds,
        "runtime_seconds_total": float(time.perf_counter() - started),
        "interpretation_guardrails": [
            "Fresh solver traversal was used only to measure target-generation variance; no model fit or optimizer step occurred.",
            "All Advantage target samples were captured in diagnostic in-memory sinks and were never inserted into a training reservoir.",
            "CHANCE_ONLY holds exact root SPNNIV3 observation, actor, legal set, and traversal RNG fixed while deck seed varies.",
            "TRAVERSAL_ONLY holds the entire deck seed fixed while only collector traversal RNG varies.",
            "K aggregation averages raw target estimators before diagnostic regret matching; it does not reinstate the rejected Phase2B0 behavior-policy algebra candidate.",
            "A Phase2B1 screen pass can authorize only precommitting one small causal pilot; production training and table readiness remain false.",
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
        "source_classification": decision["source_classification"],
        "combined_k4_screen_pass": decision["combined_k4_screen_pass"],
        "next_route": decision["next_route"],
        "pooled_k1_tv": decision["k1_source_policy_tv"],
        "combined_k4_rule": decision["k4_rules"]["COMBINED"],
        "runtime_seconds_total": result["runtime_seconds_total"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
