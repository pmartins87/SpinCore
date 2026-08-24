from __future__ import annotations

"""Phase2B15: posterior-weighted preflop-continuation chance screen.

Read-only diagnostic.  For a deterministic balanced set of heldout preflop
continuation infosets, compare two equal-compute K64 estimators built from the
same conditional-IID deals:

* UNWEIGHTED_IID64: arithmetic mean under the card prior given the current
  actor's private cards.
* POSTERIOR_WEIGHTED_IID64: self-normalized importance mean using the
  likelihood of the already-observed preflop action path under the exact final
  Phase2B13 candidate behavior ensemble (including the frozen 25% continuation
  behavior floor).

No network fit, optimizer step, reservoir mutation, or production training is
performed.
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

import r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot as b6
import r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition as b10
import r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator as b11
import r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training as b13
import r7_5_arch_reset_v1plus_phase2b7_residual_localization as b7

from spincore.r7_5_action_cfr import legal_mask, regret_matching_policy, validate_policy
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_representation_v3 import H2_FINAL
from spincore.r7_5_representation_v3_checkpoint import SCHEMA as CHECKPOINT_SCHEMA
from spincore.r7_5_representation_v3_referee_artifacts import load_heldout_v3_artifact
from spincore.r7_5_representation_v3_referee_states import effective_pf0
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    EXACT_OPPONENT_LEVELS,
    MODEL_FINGERPRINTS,
    TRAINING_SEEDS,
    EVALUATION_SEEDS,
    validate_phase2_v3_contract,
)
from spincore.solver_v3 import neural_bytes_v3


SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B15_POSTERIOR_WEIGHTED_CONTINUATION_CHANCE_V1"
PARTIAL_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B15_PARTIAL_V1"
DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL

B13_RESULT_SHA256 = "6de7996282236d34adf5e8e53416fd8a443a1fbf5abc89fc807492d0cb3dbf80"
B14_RESULT_SHA256 = "7cd1886596d345abdcdef479775498eddf7e014205de86e44afb5bb0ea291f86"
B13_EXECUTION_SHA = "2cd7d1ece46a20d2b8937fe5135a415f6bbe54c2"
B13_CANDIDATE_ARM = b13.CANDIDATE_ARM

REGIONS = ("PREFLOP_CONTINUATION_1", "PREFLOP_CONTINUATION_2PLUS")
ANCHORS_PER_REGION_PER_EVAL = 16
K = 64
BLOCKS = 2
TARGET_ITERATION = 3
MAX_WORKERS = 30
POLICY_COUNT = 1024
TAIL_TV = 0.35
MASK64 = (1 << 64) - 1

# Frozen screen gates.  A PASS only permits a separately precommitted training
# pilot; it never authorizes production training.
TV_ABS_GATE = 0.03
TV_REL_GATE = 0.15
SIGN_ABS_GATE = 0.02
SIGN_REL_GATE = 0.10
TAIL_REL_GATE = 0.10
POSTERIOR_SHIFT_TV_MIN = 0.03
ESS_MEDIAN_MIN = 16.0
ESS_P10_MIN = 8.0
MAX_WEIGHT_P95_MAX = 0.35
PER_GROUP_TV_MAX_DEGRADE = 0.01


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
        return {"count": 0, "mean": None, "p10": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p10": float(np.quantile(arr, 0.10, method="linear")),
        "p50": float(np.quantile(arr, 0.50, method="linear")),
        "p95": float(np.quantile(arr, 0.95, method="linear")),
        "max": float(arr.max()),
    }


def _policy_tv(left: Sequence[float], right: Sequence[float]) -> float:
    return float(0.5 * sum(abs(float(a) - float(b)) for a, b in zip(left, right)))


def _legal_target_mad(left: Sequence[float], right: Sequence[float], legal: Sequence[int]) -> float:
    legal = tuple(int(x) for x in legal)
    return float(sum(abs(float(left[a]) - float(right[a])) for a in legal) / len(legal))


def _sign_disagreement(left: Sequence[float], right: Sequence[float], legal: Sequence[int]) -> float:
    legal = tuple(int(x) for x in legal)
    return float(sum((float(left[a]) > 0.0) != (float(right[a]) > 0.0) for a in legal) / len(legal))


def _dominant_mismatch(left: Sequence[float], right: Sequence[float], legal: Sequence[int]) -> int:
    legal = tuple(int(x) for x in legal)
    lp = regret_matching_policy(left, legal)
    rp = regret_matching_policy(right, legal)
    la = max(legal, key=lambda a: float(lp[a]))
    ra = max(legal, key=lambda a: float(rp[a]))
    return int(la != ra)


def _mean_targets(targets: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if len(targets) != K or any(len(row) != 10 for row in targets):
        raise ValueError("Phase2B15 K/target width drift")
    return tuple(float(sum(float(row[i]) for row in targets) / len(targets)) for i in range(10))


def _self_normalized_mean(
    targets: Sequence[Sequence[float]], log_weights: Sequence[float]
) -> tuple[tuple[float, ...], dict]:
    if len(targets) != K or len(log_weights) != K:
        raise ValueError("Phase2B15 posterior estimator requires exactly K rows")
    finite = [float(x) for x in log_weights if math.isfinite(float(x))]
    if not finite:
        raise RuntimeError("Phase2B15 posterior support failure: all importance weights are zero")
    peak = max(finite)
    raw = [
        (math.exp(float(x) - peak) if math.isfinite(float(x)) else 0.0)
        for x in log_weights
    ]
    total = float(sum(raw))
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("Phase2B15 posterior weight normalization failure")
    weights = [float(x / total) for x in raw]
    estimate = tuple(
        float(sum(weights[r] * float(targets[r][i]) for r in range(K)))
        for i in range(10)
    )
    ess = float(1.0 / sum(w * w for w in weights))
    positive_logs = [float(x) for x in log_weights if math.isfinite(float(x))]
    return estimate, {
        "ess": ess,
        "max_normalized_weight": float(max(weights)),
        "zero_weight_count": int(sum(w == 0.0 for w in weights)),
        "finite_weight_count": int(sum(w > 0.0 for w in weights)),
        "log_weight_span": float(max(positive_logs) - min(positive_logs)) if len(positive_logs) > 1 else 0.0,
    }


def _mix64(*parts: int) -> int:
    x = 0x2B150FACADE00001
    for raw in parts:
        y = int(raw) & MASK64
        x ^= (y + 0x9E3779B97F4A7C15 + ((x << 6) & MASK64) + (x >> 2)) & MASK64
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & MASK64
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & MASK64
        x ^= x >> 31
    return x & MASK64


def _chance_seeds(evaluation_seed: int, state_index: int, block: int, sample_index: int) -> tuple[int, int]:
    # Deliberately independent of behavior seed: both source behaviors see the
    # same chance proposals.
    key = (int(evaluation_seed), int(state_index), int(block), int(sample_index))
    return _mix64(0x1501, *key), _mix64(0x1502, *key)


def _traversal_seed(evaluation_seed: int, state_index: int) -> int:
    # Fixed across blocks and proposals so the screen isolates chance/posterior
    # effects rather than traversal RNG.
    return _mix64(0x15A771, int(evaluation_seed), int(state_index))


def _validate_source_results(b13_result: Path, b14_result: Path) -> tuple[dict, dict]:
    if _sha256(b13_result) != B13_RESULT_SHA256:
        raise RuntimeError("Phase2B15 Phase2B13 result SHA drift")
    if _sha256(b14_result) != B14_RESULT_SHA256:
        raise RuntimeError("Phase2B15 Phase2B14 result SHA drift")
    j13 = json.loads(b13_result.read_text(encoding="utf-8"))
    j14 = json.loads(b14_result.read_text(encoding="utf-8"))
    if j13.get("schema") != b13.SCHEMA or j13.get("status") != "ROOT_IID64_TRAINING_EFFECT_NOT_SUPPORTED":
        raise RuntimeError("Phase2B15 requires exact Phase2B13 failed materiality screen")
    d13 = dict(j13.get("decision") or {})
    if bool(d13.get("causal_effect_supported")) or bool(d13.get("full_x4_confirmation_authorized")):
        raise RuntimeError("Phase2B15 source Phase2B13 unexpectedly authorizes scale-up")
    if j14.get("schema") != "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B14_B13_RESIDUAL_LOCALIZATION_V1":
        raise RuntimeError("Phase2B15 Phase2B14 schema mismatch")
    if j14.get("status") != "PREFLOP_CONTINUATION_RESIDUAL_DOMINANT_AFTER_ROOT_IID64":
        raise RuntimeError("Phase2B15 requires continuation-dominant Phase2B14 result")
    d14 = dict(j14.get("decision") or {})
    if not bool(d14.get("root_effect_consistent")):
        raise RuntimeError("Phase2B15 requires consistent localized Phase2B13 root effect")
    if d14.get("next_route") != "PRECOMMIT_POSTERIOR_WEIGHTED_PREFLOP_CONTINUATION_CHANCE_SCREEN":
        raise RuntimeError("Phase2B15 Phase2B14 route mismatch")
    return j13, j14


def _load_behavior_states(checkpoint: Path, training_seed: int) -> tuple[list[dict], dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("Phase2B15 candidate checkpoint schema mismatch")
    expected = {
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "seed": int(training_seed),
        "action_candidate": ACTION_CANDIDATE,
        "execution_sha": B13_EXECUTION_SHA,
        "architecture_fingerprint_sha256": MODEL_FINGERPRINTS[REPRESENTATION],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Phase2B15 candidate checkpoint identity drift: {key}")
    progress = dict(payload.get("progress") or {})
    extra = dict(payload.get("extra") or {})
    state = dict(extra.get("stage_state") or {})
    if progress.get("phase") != "phase2b13_resume":
        raise RuntimeError("Phase2B15 candidate checkpoint phase mismatch")
    if int(progress.get("iteration", -1)) != 3 or int(progress.get("global_root", -1)) != b13.TOTAL_ROOTS:
        raise RuntimeError("Phase2B15 candidate checkpoint final progress mismatch")
    if extra.get("schema") != b13.CHECKPOINT_EXTRA_SCHEMA:
        raise RuntimeError("Phase2B15 candidate checkpoint extra schema mismatch")
    if extra.get("arm") != B13_CANDIDATE_ARM or int(extra.get("k", -1)) != 64:
        raise RuntimeError("Phase2B15 requires exact B13 IID64 candidate checkpoint")
    if int(extra.get("stage_index", -1)) != 6:
        raise RuntimeError("Phase2B15 candidate checkpoint stage mismatch")
    if int(state.get("completed_iteration", -1)) != 3 or int(state.get("global_root", -1)) != b13.TOTAL_ROOTS:
        raise RuntimeError("Phase2B15 candidate checkpoint stage-state mismatch")
    states = list(extra.get("behavior_model_states") or [])
    if len(states) != 4:
        raise RuntimeError("Phase2B15 requires four final B13 behavior members")
    return states, {
        "training_seed": int(training_seed),
        "path": str(checkpoint),
        "sha256": _sha256(checkpoint),
        "stage_index": int(extra["stage_index"]),
        "completed_iteration": int(state["completed_iteration"]),
        "global_root": int(state["global_root"]),
        "arm": str(extra["arm"]),
        "k": int(extra["k"]),
    }


def _round_robin_scenarios(candidates: Sequence, count: int) -> list:
    groups = defaultdict(list)
    for descriptor in sorted(candidates, key=lambda d: (int(d.scenario_index), int(d.state_index))):
        groups[int(descriptor.scenario_index)].append(descriptor)
    keys = sorted(groups)
    out = []
    cursor = {key: 0 for key in keys}
    while len(out) < int(count):
        advanced = False
        for key in keys:
            idx = cursor[key]
            if idx < len(groups[key]):
                out.append(groups[key][idx])
                cursor[key] += 1
                advanced = True
                if len(out) >= int(count):
                    break
        if not advanced:
            break
    if len(out) != int(count):
        raise RuntimeError(f"Phase2B15 insufficient balanced anchors: {len(out)} != {count}")
    return out


def _select_anchors(heldout_root: Path, b14_result: dict) -> tuple[list[dict], list[dict]]:
    expected_hashes = {
        int(row["evaluation_seed"]): str(row["sha256"])
        for row in (b14_result.get("frozen_inputs") or {}).get("heldout") or []
    }
    anchors = []
    heldout_identity = []
    for evaluation_seed in map(int, EVALUATION_SEEDS):
        path = b6._find_heldout(heldout_root, evaluation_seed)
        actual = _sha256(path)
        if expected_hashes.get(evaluation_seed) != actual:
            raise RuntimeError(f"Phase2B15 heldout hash drift {evaluation_seed}")
        descriptors = load_heldout_v3_artifact(
            path,
            expected_domain=DOMAIN,
            expected_evaluation_seed=evaluation_seed,
            expected_count=2048,
        )[:POLICY_COUNT]
        heldout_identity.append({
            "evaluation_seed": evaluation_seed,
            "path": str(path),
            "sha256": actual,
        })
        for region in REGIONS:
            candidates = [
                descriptor for descriptor in descriptors
                if b7._decode_observation(descriptor.observation_v3)["region"] == region
            ]
            chosen = _round_robin_scenarios(candidates, ANCHORS_PER_REGION_PER_EVAL)
            for descriptor in chosen:
                anchors.append({
                    "evaluation_seed": evaluation_seed,
                    "state_index": int(descriptor.state_index),
                    "hand_index": int(descriptor.hand_index),
                    "scenario_index": int(descriptor.scenario_index),
                    "deck_seed": int(descriptor.deck_seed),
                    "action_path": [int(x) for x in descriptor.action_path],
                    "actor": int(descriptor.actor),
                    "observation": bytes(descriptor.observation_v3),
                    "observation_sha256": hashlib.sha256(descriptor.observation_v3).hexdigest(),
                    "active_mask": int(descriptor.active_mask),
                    "legal_slots": [int(x) for x in descriptor.legal_slots],
                    "region": region,
                })
    expected_total = len(EVALUATION_SEEDS) * len(REGIONS) * ANCHORS_PER_REGION_PER_EVAL
    if len(anchors) != expected_total:
        raise RuntimeError("Phase2B15 anchor count drift")
    return anchors, heldout_identity


def _canonical_snapshot(task: dict):
    if b10._WORKER_SOLVER is None or b10._WORKER_ACTION_SPEC is None:
        raise RuntimeError("Phase2B15 worker not initialized")
    scenarios = action_scenario_cycle(DOMAIN)
    state = b10._WORKER_SOLVER.create(
        scenarios[int(task["scenario_index"])], int(task["deck_seed"])
    )
    try:
        for action in task["action_path"]:
            if state.terminal:
                raise RuntimeError("Phase2B15 canonical path reaches terminal early")
            active_mask, legal, _exact = effective_pf0(state, b10._WORKER_ACTION_SPEC)
            if int(action) not in legal:
                raise RuntimeError("Phase2B15 canonical path action is illegal")
            state.apply_universal(active_mask, int(action))
        if state.terminal:
            raise RuntimeError("Phase2B15 canonical continuation is terminal")
        observation = neural_bytes_v3(state)
        active_mask, legal, _exact = effective_pf0(state, b10._WORKER_ACTION_SPEC)
        if observation != bytes(task["observation"]):
            raise RuntimeError("Phase2B15 canonical heldout observation drift")
        if int(state.actor) != int(task["actor"]):
            raise RuntimeError("Phase2B15 canonical heldout actor drift")
        if int(active_mask) != int(task["active_mask"]) or tuple(legal) != tuple(task["legal_slots"]):
            raise RuntimeError("Phase2B15 canonical heldout legal identity drift")
        snapshot = state.deal_snapshot()
    finally:
        state.close()
    if snapshot.visible_board_count != 0:
        raise RuntimeError("Phase2B15 anchors must be preflop with no visible board")
    return snapshot


def _variant_likelihood_and_target(task: dict, deal, traversal_seed: int) -> tuple[tuple[float, ...], float, int]:
    if b10._WORKER_SOLVER is None or b10._WORKER_COLLECTOR is None or b10._WORKER_ACTION_SPEC is None:
        raise RuntimeError("Phase2B15 worker not initialized")
    scenarios = action_scenario_cycle(DOMAIN)
    episode = scenarios[int(task["scenario_index"])]
    state = b10._WORKER_SOLVER.create_with_deal(episode, deal.holes, deal.board)
    collector = b10._WORKER_COLLECTOR
    log_likelihood = 0.0
    try:
        for action in task["action_path"]:
            if state.terminal:
                raise RuntimeError("Phase2B15 variant path reaches terminal early")
            observation = neural_bytes_v3(state)
            active_mask, legal, _exact = effective_pf0(state, b10._WORKER_ACTION_SPEC)
            if int(action) not in legal:
                raise RuntimeError("Phase2B15 variant path action became illegal")
            probabilities = validate_policy(
                collector.policy(state, observation, legal), legal
            )
            probability = float(probabilities[int(action)])
            if probability <= 0.0:
                log_likelihood = -math.inf
            elif math.isfinite(log_likelihood):
                log_likelihood += math.log(probability)
            state.apply_universal(active_mask, int(action))

        if state.terminal:
            raise RuntimeError("Phase2B15 variant continuation unexpectedly terminal")
        observation = neural_bytes_v3(state)
        active_mask, legal, _exact = effective_pf0(state, b10._WORKER_ACTION_SPEC)
        if observation != bytes(task["observation"]):
            raise RuntimeError("Phase2B15 hidden-card proposal changed target infoset observation")
        if int(state.actor) != int(task["actor"]):
            raise RuntimeError("Phase2B15 hidden-card proposal changed target actor")
        if int(active_mask) != int(task["active_mask"]) or tuple(legal) != tuple(task["legal_slots"]):
            raise RuntimeError("Phase2B15 hidden-card proposal changed target legal identity")

        sink = b10._Sink()
        collector.advantage_memory = sink
        collector.strategy_memory = b10._Sink()
        collector.rng = random.Random(int(traversal_seed))
        result = collector.collect_advantage_partial_exact(
            state,
            traverser=int(task["actor"]),
            iteration=TARGET_ITERATION,
            exact_opponent_levels=EXACT_OPPONENT_LEVELS,
        )
    finally:
        state.close()

    expected_mask = legal_mask(tuple(int(x) for x in task["legal_slots"]))
    matches = [
        sample for sample in sink.items
        if bytes(sample.observation) == bytes(task["observation"])
        and tuple(int(x) for x in sample.legal) == tuple(expected_mask)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Phase2B15 expected exactly one target continuation sample, got {len(matches)}"
        )
    return tuple(float(x) for x in matches[0].target), float(log_likelihood), int(result.nodes)


def _worker_task(task: dict) -> dict:
    snapshot = _canonical_snapshot(task)
    actor = int(task["actor"])
    block = int(task["block"])
    traversal_seed = _traversal_seed(int(task["evaluation_seed"]), int(task["state_index"]))
    targets = []
    log_weights = []
    nodes = 0
    started = time.perf_counter()
    for sample_index in range(K):
        private_seed, public_seed = _chance_seeds(
            int(task["evaluation_seed"]), int(task["state_index"]), block, sample_index
        )
        deal = b11._deal_from_factors(snapshot, actor, private_seed, public_seed)
        target, log_weight, node_count = _variant_likelihood_and_target(
            task, deal, traversal_seed
        )
        targets.append(target)
        log_weights.append(float(log_weight))
        nodes += int(node_count)

    unweighted = _mean_targets(targets)
    posterior, weight_stats = _self_normalized_mean(targets, log_weights)
    return {
        "schema": PARTIAL_SCHEMA,
        "behavior_seed": int(task["behavior_seed"]),
        "evaluation_seed": int(task["evaluation_seed"]),
        "state_index": int(task["state_index"]),
        "scenario_index": int(task["scenario_index"]),
        "region": str(task["region"]),
        "block": block,
        "k": K,
        "actor": actor,
        "action_path_length": len(task["action_path"]),
        "legal_slots": [int(x) for x in task["legal_slots"]],
        "observation_sha256": str(task["observation_sha256"]),
        "unweighted_target": [float(x) for x in unweighted],
        "posterior_target": [float(x) for x in posterior],
        "weight_stats": weight_stats,
        "target_nodes": int(nodes),
        "target_traversals": K,
        "seconds": float(time.perf_counter() - started),
    }


def _partial_path(root: Path, task: dict) -> Path:
    return (
        root
        / "partials"
        / f"behavior_{int(task['behavior_seed'])}"
        / f"eval_{int(task['evaluation_seed'])}"
        / f"state_{int(task['state_index']):04d}_block_{int(task['block'])}.json"
    )


def _valid_partial(payload: dict, task: dict) -> bool:
    return bool(
        payload.get("schema") == PARTIAL_SCHEMA
        and int(payload.get("behavior_seed", -1)) == int(task["behavior_seed"])
        and int(payload.get("evaluation_seed", -1)) == int(task["evaluation_seed"])
        and int(payload.get("state_index", -1)) == int(task["state_index"])
        and int(payload.get("block", -1)) == int(task["block"])
        and int(payload.get("k", -1)) == K
        and payload.get("observation_sha256") == task["observation_sha256"]
    )


def _run_behavior_seed(
    *,
    repo_root: Path,
    solver_path: Path,
    output_root: Path,
    behavior_seed: int,
    behavior_states: list[dict],
    anchors: Sequence[dict],
    workers: int,
) -> list[dict]:
    tasks = []
    cached = []
    for anchor in anchors:
        for block in range(BLOCKS):
            task = dict(anchor)
            task["block"] = int(block)
            task["behavior_seed"] = int(behavior_seed)
            path = _partial_path(output_root, task)
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if _valid_partial(payload, task):
                    cached.append(payload)
                    continue
            tasks.append(task)

    rows = list(cached)
    if tasks:
        with ProcessPoolExecutor(
            max_workers=min(int(workers), len(tasks)),
            initializer=b10._worker_init,
            initargs=(
                str(repo_root),
                str(solver_path),
                int(behavior_seed),
                behavior_states,
            ),
        ) as pool:
            future_map = {pool.submit(_worker_task, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                row = future.result()
                _atomic_json(row, _partial_path(output_root, task))
                rows.append(row)
                print(
                    f"[Phase2B15 task] behavior={behavior_seed} "
                    f"eval={task['evaluation_seed']} state={task['state_index']} "
                    f"{task['region']} block={task['block']} "
                    f"ESS={row['weight_stats']['ess']:.2f} seconds={row['seconds']:.2f}",
                    flush=True,
                )
    expected = len(anchors) * BLOCKS
    if len(rows) != expected:
        raise RuntimeError(f"Phase2B15 behavior task coverage drift: {len(rows)} != {expected}")
    rows.sort(key=lambda r: (int(r["evaluation_seed"]), int(r["state_index"]), int(r["block"])))
    return rows


def _pair_rows(rows: Sequence[dict]) -> list[dict]:
    groups = defaultdict(dict)
    for row in rows:
        key = (
            int(row["behavior_seed"]),
            int(row["evaluation_seed"]),
            int(row["state_index"]),
        )
        groups[key][int(row["block"])] = row
    out = []
    for key, blocks in sorted(groups.items()):
        if set(blocks) != {0, 1}:
            raise RuntimeError(f"Phase2B15 missing paired block for {key}")
        left, right = blocks[0], blocks[1]
        if left["region"] != right["region"] or left["legal_slots"] != right["legal_slots"]:
            raise RuntimeError("Phase2B15 paired task identity drift")
        legal = tuple(int(x) for x in left["legal_slots"])
        lu = tuple(float(x) for x in left["unweighted_target"])
        ru = tuple(float(x) for x in right["unweighted_target"])
        lp = tuple(float(x) for x in left["posterior_target"])
        rp = tuple(float(x) for x in right["posterior_target"])
        lu_pol = regret_matching_policy(lu, legal)
        ru_pol = regret_matching_policy(ru, legal)
        lp_pol = regret_matching_policy(lp, legal)
        rp_pol = regret_matching_policy(rp, legal)
        out.append({
            "behavior_seed": key[0],
            "evaluation_seed": key[1],
            "state_index": key[2],
            "scenario_index": int(left["scenario_index"]),
            "region": str(left["region"]),
            "actor": int(left["actor"]),
            "action_path_length": int(left["action_path_length"]),
            "legal_slots": list(legal),
            "unweighted_tv": _policy_tv(lu_pol, ru_pol),
            "posterior_tv": _policy_tv(lp_pol, rp_pol),
            "unweighted_target_mad": _legal_target_mad(lu, ru, legal),
            "posterior_target_mad": _legal_target_mad(lp, rp, legal),
            "unweighted_sign_disagreement": _sign_disagreement(lu, ru, legal),
            "posterior_sign_disagreement": _sign_disagreement(lp, rp, legal),
            "unweighted_dominant_mismatch": _dominant_mismatch(lu, ru, legal),
            "posterior_dominant_mismatch": _dominant_mismatch(lp, rp, legal),
            "posterior_shift_tv": float(
                0.5 * (
                    _policy_tv(lu_pol, lp_pol)
                    + _policy_tv(ru_pol, rp_pol)
                )
            ),
            "ess_mean": float(
                0.5 * (
                    float(left["weight_stats"]["ess"])
                    + float(right["weight_stats"]["ess"])
                )
            ),
            "ess_min": float(
                min(
                    float(left["weight_stats"]["ess"]),
                    float(right["weight_stats"]["ess"]),
                )
            ),
            "max_normalized_weight": float(
                max(
                    float(left["weight_stats"]["max_normalized_weight"]),
                    float(right["weight_stats"]["max_normalized_weight"]),
                )
            ),
            "zero_weight_count_max": int(
                max(
                    int(left["weight_stats"]["zero_weight_count"]),
                    int(right["weight_stats"]["zero_weight_count"]),
                )
            ),
        })
    return out


def _aggregate(pairs: Sequence[dict]) -> dict:
    if not pairs:
        raise RuntimeError("Phase2B15 cannot summarize empty pairs")
    u_tv = [float(r["unweighted_tv"]) for r in pairs]
    p_tv = [float(r["posterior_tv"]) for r in pairs]
    u_sign = [float(r["unweighted_sign_disagreement"]) for r in pairs]
    p_sign = [float(r["posterior_sign_disagreement"]) for r in pairs]
    u_mad = [float(r["unweighted_target_mad"]) for r in pairs]
    p_mad = [float(r["posterior_target_mad"]) for r in pairs]
    u_dom = [float(r["unweighted_dominant_mismatch"]) for r in pairs]
    p_dom = [float(r["posterior_dominant_mismatch"]) for r in pairs]
    shift = [float(r["posterior_shift_tv"]) for r in pairs]
    ess = [float(r["ess_mean"]) for r in pairs]
    maxw = [float(r["max_normalized_weight"]) for r in pairs]
    u_mean = float(sum(u_tv) / len(u_tv))
    p_mean = float(sum(p_tv) / len(p_tv))
    tv_abs = float(u_mean - p_mean)
    tv_rel = float(tv_abs / u_mean) if u_mean > 0.0 else -math.inf
    us = float(sum(u_sign) / len(u_sign))
    ps = float(sum(p_sign) / len(p_sign))
    sign_abs = float(us - ps)
    sign_rel = float(sign_abs / us) if us > 0.0 else -math.inf
    u_tail = float(sum(v >= TAIL_TV for v in u_tv) / len(u_tv))
    p_tail = float(sum(v >= TAIL_TV for v in p_tv) / len(p_tv))
    tail_rel = float((u_tail - p_tail) / u_tail) if u_tail > 0.0 else 0.0
    return {
        "count": len(pairs),
        "unweighted_tv": _summary(u_tv),
        "posterior_tv": _summary(p_tv),
        "tv_absolute_improvement": tv_abs,
        "tv_relative_improvement": tv_rel,
        "unweighted_sign_disagreement_mean": us,
        "posterior_sign_disagreement_mean": ps,
        "sign_absolute_improvement": sign_abs,
        "sign_relative_improvement": sign_rel,
        "unweighted_target_mad": _summary(u_mad),
        "posterior_target_mad": _summary(p_mad),
        "unweighted_dominant_mismatch_rate": float(sum(u_dom) / len(u_dom)),
        "posterior_dominant_mismatch_rate": float(sum(p_dom) / len(p_dom)),
        "unweighted_tail_rate_tv_ge_035": u_tail,
        "posterior_tail_rate_tv_ge_035": p_tail,
        "tail_relative_improvement": tail_rel,
        "posterior_shift_tv": _summary(shift),
        "ess": _summary(ess),
        "max_normalized_weight": _summary(maxw),
    }


def _decision(pairs: Sequence[dict], pooled: dict, by_behavior: dict, by_region: dict) -> dict:
    local_valid = bool(
        pairs
        and len(pairs)
        == len(TRAINING_SEEDS) * len(EVALUATION_SEEDS) * len(REGIONS) * ANCHORS_PER_REGION_PER_EVAL
        and all(math.isfinite(float(r["posterior_tv"])) for r in pairs)
    )
    ess_ok = bool(
        float(pooled["ess"]["p50"]) >= ESS_MEDIAN_MIN
        and float(pooled["ess"]["p10"]) >= ESS_P10_MIN
        and float(pooled["max_normalized_weight"]["p95"]) <= MAX_WEIGHT_P95_MAX
    )
    shift_material = bool(float(pooled["posterior_shift_tv"]["mean"]) >= POSTERIOR_SHIFT_TV_MIN)
    tv_material = bool(
        float(pooled["tv_absolute_improvement"]) >= TV_ABS_GATE
        or float(pooled["tv_relative_improvement"]) >= TV_REL_GATE
    )
    sign_material = bool(
        float(pooled["sign_absolute_improvement"]) >= SIGN_ABS_GATE
        or float(pooled["sign_relative_improvement"]) >= SIGN_REL_GATE
    )
    tail_material = bool(float(pooled["tail_relative_improvement"]) >= TAIL_REL_GATE)
    both_behavior_improve = bool(
        all(float(row["tv_absolute_improvement"]) > 0.0 for row in by_behavior.values())
    )
    group_guard = bool(
        all(float(row["tv_absolute_improvement"]) >= -PER_GROUP_TV_MAX_DEGRADE for row in by_region.values())
    )
    screen_pass = bool(
        local_valid
        and ess_ok
        and shift_material
        and tv_material
        and sign_material
        and tail_material
        and both_behavior_improve
        and group_guard
    )

    if not local_valid:
        classification = "PHASE2B15_INVALID_STOP_AUDIT"
        route = "STOP_AND_AUDIT_PHASE2B15_LOCAL_VALIDITY"
    elif not ess_ok:
        classification = "POSTERIOR_IMPORTANCE_WEIGHT_DEGENERACY"
        route = "DESIGN_DIRECT_POSTERIOR_PRIVATE_CARD_PROPOSAL_NO_TRAINING"
    elif not shift_material:
        classification = "POSTERIOR_CONDITIONING_EFFECT_SMALL"
        route = "REASSESS_CONTINUATION_SUPPORT_OR_UNWEIGHTED_CONDITIONAL_ESTIMATOR"
    elif screen_pass:
        classification = "POSTERIOR_WEIGHTED_CONTINUATION_ESTIMATOR_SUPPORTED"
        route = "PRECOMMIT_SMALL_POSTERIOR_WEIGHTED_CONTINUATION_TARGET_TRAINING_PILOT"
    else:
        classification = "POSTERIOR_WEIGHTING_MATERIAL_BUT_STABILITY_NOT_SUPPORTED"
        route = "DESIGN_BETTER_POSTERIOR_PROPOSAL_OR_REPRESENTATION_SUPPORT_NO_TRAINING"

    return {
        "classification": classification,
        "next_route": route,
        "local_valid": local_valid,
        "importance_weight_health_pass": ess_ok,
        "posterior_shift_material": shift_material,
        "tv_materiality_pass": tv_material,
        "sign_materiality_pass": sign_material,
        "tail_materiality_pass": tail_material,
        "both_behavior_seeds_tv_improve": both_behavior_improve,
        "continuation_region_non_degradation_pass": group_guard,
        "screen_pass": screen_pass,
        "small_training_pilot_precommit_allowed": bool(screen_pass),
        "training_authorized": False,
        "full_x4_confirmation_authorized": False,
        "architecture_winner_selected": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def run(args) -> dict:
    repo_root = Path(args.repo_root).resolve()
    solver_path = Path(args.solver).resolve()
    heldout_root = Path(args.heldout_root).resolve()
    b13_root = Path(args.phase2b13_root).resolve()
    b13_result_path = Path(args.phase2b13_result).resolve()
    b14_result_path = Path(args.phase2b14_result).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    _j13, j14 = _validate_source_results(b13_result_path, b14_result_path)
    for seed in map(int, TRAINING_SEEDS):
        validate_phase2_v3_contract(
            repo_root,
            representation=REPRESENTATION,
            domain=DOMAIN,
            training_seed=seed,
        )
    anchors, heldout_identity = _select_anchors(heldout_root, j14)

    all_rows = []
    behavior_identity = []
    for behavior_seed in map(int, TRAINING_SEEDS):
        checkpoint = (
            b13_root
            / B13_CANDIDATE_ARM
            / f"seed_{behavior_seed}"
            / "resume_checkpoint.pt"
        )
        if not checkpoint.is_file():
            raise RuntimeError(f"Phase2B15 missing B13 candidate checkpoint {behavior_seed}")
        states, identity = _load_behavior_states(checkpoint, behavior_seed)
        behavior_identity.append(identity)
        rows = _run_behavior_seed(
            repo_root=repo_root,
            solver_path=solver_path,
            output_root=output_root,
            behavior_seed=behavior_seed,
            behavior_states=states,
            anchors=anchors,
            workers=int(args.workers),
        )
        all_rows.extend(rows)

    pairs = _pair_rows(all_rows)
    pooled = _aggregate(pairs)
    by_behavior = {
        str(seed): _aggregate([r for r in pairs if int(r["behavior_seed"]) == int(seed)])
        for seed in map(int, TRAINING_SEEDS)
    }
    by_region = {
        region: _aggregate([r for r in pairs if r["region"] == region])
        for region in REGIONS
    }
    by_evaluation = {
        str(seed): _aggregate([r for r in pairs if int(r["evaluation_seed"]) == int(seed)])
        for seed in map(int, EVALUATION_SEEDS)
    }
    decision = _decision(pairs, pooled, by_behavior, by_region)

    result = {
        "schema": SCHEMA,
        "status": decision["classification"],
        "execution_sha": str(args.execution_sha),
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "source_phase2b13_result_sha256": B13_RESULT_SHA256,
        "source_phase2b14_result_sha256": B14_RESULT_SHA256,
        "contract": {
            "regions": list(REGIONS),
            "anchors_per_region_per_evaluation_seed": ANCHORS_PER_REGION_PER_EVAL,
            "total_anchors": len(anchors),
            "behavior_seeds": list(map(int, TRAINING_SEEDS)),
            "evaluation_seeds": list(map(int, EVALUATION_SEEDS)),
            "blocks": BLOCKS,
            "k_per_block": K,
            "total_target_traversals": len(anchors) * len(TRAINING_SEEDS) * BLOCKS * K,
            "target_iteration": TARGET_ITERATION,
            "exact_opponent_levels": EXACT_OPPONENT_LEVELS,
            "proposal": "UNIFORM_PRIVATE_AND_FUTURE_BOARD_GIVEN_CURRENT_ACTOR_HOLE_CARDS",
            "posterior_weight": "PRODUCT_OF_FROZEN_BEHAVIOR_PROBABILITIES_FOR_OBSERVED_PREFLOP_ACTION_PATH",
            "behavior_floor": 0.25,
            "same_chance_proposals_for_unweighted_and_posterior": True,
            "same_chance_proposals_across_behavior_seeds": True,
            "fixed_traversal_rng_within_anchor": True,
        },
        "frozen_inputs": {
            "heldout": heldout_identity,
            "behavior_checkpoints": behavior_identity,
            "anchors": [
                {
                    k: v for k, v in row.items()
                    if k not in ("observation",)
                }
                for row in anchors
            ],
        },
        "pooled": pooled,
        "by_behavior_seed": by_behavior,
        "by_region": by_region,
        "by_evaluation_seed": by_evaluation,
        "decision": decision,
        "training_authorized": False,
        "full_x4_confirmation_authorized": False,
        "architecture_winner_selected": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="R7.5 architecture-reset Phase2B15 posterior-weighted continuation chance screen"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--phase2b13-root", type=Path, required=True)
    parser.add_argument("--phase2b13-result", type=Path, required=True)
    parser.add_argument("--phase2b14-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-sha", required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    if int(args.workers) < 1 or int(args.workers) > MAX_WORKERS:
        raise RuntimeError(f"Phase2B15 workers must be in 1..{MAX_WORKERS}")
    if K != 64 or BLOCKS != 2 or ANCHORS_PER_REGION_PER_EVAL != 16:
        raise RuntimeError("Phase2B15 frozen budget drift")

    result = run(args)
    out = Path(args.output_root).resolve() / "R7_5_ARCH_RESET_V1PLUS_PHASE2B15_POSTERIOR_WEIGHTED_CONTINUATION_CHANCE.json"
    _atomic_json(result, out)
    print(json.dumps({
        "status": result["status"],
        "unweighted_mean_tv": result["pooled"]["unweighted_tv"]["mean"],
        "posterior_mean_tv": result["pooled"]["posterior_tv"]["mean"],
        "tv_absolute_improvement": result["pooled"]["tv_absolute_improvement"],
        "posterior_shift_tv": result["pooled"]["posterior_shift_tv"]["mean"],
        "ess_median": result["pooled"]["ess"]["p50"],
        "ess_p10": result["pooled"]["ess"]["p10"],
        "screen_pass": result["decision"]["screen_pass"],
        "next_route": result["decision"]["next_route"],
        "result": str(out),
        "result_sha256": _sha256(out),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
