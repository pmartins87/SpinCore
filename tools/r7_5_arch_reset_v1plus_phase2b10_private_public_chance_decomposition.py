from __future__ import annotations

"""Phase2B10: decompose preflop root chance variance into private-hole vs public-board components.

This is a read-only solver diagnostic over the exact completed Phase2B6 behavior
ensembles. It uses the additive explicit-deal diagnostic solver API to hold the
acting player's exact root infoset fixed while independently resampling opponent
hole cards, future board cards, or both. No model fit or reservoir mutation occurs.
"""

import argparse
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

import r7_5_arch_reset_v1plus_phase2b1_target_variance as b1
import r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot as b6
from spincore.deep_cfr import icm_delta_utility
from spincore.r7_5_action_cfr import regret_matching_policy
from spincore.r7_5_action_scenarios import action_scenario_cycle
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
from spincore.solver import DealSnapshot, SolverLibrary
from spincore_nn.models_v3_final import make_h2_final_v3

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B10_PRIVATE_PUBLIC_CHANCE_DECOMPOSITION_V1"
DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL
PHASE2B1_RESULT_SHA256 = "f95751afeb17fcd5844bfcb2971577b92a400750444e5dabe2f4ddb5718ba6ef"
PHASE2B6_RESULT_SHA256 = "33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a"
PHASE2B9_RESULT_SHA256 = "71f77b6921597c7b1d048f8fb3e448f5fce74a974b247ac4ca88383fece5c64a"
PHASE2B6_EXECUTION_SHA = "4fa96434321c32efc734a55ae75982018ff2d091"
FINAL_ITERATION = 3
FINAL_GLOBAL_ROOT = 768
FINAL_STAGE_INDEX = 12
TARGET_ITERATION = 3
ANCHORS_PER_SCENARIO = 4
REPLICATES = 8
ARMS = ("TRAVERSAL_ONLY", "PRIVATE_ONLY", "PUBLIC_ONLY", "COMBINED")
MATERIAL_EXCESS_TV = 0.10
DOMINANCE_RATIO = 1.50
MAX_WORKERS = 30
MASK64 = (1 << 64) - 1

_WORKER_SOLVER = None
_WORKER_COLLECTOR = None
_WORKER_ACTION_SPEC = None
_WORKER_BEHAVIOR_SEED = None


class _Sink:
    def __init__(self):
        self.items = []

    def add(self, item) -> None:
        self.items.append(item)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _summary(values: Sequence[float]) -> dict:
    rows = np.asarray([float(v) for v in values], dtype=np.float64)
    if rows.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(rows.size),
        "mean": float(rows.mean()),
        "p50": float(np.quantile(rows, 0.50, method="linear")),
        "p95": float(np.quantile(rows, 0.95, method="linear")),
        "max": float(rows.max()),
    }


def _policy_tv(left: Sequence[float], right: Sequence[float]) -> float:
    return float(0.5 * sum(abs(float(a) - float(b)) for a, b in zip(left, right)))


def _pair_metrics(targets: Sequence[Sequence[float]], legal_mask_row: Sequence[int]) -> list[dict]:
    if len(targets) != REPLICATES:
        raise ValueError(f"Phase2B10 requires exactly {REPLICATES} targets per task")
    legal = tuple(index for index, enabled in enumerate(legal_mask_row) if int(enabled))
    if not legal:
        raise ValueError("Phase2B10 empty legal mask")
    out = []
    for start in range(0, REPLICATES, 2):
        left = tuple(float(x) for x in targets[start])
        right = tuple(float(x) for x in targets[start + 1])
        mad = float(sum(abs(left[a] - right[a]) for a in legal) / len(legal))
        sign = float(sum((left[a] > 0.0) != (right[a] > 0.0) for a in legal) / len(legal))
        lp = regret_matching_policy(left, legal)
        rp = regret_matching_policy(right, legal)
        ldom = max(legal, key=lambda a: float(lp[a]))
        rdom = max(legal, key=lambda a: float(rp[a]))
        out.append({
            "pair_start": int(start),
            "target_mean_abs_diff": mad,
            "legal_sign_disagreement_fraction": sign,
            "regret_matching_policy_tv": _policy_tv(lp, rp),
            "dominant_legal_action_mismatch": int(ldom != rdom),
        })
    return out


def _variant_seed(scenario_index: int, anchor_index: int, arm: str, replicate: int) -> int:
    arm_code = {"PRIVATE_ONLY": 1, "PUBLIC_ONLY": 2, "COMBINED": 3, "TRAVERSAL_ONLY": 4}[str(arm)]
    return (
        0x2B10C0DEC0FFEE01
        ^ (int(scenario_index) * 0x9E3779B97F4A7C15)
        ^ (int(anchor_index) * 0xD1B54A32D192ED03)
        ^ (int(arm_code) * 0x94D049BB133111EB)
        ^ (int(replicate) * 0xBF58476D1CE4E5B9)
    ) & MASK64


def _fixed_traversal_seed(scenario_index: int, anchor_index: int) -> int:
    return (
        0x2B10A7710A5E0001
        ^ (int(scenario_index) * 0xD6E8FEB86659FD93)
        ^ (int(anchor_index) * 0xA5A3564E27F8862B)
    ) & MASK64


def _resample_deal(snapshot: DealSnapshot, actor: int, arm: str, replicate: int, scenario_index: int, anchor_index: int) -> DealSnapshot:
    if snapshot.visible_board_count != 0:
        raise ValueError("Phase2B10 explicit anchor must be a preflop root with zero visible board cards")
    if actor not in (0, 1, 2):
        raise ValueError("Phase2B10 invalid actor")
    holes = [list(row) for row in snapshot.holes]
    board = list(snapshot.board)
    actor_cards = tuple(int(x) for x in holes[actor])
    if any(x < 0 for x in actor_cards):
        raise ValueError("Phase2B10 actor has invalid hole cards")
    rng = random.Random(_variant_seed(scenario_index, anchor_index, arm, replicate))
    opponents = [seat for seat in range(3) if seat != actor]

    if arm == "TRAVERSAL_ONLY":
        return snapshot
    if arm == "PRIVATE_ONLY":
        fixed = set(actor_cards) | set(board)
        pool = [card for card in range(52) if card not in fixed]
        rng.shuffle(pool)
        cursor = 0
        for seat in opponents:
            holes[seat] = [pool[cursor], pool[cursor + 1]]
            cursor += 2
    elif arm == "PUBLIC_ONLY":
        fixed = {int(card) for row in holes for card in row if int(card) >= 0}
        pool = [card for card in range(52) if card not in fixed]
        rng.shuffle(pool)
        board = pool[:5]
    elif arm == "COMBINED":
        pool = [card for card in range(52) if card not in set(actor_cards)]
        rng.shuffle(pool)
        cursor = 0
        for seat in opponents:
            holes[seat] = [pool[cursor], pool[cursor + 1]]
            cursor += 2
        board = pool[cursor : cursor + 5]
    else:
        raise ValueError(f"unknown Phase2B10 arm {arm}")

    cards = [int(card) for row in holes for card in row if int(card) >= 0] + [int(card) for card in board]
    if len(cards) != 11 or len(cards) != len(set(cards)):
        raise AssertionError("Phase2B10 resampling produced malformed three-handed deal")
    if tuple(holes[actor]) != actor_cards:
        raise AssertionError("Phase2B10 resampling changed acting-player hole cards")
    if arm == "PRIVATE_ONLY" and tuple(board) != snapshot.board:
        raise AssertionError("Phase2B10 PRIVATE_ONLY changed board")
    if arm == "PUBLIC_ONLY" and tuple(tuple(row) for row in holes) != snapshot.holes:
        raise AssertionError("Phase2B10 PUBLIC_ONLY changed private cards")
    return DealSnapshot(
        holes=tuple(tuple(int(x) for x in row) for row in holes),
        board=tuple(int(x) for x in board),
        visible_board_count=0,
    )


def _load_b6_behavior_states(checkpoint: Path, training_seed: int) -> list[dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "action_candidate": ACTION_CANDIDATE,
        "execution_sha": PHASE2B6_EXECUTION_SHA,
        "architecture_fingerprint_sha256": MODEL_FINGERPRINTS[REPRESENTATION],
        "seed": int(training_seed),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Phase2B10 Phase2B6 checkpoint identity mismatch {training_seed}/{key}")
    progress = dict(payload.get("progress") or {})
    extra = dict(payload.get("extra") or {})
    if progress.get("phase") != "phase2b6_resume":
        raise RuntimeError("Phase2B10 source checkpoint phase mismatch")
    if int(progress.get("iteration", -1)) != FINAL_ITERATION or int(progress.get("global_root", -1)) != FINAL_GLOBAL_ROOT:
        raise RuntimeError("Phase2B10 source checkpoint progress mismatch")
    if extra.get("schema") != b6.CHECKPOINT_EXTRA_SCHEMA or int(extra.get("stage_index", -1)) != FINAL_STAGE_INDEX:
        raise RuntimeError("Phase2B10 source checkpoint extra/stage mismatch")
    intervention = dict(extra.get("intervention") or {})
    if float(intervention.get("floor", -1.0)) != 0.25:
        raise RuntimeError("Phase2B10 requires exact Phase2B6 25% continuation floor source")
    states = list(extra.get("behavior_model_states") or [])
    if len(states) != ENSEMBLE_SIZE:
        raise RuntimeError("Phase2B10 requires four final Phase2B6 behavior states")
    return states


def _worker_init(repo_root: str, solver_path: str, behavior_seed: int, behavior_states: list[dict]) -> None:
    global _WORKER_SOLVER, _WORKER_COLLECTOR, _WORKER_ACTION_SPEC, _WORKER_BEHAVIOR_SEED
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError("Phase2B10 worker Torch thread contract drift")
    contract = validate_phase2_v3_contract(
        repo_root,
        representation=REPRESENTATION,
        domain=DOMAIN,
        training_seed=int(behavior_seed),
    )
    native = V3UncertaintyDampedPolicyMixture(
        representation=REPRESENTATION,
        device="cpu",
        epsilon_scale=EPSILON_SCALE,
        epsilon_cap=EPSILON_CAP,
    )
    models = []
    for index, state in enumerate(behavior_states):
        _cfg, model = make_h2_final_v3(device="cpu", seed=0x2B10000 + index)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
    native.models = models
    policy = b6.PreflopContinuationFloorPolicy(native, floor=0.25)
    _WORKER_SOLVER = SolverLibrary(solver_path)
    if not _WORKER_SOLVER.explicit_deal_available:
        raise RuntimeError("Phase2B10 solver lacks explicit-deal diagnostic extension")
    _WORKER_ACTION_SPEC = contract["action_spec"]
    _WORKER_COLLECTOR = UniversalPartialExactCollectorV3(
        action_spec=_WORKER_ACTION_SPEC,
        policy=policy,
        terminal_utility=icm_delta_utility(PAYOUT),
        rng=random.Random(0),
        advantage_memory=_Sink(),
        strategy_memory=_Sink(),
    )
    _WORKER_BEHAVIOR_SEED = int(behavior_seed)


def _one_target(episode, deal: DealSnapshot, traversal_seed: int, expected: dict) -> tuple[tuple[float, ...], int]:
    if _WORKER_SOLVER is None or _WORKER_COLLECTOR is None:
        raise RuntimeError("Phase2B10 worker not initialized")
    root = _WORKER_SOLVER.create_with_deal(episode, deal.holes, deal.board)
    sink = _Sink()
    _WORKER_COLLECTOR.advantage_memory = sink
    _WORKER_COLLECTOR.strategy_memory = _Sink()
    _WORKER_COLLECTOR.rng = random.Random(int(traversal_seed))
    try:
        observation, actor, legal, mask = b1._root_identity(root, _WORKER_ACTION_SPEC)
        if hashlib.sha256(observation).hexdigest() != str(expected["observation_sha256"]):
            raise RuntimeError("Phase2B10 explicit variant changed exact root SPNNIV3 observation")
        if actor != int(expected["actor"]) or legal != tuple(expected["legal"]) or mask != tuple(expected["legal_mask"]):
            raise RuntimeError("Phase2B10 explicit variant changed root actor/legal identity")
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
        if hashlib.sha256(sample.observation).hexdigest() == str(expected["observation_sha256"])
        and tuple(sample.legal) == tuple(expected["legal_mask"])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Phase2B10 expected exactly one root target sample, got {len(matches)}")
    return tuple(float(v) for v in matches[0].target), int(result.nodes)


def _worker_task(task: dict) -> dict:
    scenarios = action_scenario_cycle(DOMAIN)
    scenario_index = int(task["scenario_index"])
    anchor_index = int(task["anchor_index"])
    arm = str(task["arm"])
    episode = scenarios[scenario_index]
    expected = {
        "observation_sha256": str(task["observation_sha256"]),
        "actor": int(task["actor"]),
        "legal": tuple(int(x) for x in task["legal"]),
        "legal_mask": tuple(int(x) for x in task["legal_mask"]),
    }
    anchor = _WORKER_SOLVER.create(episode, int(task["anchor_deck_seed"]))
    try:
        observation, actor, legal, mask = b1._root_identity(anchor, _WORKER_ACTION_SPEC)
        if hashlib.sha256(observation).hexdigest() != expected["observation_sha256"] or actor != expected["actor"] or legal != expected["legal"] or mask != expected["legal_mask"]:
            raise RuntimeError("Phase2B10 stored Phase2B1 anchor identity drift")
        snapshot = anchor.deal_snapshot()
    finally:
        anchor.close()
    if snapshot.visible_board_count != 0:
        raise RuntimeError("Phase2B10 anchor is not a preflop root")

    targets = []
    nodes = []
    started = time.perf_counter()
    fixed_traversal = _fixed_traversal_seed(scenario_index, anchor_index)
    for replicate in range(REPLICATES):
        deal = _resample_deal(snapshot, actor, arm, replicate, scenario_index, anchor_index)
        traversal_seed = (
            _variant_seed(scenario_index, anchor_index, arm, replicate)
            if arm == "TRAVERSAL_ONLY"
            else fixed_traversal
        )
        target, node_count = _one_target(episode, deal, traversal_seed, expected)
        targets.append([float(v) for v in target])
        nodes.append(int(node_count))
    return {
        "source_behavior_seed": int(_WORKER_BEHAVIOR_SEED),
        "scenario_index": scenario_index,
        "anchor_index": anchor_index,
        "anchor_deck_seed": int(task["anchor_deck_seed"]),
        "arm": arm,
        "targets": targets,
        "legal_mask": list(expected["legal_mask"]),
        "nodes": nodes,
        "seconds": float(time.perf_counter() - started),
    }


def _tasks(collision_groups: Sequence[dict]) -> list[dict]:
    tasks = []
    if len(collision_groups) != 15:
        raise RuntimeError("Phase2B10 requires exactly 15 Phase2B1 collision groups")
    for group in collision_groups:
        deck_seeds = [int(x) for x in group.get("deck_seeds") or []]
        if len(deck_seeds) < ANCHORS_PER_SCENARIO:
            raise RuntimeError("Phase2B10 collision group does not contain four frozen anchors")
        for anchor_index in range(ANCHORS_PER_SCENARIO):
            for arm in ARMS:
                tasks.append({
                    "scenario_index": int(group["scenario_index"]),
                    "anchor_index": int(anchor_index),
                    "anchor_deck_seed": int(deck_seeds[anchor_index]),
                    "arm": arm,
                    "observation_sha256": str(group["observation_sha256"]),
                    "actor": int(group["actor"]),
                    "legal": tuple(int(x) for x in group["legal"]),
                    "legal_mask": tuple(int(x) for x in group["legal_mask"]),
                })
    return tasks


def _run_behavior_seed(repo_root: Path, solver_path: Path, behavior_seed: int, behavior_states: list[dict], collision_groups: Sequence[dict], workers: int) -> list[dict]:
    tasks = _tasks(collision_groups)
    results = []
    with ProcessPoolExecutor(
        max_workers=int(workers),
        initializer=_worker_init,
        initargs=(str(repo_root), str(solver_path), int(behavior_seed), behavior_states),
    ) as pool:
        future_map = {pool.submit(_worker_task, task): task for task in tasks}
        for future in as_completed(future_map):
            task = future_map[future]
            row = future.result()
            results.append(row)
            print(
                f"[Phase2B10 target] behavior={behavior_seed} scenario={task['scenario_index']:02d} "
                f"anchor={task['anchor_index']} arm={task['arm']} seconds={row['seconds']:.2f}",
                flush=True,
            )
    results.sort(key=lambda r: (int(r["source_behavior_seed"]), int(r["scenario_index"]), int(r["anchor_index"]), str(r["arm"])))
    return results


def _summaries(task_rows: Sequence[dict]) -> tuple[list[dict], dict, dict]:
    pair_rows = []
    for task in task_rows:
        for metric in _pair_metrics(task["targets"], task["legal_mask"]):
            pair_rows.append({
                "source_behavior_seed": int(task["source_behavior_seed"]),
                "scenario_index": int(task["scenario_index"]),
                "anchor_index": int(task["anchor_index"]),
                "arm": str(task["arm"]),
                **metric,
            })

    def summarize(rows: Sequence[dict]) -> dict:
        return {
            "pair_count": len(rows),
            "target_mean_abs_diff": _summary([r["target_mean_abs_diff"] for r in rows]),
            "legal_sign_disagreement_fraction": _summary([r["legal_sign_disagreement_fraction"] for r in rows]),
            "regret_matching_policy_tv": _summary([r["regret_matching_policy_tv"] for r in rows]),
            "dominant_legal_action_mismatch_rate": float(sum(int(r["dominant_legal_action_mismatch"]) for r in rows) / len(rows)) if rows else None,
        }

    by_seed = {}
    for seed in map(int, TRAINING_SEEDS):
        by_seed[str(seed)] = {}
        for arm in ARMS:
            rows = [r for r in pair_rows if int(r["source_behavior_seed"]) == seed and r["arm"] == arm]
            by_seed[str(seed)][arm] = summarize(rows)

    pooled = {}
    for arm in ARMS:
        pooled[arm] = summarize([r for r in pair_rows if r["arm"] == arm])
    return pair_rows, by_seed, pooled


def _decision(by_seed: dict, pooled: dict) -> dict:
    tv = {arm: float(pooled[arm]["regret_matching_policy_tv"]["mean"]) for arm in ARMS}
    traversal = tv["TRAVERSAL_ONLY"]
    private_excess = tv["PRIVATE_ONLY"] - traversal
    public_excess = tv["PUBLIC_ONLY"] - traversal
    combined_excess = tv["COMBINED"] - traversal
    private_pos = max(0.0, private_excess)
    public_pos = max(0.0, public_excess)
    public_both = all(
        float(by_seed[str(seed)]["PUBLIC_ONLY"]["regret_matching_policy_tv"]["mean"])
        > float(by_seed[str(seed)]["PRIVATE_ONLY"]["regret_matching_policy_tv"]["mean"])
        for seed in map(int, TRAINING_SEEDS)
    )
    private_both = all(
        float(by_seed[str(seed)]["PRIVATE_ONLY"]["regret_matching_policy_tv"]["mean"])
        > float(by_seed[str(seed)]["PUBLIC_ONLY"]["regret_matching_policy_tv"]["mean"])
        for seed in map(int, TRAINING_SEEDS)
    )
    if public_excess >= MATERIAL_EXCESS_TV and public_pos >= DOMINANCE_RATIO * max(private_pos, 1e-12) and public_both:
        classification = "PUBLIC_BOARD_CHANCE_DOMINANT"
        route = "PRECOMMIT_PUBLIC_CHANCE_SAMPLING_OR_STRATIFIED_BOARD_DIAGNOSTIC"
    elif private_excess >= MATERIAL_EXCESS_TV and private_pos >= DOMINANCE_RATIO * max(public_pos, 1e-12) and private_both:
        classification = "PRIVATE_HOLE_CHANCE_DOMINANT"
        route = "PRECOMMIT_PRIVATE_HAND_STRATIFIED_CHANCE_DIAGNOSTIC"
    elif combined_excess >= MATERIAL_EXCESS_TV:
        classification = "MIXED_PRIVATE_PUBLIC_CHANCE"
        route = "PRECOMMIT_FACTORIZED_PRIVATE_PUBLIC_CHANCE_VARIANCE_REDUCTION_DIAGNOSTIC"
    else:
        classification = "CHANCE_COMPONENT_DECOMPOSITION_UNRESOLVED"
        route = "REASSESS_REPRESENTATION_SUPPORT_AND_CHANCE_INTERACTION_BEFORE_TRAINING"
    return {
        "classification": classification,
        "arm_k1_policy_tv": tv,
        "private_excess_vs_traversal": float(private_excess),
        "public_excess_vs_traversal": float(public_excess),
        "combined_excess_vs_traversal": float(combined_excess),
        "interaction_excess_combined_minus_max_single": float(tv["COMBINED"] - max(tv["PRIVATE_ONLY"], tv["PUBLIC_ONLY"])),
        "public_gt_private_both_source_behavior_seeds": bool(public_both),
        "private_gt_public_both_source_behavior_seeds": bool(private_both),
        "material_excess_tv_threshold": MATERIAL_EXCESS_TV,
        "dominance_ratio_threshold": DOMINANCE_RATIO,
        "next_route": route,
        "training_pilot_authorized": False,
        "architecture_winner_selected": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def _validate_result(path: Path, expected_sha: str, expected_status: str | None = None) -> dict:
    if _sha256(path) != expected_sha:
        raise RuntimeError(f"Phase2B10 prerequisite SHA mismatch for {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if expected_status is not None and payload.get("status") != expected_status:
        raise RuntimeError(f"Phase2B10 prerequisite status mismatch for {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="R7.5 Phase2B10 private/public chance decomposition")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--phase2b1-result", type=Path, required=True)
    parser.add_argument("--phase2b6-root", type=Path, required=True)
    parser.add_argument("--phase2b6-result", type=Path, required=True)
    parser.add_argument("--phase2b9-result", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    solver_path = args.solver.resolve()
    b1_result = _validate_result(args.phase2b1_result.resolve(), PHASE2B1_RESULT_SHA256)
    _validate_result(args.phase2b6_result.resolve(), PHASE2B6_RESULT_SHA256, "PREFLOP_DAMPING_CAUSAL_EFFECT_SUPPORTED_BUT_STILL_UNSTABLE")
    _validate_result(args.phase2b9_result.resolve(), PHASE2B9_RESULT_SHA256, "HUBER_ROBUSTNESS_SCREEN_FAIL_DO_NOT_TRAIN")
    if b1_result.get("schema") != b1.SCHEMA or (b1_result.get("decision") or {}).get("source_classification") != "CHANCE_DOMINANT":
        raise RuntimeError("Phase2B10 requires exact chance-dominant Phase2B1 result")
    collision_groups = list(b1_result.get("collision_groups") or [])
    workers = max(1, min(int(args.workers), MAX_WORKERS, os.cpu_count() or MAX_WORKERS))
    torch.set_num_threads(1)

    b6_root = args.phase2b6_root.resolve()
    states_by_seed = {}
    checkpoint_identity = []
    for seed in map(int, TRAINING_SEEDS):
        validate_phase2_v3_contract(repo_root, representation=REPRESENTATION, domain=DOMAIN, training_seed=seed)
        checkpoint = b6_root / f"seed_{seed}" / "resume_checkpoint.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"Phase2B10 missing Phase2B6 checkpoint {checkpoint}")
        states_by_seed[seed] = _load_b6_behavior_states(checkpoint, seed)
        checkpoint_identity.append({"training_seed": seed, "path": str(checkpoint), "sha256": _sha256(checkpoint)})

    probe = SolverLibrary(solver_path)
    if not probe.explicit_deal_available:
        raise RuntimeError("Phase2B10 requires explicit-deal diagnostic solver extension")

    started = time.perf_counter()
    task_rows = []
    runtime_by_seed = {}
    for seed in map(int, TRAINING_SEEDS):
        local = time.perf_counter()
        print(f"[Phase2B10] behavior seed {seed}: 240 tasks / 1920 root traversals with {workers} workers...", flush=True)
        task_rows.extend(_run_behavior_seed(repo_root, solver_path, seed, states_by_seed[seed], collision_groups, workers))
        runtime_by_seed[str(seed)] = float(time.perf_counter() - local)

    pair_rows, by_seed, pooled = _summaries(task_rows)
    decision = _decision(by_seed, pooled)
    result = {
        "schema": SCHEMA,
        "status": decision["classification"],
        "representation": REPRESENTATION,
        "domain": DOMAIN,
        "source_behavior": "EXACT_COMPLETED_PHASE2B6_WITH_25_PERCENT_PREFLOP_CONTINUATION_FLOOR",
        "training_seeds": list(map(int, TRAINING_SEEDS)),
        "action_candidate": ACTION_CANDIDATE,
        "exact_opponent_levels": EXACT_OPPONENT_LEVELS,
        "target_iteration": TARGET_ITERATION,
        "anchors_per_scenario": ANCHORS_PER_SCENARIO,
        "replicates_per_arm": REPLICATES,
        "arms": list(ARMS),
        "worker_processes": workers,
        "torch_threads_per_worker": 1,
        "total_root_target_traversals": len(task_rows) * REPLICATES,
        "pair_metric_row_count": len(pair_rows),
        "frozen_inputs": {
            "phase2b1_result_sha256": PHASE2B1_RESULT_SHA256,
            "phase2b6_result_sha256": PHASE2B6_RESULT_SHA256,
            "phase2b9_result_sha256": PHASE2B9_RESULT_SHA256,
            "phase2b6_checkpoints": checkpoint_identity,
        },
        "by_source_behavior_seed": by_seed,
        "pooled": pooled,
        "decision": decision,
        "runtime_seconds_by_source_behavior_seed": runtime_by_seed,
        "runtime_seconds_total": float(time.perf_counter() - started),
        "guardrails": [
            "Explicit-deal creation is used only for this read-only diagnostic and does not authorize explicit-deal production training.",
            "Every variant preserves the acting player's exact root SPNNIV3 observation, actor, and legal mask.",
            "PRIVATE_ONLY changes opponent holes while holding actor holes and the complete board fixed.",
            "PUBLIC_ONLY changes the complete board while holding all player hole cards fixed.",
            "COMBINED changes opponent holes and board while holding actor holes fixed.",
            "No model fit, optimizer step, reservoir insertion, Strategy collection, or checkpoint mutation occurs.",
        ],
        "architecture_winner_selected": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    print(json.dumps({
        "status": result["status"],
        "arm_k1_policy_tv": decision["arm_k1_policy_tv"],
        "private_excess": decision["private_excess_vs_traversal"],
        "public_excess": decision["public_excess_vs_traversal"],
        "combined_excess": decision["combined_excess_vs_traversal"],
        "interaction_excess": decision["interaction_excess_combined_minus_max_single"],
        "next_route": decision["next_route"],
        "runtime_seconds_total": result["runtime_seconds_total"],
        "result": str(out),
        "result_sha256": _sha256(out),
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
