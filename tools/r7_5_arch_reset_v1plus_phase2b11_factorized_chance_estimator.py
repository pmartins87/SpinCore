from __future__ import annotations

"""Phase2B11: equal-compute IID vs factorized private/public chance estimators.

Read-only diagnostic over the exact Phase2B6 behavior ensembles.  Every
individual explicit deal has the correct conditional distribution given the
acting player's root hole cards.  Factorized arms reuse private rows and public
random-order columns to create a crossed Monte-Carlo estimator; IID arms use the
same traversal budgets with independent private/public pairs.
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

import r7_5_arch_reset_v1plus_phase2b1_target_variance as b1
import r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition as b10
from spincore.r7_5_action_cfr import regret_matching_policy
from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_representation_v3_stage_contract import TRAINING_SEEDS
from spincore.solver import DealSnapshot

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B11_FACTORIZED_CHANCE_ESTIMATOR_V1"
DOMAIN = "THREE_HANDED"
PHASE2B1_RESULT_SHA256 = "f95751afeb17fcd5844bfcb2971577b92a400750444e5dabe2f4ddb5718ba6ef"
PHASE2B6_RESULT_SHA256 = "33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a"
PHASE2B10_RESULT_SHA256 = "0295574c6133eb05866ecbdccf7e31efa4e6e8936dbd8bb7e375e166b27fe4dc"
ANCHORS_PER_SCENARIO = 4
BLOCKS = 4
DESIGNS = ("IID4", "FACTOR2X2", "IID16", "FACTOR4X4")
TRAVERSALS_PER_DESIGN = {"IID4": 4, "FACTOR2X2": 4, "IID16": 16, "FACTOR4X4": 16}
MAX_WORKERS = 30
TV_ABS_GATE = 0.05
TV_REL_GATE = 0.20
SIGN_ABS_GATE = 0.05
SIGN_REL_GATE = 0.15
P95_TOLERANCE = 0.02
DOMINANT_MISMATCH_TOLERANCE = 0.02
FACTOR_SIZE_REVERSAL_TOLERANCE = 0.01
MASK64 = (1 << 64) - 1


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


def _seed(namespace: int, scenario_index: int, anchor_index: int, block: int, index: int) -> int:
    return (
        0x2B11FACADE000001
        ^ (int(namespace) * 0x9E3779B97F4A7C15)
        ^ (int(scenario_index) * 0xD1B54A32D192ED03)
        ^ (int(anchor_index) * 0x94D049BB133111EB)
        ^ (int(block) * 0xBF58476D1CE4E5B9)
        ^ (int(index) * 0xA24BAED4963EE407)
    ) & MASK64


def _traversal_seed(scenario_index: int, anchor_index: int) -> int:
    return (
        0x2B11A7710A5E0001
        ^ (int(scenario_index) * 0xD6E8FEB86659FD93)
        ^ (int(anchor_index) * 0xA5A3564E27F8862B)
    ) & MASK64


def _actor_cards(snapshot: DealSnapshot, actor: int) -> tuple[int, int]:
    cards = tuple(int(x) for x in snapshot.holes[int(actor)])
    if len(cards) != 2 or any(x < 0 or x >= 52 for x in cards) or cards[0] == cards[1]:
        raise ValueError("Phase2B11 invalid acting-player hole cards")
    return cards


def _private_holes(snapshot: DealSnapshot, actor: int, private_seed: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    actor_cards = _actor_cards(snapshot, actor)
    pool = [card for card in range(52) if card not in set(actor_cards)]
    rng = random.Random(int(private_seed))
    rng.shuffle(pool)
    holes = [[-1, -1] for _ in range(3)]
    holes[int(actor)] = [actor_cards[0], actor_cards[1]]
    cursor = 0
    for seat in range(3):
        if seat == int(actor):
            continue
        holes[seat] = [int(pool[cursor]), int(pool[cursor + 1])]
        cursor += 2
    return tuple(tuple(int(x) for x in row) for row in holes)


def _public_board(actor_cards: Sequence[int], holes: Sequence[Sequence[int]], public_seed: int) -> tuple[int, int, int, int, int]:
    actor_set = {int(x) for x in actor_cards}
    opponent_set = {
        int(card)
        for row in holes
        for card in row
        if int(card) >= 0 and int(card) not in actor_set
    }
    if len(opponent_set) != 4:
        raise ValueError("Phase2B11 requires four distinct opponent private cards")
    # A shared public key is a random ordering of the 50 actor-excluded cards.
    # Filtering the four row-specific private cards leaves a uniform ordering of
    # the valid 46-card board deck for that row.
    ordering = [card for card in range(52) if card not in actor_set]
    rng = random.Random(int(public_seed))
    rng.shuffle(ordering)
    board = tuple(int(card) for card in ordering if card not in opponent_set)[:5]
    if len(board) != 5 or len(set(board)) != 5:
        raise AssertionError("Phase2B11 public-key board generation failed")
    return board


def _deal_from_factors(snapshot: DealSnapshot, actor: int, private_seed: int, public_seed: int) -> DealSnapshot:
    holes = _private_holes(snapshot, actor, private_seed)
    actor_cards = _actor_cards(snapshot, actor)
    board = _public_board(actor_cards, holes, public_seed)
    cards = [int(card) for row in holes for card in row] + [int(card) for card in board]
    if len(cards) != 11 or len(cards) != len(set(cards)) or any(card < 0 or card >= 52 for card in cards):
        raise AssertionError("Phase2B11 factor generation produced malformed deal")
    if tuple(holes[int(actor)]) != tuple(actor_cards):
        raise AssertionError("Phase2B11 factor generation changed actor holes")
    return DealSnapshot(holes=holes, board=board, visible_board_count=0)


def _iid_deals(snapshot: DealSnapshot, actor: int, *, scenario_index: int, anchor_index: int, block: int, count: int, namespace: int) -> list[DealSnapshot]:
    out = []
    for index in range(int(count)):
        private_seed = _seed(namespace, scenario_index, anchor_index, block, 2 * index)
        public_seed = _seed(namespace + 1, scenario_index, anchor_index, block, 2 * index + 1)
        out.append(_deal_from_factors(snapshot, actor, private_seed, public_seed))
    return out


def _factorized_deals(snapshot: DealSnapshot, actor: int, *, scenario_index: int, anchor_index: int, block: int, side: int, namespace: int) -> list[DealSnapshot]:
    private_seeds = [
        _seed(namespace, scenario_index, anchor_index, block, row)
        for row in range(int(side))
    ]
    public_seeds = [
        _seed(namespace + 1, scenario_index, anchor_index, block, col)
        for col in range(int(side))
    ]
    return [
        _deal_from_factors(snapshot, actor, private_seed, public_seed)
        for private_seed in private_seeds
        for public_seed in public_seeds
    ]


def _mean_targets(targets: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not targets:
        raise ValueError("Phase2B11 cannot average an empty target set")
    width = len(targets[0])
    if width != 10 or any(len(row) != width for row in targets):
        raise ValueError("Phase2B11 target width drift")
    return tuple(float(sum(float(row[i]) for row in targets) / len(targets)) for i in range(width))


def _worker_init(repo_root: str, solver_path: str, behavior_seed: int, behavior_states: list[dict]) -> None:
    b10._worker_init(repo_root, solver_path, int(behavior_seed), behavior_states)


def _design_deals(snapshot: DealSnapshot, actor: int, design: str, scenario_index: int, anchor_index: int, block: int) -> list[DealSnapshot]:
    if design == "IID4":
        return _iid_deals(snapshot, actor, scenario_index=scenario_index, anchor_index=anchor_index, block=block, count=4, namespace=101)
    if design == "FACTOR2X2":
        return _factorized_deals(snapshot, actor, scenario_index=scenario_index, anchor_index=anchor_index, block=block, side=2, namespace=201)
    if design == "IID16":
        return _iid_deals(snapshot, actor, scenario_index=scenario_index, anchor_index=anchor_index, block=block, count=16, namespace=301)
    if design == "FACTOR4X4":
        return _factorized_deals(snapshot, actor, scenario_index=scenario_index, anchor_index=anchor_index, block=block, side=4, namespace=401)
    raise ValueError(f"unknown Phase2B11 design {design}")


def _worker_task(task: dict) -> dict:
    if b10._WORKER_SOLVER is None or b10._WORKER_ACTION_SPEC is None:
        raise RuntimeError("Phase2B11 worker not initialized")
    scenarios = action_scenario_cycle(DOMAIN)
    scenario_index = int(task["scenario_index"])
    anchor_index = int(task["anchor_index"])
    block = int(task["block"])
    episode = scenarios[scenario_index]
    expected = {
        "observation_sha256": str(task["observation_sha256"]),
        "actor": int(task["actor"]),
        "legal": tuple(int(x) for x in task["legal"]),
        "legal_mask": tuple(int(x) for x in task["legal_mask"]),
    }
    anchor = b10._WORKER_SOLVER.create(episode, int(task["anchor_deck_seed"]))
    try:
        observation, actor, legal, mask = b1._root_identity(anchor, b10._WORKER_ACTION_SPEC)
        if hashlib.sha256(observation).hexdigest() != expected["observation_sha256"] or actor != expected["actor"] or legal != expected["legal"] or mask != expected["legal_mask"]:
            raise RuntimeError("Phase2B11 stored anchor identity drift")
        snapshot = anchor.deal_snapshot()
    finally:
        anchor.close()
    if snapshot.visible_board_count != 0:
        raise RuntimeError("Phase2B11 anchor must be an initial preflop root")

    fixed_traversal = _traversal_seed(scenario_index, anchor_index)
    estimators = {}
    node_totals = {}
    started = time.perf_counter()
    for design in DESIGNS:
        deals = _design_deals(snapshot, actor, design, scenario_index, anchor_index, block)
        if len(deals) != TRAVERSALS_PER_DESIGN[design]:
            raise AssertionError("Phase2B11 equal-compute design count drift")
        targets = []
        nodes = 0
        for deal in deals:
            target, node_count = b10._one_target(episode, deal, fixed_traversal, expected)
            targets.append(target)
            nodes += int(node_count)
        estimators[design] = [float(x) for x in _mean_targets(targets)]
        node_totals[design] = int(nodes)
    return {
        "source_behavior_seed": int(b10._WORKER_BEHAVIOR_SEED),
        "scenario_index": scenario_index,
        "anchor_index": anchor_index,
        "block": block,
        "anchor_deck_seed": int(task["anchor_deck_seed"]),
        "legal_mask": list(expected["legal_mask"]),
        "estimators": estimators,
        "node_totals": node_totals,
        "seconds": float(time.perf_counter() - started),
    }


def _tasks(collision_groups: Sequence[dict]) -> list[dict]:
    if len(collision_groups) != 15:
        raise RuntimeError("Phase2B11 requires exactly 15 Phase2B1 collision groups")
    tasks = []
    for group in collision_groups:
        seeds = [int(x) for x in group.get("deck_seeds") or []]
        if len(seeds) < ANCHORS_PER_SCENARIO:
            raise RuntimeError("Phase2B11 collision group lacks four frozen anchors")
        for anchor_index in range(ANCHORS_PER_SCENARIO):
            for block in range(BLOCKS):
                tasks.append({
                    "scenario_index": int(group["scenario_index"]),
                    "anchor_index": int(anchor_index),
                    "block": int(block),
                    "anchor_deck_seed": int(seeds[anchor_index]),
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
                f"[Phase2B11 target] behavior={behavior_seed} scenario={task['scenario_index']:02d} "
                f"anchor={task['anchor_index']} block={task['block']} seconds={row['seconds']:.2f}",
                flush=True,
            )
    results.sort(key=lambda row: (int(row["source_behavior_seed"]), int(row["scenario_index"]), int(row["anchor_index"]), int(row["block"])))
    return results


def _pair_metric(left: Sequence[float], right: Sequence[float], legal_mask_row: Sequence[int]) -> dict:
    legal = tuple(index for index, enabled in enumerate(legal_mask_row) if int(enabled))
    if not legal:
        raise ValueError("Phase2B11 empty legal mask")
    mad = float(sum(abs(float(left[a]) - float(right[a])) for a in legal) / len(legal))
    sign = float(sum((float(left[a]) > 0.0) != (float(right[a]) > 0.0) for a in legal) / len(legal))
    lp = regret_matching_policy(left, legal)
    rp = regret_matching_policy(right, legal)
    ldom = max(legal, key=lambda a: float(lp[a]))
    rdom = max(legal, key=lambda a: float(rp[a]))
    return {
        "target_mean_abs_diff": mad,
        "legal_sign_disagreement_fraction": sign,
        "regret_matching_policy_tv": _policy_tv(lp, rp),
        "dominant_legal_action_mismatch": int(ldom != rdom),
    }


def _summaries(task_rows: Sequence[dict]) -> tuple[list[dict], dict, dict]:
    index = {
        (int(row["source_behavior_seed"]), int(row["scenario_index"]), int(row["anchor_index"]), int(row["block"])): row
        for row in task_rows
    }
    pair_rows = []
    for seed in map(int, TRAINING_SEEDS):
        for scenario_index in range(15):
            for anchor_index in range(ANCHORS_PER_SCENARIO):
                for pair_start in (0, 2):
                    left_row = index[(seed, scenario_index, anchor_index, pair_start)]
                    right_row = index[(seed, scenario_index, anchor_index, pair_start + 1)]
                    if tuple(left_row["legal_mask"]) != tuple(right_row["legal_mask"]):
                        raise RuntimeError("Phase2B11 paired block legal-mask drift")
                    for design in DESIGNS:
                        metric = _pair_metric(
                            left_row["estimators"][design],
                            right_row["estimators"][design],
                            left_row["legal_mask"],
                        )
                        pair_rows.append({
                            "source_behavior_seed": seed,
                            "scenario_index": scenario_index,
                            "anchor_index": anchor_index,
                            "pair_start": pair_start,
                            "design": design,
                            **metric,
                        })

    def summarize(rows: Sequence[dict]) -> dict:
        return {
            "pair_count": len(rows),
            "target_mean_abs_diff": _summary([row["target_mean_abs_diff"] for row in rows]),
            "legal_sign_disagreement_fraction": _summary([row["legal_sign_disagreement_fraction"] for row in rows]),
            "regret_matching_policy_tv": _summary([row["regret_matching_policy_tv"] for row in rows]),
            "dominant_legal_action_mismatch_rate": float(sum(int(row["dominant_legal_action_mismatch"]) for row in rows) / len(rows)) if rows else None,
        }

    by_seed = {}
    for seed in map(int, TRAINING_SEEDS):
        by_seed[str(seed)] = {}
        for design in DESIGNS:
            rows = [row for row in pair_rows if int(row["source_behavior_seed"]) == seed and row["design"] == design]
            by_seed[str(seed)][design] = summarize(rows)
    pooled = {}
    for design in DESIGNS:
        pooled[design] = summarize([row for row in pair_rows if row["design"] == design])
    return pair_rows, by_seed, pooled


def _comparison(control: dict, candidate: dict) -> dict:
    control_tv = float(control["regret_matching_policy_tv"]["mean"])
    candidate_tv = float(candidate["regret_matching_policy_tv"]["mean"])
    tv_abs = control_tv - candidate_tv
    tv_rel = tv_abs / control_tv if control_tv > 0.0 else 0.0
    control_sign = float(control["legal_sign_disagreement_fraction"]["mean"])
    candidate_sign = float(candidate["legal_sign_disagreement_fraction"]["mean"])
    sign_abs = control_sign - candidate_sign
    sign_rel = sign_abs / control_sign if control_sign > 0.0 else 0.0
    return {
        "control_mean_tv": control_tv,
        "candidate_mean_tv": candidate_tv,
        "tv_absolute_improvement": float(tv_abs),
        "tv_relative_improvement": float(tv_rel),
        "control_sign_disagreement": control_sign,
        "candidate_sign_disagreement": candidate_sign,
        "sign_absolute_improvement": float(sign_abs),
        "sign_relative_improvement": float(sign_rel),
        "control_target_mad": float(control["target_mean_abs_diff"]["mean"]),
        "candidate_target_mad": float(candidate["target_mean_abs_diff"]["mean"]),
        "control_p95_tv": float(control["regret_matching_policy_tv"]["p95"]),
        "candidate_p95_tv": float(candidate["regret_matching_policy_tv"]["p95"]),
        "control_dominant_mismatch": float(control["dominant_legal_action_mismatch_rate"]),
        "candidate_dominant_mismatch": float(candidate["dominant_legal_action_mismatch_rate"]),
    }


def _decision(by_seed: dict, pooled: dict) -> dict:
    primary = _comparison(pooled["IID16"], pooled["FACTOR4X4"])
    secondary = _comparison(pooled["IID4"], pooled["FACTOR2X2"])
    tv_material = bool(primary["tv_absolute_improvement"] >= TV_ABS_GATE or primary["tv_relative_improvement"] >= TV_REL_GATE)
    sign_material = bool(primary["sign_absolute_improvement"] >= SIGN_ABS_GATE or primary["sign_relative_improvement"] >= SIGN_REL_GATE)
    both_seed = all(
        float(by_seed[str(seed)]["FACTOR4X4"]["regret_matching_policy_tv"]["mean"])
        < float(by_seed[str(seed)]["IID16"]["regret_matching_policy_tv"]["mean"])
        for seed in map(int, TRAINING_SEEDS)
    )
    p95_ok = bool(primary["candidate_p95_tv"] <= primary["control_p95_tv"] + P95_TOLERANCE)
    dom_ok = bool(primary["candidate_dominant_mismatch"] <= primary["control_dominant_mismatch"] + DOMINANT_MISMATCH_TOLERANCE)
    no_reversal = bool(
        float(pooled["FACTOR4X4"]["regret_matching_policy_tv"]["mean"])
        <= float(pooled["FACTOR2X2"]["regret_matching_policy_tv"]["mean"]) + FACTOR_SIZE_REVERSAL_TOLERANCE
    )
    screen_pass = bool(tv_material and sign_material and both_seed and p95_ok and dom_ok and no_reversal)
    raw_coherent = bool(
        primary["candidate_target_mad"] < primary["control_target_mad"]
        and primary["candidate_sign_disagreement"] < primary["control_sign_disagreement"]
    )
    if screen_pass:
        status = "FACTORIZED_CHANCE_ESTIMATOR_SCREEN_PASS"
        route = "PRECOMMIT_SMALL_FACTORIZED_CHANCE_TARGET_TRAINING_PILOT_WITH_EQUAL_COMPUTE_CONTROL"
    elif raw_coherent:
        status = "FACTORIZED_TARGETS_IMPROVE_BUT_POLICY_TV_GATE_FAIL"
        route = "INVESTIGATE_REGRET_MATCHING_SENSITIVITY_AFTER_FACTORIZED_TARGET_ESTIMATION_NO_TRAINING"
    else:
        status = "FACTORIZED_CHANCE_ESTIMATOR_SCREEN_FAIL"
        route = "REASSESS_SOLVER_LEVEL_CHANCE_EXPECTATION_OR_REPRESENTATION_SUPPORT_NO_TRAINING"
    return {
        "classification": status,
        "primary_factor4x4_vs_iid16": primary,
        "secondary_factor2x2_vs_iid4": secondary,
        "tv_materiality_pass": tv_material,
        "sign_materiality_pass": sign_material,
        "both_source_behavior_seeds_directionally_improve": both_seed,
        "p95_non_degradation_pass": p95_ok,
        "dominant_action_mismatch_non_degradation_pass": dom_ok,
        "factor_size_nonreversal_pass": no_reversal,
        "raw_target_and_sign_directionally_improve": raw_coherent,
        "screen_pass": screen_pass,
        "next_route": route,
        "small_causal_training_pilot_precommit_allowed": screen_pass,
        "training_pilot_authorized": False,
        "architecture_winner_selected": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }


def _validate_result(path: Path, sha: str, *, status: str | None = None) -> dict:
    if _sha256(path) != sha:
        raise RuntimeError(f"Phase2B11 prerequisite SHA mismatch for {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if status is not None and payload.get("status") != status:
        raise RuntimeError(f"Phase2B11 prerequisite status mismatch for {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="R7.5 Phase2B11 factorized private/public chance estimator screen")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--phase2b1-result", type=Path, required=True)
    parser.add_argument("--phase2b6-root", type=Path, required=True)
    parser.add_argument("--phase2b6-result", type=Path, required=True)
    parser.add_argument("--phase2b10-result", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    solver_path = args.solver.resolve()
    b1_result = _validate_result(args.phase2b1_result.resolve(), PHASE2B1_RESULT_SHA256)
    _validate_result(args.phase2b6_result.resolve(), PHASE2B6_RESULT_SHA256, status="PREFLOP_DAMPING_CAUSAL_EFFECT_SUPPORTED_BUT_STILL_UNSTABLE")
    b10_result = _validate_result(args.phase2b10_result.resolve(), PHASE2B10_RESULT_SHA256, status="MIXED_PRIVATE_PUBLIC_CHANCE")
    if b1_result.get("schema") != b1.SCHEMA:
        raise RuntimeError("Phase2B11 Phase2B1 schema mismatch")
    if b10_result.get("schema") != b10.SCHEMA or (b10_result.get("decision") or {}).get("next_route") != "PRECOMMIT_FACTORIZED_PRIVATE_PUBLIC_CHANCE_VARIANCE_REDUCTION_DIAGNOSTIC":
        raise RuntimeError("Phase2B11 requires exact mixed Phase2B10 routing")
    collision_groups = list(b1_result.get("collision_groups") or [])
    workers = max(1, min(int(args.workers), MAX_WORKERS, os.cpu_count() or MAX_WORKERS))
    torch.set_num_threads(1)

    b6_root = args.phase2b6_root.resolve()
    states_by_seed = {}
    checkpoint_identity = []
    for seed in map(int, TRAINING_SEEDS):
        checkpoint = b6_root / f"seed_{seed}" / "resume_checkpoint.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"Phase2B11 missing Phase2B6 checkpoint {checkpoint}")
        states = b10._load_b6_behavior_states(checkpoint, seed)
        states_by_seed[seed] = states
        checkpoint_identity.append({"training_seed": seed, "path": str(checkpoint), "sha256": _sha256(checkpoint), "behavior_members": len(states)})

    started = time.perf_counter()
    task_rows = []
    seconds_by_seed = {}
    for seed in map(int, TRAINING_SEEDS):
        local = time.perf_counter()
        print(
            f"[Phase2B11] behavior seed {seed}: 240 tasks / 9600 root traversals with {workers} workers...",
            flush=True,
        )
        rows = _run_behavior_seed(repo_root, solver_path, seed, states_by_seed[seed], collision_groups, workers)
        task_rows.extend(rows)
        seconds_by_seed[str(seed)] = float(time.perf_counter() - local)

    pair_rows, by_seed, pooled = _summaries(task_rows)
    decision = _decision(by_seed, pooled)
    result = {
        "schema": SCHEMA,
        "status": decision["classification"],
        "representation": b10.REPRESENTATION,
        "domain": DOMAIN,
        "source_behavior": "EXACT_COMPLETED_PHASE2B6_WITH_25_PERCENT_PREFLOP_CONTINUATION_FLOOR",
        "training_seeds": [int(seed) for seed in TRAINING_SEEDS],
        "anchors_per_scenario": ANCHORS_PER_SCENARIO,
        "blocks": BLOCKS,
        "designs": list(DESIGNS),
        "traversals_per_estimator": dict(TRAVERSALS_PER_DESIGN),
        "total_root_target_traversals": 19200,
        "worker_processes": workers,
        "torch_threads_per_worker": 1,
        "frozen_inputs": {
            "phase2b1_result_sha256": PHASE2B1_RESULT_SHA256,
            "phase2b6_result_sha256": PHASE2B6_RESULT_SHA256,
            "phase2b10_result_sha256": PHASE2B10_RESULT_SHA256,
            "phase2b6_checkpoints": checkpoint_identity,
        },
        "by_source_behavior_seed": by_seed,
        "pooled": pooled,
        "decision": decision,
        "pair_metric_row_count": len(pair_rows),
        "task_count": len(task_rows),
        "runtime_seconds_by_source_behavior_seed": seconds_by_seed,
        "runtime_seconds_total": float(time.perf_counter() - started),
        "guardrails": [
            "Every explicit deal preserves the actor's exact root hole cards, SPNNIV3 observation, actor and legal identity.",
            "Every crossed cell has the correct conditional private/public deal marginal; factorization changes coupling only.",
            "IID and factorized estimators are compared at equal root-traversal budgets of 4 and 16.",
            "Raw target vectors are averaged before diagnostic regret matching.",
            "No model fit, optimizer step, reservoir insertion, Strategy collection, AveragePolicy fit or checkpoint mutation occurs.",
            "A screen PASS permits only a separately precommitted small causal training pilot; it does not authorize training here.",
        ],
        "training_pilot_authorized": False,
        "architecture_winner_selected": False,
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
        "screen_pass": decision["screen_pass"],
        "primary": decision["primary_factor4x4_vs_iid16"],
        "secondary": decision["secondary_factor2x2_vs_iid4"],
        "next_route": decision["next_route"],
        "runtime_seconds_total": result["runtime_seconds_total"],
    }, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
