from __future__ import annotations

"""Phase2C2: bounded structural range/reach target-kernel causal pilot.

Both arms spend identical K64 auxiliary root and continuation target compute.
The already-supported B13 root IID64 mean is used in both arms.  One guaranteed
two-action preflop continuation Advantage sample per logical root is also
replaced.  The control uses the first target from an exact Phase2C1
range-stratified K64 proposal set; the candidate uses the arithmetic mean of the
same 64 targets.

Fresh reservoirs/optimizers are used, but iteration-1 behavior is bootstrapped
from the exact final B13 IID64 candidate four-member ensemble.  Two iterations
allow the continuation target intervention to affect behavior and then the
AveragePolicy collection in the second iteration.
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

import numpy as np
import torch

import r7_5_3d_v1plus_phase2a_strategy_capacity as phase2a
import r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot as b6
import r7_5_arch_reset_v1plus_phase2b7_residual_localization as b7
import r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition as b10
import r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training as b13
import r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance as b15
import r7_5_arch_reset_v1plus_phase2c0_structural_reach_factorization as c0
import r7_5_arch_reset_v1plus_phase2c1_exact_range_reach_solver_prototype as c1
import spincore.r7_5_representation_v3_stage as stage

from spincore.r7_5_action_cfr import ActionAdvantageSample, legal_mask, validate_policy
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
from spincore.r7_5_representation_v3_referee_states import effective_pf0
from spincore.r7_5_representation_v3_stage import frozen_config, new_phase2_v3_runtime
from spincore.r7_5_representation_v3_stage_contract import (
    ACTION_CANDIDATE,
    ADVANTAGE_NRMSE_MAX,
    BATCH_SIZE,
    CROSS_SEED_MEAN_TV_MAX,
    CROSS_SEED_P95_TV_MAX,
    EVALUATION_SEEDS,
    EXACT_OPPONENT_LEVELS,
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
from spincore.solver import DealSnapshot, SolverLibrary
from spincore.solver_v3 import neural_bytes_v3

SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2C2_RANGE_REACH_TARGET_KERNEL_CAUSAL_PILOT_V1"
SEED_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2C2_SEED_V1"
CHECKPOINT_EXTRA_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2C2_RESUME_V1"

DOMAIN = "THREE_HANDED"
REPRESENTATION = H2_FINAL
FLOOR = 0.25
K = 64
STRATA_SIDE = 8
ARMS = ("RANGE1_EQUAL_COMPUTE_CONTROL", "RANGE64_MEAN_CANDIDATE")
CONTROL_ARM, CANDIDATE_ARM = ARMS

PILOT_ITERATIONS = 2
CHUNKS_PER_ITERATION = 2
ROOTS_PER_CHUNK = ROOTS_PER_ITERATION // CHUNKS_PER_ITERATION
if ROOTS_PER_ITERATION % CHUNKS_PER_ITERATION:
    raise RuntimeError("Phase2C2 requires ROOTS_PER_ITERATION divisible by chunk count")
ROOTS_PER_ITERATION_EFFECTIVE = ROOTS_PER_CHUNK * CHUNKS_PER_ITERATION
TOTAL_ROOTS = PILOT_ITERATIONS * ROOTS_PER_ITERATION_EFFECTIVE
POLICY_COUNT = 1024

C1_RESULT_SHA256 = "62ad2352c807a3b046bc84df2cbdf66cc8e0217e3422d01f2bcd9ddeafe7875b"
C1_STATUS = "EXACT_RANGE_REACH_TRANSITION_PROTOTYPE_FEASIBLE"
B13_RESULT_SHA256 = "6de7996282236d34adf5e8e53416fd8a443a1fbf5abc89fc807492d0cb3dbf80"
B14_RESULT_SHA256 = "7cd1886596d345abdcdef479775498eddf7e014205de86e44afb5bb0ea291f86"

COMMON_POLICY_INIT_SEED = b13.COMMON_POLICY_INIT_SEED
COMMON_BATCH_SEED = b13.COMMON_BATCH_SEED
BOOTSTRAP_REPLICATES = 2000
CAUSAL_ABS_MIN = 0.020
CAUSAL_REL_MIN = 0.10
COMMON_P95_MAX_DEGRADE = 0.020
NATIVE_MEAN_MAX_DEGRADE = 0.010
ROOT_MEAN_MAX_DEGRADE = 0.020

MASK64 = (1 << 64) - 1
ACTION_PREFERENCE = (1, 2, 3, 4, 5, 6, 7, 8, 9, 0)
RANGE_NAMESPACE = 0x2C020A11CE000001
BOARD_NAMESPACE = 0x2C020B0A4D000001
TRAVERSAL_NAMESPACE = 0x2C02A7710A5E0001


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


def _mix64(*parts: int) -> int:
    x = RANGE_NAMESPACE
    for raw in parts:
        y = int(raw) & MASK64
        x ^= (y + 0x9E3779B97F4A7C15 + ((x << 6) & MASK64) + (x >> 2)) & MASK64
        x ^= x >> 30
        x = (x * 0xBF58476D1CE4E5B9) & MASK64
        x ^= x >> 27
        x = (x * 0x94D049BB133111EB) & MASK64
        x ^= x >> 31
    return x & MASK64


def _mean_targets(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if len(rows) != K or any(len(row) != 10 for row in rows):
        raise RuntimeError("Phase2C2 requires exactly K ten-action targets")
    return tuple(float(sum(float(row[i]) for row in rows) / K) for i in range(10))


def _behavior_states(behavior) -> list[dict]:
    if len(behavior.models) != 4:
        raise RuntimeError("Phase2C2 requires a four-member behavior ensemble")
    return [model.state_dict() for model in behavior.models]


def _preferred_positive_actions(state) -> list[int]:
    if b10._WORKER_COLLECTOR is None or b10._WORKER_ACTION_SPEC is None:
        raise RuntimeError("Phase2C2 worker behavior is not initialized")
    observation = neural_bytes_v3(state)
    active_mask, legal, _exact = effective_pf0(state, b10._WORKER_ACTION_SPEC)
    probabilities = validate_policy(
        b10._WORKER_COLLECTOR.policy(state, observation, legal), legal
    )
    return [
        action
        for action in ACTION_PREFERENCE
        if action in legal and float(probabilities[action]) > 0.0
    ]


def _select_depth2_continuation(task: dict) -> dict:
    """Find a deterministic positive-support two-action preflop continuation."""
    if b10._WORKER_SOLVER is None or b10._WORKER_ACTION_SPEC is None:
        raise RuntimeError("Phase2C2 worker solver is not initialized")
    episode = action_scenario_cycle(DOMAIN)[int(task["scenario_index"])]
    root = b10._WORKER_SOLVER.create(episode, int(task["anchor_deck_seed"]))
    try:
        for a0 in _preferred_positive_actions(root):
            active0, legal0, _ = effective_pf0(root, b10._WORKER_ACTION_SPEC)
            if a0 not in legal0:
                continue
            s1 = root.child_universal(active0, int(a0))
            try:
                if s1.terminal:
                    continue
                obs1 = neural_bytes_v3(s1)
                street1, nonforced1 = b6._v3_street_and_nonforced_preflop(obs1)
                if street1 != 0 or nonforced1 < 1:
                    continue
                for a1 in _preferred_positive_actions(s1):
                    active1, legal1, _ = effective_pf0(s1, b10._WORKER_ACTION_SPEC)
                    if a1 not in legal1:
                        continue
                    s2 = s1.child_universal(active1, int(a1))
                    try:
                        if s2.terminal:
                            continue
                        observation = neural_bytes_v3(s2)
                        street2, nonforced2 = b6._v3_street_and_nonforced_preflop(observation)
                        if street2 != 0 or nonforced2 < 2:
                            continue
                        active2, legal2, _ = effective_pf0(s2, b10._WORKER_ACTION_SPEC)
                        actor = int(s2.actor)
                        snapshot = s2.deal_snapshot()
                        if snapshot.visible_board_count != 0:
                            raise RuntimeError("Phase2C2 selected continuation has visible board")
                        return {
                            "training_seed": int(task["training_seed"]),
                            "scenario_index": int(task["scenario_index"]),
                            "global_root": int(task["global_root"]),
                            "iteration": int(task["iteration"]),
                            "deck_seed": int(task["anchor_deck_seed"]),
                            "action_path": [int(a0), int(a1)],
                            "actor": actor,
                            "observation": bytes(observation),
                            "observation_sha256": hashlib.sha256(observation).hexdigest(),
                            "active_mask": int(active2),
                            "legal_slots": [int(x) for x in legal2],
                            "legal_mask": list(legal_mask(tuple(int(x) for x in legal2))),
                            "region": "PREFLOP_CONTINUATION_2PLUS",
                            "_snapshot": snapshot,
                        }
                    finally:
                        s2.close()
            finally:
                s1.close()
    finally:
        root.close()
    raise RuntimeError(
        f"Phase2C2 could not find positive-support depth-2 continuation "
        f"scenario={task['scenario_index']} root={task['global_root']}"
    )


def _range_vectors(cont: dict) -> tuple[list[tuple[int, int]], list[int], dict[int, np.ndarray], int]:
    snapshot: DealSnapshot = cont["_snapshot"]
    actor = int(cont["actor"])
    opponents = [seat for seat in range(3) if seat != actor]
    hands = c0._ordered_hands(snapshot.holes[actor])
    reaches = {seat: np.ones(len(hands), dtype=np.float64) for seat in opponents}
    episode = action_scenario_cycle(DOMAIN)[int(cont["scenario_index"])]
    canonical = b10._WORKER_SOLVER.create_with_deal(episode, snapshot.holes, snapshot.board)
    evals = 0
    try:
        for event_index, action in enumerate(cont["action_path"]):
            if canonical.terminal:
                raise RuntimeError("Phase2C2 reach replay reached terminal early")
            acting = int(canonical.actor)
            active_mask, legal, _ = effective_pf0(canonical, b10._WORKER_ACTION_SPEC)
            if int(action) not in legal:
                raise RuntimeError("Phase2C2 selected public action became illegal")
            if acting != actor:
                if acting not in reaches:
                    raise RuntimeError("Phase2C2 acting seat outside opponent reach state")
                table = reaches[acting]
                for hand_index, hand in enumerate(hands):
                    p = c1._event_probability(
                        cont,
                        snapshot,
                        target_seat=acting,
                        hand=hand,
                        event_index=event_index,
                    )
                    if not math.isfinite(p) or p < 0.0 or p > 1.0 + 1e-12:
                        raise RuntimeError("Phase2C2 invalid reach update probability")
                    table[hand_index] *= float(p)
                evals += len(hands)
            canonical.apply_universal(active_mask, int(action))
        observation = neural_bytes_v3(canonical)
        active_mask, legal, _ = effective_pf0(canonical, b10._WORKER_ACTION_SPEC)
        if (
            observation != bytes(cont["observation"])
            or int(canonical.actor) != actor
            or int(active_mask) != int(cont["active_mask"])
            or tuple(legal) != tuple(cont["legal_slots"])
        ):
            raise RuntimeError("Phase2C2 reach propagation target identity drift")
    finally:
        canonical.close()
    return hands, opponents, reaches, int(evals)


def _cdf_draw(weights: np.ndarray, u: float) -> int:
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("Phase2C2 cannot draw from zero/nonfinite mass")
    if not 0.0 < float(u) < 1.0:
        raise ValueError("Phase2C2 stratified uniform must lie inside (0,1)")
    cdf = np.cumsum(weights, dtype=np.float64)
    target = float(u) * float(cdf[-1])
    idx = int(np.searchsorted(cdf, target, side="right"))
    return min(idx, len(weights) - 1)


def _stratified_joint_indices(
    hands: Sequence[tuple[int, int]],
    wa: np.ndarray,
    wb: np.ndarray,
    *,
    seed: int,
) -> tuple[list[tuple[int, int]], dict]:
    if len(hands) != 2450 or wa.shape != (2450,) or wb.shape != (2450,):
        raise RuntimeError("Phase2C2 reach table dimensions drift")
    b0 = np.asarray([h[0] for h in hands], dtype=np.int16)
    b1 = np.asarray([h[1] for h in hands], dtype=np.int16)
    valid_sums = np.empty(len(hands), dtype=np.float64)
    for i, (a0, a1) in enumerate(hands):
        mask = (b0 != a0) & (b1 != a0) & (b0 != a1) & (b1 != a1)
        valid_sums[i] = float(wb[mask].sum())
    marginal = wa * valid_sums
    normalizer = float(marginal.sum())
    if not math.isfinite(normalizer) or normalizer <= 0.0:
        raise RuntimeError("Phase2C2 structural joint posterior has zero/nonfinite mass")

    rng_a = random.Random(int(seed))
    selected: list[tuple[int, int]] = []
    a_indices = []
    for stratum_a in range(STRATA_SIDE):
        u_a = (float(stratum_a) + rng_a.random()) / STRATA_SIDE
        ia = _cdf_draw(marginal, u_a)
        a_indices.append(ia)
        a0, a1 = hands[ia]
        valid = (b0 != a0) & (b1 != a0) & (b0 != a1) & (b1 != a1)
        valid_indices = np.flatnonzero(valid)
        conditional = wb[valid_indices]
        rng_b = random.Random(_mix64(int(seed), int(stratum_a), int(ia), 0xBEEF))
        for stratum_b in range(STRATA_SIDE):
            u_b = (float(stratum_b) + rng_b.random()) / STRATA_SIDE
            local = _cdf_draw(conditional, u_b)
            ib = int(valid_indices[local])
            ha, hb = hands[ia], hands[ib]
            if len({ha[0], ha[1], hb[0], hb[1]}) != 4:
                raise RuntimeError("Phase2C2 stratified sampler produced card collision")
            selected.append((ia, ib))
    if len(selected) != K:
        raise RuntimeError("Phase2C2 structural stratification did not produce K assignments")
    return selected, {
        "joint_normalizer": normalizer,
        "unique_seat_a_indices": len(set(a_indices)),
        "unique_joint_assignments": len(set(selected)),
    }


def _board_for_joint(
    actor_cards: Sequence[int],
    hand_a: tuple[int, int],
    hand_b: tuple[int, int],
    *,
    seed: int,
) -> tuple[int, int, int, int, int]:
    used = {int(x) for x in actor_cards} | {int(x) for x in hand_a} | {int(x) for x in hand_b}
    if len(used) != 6:
        raise RuntimeError("Phase2C2 joint private assignment collision")
    pool = [card for card in range(52) if card not in used]
    rng = random.Random(int(seed))
    rng.shuffle(pool)
    return tuple(int(x) for x in pool[:5])


def _deal_from_joint(
    snapshot: DealSnapshot,
    actor: int,
    opponents: Sequence[int],
    hand_a: tuple[int, int],
    hand_b: tuple[int, int],
    board: Sequence[int],
) -> DealSnapshot:
    holes = [[-1, -1] for _ in range(3)]
    actor_cards = tuple(int(x) for x in snapshot.holes[int(actor)])
    holes[int(actor)] = [actor_cards[0], actor_cards[1]]
    holes[int(opponents[0])] = [int(hand_a[0]), int(hand_a[1])]
    holes[int(opponents[1])] = [int(hand_b[0]), int(hand_b[1])]
    flat = [int(x) for row in holes for x in row] + [int(x) for x in board]
    if len(flat) != 11 or len(flat) != len(set(flat)):
        raise RuntimeError("Phase2C2 explicit structural deal malformed")
    return DealSnapshot(
        holes=tuple(tuple(int(x) for x in row) for row in holes),
        board=tuple(int(x) for x in board),
        visible_board_count=0,
    )


def _continuation_target_task(task: dict) -> dict:
    cont = _select_depth2_continuation(task)
    snapshot: DealSnapshot = cont["_snapshot"]
    hands, opponents, reaches, table_evals = _range_vectors(cont)
    seat_a, seat_b = opponents
    assignments, support = _stratified_joint_indices(
        hands,
        reaches[seat_a],
        reaches[seat_b],
        seed=_mix64(
            int(task["training_seed"]),
            int(task["global_root"]),
            int(task["iteration"]),
            0xA64,
        ),
    )
    traversal_seed = _mix64(
        TRAVERSAL_NAMESPACE,
        int(task["training_seed"]),
        int(task["global_root"]),
        int(task["iteration"]),
    )
    targets = []
    nodes = 0
    started = time.perf_counter()
    for sample_index, (ia, ib) in enumerate(assignments):
        hand_a = hands[ia]
        hand_b = hands[ib]
        board = _board_for_joint(
            snapshot.holes[int(cont["actor"])],
            hand_a,
            hand_b,
            seed=_mix64(
                BOARD_NAMESPACE,
                int(task["training_seed"]),
                int(task["global_root"]),
                int(task["iteration"]),
                int(sample_index),
            ),
        )
        deal = _deal_from_joint(
            snapshot,
            int(cont["actor"]),
            opponents,
            hand_a,
            hand_b,
            board,
        )
        target, _logp, node_count = b15._variant_likelihood_and_target(
            cont, deal, int(traversal_seed)
        )
        targets.append(tuple(float(x) for x in target))
        nodes += int(node_count)

    result = {
        key: value for key, value in cont.items() if key != "_snapshot"
    }
    result.update(
        {
            "kind": "CONTINUATION",
            "first_target": [float(x) for x in targets[0]],
            "mean_target": [float(x) for x in _mean_targets(targets)],
            "target_traversals": K,
            "target_nodes": int(nodes),
            "reach_table_policy_evaluations": int(table_evals),
            "joint_normalizer": float(support["joint_normalizer"]),
            "unique_seat_a_indices": int(support["unique_seat_a_indices"]),
            "unique_joint_assignments": int(support["unique_joint_assignments"]),
            "seconds": float(time.perf_counter() - started),
        }
    )
    return result


def _combined_aux_task(task: dict) -> dict:
    root = b13._root_target_task(task)
    cont = _continuation_target_task(task)
    return {
        "training_seed": int(task["training_seed"]),
        "global_root": int(task["global_root"]),
        "iteration": int(task["iteration"]),
        "scenario_index": int(task["scenario_index"]),
        "anchor_deck_seed": int(task["anchor_deck_seed"]),
        "root": root,
        "continuation": cont,
    }


class MultiReplacingAdvantageMemory:
    """Replace exact root + exact continuation samples without changing add count/order."""

    def __init__(self, delegate, *, iteration: int, replacements: Sequence[dict]):
        self.delegate = delegate
        self.iteration = int(iteration)
        self.entries = {}
        self.counts = {}
        for row in replacements:
            obs = bytes(row["observation"])
            if obs in self.entries:
                raise RuntimeError("Phase2C2 duplicate replacement observation")
            target = tuple(float(x) for x in row["target"])
            legal = tuple(int(x) for x in row["legal_mask"])
            if len(target) != 10 or len(legal) != 10:
                raise RuntimeError("Phase2C2 replacement width drift")
            self.entries[obs] = (target, legal, str(row["label"]))
            self.counts[obs] = 0

    def add(self, sample) -> None:
        obs = bytes(sample.observation)
        if int(sample.iteration) == self.iteration and obs in self.entries:
            target, expected_legal, _label = self.entries[obs]
            self.counts[obs] += 1
            if self.counts[obs] != 1:
                raise RuntimeError("Phase2C2 replacement observation inserted multiple times")
            legal = tuple(int(x) for x in sample.legal)
            if legal != expected_legal:
                raise RuntimeError("Phase2C2 replacement legal-mask drift")
            self.delegate.add(
                ActionAdvantageSample(
                    observation=obs,
                    legal=legal,
                    target=target,
                    weight=float(sample.weight),
                    iteration=int(sample.iteration),
                )
            )
            return
        self.delegate.add(sample)

    def assert_complete(self) -> None:
        missing = [
            self.entries[obs][2]
            for obs, count in self.counts.items()
            if int(count) != 1
        ]
        if missing:
            raise RuntimeError(f"Phase2C2 missing exact replacements: {missing}")


def _aux_rows(
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
        max_workers=min(int(chance_workers), len(tasks)),
        initializer=b10._worker_init,
        initargs=(str(repo_root), str(solver_path), int(training_seed), states),
    ) as pool:
        fmap = {pool.submit(_combined_aux_task, task): task for task in tasks}
        for future in as_completed(fmap):
            task = fmap[future]
            row = future.result()
            rows.append(row)
            print(
                f"[Phase2C2 aux] seed={training_seed} i{target_iteration} "
                f"root={task['global_root']} root_s={row['root']['seconds']:.2f} "
                f"cont_s={row['continuation']['seconds']:.2f} "
                f"joint_unique={row['continuation']['unique_joint_assignments']}",
                flush=True,
            )
    rows.sort(key=lambda r: int(r["global_root"]))
    expected = list(range(int(start_global_root), int(start_global_root) + ROOTS_PER_CHUNK))
    if [int(r["global_root"]) for r in rows] != expected:
        raise RuntimeError("Phase2C2 auxiliary root coverage drift")
    return rows


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

    aux_rows = _aux_rows(
        repo_root=repo_root,
        solver_path=solver_path,
        training_seed=int(state["training_seed"]),
        behavior=behavior,
        start_global_root=start_global_root,
        target_iteration=int(target_iteration),
        chance_workers=int(chance_workers),
    )
    aux_index = {int(r["global_root"]): r for r in aux_rows}

    session.collector.reset_telemetry()
    roots_before = int(bundle.counters["roots"])
    nodes_before = int(bundle.counters["nodes"])
    adv_before = int(bundle.adv_mem.seen)
    pol_before = int(bundle.pol_mem.seen)
    floor_before = floor_policy.stats()
    root_replacements = 0
    continuation_replacements = 0
    started = time.perf_counter()

    for _ in range(ROOTS_PER_CHUNK):
        scenario_index = global_root % len(scenarios)
        row = aux_index[global_root]
        if int(row["scenario_index"]) != scenario_index:
            raise RuntimeError("Phase2C2 scenario scheduling drift")
        scenario_counts[scenario_index] += 1

        root_row = row["root"]
        cont_row = row["continuation"]
        root_target = root_row["mean_target"]
        cont_target = (
            cont_row["first_target"] if arm == CONTROL_ARM else cont_row["mean_target"]
        )
        proxy = MultiReplacingAdvantageMemory(
            bundle.adv_mem,
            iteration=int(target_iteration),
            replacements=[
                {
                    "label": "ROOT_IID64_MEAN",
                    "observation": bytes(root_row["root_observation"]),
                    "target": root_target,
                    "legal_mask": root_row["legal_mask"],
                },
                {
                    "label": "DEPTH2_RANGE_TARGET",
                    "observation": bytes(cont_row["observation"]),
                    "target": cont_target,
                    "legal_mask": cont_row["legal_mask"],
                },
            ],
        )
        original = session.collector.advantage_memory
        session.collector.advantage_memory = proxy
        try:
            session.collect_root(
                scenarios[scenario_index],
                iteration=int(target_iteration),
                exact_opponent_levels=EXACT_OPPONENT_LEVELS,
                deck_seed=int(row["anchor_deck_seed"]),
            )
        finally:
            session.collector.advantage_memory = original
        proxy.assert_complete()
        root_replacements += 1
        continuation_replacements += 1
        global_root += 1

    state["global_root"] = int(global_root)
    state["scenario_counts"] = scenario_counts
    tree_seconds = float(time.perf_counter() - started)

    root_aux = sum(int(r["root"]["aux_traversals"]) for r in aux_rows)
    cont_aux = sum(int(r["continuation"]["target_traversals"]) for r in aux_rows)
    if root_aux != ROOTS_PER_CHUNK * K or cont_aux != ROOTS_PER_CHUNK * K:
        raise RuntimeError("Phase2C2 auxiliary traversal-count drift")

    report = {
        "arm": str(arm),
        "roots": int(bundle.counters["roots"]) - roots_before,
        "nodes": int(bundle.counters["nodes"]) - nodes_before,
        "advantage_seen": int(bundle.adv_mem.seen) - adv_before,
        "strategy_seen": int(bundle.pol_mem.seen) - pol_before,
        "tree_collection_seconds": tree_seconds,
        "branch_geometry": session.collector.telemetry_snapshot(),
        "floor_policy_delta": b6._stats_delta(floor_policy.stats(), floor_before),
        "root_replacements": int(root_replacements),
        "continuation_replacements": int(continuation_replacements),
        "root_aux_target_traversals": int(root_aux),
        "continuation_aux_target_traversals": int(cont_aux),
        "root_aux_target_nodes": int(sum(int(r["root"]["aux_nodes"]) for r in aux_rows)),
        "continuation_aux_target_nodes": int(
            sum(int(r["continuation"]["target_nodes"]) for r in aux_rows)
        ),
        "continuation_reach_table_policy_evaluations": int(
            sum(int(r["continuation"]["reach_table_policy_evaluations"]) for r in aux_rows)
        ),
        "continuation_unique_joint_assignments_min": int(
            min(int(r["continuation"]["unique_joint_assignments"]) for r in aux_rows)
        ),
        "continuation_paths": [
            {
                "global_root": int(r["global_root"]),
                "scenario_index": int(r["scenario_index"]),
                "action_path": [int(x) for x in r["continuation"]["action_path"]],
                "actor": int(r["continuation"]["actor"]),
                "observation_sha256": str(r["continuation"]["observation_sha256"]),
            }
            for r in aux_rows
        ],
    }
    if report["roots"] != ROOTS_PER_CHUNK:
        raise RuntimeError("Phase2C2 chunk logical-root count drift")
    if report["root_replacements"] != ROOTS_PER_CHUNK:
        raise RuntimeError("Phase2C2 root replacement count drift")
    if report["continuation_replacements"] != ROOTS_PER_CHUNK:
        raise RuntimeError("Phase2C2 continuation replacement count drift")
    return report


def _stage_coords(stage_index: int) -> tuple[int, int]:
    total = PILOT_ITERATIONS * CHUNKS_PER_ITERATION
    if not 1 <= int(stage_index) <= total:
        raise ValueError("Phase2C2 stage index outside pilot range")
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
    source_checkpoint_sha256: str,
    last_stage_report: dict,
) -> None:
    extra = {
        "schema": CHECKPOINT_EXTRA_SCHEMA,
        "arm": str(arm),
        "k": K,
        "pilot_iterations": PILOT_ITERATIONS,
        "chunks_per_iteration": CHUNKS_PER_ITERATION,
        "stage_config": config.to_dict(),
        "stage_state": dict(state),
        "stage_index": int(stage_index),
        "behavior_model_states": _behavior_states(behavior),
        "behavior_stats": behavior.stats(),
        "floor_policy_stats": floor_policy.stats(),
        "source_b13_checkpoint_sha256": str(source_checkpoint_sha256),
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
            phase="phase2c2_resume",
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
    source_checkpoint_sha256: str,
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
    if progress.phase != "phase2c2_resume":
        raise RuntimeError("Phase2C2 resume phase mismatch")
    if extra.get("schema") != CHECKPOINT_EXTRA_SCHEMA:
        raise RuntimeError("Phase2C2 resume schema mismatch")
    if extra.get("arm") != str(arm) or int(extra.get("k", -1)) != K:
        raise RuntimeError("Phase2C2 resume arm/K mismatch")
    if int(extra.get("pilot_iterations", -1)) != PILOT_ITERATIONS:
        raise RuntimeError("Phase2C2 resume iteration contract drift")
    if int(extra.get("chunks_per_iteration", -1)) != CHUNKS_PER_ITERATION:
        raise RuntimeError("Phase2C2 resume chunk contract drift")
    if dict(extra.get("stage_config") or {}) != config.to_dict():
        raise RuntimeError("Phase2C2 resume stage config drift")
    if extra.get("source_b13_checkpoint_sha256") != str(source_checkpoint_sha256):
        raise RuntimeError("Phase2C2 source bootstrap checkpoint drift")
    state = dict(extra.get("stage_state") or {})
    if int(progress.iteration) != int(state.get("completed_iteration", -1)):
        raise RuntimeError("Phase2C2 resume completed-iteration mismatch")
    if int(progress.global_root) != int(state.get("global_root", -1)):
        raise RuntimeError("Phase2C2 resume global-root mismatch")
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
                raise RuntimeError(f"Phase2C2 completed stage report missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("stage_index", -1)) != index:
            raise RuntimeError("Phase2C2 stage report identity mismatch")


def _run_arm_seed_trajectory(
    *,
    repo_root: Path,
    solver_path: Path,
    b13_root: Path,
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
        raise RuntimeError("Phase2C2 Torch-thread contract drift")
    solver = SolverLibrary(solver_path)
    if not solver.explicit_deal_available:
        raise RuntimeError("Phase2C2 requires explicit-deal solver extension")

    source_checkpoint = (
        b13_root / b13.CANDIDATE_ARM / f"seed_{int(training_seed)}" / "resume_checkpoint.pt"
    )
    source_states, source_identity = b15._load_behavior_states(
        source_checkpoint, int(training_seed)
    )
    source_checkpoint_sha256 = str(source_identity["sha256"])

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
            source_checkpoint_sha256=source_checkpoint_sha256,
        )
        _validate_stage_prefix(seed_root, completed_stages, last_report)
        print(
            f"[Phase2C2 resume] arm={arm} seed={training_seed} "
            f"completed_stages={completed_stages}/{PILOT_ITERATIONS * CHUNKS_PER_ITERATION}",
            flush=True,
        )
    else:
        bundle, _unused_session, _unused_behavior, spec, state = new_phase2_v3_runtime(
            repo_root,
            solver=solver,
            representation=REPRESENTATION,
            domain=DOMAIN,
            training_seed=int(training_seed),
            config=base_config,
        )
        behavior = b6._make_behavior_from_states(source_states, config=base_config)
        session = stage._make_session(solver, bundle, spec, behavior)
        floor_policy = b6.PreflopContinuationFloorPolicy(behavior, floor=FLOOR)
        session.collector.policy = floor_policy
        state["phase2c2"] = {
            "schema": CHECKPOINT_EXTRA_SCHEMA,
            "arm": str(arm),
            "k": K,
            "pilot_iterations": PILOT_ITERATIONS,
            "chunks_per_iteration": CHUNKS_PER_ITERATION,
            "bootstrap_behavior": "EXACT_FINAL_PHASE2B13_IID64_MEAN_CANDIDATE",
            "source_b13_checkpoint_sha256": source_checkpoint_sha256,
            "fresh_reservoirs": True,
            "fresh_optimizers": True,
            "root_target": "IID64_MEAN_BOTH_ARMS",
            "continuation_target": (
                "FIRST_OF_RANGE_STRATIFIED_K64" if arm == CONTROL_ARM
                else "MEAN_OF_RANGE_STRATIFIED_K64"
            ),
        }
        completed_stages = 0
        last_report = {}

    total_stages = PILOT_ITERATIONS * CHUNKS_PER_ITERATION
    for stage_index in range(completed_stages + 1, total_stages + 1):
        iteration, chunk = _stage_coords(stage_index)
        if chunk == 1:
            if int(state["completed_iteration"]) != iteration - 1:
                raise RuntimeError("Phase2C2 iteration-start identity drift")
            state["phase2c2_pending_iteration"] = {
                "iteration": int(iteration),
                "roots_before": int(bundle.counters["roots"]),
                "nodes_before": int(bundle.counters["nodes"]),
                "advantage_seen_before": int(bundle.adv_mem.seen),
                "strategy_seen_before": int(bundle.pol_mem.seen),
                "chunks": [],
            }

        pending = dict(state.get("phase2c2_pending_iteration") or {})
        if int(pending.get("iteration", -1)) != iteration:
            raise RuntimeError("Phase2C2 missing pending iteration")
        chunks = list(pending.get("chunks") or [])
        if len(chunks) != chunk - 1:
            raise RuntimeError("Phase2C2 pending chunk history drift")

        print(
            f"[Phase2C2 train] arm={arm} seed={training_seed} i{iteration}c{chunk}",
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
        state["phase2c2_pending_iteration"] = pending

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
            if roots_added != ROOTS_PER_ITERATION_EFFECTIVE:
                raise RuntimeError("Phase2C2 iteration root total drift")
            patched = dict(iteration_report)
            patched.update(
                {
                    "roots_added": int(roots_added),
                    "nodes_added": int(bundle.counters["nodes"]) - int(pending["nodes_before"]),
                    "advantage_seen_added": int(bundle.adv_mem.seen)
                    - int(pending["advantage_seen_before"]),
                    "strategy_seen_added": int(bundle.pol_mem.seen)
                    - int(pending["strategy_seen_before"]),
                    "root_replacements": sum(int(x["root_replacements"]) for x in chunks),
                    "continuation_replacements": sum(
                        int(x["continuation_replacements"]) for x in chunks
                    ),
                    "root_aux_target_traversals": sum(
                        int(x["root_aux_target_traversals"]) for x in chunks
                    ),
                    "continuation_aux_target_traversals": sum(
                        int(x["continuation_aux_target_traversals"]) for x in chunks
                    ),
                    "chance_chunks": chunks,
                }
            )
            state["iteration_reports"][-1] = patched
            state.pop("phase2c2_pending_iteration", None)
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
            source_checkpoint_sha256=source_checkpoint_sha256,
            last_stage_report=stage_report,
        )
        _atomic_json(stage_report, _stage_path(seed_root, stage_index))
        print(
            f"[Phase2C2 stage complete] arm={arm} seed={training_seed} "
            f"i{iteration}c{chunk} roots={chunk_report['roots']} "
            f"root_aux={chunk_report['root_aux_target_traversals']} "
            f"cont_aux={chunk_report['continuation_aux_target_traversals']}",
            flush=True,
        )

    if int(bundle.counters["roots"]) != TOTAL_ROOTS:
        raise RuntimeError("Phase2C2 final logical-root count drift")
    if int(state["completed_iteration"]) != PILOT_ITERATIONS:
        raise RuntimeError("Phase2C2 final iteration count drift")
    return bundle, state, floor_policy, source_identity


def _fit_policies(*, seed_root: Path, arm: str, training_seed: int, execution_sha: str, bundle):
    return b13._fit_policies(
        seed_root=seed_root,
        arm=str(arm),
        training_seed=int(training_seed),
        execution_sha=str(execution_sha),
        bundle=bundle,
    )


def _run_single(args, arm: str, training_seed: int) -> int:
    output_root = Path(args.output_root).resolve()
    seed_root = output_root / str(arm) / f"seed_{int(training_seed)}"
    result_path = seed_root / "seed_result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("schema") == SEED_SCHEMA
            and existing.get("status") == "SEED_COMPLETE"
            and existing.get("execution_sha") == str(args.execution_sha)
            and existing.get("arm") == str(arm)
        ):
            print(
                f"[Phase2C2 seed resume] arm={arm} seed={training_seed} already complete",
                flush=True,
            )
            return 0

    bundle, state, floor_policy, source_identity = _run_arm_seed_trajectory(
        repo_root=Path(args.repo_root).resolve(),
        solver_path=Path(args.solver).resolve(),
        b13_root=Path(args.phase2b13_root).resolve(),
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
        advantage_rows.append(
            {
                "iteration": int(row.get("iteration", -1)),
                "ensemble_weighted_nrmse": nrmse,
                "gate_max": ADVANTAGE_NRMSE_MAX,
                "gate_pass": bool(
                    nrmse <= ADVANTAGE_NRMSE_MAX
                    and bool(row.get("ensemble_advantage_gate_pass"))
                ),
                "logical_roots": int(row.get("roots_added", -1)),
                "root_replacements": int(row.get("root_replacements", -1)),
                "continuation_replacements": int(
                    row.get("continuation_replacements", -1)
                ),
                "root_aux_target_traversals": int(
                    row.get("root_aux_target_traversals", -1)
                ),
                "continuation_aux_target_traversals": int(
                    row.get("continuation_aux_target_traversals", -1)
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
        "source_b13_behavior_checkpoint": source_identity,
        "advantage_gates": advantage_rows,
        "all_advantage_gates_pass": bool(
            len(advantage_rows) == PILOT_ITERATIONS
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
                "status": "SEED_COMPLETE",
                "arm": str(arm),
                "training_seed": int(training_seed),
                "roots": int(result["roots"]),
                "advantage_pass": bool(result["all_advantage_gates_pass"]),
                "strategy_seen": int(result["strategy_memory"]["seen"]),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _validate_prerequisites(c1_result: Path, b13_result: Path, b14_result: Path) -> dict:
    if _sha256(c1_result) != C1_RESULT_SHA256:
        raise RuntimeError("Phase2C2 Phase2C1 result SHA drift")
    j1 = json.loads(c1_result.read_text(encoding="utf-8"))
    if (
        j1.get("schema") != c1.SCHEMA
        or j1.get("status") != C1_STATUS
        or not bool((j1.get("decision") or {}).get("screen_pass"))
        or (j1.get("decision") or {}).get("next_route")
        != "PRECOMMIT_SINGLE_BOUNDED_RANGE_REACH_TARGET_KERNEL_CAUSAL_PILOT"
    ):
        raise RuntimeError("Phase2C2 requires exact successful Phase2C1 route")
    if _sha256(b13_result) != B13_RESULT_SHA256:
        raise RuntimeError("Phase2C2 Phase2B13 result SHA drift")
    if _sha256(b14_result) != B14_RESULT_SHA256:
        raise RuntimeError("Phase2C2 Phase2B14 result SHA drift")
    return {
        "phase2c1_result_sha256": C1_RESULT_SHA256,
        "phase2b13_result_sha256": B13_RESULT_SHA256,
        "phase2b14_result_sha256": B14_RESULT_SHA256,
    }


def _region_indices(descriptors) -> dict[str, list[int]]:
    out = {"PREFLOP_ROOT": [], "PREFLOP_CONTINUATION_2PLUS": []}
    for index, descriptor in enumerate(descriptors):
        region = str(b7._decode_observation(descriptor.observation_v3)["region"])
        if region in out:
            out[region].append(index)
    if not out["PREFLOP_ROOT"] or not out["PREFLOP_CONTINUATION_2PLUS"]:
        raise RuntimeError("Phase2C2 heldout region coverage missing")
    return out


def _subset_mean(values: Sequence[float], indices: Sequence[int]) -> float:
    if not indices:
        raise ValueError("Phase2C2 empty metric subset")
    return float(sum(float(values[i]) for i in indices) / len(indices))


def _evaluate(args) -> dict:
    output_root = Path(args.output_root).resolve()
    heldout_root = Path(args.heldout_root).resolve()
    prerequisites = _validate_prerequisites(
        Path(args.phase2c1_result).resolve(),
        Path(args.phase2b13_result).resolve(),
        Path(args.phase2b14_result).resolve(),
    )
    torch.set_num_threads(TORCH_THREADS)

    seed_results = {}
    for arm in ARMS:
        seed_results[arm] = {}
        for seed in map(int, TRAINING_SEEDS):
            path = output_root / arm / f"seed_{seed}" / "seed_result.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("schema") != SEED_SCHEMA
                or payload.get("status") != "SEED_COMPLETE"
                or payload.get("execution_sha") != str(args.execution_sha)
                or payload.get("arm") != arm
                or int(payload.get("roots", -1)) != TOTAL_ROOTS
                or int(payload.get("iterations", -1)) != PILOT_ITERATIONS
            ):
                raise RuntimeError(f"Phase2C2 seed result invalid: {arm}/{seed}")
            seed_results[arm][seed] = payload

    descriptors = {}
    region_indices = {}
    heldout_identity = []
    for evaluation_seed in map(int, EVALUATION_SEEDS):
        heldout = b6._find_heldout(heldout_root, evaluation_seed)
        rows = load_heldout_v3_artifact(
            heldout,
            expected_domain=DOMAIN,
            expected_evaluation_seed=evaluation_seed,
            expected_count=2048,
        )[:POLICY_COUNT]
        if len(rows) != POLICY_COUNT:
            raise RuntimeError("Phase2C2 heldout policy-count drift")
        descriptors[evaluation_seed] = rows
        region_indices[evaluation_seed] = _region_indices(rows)
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
            all_local_valid = all_local_valid and bool(sr["all_advantage_gates_pass"])
            for row in sr["advantage_gates"]:
                local_advantage.append({"arm": arm, "training_seed": seed, **dict(row)})
                if (
                    int(row.get("logical_roots", -1)) != ROOTS_PER_ITERATION_EFFECTIVE
                    or int(row.get("root_replacements", -1)) != ROOTS_PER_ITERATION_EFFECTIVE
                    or int(row.get("continuation_replacements", -1))
                    != ROOTS_PER_ITERATION_EFFECTIVE
                    or int(row.get("root_aux_target_traversals", -1))
                    != ROOTS_PER_ITERATION_EFFECTIVE * K
                    or int(row.get("continuation_aux_target_traversals", -1))
                    != ROOTS_PER_ITERATION_EFFECTIVE * K
                ):
                    all_local_valid = False

            models[arm][seed] = {}
            for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
                artifact = (
                    output_root / arm / f"seed_{seed}" / "policies" / f"{mode}.pt"
                )
                model, payload = b13._load_policy(
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
    paired = {"COMMON_LEARNER": {}, "NATIVE_LEARNER": {}}
    pooled = {}

    for mode in ("COMMON_LEARNER", "NATIVE_LEARNER"):
        control_means = []
        candidate_means = []
        for evaluation_seed in map(int, EVALUATION_SEEDS):
            desc = descriptors[evaluation_seed]
            control_left = b6._probabilities_fixed(models[CONTROL_ARM][seed_a][mode], desc)
            control_right = b6._probabilities_fixed(models[CONTROL_ARM][seed_b][mode], desc)
            candidate_left = b6._probabilities_fixed(models[CANDIDATE_ARM][seed_a][mode], desc)
            candidate_right = b6._probabilities_fixed(models[CANDIDATE_ARM][seed_b][mode], desc)
            control_metric = cross_seed_policy_stability(control_left, control_right)
            candidate_metric = cross_seed_policy_stability(candidate_left, candidate_right)
            control_tv = b6._tv_vector(control_left, control_right)
            candidate_tv = b6._tv_vector(candidate_left, candidate_right)
            paired[mode][str(evaluation_seed)] = [
                float(c - k) for c, k in zip(control_tv, candidate_tv)
            ]
            idx = region_indices[evaluation_seed]
            control_root = _subset_mean(control_tv, idx["PREFLOP_ROOT"])
            candidate_root = _subset_mean(candidate_tv, idx["PREFLOP_ROOT"])
            control_cont2 = _subset_mean(
                control_tv, idx["PREFLOP_CONTINUATION_2PLUS"]
            )
            candidate_cont2 = _subset_mean(
                candidate_tv, idx["PREFLOP_CONTINUATION_2PLUS"]
            )
            comparisons.append(
                {
                    "learner_mode": mode,
                    "evaluation_seed": int(evaluation_seed),
                    "control": {
                        "mean": float(control_metric["mean"]),
                        "p95": float(control_metric["p95"]),
                        "root_mean": float(control_root),
                        "continuation_2plus_mean": float(control_cont2),
                    },
                    "candidate": {
                        "mean": float(candidate_metric["mean"]),
                        "p95": float(candidate_metric["p95"]),
                        "root_mean": float(candidate_root),
                        "continuation_2plus_mean": float(candidate_cont2),
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
                    "root_change_candidate_minus_control": float(
                        candidate_root - control_root
                    ),
                    "continuation_2plus_improvement": float(
                        control_cont2 - candidate_cont2
                    ),
                }
            )
            control_means.append(float(control_metric["mean"]))
            candidate_means.append(float(candidate_metric["mean"]))
        pooled[mode] = {
            "control_mean_tv": float(sum(control_means) / len(control_means)),
            "candidate_mean_tv": float(sum(candidate_means) / len(candidate_means)),
        }
        pooled[mode]["absolute_improvement"] = float(
            pooled[mode]["control_mean_tv"] - pooled[mode]["candidate_mean_tv"]
        )
        pooled[mode]["relative_improvement"] = (
            float(
                pooled[mode]["absolute_improvement"]
                / pooled[mode]["control_mean_tv"]
            )
            if pooled[mode]["control_mean_tv"] > 0.0
            else -math.inf
        )

    common_boot = equal_group_stratified_bootstrap_mean_ci(
        paired["COMMON_LEARNER"],
        seed_parts=("R7.5_ARCH_RESET", "PHASE2C2", "COMMON", "CONTROL_MINUS_CANDIDATE"),
        replicates=BOOTSTRAP_REPLICATES,
        confidence_level=0.95,
    )
    native_boot = equal_group_stratified_bootstrap_mean_ci(
        paired["NATIVE_LEARNER"],
        seed_parts=("R7.5_ARCH_RESET", "PHASE2C2", "NATIVE", "CONTROL_MINUS_CANDIDATE"),
        replicates=BOOTSTRAP_REPLICATES,
        confidence_level=0.95,
    )

    common_rows = [r for r in comparisons if r["learner_mode"] == "COMMON_LEARNER"]
    native_rows = [r for r in comparisons if r["learner_mode"] == "NATIVE_LEARNER"]
    common_material = bool(
        pooled["COMMON_LEARNER"]["absolute_improvement"] >= CAUSAL_ABS_MIN
        or pooled["COMMON_LEARNER"]["relative_improvement"] >= CAUSAL_REL_MIN
    )
    common_ci_positive = bool(float(common_boot["ci_low"]) > 0.0)
    common_both_improve = bool(
        all(float(r["mean_improvement_control_minus_candidate"]) > 0.0 for r in common_rows)
    )
    common_p95_ok = bool(
        all(
            float(r["p95_change_candidate_minus_control"]) <= COMMON_P95_MAX_DEGRADE
            for r in common_rows
        )
    )
    continuation_2plus_both_improve = bool(
        all(float(r["continuation_2plus_improvement"]) > 0.0 for r in common_rows)
    )
    root_non_degrade = bool(
        all(
            float(r["root_change_candidate_minus_control"]) <= ROOT_MEAN_MAX_DEGRADE
            for r in common_rows
        )
    )
    native_noncontradiction = bool(
        pooled["NATIVE_LEARNER"]["absolute_improvement"] >= 0.0
        and all(
            float(r["mean_improvement_control_minus_candidate"])
            >= -NATIVE_MEAN_MAX_DEGRADE
            for r in native_rows
        )
    )
    causal_supported = bool(
        all_local_valid
        and common_material
        and common_ci_positive
        and common_both_improve
        and common_p95_ok
        and continuation_2plus_both_improve
        and root_non_degrade
        and native_noncontradiction
    )
    hard_stability = bool(
        common_rows
        and all(
            bool(r["candidate"]["hard_mean_gate_pass"])
            and bool(r["candidate"]["hard_p95_gate_pass"])
            for r in common_rows
        )
    )

    if not all_local_valid:
        status = "PHASE2C2_INVALID_STOP_AUDIT"
        route = "STOP_AND_AUDIT_PHASE2C2_LOCAL_VALIDITY"
    elif causal_supported:
        status = (
            "STRUCTURAL_RANGE_REACH_CAUSAL_EFFECT_SUPPORTED_HARD_STABILITY_SMALL_PILOT"
            if hard_stability
            else "STRUCTURAL_RANGE_REACH_CAUSAL_EFFECT_SUPPORTED_SMALL_PILOT"
        )
        route = "PRECOMMIT_FULL_X4_STRUCTURAL_RANGE_REACH_CONFIRMATION"
    else:
        status = "STRUCTURAL_RANGE_REACH_CAUSAL_EFFECT_NOT_SUPPORTED_SELECT_V1_FALLBACK"
        route = "SELECT_CERTIFIED_STABLE_V1_FALLBACK_AND_CLOSE_V1PLUS_ARCHITECTURE_RESET"

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
            "source_bootstrap_behavior": "EXACT_FINAL_PHASE2B13_IID64_MEAN_CANDIDATE",
            "fresh_reservoirs": True,
            "fresh_optimizers": True,
            "iterations": PILOT_ITERATIONS,
            "chunks_per_iteration": CHUNKS_PER_ITERATION,
            "roots_per_chunk": ROOTS_PER_CHUNK,
            "roots_per_iteration_effective": ROOTS_PER_ITERATION_EFFECTIVE,
            "roots_per_arm_seed": TOTAL_ROOTS,
            "root_k": K,
            "continuation_k": K,
            "continuation_stratification": "RANDOMIZED_8X8_EXACT_RANGE_REACH",
            "control_continuation_target_used": "FIRST_OF_K64",
            "candidate_continuation_target_used": "ARITHMETIC_MEAN_OF_SAME_K64",
            "continuation_floor": FLOOR,
            "root_floor": 0.0,
            "postflop_floor": 0.0,
            "heldout_inference_floor": 0.0,
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
            "continuation_2plus_both_heldouts_improve": continuation_2plus_both_improve,
            "root_non_degradation_pass": root_non_degrade,
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
    jobs = []
    entrypoint = str(Path(__file__).resolve())
    for seed in map(int, TRAINING_SEEDS):
        for arm in ARMS:
            cmd = [
                sys.executable,
                entrypoint,
                "--repo-root", str(Path(args.repo_root).resolve()),
                "--solver", str(Path(args.solver).resolve()),
                "--heldout-root", str(Path(args.heldout_root).resolve()),
                "--phase2b13-root", str(Path(args.phase2b13_root).resolve()),
                "--phase2b13-result", str(Path(args.phase2b13_result).resolve()),
                "--phase2b14-result", str(Path(args.phase2b14_result).resolve()),
                "--phase2c1-result", str(Path(args.phase2c1_result).resolve()),
                "--output-root", str(output_root),
                "--execution-sha", str(args.execution_sha),
                "--chance-workers", str(int(args.chance_workers)),
                "--single-seed", str(seed),
                "--arm", str(arm),
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
                    f"Phase2C2 worker failed arm={arm} seed={seed} exit={completed.returncode}"
                )

    result = _evaluate(args)
    out = output_root / "R7_5_ARCH_RESET_V1PLUS_PHASE2C2_RANGE_REACH_TARGET_KERNEL_CAUSAL_PILOT.json"
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


def _path_preflight(args) -> int:
    repo_root = Path(args.repo_root).resolve()
    solver_path = Path(args.solver).resolve()
    b13_root = Path(args.phase2b13_root).resolve()
    scenarios = action_scenario_cycle(DOMAIN)
    checked = 0
    for seed in map(int, TRAINING_SEEDS):
        checkpoint = b13_root / b13.CANDIDATE_ARM / f"seed_{seed}" / "resume_checkpoint.pt"
        states, _identity = b15._load_behavior_states(checkpoint, seed)
        b10._worker_init(str(repo_root), str(solver_path), seed, states)
        for scenario_index, _episode in enumerate(scenarios):
            task = {
                "training_seed": seed,
                "scenario_index": scenario_index,
                "global_root": scenario_index,
                "iteration": 1,
                "anchor_deck_seed": int(deck_seed(seed, scenario_index, 1)),
            }
            row = _select_depth2_continuation(task)
            if len(row["action_path"]) != 2 or row["region"] != "PREFLOP_CONTINUATION_2PLUS":
                raise RuntimeError("Phase2C2 path preflight continuation contract drift")
            checked += 1
    print(f"Phase2C2 depth-2 continuation path preflight PASS cases={checked}", flush=True)
    return 0


def _kernel_preflight(args) -> int:
    repo_root = Path(args.repo_root).resolve()
    solver_path = Path(args.solver).resolve()
    b13_root = Path(args.phase2b13_root).resolve()
    seed = int(TRAINING_SEEDS[0])
    checkpoint = b13_root / b13.CANDIDATE_ARM / f"seed_{seed}" / "resume_checkpoint.pt"
    states, _identity = b15._load_behavior_states(checkpoint, seed)
    b10._worker_init(str(repo_root), str(solver_path), seed, states)
    task = {
        "training_seed": seed,
        "scenario_index": 0,
        "global_root": 0,
        "iteration": 1,
        "anchor_deck_seed": int(deck_seed(seed, 0, 1)),
    }
    row = _combined_aux_task(task)
    if int(row["root"]["aux_traversals"]) != K:
        raise RuntimeError("Phase2C2 kernel preflight root K drift")
    if int(row["continuation"]["target_traversals"]) != K:
        raise RuntimeError("Phase2C2 kernel preflight continuation K drift")
    if row["continuation"]["region"] != "PREFLOP_CONTINUATION_2PLUS":
        raise RuntimeError("Phase2C2 kernel preflight region drift")
    if int(row["continuation"]["unique_joint_assignments"]) <= 0:
        raise RuntimeError("Phase2C2 kernel preflight empty structural support")
    print(
        "Phase2C2 structural kernel preflight PASS "
        f"root_aux={row['root']['aux_traversals']} "
        f"cont_aux={row['continuation']['target_traversals']} "
        f"joint_unique={row['continuation']['unique_joint_assignments']}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="R7.5 architecture-reset Phase2C2 structural range/reach causal pilot"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path)
    parser.add_argument("--phase2b13-root", type=Path, required=True)
    parser.add_argument("--phase2b13-result", type=Path)
    parser.add_argument("--phase2b14-result", type=Path)
    parser.add_argument("--phase2c1-result", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--execution-sha")
    parser.add_argument("--arm-workers", type=int, default=2)
    parser.add_argument("--chance-workers", type=int, default=14)
    parser.add_argument("--single-seed", type=int, choices=TRAINING_SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--path-preflight-only", action="store_true")
    parser.add_argument("--kernel-preflight-only", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    validate_phase2_v3_contract(
        repo_root,
        representation=REPRESENTATION,
        domain=DOMAIN,
        training_seed=int(TRAINING_SEEDS[0]),
    )
    if args.path_preflight_only:
        return _path_preflight(args)
    if args.kernel_preflight_only:
        return _kernel_preflight(args)

    required = (
        args.heldout_root,
        args.phase2b13_result,
        args.phase2b14_result,
        args.phase2c1_result,
        args.output_root,
        args.execution_sha,
    )
    if any(x is None for x in required):
        raise RuntimeError("Phase2C2 full run missing required arguments")
    if not 1 <= int(args.arm_workers) <= 2:
        raise RuntimeError("Phase2C2 arm-workers must be 1..2")
    if not 1 <= int(args.chance_workers) <= 14:
        raise RuntimeError("Phase2C2 chance-workers must be 1..14")
    if (args.single_seed is None) != (args.arm is None):
        raise RuntimeError("Phase2C2 --single-seed and --arm must be supplied together")

    _validate_prerequisites(
        Path(args.phase2c1_result).resolve(),
        Path(args.phase2b13_result).resolve(),
        Path(args.phase2b14_result).resolve(),
    )
    if args.single_seed is not None:
        return _run_single(args, str(args.arm), int(args.single_seed))
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
