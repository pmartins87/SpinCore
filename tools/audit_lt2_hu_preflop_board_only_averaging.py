#!/usr/bin/env python3
from __future__ import annotations

"""HU-preflop board-only target averaging audit for LT2 Stage B.

Purpose:
  Measure how much of the full hidden-deal averaging benefit can be captured by
  the production-feasible intervention of resampling FUTURE BOARDS while holding
  the currently sampled opponent hand fixed.

For each deterministic HU-preflop anchor:
  * build an independent 64-deal conditional reference using 16 posterior hands
    x 4 future boards at exact opponent level 1;
  * select 16 independent posterior candidate hands;
  * for each candidate hand evaluate 8 independent future boards at exact level 0;
  * form board-only averages K = 1,2,4,8 within the same hidden opponent hand;
  * compare target MSE, regret-matching policy TV, argmax, branch mismatch,
    value gap/regret, and actual nodes to the full conditional reference.

This remains read-only: no roots are collected into training memory and no
optimizer steps are run.
"""

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import torch

import audit_lt2_hu_preflop_conditional_resampling as cond
import audit_lt2_hu_preflop_target_estimator_budget as base
from audit_lt2_checkpoint_fit import _selftest_lean_rm_policy_tensor
from audit_lt2_repeated_target_variance import _capture_root_target
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.lean_functional_training import load_checkpoint
from spincore.solver import SolverLibrary

DOMAIN = "TRUE_HEADS_UP"
REFERENCE_LEVEL = 1
CANDIDATE_LEVEL = 0
BUDGETS = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--reference-hands", type=int, default=16)
    p.add_argument("--reference-boards-per-hand", type=int, default=4)
    p.add_argument("--candidate-hands", type=int, default=16)
    p.add_argument("--candidate-boards-per-hand", type=int, default=8)
    p.add_argument("--root-states", type=int, default=16)
    p.add_argument("--cont1-states", type=int, default=32)
    p.add_argument("--cont2-states", type=int, default=16)
    p.add_argument("--max-episodes", type=int, default=50000)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _targets_for_hand_boards(
    *,
    solver: SolverLibrary,
    runtime,
    anchor: dict[str, Any],
    hand: tuple[int, int],
    boards: int,
    exact_level: int,
    seed: int,
    tag: int,
    hand_position: int,
    completed: int,
) -> tuple[list[tuple[float, ...]], list[int]]:
    hero_cards = tuple(int(x) for x in anchor["hero_cards"])
    holes = cond._holes_for(
        anchor["episode"],
        hero_seat=int(anchor["actor"]),
        hero_cards=hero_cards,
        opponent_seat=int(anchor["opponent_seat"]),
        opponent_hand=hand,
    )
    targets: list[tuple[float, ...]] = []
    nodes: list[int] = []
    for bpos in range(int(boards)):
        board_rng = random.Random(
            (
                int(seed)
                ^ int(tag)
                ^ 0xB04D0000
                ^ (int(anchor["anchor_index"]) * 0x45D9F3B)
                ^ (int(hand_position) * 0x13579B)
                ^ (int(bpos) * 0x2468D)
            )
            & ((1 << 63) - 1)
        )
        board = cond._board_for(hero_cards, hand, rng=board_rng)
        state, _ = cond._replay_to_anchor(
            solver=solver,
            runtime=runtime,
            episode=anchor["episode"],
            holes=holes,
            board=board,
            action_path=tuple(anchor["action_path"]),
            target_actor=int(anchor["actor"]),
            target_observation=anchor["observation"],
            target_legal_mask=tuple(anchor["legal_mask"]),
            compute_opponent_log_reach=False,
        )
        try:
            rng_seed = (
                int(seed)
                ^ int(tag)
                ^ 0xA17E0000
                ^ (int(anchor["anchor_index"]) * 0x7F4A7C15)
                ^ (int(hand_position) * 0x94D049BB)
                ^ (int(bpos) * 0x369DEA0F)
                ^ (int(exact_level) * 0x51ED270B)
            ) & ((1 << 63) - 1)
            sample, node_count = _capture_root_target(
                runtime,
                state,
                iteration=int(completed) + 1,
                exact_level=int(exact_level),
                rng_seed=int(rng_seed),
            )
            if sample.observation != anchor["observation"] or tuple(sample.legal) != tuple(anchor["legal_mask"]):
                raise RuntimeError("board-only target sample identity drift")
            targets.append(tuple(float(x) for x in sample.target))
            nodes.append(int(node_count))
        finally:
            state.close()
    return targets, nodes


def _reference(
    *,
    solver: SolverLibrary,
    runtime,
    anchor: dict[str, Any],
    hands: list[tuple[int, int]],
    probs: list[float],
    hand_count: int,
    boards_per_hand: int,
    seed: int,
    completed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    selected = cond._stratified_hand_indices(
        probs,
        int(hand_count),
        seed=(int(seed) ^ 0x11335577 ^ (int(anchor["anchor_index"]) * 0x9E3779B1)) & ((1 << 63) - 1),
    )
    all_targets: list[tuple[float, ...]] = []
    all_nodes: list[int] = []
    for hpos, hidx in enumerate(selected):
        t, n = _targets_for_hand_boards(
            solver=solver,
            runtime=runtime,
            anchor=anchor,
            hand=hands[int(hidx)],
            boards=int(boards_per_hand),
            exact_level=REFERENCE_LEVEL,
            seed=int(seed),
            tag=0x11000000,
            hand_position=hpos,
            completed=int(completed),
        )
        all_targets.extend(t)
        all_nodes.extend(n)
    tensor = torch.tensor(all_targets, dtype=torch.float32)
    return tensor.mean(dim=0), {
        "hands": int(hand_count),
        "boards_per_hand": int(boards_per_hand),
        "deals": len(all_targets),
        "mean_nodes_per_deal": float(statistics.fmean(all_nodes)),
        "total_nodes": int(sum(all_nodes)),
        "unique_selected_ordered_hands": int(len(set(selected))),
    }


def _candidate_hand_rows(
    *,
    solver: SolverLibrary,
    runtime,
    anchor: dict[str, Any],
    hands: list[tuple[int, int]],
    probs: list[float],
    hand_count: int,
    boards_per_hand: int,
    reference_target: torch.Tensor,
    legal: torch.Tensor,
    seed: int,
    completed: int,
) -> dict[str, Any]:
    selected = cond._stratified_hand_indices(
        probs,
        int(hand_count),
        seed=(int(seed) ^ 0x88776655 ^ (int(anchor["anchor_index"]) * 0x9E3779B1)) & ((1 << 63) - 1),
    )
    by_budget: dict[str, list[dict[str, Any]]] = {str(k): [] for k in BUDGETS}

    for hpos, hidx in enumerate(selected):
        targets, nodes = _targets_for_hand_boards(
            solver=solver,
            runtime=runtime,
            anchor=anchor,
            hand=hands[int(hidx)],
            boards=int(boards_per_hand),
            exact_level=CANDIDATE_LEVEL,
            seed=int(seed),
            tag=0x22000000,
            hand_position=hpos,
            completed=int(completed),
        )
        tensor = torch.tensor(targets, dtype=torch.float32)
        for k in BUDGETS:
            if int(boards_per_hand) % int(k) != 0:
                raise RuntimeError("candidate boards per hand must be divisible by every budget")
            blocks = int(boards_per_hand) // int(k)
            for block in range(blocks):
                lo = block * int(k)
                hi = lo + int(k)
                estimate = tensor[lo:hi].mean(dim=0)
                m = base._policy_metrics(estimate, reference_target, legal)
                m["nodes"] = int(sum(nodes[lo:hi]))
                by_budget[str(k)].append(m)

    out: dict[str, Any] = {
        "posterior_hand_draws": int(hand_count),
        "unique_selected_ordered_hands": int(len(set(selected))),
        "boards_per_hand": int(boards_per_hand),
        "budgets": {},
    }
    for k in BUDGETS:
        rows = by_budget[str(k)]
        fields = (
            "target_mse_to_reference",
            "policy_tv_to_reference",
            "signed_reference_policy_minus_candidate_policy_value_gap_chips",
            "candidate_policy_regret_to_reference_best_action_chips",
            "nodes",
        )
        block = {
            field: float(statistics.fmean(float(r[field]) for r in rows))
            for field in fields
        }
        block["estimates"] = len(rows)
        block["argmax_agreement_rate"] = float(
            statistics.fmean(1.0 if bool(r["argmax_agreement"]) else 0.0 for r in rows)
        )
        block["branch_mismatch_rate"] = float(
            statistics.fmean(1.0 if bool(r["branch_mismatch"]) else 0.0 for r in rows)
        )
        out["budgets"][str(k)] = block
    return out


def _aggregate(rows: list[dict[str, Any]], k: str) -> dict[str, Any]:
    fields = (
        "target_mse_to_reference",
        "policy_tv_to_reference",
        "signed_reference_policy_minus_candidate_policy_value_gap_chips",
        "candidate_policy_regret_to_reference_best_action_chips",
        "nodes",
        "argmax_agreement_rate",
        "branch_mismatch_rate",
    )
    return {
        field: base._mean_ci([float(r["candidate"]["budgets"][k][field]) for r in rows])
        for field in fields
    }


def main() -> int:
    args = parse_args()
    if args.reference_hands <= 0 or args.reference_boards_per_hand <= 0:
        raise SystemExit("positive reference design required")
    if args.candidate_hands <= 0 or args.candidate_boards_per_hand < max(BUDGETS):
        raise SystemExit("candidate board pool too small")
    if any(args.candidate_boards_per_hand % k != 0 for k in BUDGETS):
        raise SystemExit(f"candidate boards per hand must be divisible by {BUDGETS}")
    if min(args.root_states, args.cont1_states, args.cont2_states) < 0 or args.threads <= 0:
        raise SystemExit("invalid anchor/thread configuration")

    checkpoint = args.checkpoint.resolve(strict=True)
    solver_path = args.solver.resolve(strict=True)
    torch.set_num_threads(int(args.threads))
    _selftest_lean_rm_policy_tensor()

    solver = SolverLibrary(solver_path)
    if not solver.explicit_deal_available:
        raise SystemExit("solver explicit-deal diagnostic API is required")

    seed, config, completed, _sampler0, runtimes, _history, finalized = load_checkpoint(
        checkpoint, solver=solver
    )
    if not finalized or int(completed) != int(config.iterations):
        raise SystemExit("expected finalized Stage-B checkpoint")
    runtime = runtimes[DOMAIN]

    quotas = {
        "PREFLOP_ROOT": int(args.root_states),
        "PREFLOP_CONTINUATION_1": int(args.cont1_states),
        "PREFLOP_CONTINUATION_2PLUS": int(args.cont2_states),
    }
    sampler = LegacyScenarioSampler(
        seed=int(seed) ^ 0xC0D1710A,
        config=LegacyScenarioConfig(heads_up_prob=1.0),
    )
    anchors, episodes_seen, counts = base._collect_anchors(
        solver=solver,
        runtime=runtime,
        seed=int(seed),
        completed=int(completed),
        sampler=sampler,
        quotas=quotas,
        max_episodes=int(args.max_episodes),
    )

    rows: list[dict[str, Any]] = []
    for idx, anchor in enumerate(anchors):
        print(
            f"BOARD_ONLY anchor={idx+1}/{len(anchors)} "
            f"region={anchor['region']} path={len(anchor['action_path'])} "
            f"last={None if not anchor['action_path'] else anchor['action_path'][-1]}",
            flush=True,
        )
        hands, probs, posterior = cond._posterior_hand_distribution(
            solver=solver,
            runtime=runtime,
            anchor=anchor,
        )
        reference_target, reference_meta = _reference(
            solver=solver,
            runtime=runtime,
            anchor=anchor,
            hands=hands,
            probs=probs,
            hand_count=int(args.reference_hands),
            boards_per_hand=int(args.reference_boards_per_hand),
            seed=int(seed) ^ 0x51A7C0DE,
            completed=int(completed),
        )
        legal = base._legal_tensor(anchor)
        current_model = base._model_metrics(runtime, anchor, reference_target, legal)
        candidate = _candidate_hand_rows(
            solver=solver,
            runtime=runtime,
            anchor=anchor,
            hands=hands,
            probs=probs,
            hand_count=int(args.candidate_hands),
            boards_per_hand=int(args.candidate_boards_per_hand),
            reference_target=reference_target,
            legal=legal,
            seed=int(seed) ^ 0x51A7C0DE,
            completed=int(completed),
        )
        rows.append({
            "anchor_index": int(anchor["anchor_index"]),
            "region": str(anchor["region"]),
            "path_length": len(anchor["action_path"]),
            "last_action_slot": None if not anchor["action_path"] else int(anchor["action_path"][-1]),
            "facing_all_in": bool(anchor["action_path"] and int(anchor["action_path"][-1]) == 9),
            "posterior": posterior,
            "reference": reference_meta,
            "current_model": current_model,
            "candidate": candidate,
        })

    subsets = {
        "ALL_ANCHORS_EQUAL_WEIGHT": lambda r: True,
        "PREFLOP_ROOT": lambda r: r["region"] == "PREFLOP_ROOT",
        "PREFLOP_CONTINUATION_1": lambda r: r["region"] == "PREFLOP_CONTINUATION_1",
        "PREFLOP_CONTINUATION_2PLUS": lambda r: r["region"] == "PREFLOP_CONTINUATION_2PLUS",
        "FACING_ALL_IN_LAST_ACTION": lambda r: bool(r["facing_all_in"]),
    }
    summaries: dict[str, Any] = {}
    for name, pred in subsets.items():
        rr = [r for r in rows if pred(r)]
        summaries[name] = {
            "n": len(rr),
            "current_model": base._aggregate_model([r["current_model"] for r in rr]) if rr else {},
            "board_only_exact0": {
                str(k): _aggregate(rr, str(k))
                for k in BUDGETS
            },
        }

    report = {
        "schema": "SPINCORE_LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_V1",
        "checkpoint": str(checkpoint),
        "completed_iteration": int(completed),
        "seed": int(seed),
        "method": {
            "read_only": True,
            "new_training_roots": 0,
            "optimizer_steps": 0,
            "domain": DOMAIN,
            "street": "PREFLOP",
            "anchor_region_quotas": quotas,
            "reference": {
                "posterior_hands": int(args.reference_hands),
                "future_boards_per_hand": int(args.reference_boards_per_hand),
                "total_hidden_deals_per_anchor": int(args.reference_hands * args.reference_boards_per_hand),
                "exact_opponent_level": int(REFERENCE_LEVEL),
            },
            "candidate": {
                "posterior_hands_evaluated_per_anchor": int(args.candidate_hands),
                "future_boards_per_fixed_hand": int(args.candidate_boards_per_hand),
                "exact_opponent_level": int(CANDIDATE_LEVEL),
                "board_average_budgets": list(BUDGETS),
                "one_fixed_opponent_hand_per_estimator": True,
            },
            "candidate_reference_independent_sampling_streams": True,
            "interpretation": "candidate isolates production-feasible future-board averaging while leaving sampled opponent-hand uncertainty intact",
            "no_arbitrary_pass_threshold": True,
        },
        "episodes_sampled_for_anchors": int(episodes_seen),
        "anchor_counts": counts,
        "summaries": summaries,
        "rows": rows,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    overall = summaries["ALL_ANCHORS_EQUAL_WEIGHT"]
    print("=== HU PREFLOP BOARD-ONLY AVERAGING ===")
    print(
        f"current_model_tv={overall['current_model']['policy_tv_to_reference']['mean']:.4f}"
    )
    for k in BUDGETS:
        x = overall["board_only_exact0"][str(k)]
        print(
            f"K={k:02d} nodes={x['nodes']['mean']:.1f} "
            f"mse={x['target_mse_to_reference']['mean']:.6f} "
            f"tv={x['policy_tv_to_reference']['mean']:.4f} "
            f"regret={x['candidate_policy_regret_to_reference_best_action_chips']['mean']:.2f}"
        )
    print("LT2_HU_PREFLOP_BOARD_ONLY_AVERAGING_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
