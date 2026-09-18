#!/usr/bin/env python3
from __future__ import annotations

"""HU-preflop target-estimator budget sweep for LT2 Stage B.

This read-only audit follows the conditional-resampling result that attributed
~94% of sampled-target MSE to hidden/chance variation rather than current-model
conditional-mean error.

For the same deterministic HU-preflop anchor design it:
  * builds an independent high-budget exact-level-1 conditional reference;
  * builds paired candidate pools on independent hidden deals at exact levels 0/1;
  * averages K independent hidden-deal targets for K in 1,2,4,8,16,32,64;
  * scores target MSE, regret-matching policy TV, argmax agreement, branch
    mismatch, reference-target value gap/regret, and actual traversal nodes.

The reference is still a Monte-Carlo approximation to the current Stage-B
target-process conditional mean, not a GTO oracle.
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
from audit_lt2_checkpoint_fit import _lean_rm_policy_tensor, _selftest_lean_rm_policy_tensor
from audit_lt2_repeated_target_variance import _capture_root_target
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_functional_training import (
    _root_deck_seed,
    load_checkpoint,
)
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.solver import SolverLibrary

DOMAIN = "TRUE_HEADS_UP"
CHIP_SCALE = 1500.0
BUDGETS = (1, 2, 4, 8, 16, 32, 64)
REFERENCE_LEVEL = 1
CANDIDATE_LEVELS = (0, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--reference-deals", type=int, default=64)
    p.add_argument("--candidate-deals", type=int, default=64)
    p.add_argument("--root-states", type=int, default=16)
    p.add_argument("--cont1-states", type=int, default=32)
    p.add_argument("--cont2-states", type=int, default=16)
    p.add_argument("--max-episodes", type=int, default=50000)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _collect_anchors(
    *,
    solver: SolverLibrary,
    runtime,
    seed: int,
    completed: int,
    sampler,
    quotas: dict[str, int],
    max_episodes: int,
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    counts = {name: 0 for name in quotas}
    anchors: list[dict[str, Any]] = []
    episodes_seen = 0

    while any(counts[name] < quotas[name] for name in quotas) and episodes_seen < int(max_episodes):
        episode_index = episodes_seen
        episodes_seen += 1
        episode = sampler.sample_episode()
        if not episode.game_is_hu:
            continue

        deck_seed = _root_deck_seed(
            int(seed) ^ 0xC0D1,
            DOMAIN,
            int(completed) + 1,
            episode_index,
        )
        trajectory_rng = random.Random(
            (int(seed) ^ 0x771A0000 ^ (episode_index * 0x9E3779B1)) & ((1 << 63) - 1)
        )
        state = solver.create(episode, int(deck_seed))
        action_path: list[int] = []
        captured_regions: set[str] = set()
        try:
            while not state.terminal and cond._street(state) == 0:
                region = cond._region(len(action_path))
                if region in quotas and counts[region] < quotas[region] and region not in captured_regions:
                    actor = int(state.actor)
                    live = cond._live_seats(episode)
                    opponent = live[1] if actor == live[0] else live[0]
                    snapshot = state.deal_snapshot()
                    active_mask = FIRST_RELEASE_ACTION_SPEC.active_mask(0)
                    legal = cond.lean_legal_actions(state, active_mask)
                    anchors.append(
                        {
                            "anchor_index": len(anchors),
                            "episode_index": int(episode_index),
                            "episode": episode,
                            "region": region,
                            "actor": actor,
                            "opponent_seat": int(opponent),
                            "hero_cards": tuple(int(x) for x in snapshot.holes[actor]),
                            "action_path": tuple(int(x) for x in action_path),
                            "observation": state.neural_bytes(),
                            "legal_mask": cond._legal_mask_tuple(legal),
                        }
                    )
                    counts[region] += 1
                    captured_regions.add(region)

                slot = cond._sample_current_action(runtime, state, trajectory_rng)
                action_path.append(int(slot))
                if len(action_path) > 20:
                    raise RuntimeError("HU preflop estimator anchor path exceeded 20 decisions")
        finally:
            state.close()

    if any(counts[name] < quotas[name] for name in quotas):
        missing = {name: quotas[name] - counts[name] for name in quotas if counts[name] < quotas[name]}
        raise RuntimeError(f"failed to fill anchor quotas after {episodes_seen} episodes: {missing}")
    return anchors, int(episodes_seen), counts


def _pool(
    *,
    solver: SolverLibrary,
    runtime,
    anchor: dict[str, Any],
    hands: list[tuple[int, int]],
    probs: list[float],
    count: int,
    levels: tuple[int, ...],
    seed: int,
    tag: int,
    completed: int,
) -> dict[int, dict[str, Any]]:
    indices = cond._stratified_hand_indices(
        probs,
        int(count),
        seed=(int(seed) ^ int(tag) ^ (int(anchor["anchor_index"]) * 0x9E3779B1)) & ((1 << 63) - 1),
    )
    hero_cards = tuple(int(x) for x in anchor["hero_cards"])
    out = {
        int(level): {"targets": [], "nodes": [], "hand_indices": list(map(int, indices))}
        for level in levels
    }

    for pos, hand_idx in enumerate(indices):
        hand = hands[int(hand_idx)]
        board_rng = random.Random(
            (
                int(seed)
                ^ int(tag)
                ^ 0xB04D5000
                ^ (int(anchor["anchor_index"]) * 0x45D9F3B)
                ^ (pos * 0x13579B)
            )
            & ((1 << 63) - 1)
        )
        board = cond._board_for(hero_cards, hand, rng=board_rng)
        holes = cond._holes_for(
            anchor["episode"],
            hero_seat=int(anchor["actor"]),
            hero_cards=hero_cards,
            opponent_seat=int(anchor["opponent_seat"]),
            opponent_hand=hand,
        )

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
            for level in levels:
                rng_seed = (
                    int(seed)
                    ^ int(tag)
                    ^ 0xA17E1000
                    ^ (int(anchor["anchor_index"]) * 0x7F4A7C15)
                    ^ (pos * 0x94D049BB)
                    ^ (int(level) * 0x369DEA0F)
                ) & ((1 << 63) - 1)
                sample, nodes = _capture_root_target(
                    runtime,
                    state,
                    iteration=int(completed) + 1,
                    exact_level=int(level),
                    rng_seed=int(rng_seed),
                )
                if sample.observation != anchor["observation"] or tuple(sample.legal) != tuple(anchor["legal_mask"]):
                    raise RuntimeError("estimator target sample identity drift")
                out[int(level)]["targets"].append(tuple(float(x) for x in sample.target))
                out[int(level)]["nodes"].append(int(nodes))
        finally:
            state.close()
    return out


def _legal_tensor(anchor: dict[str, Any]) -> torch.Tensor:
    legal = torch.tensor(anchor["legal_mask"], dtype=torch.bool)
    if int(legal.sum().item()) <= 0:
        raise RuntimeError("empty legal mask")
    return legal


def _mse(a: torch.Tensor, b: torch.Tensor, legal: torch.Tensor) -> float:
    diff = (a - b)[legal]
    return float((diff * diff).mean().item())


def _policy(target: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    return _lean_rm_policy_tensor(target.unsqueeze(0), legal.unsqueeze(0))[0]


def _policy_metrics(candidate_target: torch.Tensor, reference_target: torch.Tensor, legal: torch.Tensor) -> dict[str, Any]:
    cp = _policy(candidate_target, legal)
    rp = _policy(reference_target, legal)
    tv = float((0.5 * torch.abs(cp - rp).sum()).item())
    carg = int(cp.masked_fill(~legal, -1.0).argmax().item())
    rarg = int(rp.masked_fill(~legal, -1.0).argmax().item())
    cpos = bool((torch.clamp(candidate_target, min=0.0) * legal.float()).sum().item() > 0.0)
    rpos = bool((torch.clamp(reference_target, min=0.0) * legal.float()).sum().item() > 0.0)
    ref_policy_value = float((rp * reference_target).sum().item())
    cand_policy_value = float((cp * reference_target).sum().item())
    best_value = float(reference_target.masked_fill(~legal, float("-inf")).max().item())
    return {
        "target_mse_to_reference": _mse(candidate_target, reference_target, legal),
        "policy_tv_to_reference": tv,
        "argmax_agreement": bool(carg == rarg),
        "branch_mismatch": bool(cpos != rpos),
        "signed_reference_policy_minus_candidate_policy_value_gap_chips": float(
            (ref_policy_value - cand_policy_value) * CHIP_SCALE
        ),
        "candidate_policy_regret_to_reference_best_action_chips": float(
            max(0.0, best_value - cand_policy_value) * CHIP_SCALE
        ),
        "candidate_action_mass": [float(x) for x in cp.tolist()],
        "reference_action_mass": [float(x) for x in rp.tolist()],
    }


def _model_metrics(runtime, anchor: dict[str, Any], reference_target: torch.Tensor, legal: torch.Tensor) -> dict[str, Any]:
    raw = cond._model_raw(runtime, anchor["observation"], tuple(anchor["legal_mask"]))
    return _policy_metrics(raw, reference_target, legal)


def _anchor_budget_metrics(
    *,
    candidate_targets: torch.Tensor,
    candidate_nodes: list[int],
    reference_target: torch.Tensor,
    legal: torch.Tensor,
    k: int,
) -> dict[str, Any]:
    n = int(candidate_targets.shape[0])
    if n % int(k) != 0:
        raise RuntimeError(f"candidate pool size {n} not divisible by budget {k}")
    blocks = n // int(k)
    rows = []
    for b in range(blocks):
        lo = b * int(k)
        hi = lo + int(k)
        estimate = candidate_targets[lo:hi].mean(dim=0)
        metrics = _policy_metrics(estimate, reference_target, legal)
        metrics["nodes"] = int(sum(candidate_nodes[lo:hi]))
        rows.append(metrics)

    fields = (
        "target_mse_to_reference",
        "policy_tv_to_reference",
        "signed_reference_policy_minus_candidate_policy_value_gap_chips",
        "candidate_policy_regret_to_reference_best_action_chips",
        "nodes",
    )
    out = {
        name: float(statistics.fmean(float(r[name]) for r in rows))
        for name in fields
    }
    out["blocks"] = int(blocks)
    out["argmax_agreement_rate"] = float(
        statistics.fmean(1.0 if bool(r["argmax_agreement"]) else 0.0 for r in rows)
    )
    out["branch_mismatch_rate"] = float(
        statistics.fmean(1.0 if bool(r["branch_mismatch"]) else 0.0 for r in rows)
    )
    out["candidate_action_mass"] = [
        float(statistics.fmean(float(r["candidate_action_mass"][i]) for r in rows))
        for i in range(10)
    ]
    out["reference_action_mass"] = list(rows[0]["reference_action_mass"])
    return out


def _mean_ci(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sem": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(statistics.fmean(values))
    sem = 0.0 if n == 1 else float(statistics.stdev(values) / math.sqrt(n))
    half = 1.96 * sem
    return {"n": n, "mean": mean, "sem": sem, "ci95_low": mean - half, "ci95_high": mean + half}


def _aggregate_budget(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "target_mse_to_reference",
        "policy_tv_to_reference",
        "signed_reference_policy_minus_candidate_policy_value_gap_chips",
        "candidate_policy_regret_to_reference_best_action_chips",
        "nodes",
        "argmax_agreement_rate",
        "branch_mismatch_rate",
    )
    out = {name: _mean_ci([float(r[name]) for r in rows]) for name in fields}
    out["candidate_action_mass"] = [
        float(statistics.fmean(float(r["candidate_action_mass"][i]) for r in rows))
        for i in range(10)
    ]
    out["reference_action_mass"] = [
        float(statistics.fmean(float(r["reference_action_mass"][i]) for r in rows))
        for i in range(10)
    ]
    return out


def _aggregate_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mapped = []
    for r in rows:
        mapped.append({
            "target_mse_to_reference": float(r["target_mse_to_reference"]),
            "policy_tv_to_reference": float(r["policy_tv_to_reference"]),
            "signed_reference_policy_minus_candidate_policy_value_gap_chips": float(
                r["signed_reference_policy_minus_candidate_policy_value_gap_chips"]
            ),
            "candidate_policy_regret_to_reference_best_action_chips": float(
                r["candidate_policy_regret_to_reference_best_action_chips"]
            ),
            "nodes": 0.0,
            "argmax_agreement_rate": 1.0 if bool(r["argmax_agreement"]) else 0.0,
            "branch_mismatch_rate": 1.0 if bool(r["branch_mismatch"]) else 0.0,
            "candidate_action_mass": list(r["candidate_action_mass"]),
            "reference_action_mass": list(r["reference_action_mass"]),
        })
    return _aggregate_budget(mapped)


def main() -> int:
    args = parse_args()
    if args.reference_deals < 2 or args.candidate_deals < max(BUDGETS):
        raise SystemExit("reference/candidate pools too small")
    if args.reference_deals % 2 != 0:
        raise SystemExit("reference-deals must be even for split-half diagnostic")
    if any(args.candidate_deals % k != 0 for k in BUDGETS):
        raise SystemExit(f"candidate-deals must be divisible by every budget {BUDGETS}")
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
    anchors, episodes_seen, counts = _collect_anchors(
        solver=solver,
        runtime=runtime,
        seed=int(seed),
        completed=int(completed),
        sampler=sampler,
        quotas=quotas,
        max_episodes=int(args.max_episodes),
    )

    anchor_rows: list[dict[str, Any]] = []
    for idx, anchor in enumerate(anchors):
        print(
            f"TARGET_ESTIMATOR anchor={idx+1}/{len(anchors)} "
            f"region={anchor['region']} path={len(anchor['action_path'])} "
            f"last={None if not anchor['action_path'] else anchor['action_path'][-1]}",
            flush=True,
        )

        hands, probs, posterior = cond._posterior_hand_distribution(
            solver=solver,
            runtime=runtime,
            anchor=anchor,
        )
        ref_pool = _pool(
            solver=solver,
            runtime=runtime,
            anchor=anchor,
            hands=hands,
            probs=probs,
            count=int(args.reference_deals),
            levels=(REFERENCE_LEVEL,),
            seed=int(seed) ^ 0x51A7C0DE,
            tag=0x11000000,
            completed=int(completed),
        )[REFERENCE_LEVEL]
        cand_pool = _pool(
            solver=solver,
            runtime=runtime,
            anchor=anchor,
            hands=hands,
            probs=probs,
            count=int(args.candidate_deals),
            levels=CANDIDATE_LEVELS,
            seed=int(seed) ^ 0x51A7C0DE,
            tag=0x22000000,
            completed=int(completed),
        )

        legal = _legal_tensor(anchor)
        ref_targets = torch.tensor(ref_pool["targets"], dtype=torch.float32)
        reference_target = ref_targets.mean(dim=0)
        half = int(args.reference_deals) // 2
        ref_half_a = ref_targets[:half].mean(dim=0)
        ref_half_b = ref_targets[half:].mean(dim=0)
        split = _policy_metrics(ref_half_a, ref_half_b, legal)
        split["nodes_per_half"] = float(
            0.5 * (
                sum(int(x) for x in ref_pool["nodes"][:half])
                + sum(int(x) for x in ref_pool["nodes"][half:])
            )
        )

        model = _model_metrics(runtime, anchor, reference_target, legal)
        budgets: dict[str, Any] = {}
        for level in CANDIDATE_LEVELS:
            targets = torch.tensor(cand_pool[level]["targets"], dtype=torch.float32)
            nodes = list(map(int, cand_pool[level]["nodes"]))
            budgets[f"exact_level_{level}"] = {
                str(k): _anchor_budget_metrics(
                    candidate_targets=targets,
                    candidate_nodes=nodes,
                    reference_target=reference_target,
                    legal=legal,
                    k=int(k),
                )
                for k in BUDGETS
            }

        anchor_rows.append({
            "anchor_index": int(anchor["anchor_index"]),
            "episode_index": int(anchor["episode_index"]),
            "region": str(anchor["region"]),
            "path_length": len(anchor["action_path"]),
            "last_action_slot": None if not anchor["action_path"] else int(anchor["action_path"][-1]),
            "facing_all_in": bool(anchor["action_path"] and int(anchor["action_path"][-1]) == 9),
            "actor": int(anchor["actor"]),
            "opponent_seat": int(anchor["opponent_seat"]),
            "legal_action_count": int(legal.sum().item()),
            "posterior": posterior,
            "reference": {
                "exact_level": int(REFERENCE_LEVEL),
                "deals": int(args.reference_deals),
                "split_half_target_mse": float(split["target_mse_to_reference"]),
                "split_half_policy_tv": float(split["policy_tv_to_reference"]),
                "split_half_argmax_agreement": bool(split["argmax_agreement"]),
                "split_half_branch_mismatch": bool(split["branch_mismatch"]),
                "split_half_nodes_per_half": float(split["nodes_per_half"]),
                "reference_action_mass": list(split["reference_action_mass"]),
            },
            "current_model": model,
            "budgets": budgets,
        })

    summaries: dict[str, Any] = {}
    subsets = {
        "ALL_ANCHORS_EQUAL_WEIGHT": lambda r: True,
        "PREFLOP_ROOT": lambda r: r["region"] == "PREFLOP_ROOT",
        "PREFLOP_CONTINUATION_1": lambda r: r["region"] == "PREFLOP_CONTINUATION_1",
        "PREFLOP_CONTINUATION_2PLUS": lambda r: r["region"] == "PREFLOP_CONTINUATION_2PLUS",
        "FACING_ALL_IN_LAST_ACTION": lambda r: bool(r["facing_all_in"]),
    }

    for name, pred in subsets.items():
        rr = [r for r in anchor_rows if pred(r)]
        block: dict[str, Any] = {
            "n": len(rr),
            "reference_split_half": {
                "target_mse": _mean_ci([float(r["reference"]["split_half_target_mse"]) for r in rr]),
                "policy_tv": _mean_ci([float(r["reference"]["split_half_policy_tv"]) for r in rr]),
                "argmax_agreement_rate": float(statistics.fmean(
                    1.0 if bool(r["reference"]["split_half_argmax_agreement"]) else 0.0 for r in rr
                )) if rr else float("nan"),
                "branch_mismatch_rate": float(statistics.fmean(
                    1.0 if bool(r["reference"]["split_half_branch_mismatch"]) else 0.0 for r in rr
                )) if rr else float("nan"),
                "nodes_per_half": _mean_ci([float(r["reference"]["split_half_nodes_per_half"]) for r in rr]),
            },
            "current_model": _aggregate_model([r["current_model"] for r in rr]) if rr else {},
            "candidate_estimators": {},
        }
        for level in CANDIDATE_LEVELS:
            level_key = f"exact_level_{level}"
            block["candidate_estimators"][level_key] = {}
            for k in BUDGETS:
                rows_k = [r["budgets"][level_key][str(k)] for r in rr]
                block["candidate_estimators"][level_key][str(k)] = _aggregate_budget(rows_k)
        summaries[name] = block

    report = {
        "schema": "SPINCORE_LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_V1",
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
            "anchors_reused_design": "same deterministic anchor generator as prior HU-preflop conditional audit",
            "posterior": "exact enumeration of all 2450 ordered opponent hands under current Stage-B behavior reach",
            "reference_exact_level": int(REFERENCE_LEVEL),
            "reference_independent_hidden_deals": int(args.reference_deals),
            "candidate_independent_hidden_deals": int(args.candidate_deals),
            "candidate_exact_levels": list(CANDIDATE_LEVELS),
            "budgets": list(BUDGETS),
            "candidate_pairing": "exact levels 0 and 1 use the same candidate hidden hand/future-board deals",
            "reference_independence": "reference hidden deals use a disjoint deterministic sampling stream from candidate deals",
            "budget_blocks": "non-overlapping blocks within each candidate pool; metrics averaged within anchor then anchors equally weighted",
            "reference_warning": "reference is a high-budget current-target-process conditional estimate, not a GTO oracle",
            "decision_metrics": [
                "target_mse_to_reference",
                "policy_tv_to_reference",
                "argmax_agreement_rate",
                "branch_mismatch_rate",
                "signed_reference_policy_minus_candidate_policy_value_gap_chips",
                "candidate_policy_regret_to_reference_best_action_chips",
                "actual traversal nodes",
            ],
            "no_arbitrary_pass_threshold": True,
        },
        "episodes_sampled_for_anchors": int(episodes_seen),
        "anchor_counts": counts,
        "summaries": summaries,
        "rows": anchor_rows,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    overall = summaries["ALL_ANCHORS_EQUAL_WEIGHT"]
    print("=== HU PREFLOP TARGET-ESTIMATOR BUDGET SWEEP ===")
    print(
        f"reference_split_half_tv={overall['reference_split_half']['policy_tv']['mean']:.4f} "
        f"current_model_tv={overall['current_model']['policy_tv_to_reference']['mean']:.4f}"
    )
    for level in CANDIDATE_LEVELS:
        lk = f"exact_level_{level}"
        for k in BUDGETS:
            x = overall["candidate_estimators"][lk][str(k)]
            print(
                f"level={level} K={k:02d} "
                f"nodes={x['nodes']['mean']:.1f} "
                f"mse={x['target_mse_to_reference']['mean']:.6f} "
                f"tv={x['policy_tv_to_reference']['mean']:.4f} "
                f"regret={x['candidate_policy_regret_to_reference_best_action_chips']['mean']:.2f}"
            )
    print("LT2_HU_PREFLOP_TARGET_ESTIMATOR_BUDGET_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
