#!/usr/bin/env python3
from __future__ import annotations

"""Broad Jammer-facing action-gap / regret-matching calibration audit.

Read-only diagnostic on broad, NON-DIVERGENCE-SELECTED HU states where the hero
faces a JAMMER all-in preflop.

State selection:
  * forensic seeds 20260920..20260925;
  * same empirical HU scenarios;
  * Stage A and Stage B current behavior run in lock-step only until a common
    Jammer-facing-all-in state is reached;
  * the anchor is recorded BEFORE sampling the hero action at that state;
  * selection does NOT depend on whether A/B subsequently choose different
    actions or on terminal B-A outcome;
  * deterministic balanced sample per seed.

For each anchor, build a common low-noise Q-like action-gap reference because
JAMMER is hand-independent and is already all-in:
  * uniform compatible opponent hands;
  * uniform future boards;
  * canonical legal-action gauge Q(a)-mean_legal(Q).

Then, for each stage:
  * read raw Advantage outputs;
  * derive the exact production lean regret-matching behavior;
  * derive the stage-specific true Advantage target
      A_sigma(a) = Q(a) - sum_b sigma(b) Q(b)
    from the common Q-like reference and that stage's own current sigma;
  * measure raw-target MSE, canonical action-gap MSE, sign/support mistakes,
    fallback incidence, best-action agreement, true-negative policy mass and
    expected policy regret.

This directly tests how similar MSE can still yield very different policies due
to sign / ranking / regret-matching nonlinearities.

No training roots, optimizer steps, or memory writes.
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

import audit_lt2_hu_behavior_first_divergence as behfd
import audit_lt2_hu_preflop_conditional_resampling as cond
import audit_lt2_jammer_facing_allin_common_reference_v2 as common
import audit_lt2_jammer_facing_allin_target_overlay as overlay
import audit_lt2_stage_a_b_first_divergence as fd
from audit_lt2_hu_preflop_target_estimator_budget import _mean_ci
from spincore.lean_action_policy import lean_regret_matching_policy
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import apply_lean, lean_legal_actions
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.solver import Episode, SolverLibrary

FORENSIC_SEEDS = (20260920, 20260921, 20260922, 20260923, 20260924, 20260925)
CHIP_SCALE = 1500.0
SIGN_EPS = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--stage-b", type=Path, required=True)
    p.add_argument("--scenarios-per-seed", type=int, default=5000)
    p.add_argument("--anchors-per-seed", type=int, default=8)
    p.add_argument("--reference-hands", type=int, default=32)
    p.add_argument("--reference-boards-per-hand", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _legal_context(state):
    street = int(cond._street(state))
    active_mask = FIRST_RELEASE_ACTION_SPEC.active_mask(street)
    legal = tuple(int(x) for x in lean_legal_actions(state, active_mask))
    if not legal:
        raise RuntimeError("nonterminal state has no legal actions")
    return int(active_mask), legal


def _sample_probs(legal, probs, rng):
    x = rng.random()
    cumulative = 0.0
    for slot in legal:
        cumulative += float(probs[slot])
        if x < cumulative:
            return int(slot)
    return int(legal[-1])


def _behavior_distribution(stage, state):
    active_mask, legal = _legal_context(state)
    obs = state.neural_bytes()
    probs = stage.runtime.session.behavior(state, obs, legal)
    out = tuple(float(x) for x in probs)
    mass = sum(out[a] for a in legal)
    if not (0.999 <= mass <= 1.001):
        raise RuntimeError(f"behavior probability mass drift: {mass}")
    return active_mask, legal, out


def _jammer_action(state):
    active_mask, legal = _legal_context(state)
    if 9 in legal:
        return active_mask, 9
    if 1 in legal:
        return active_mask, 1
    if 0 in legal:
        return active_mask, 0
    return active_mask, int(max(legal))


def _anchor_from_state(
    *,
    state,
    episode: Episode,
    seed: int,
    scenario_index: int,
    hero: int,
    action_path: list[int],
) -> dict[str, Any]:
    snapshot = state.deal_snapshot()
    active_mask, legal = _legal_context(state)
    live = [s for s, stack in enumerate(episode.stacks) if int(stack) > 0]
    opponent = live[1] if int(hero) == int(live[0]) else live[0]
    return {
        "seed": int(seed),
        "scenario_index": int(scenario_index),
        "episode": episode,
        "actor": int(hero),
        "opponent_seat": int(opponent),
        "hero_cards": tuple(int(x) for x in snapshot.holes[int(hero)]),
        "action_path": tuple(int(x) for x in action_path),
        "observation": state.neural_bytes(),
        "legal_mask": cond._legal_mask_tuple(tuple(legal)),
        "active_mask": int(active_mask),
        "legal": tuple(int(x) for x in legal),
        "common_public_action_count": len(action_path),
        "blind": f"{episode.small_blind}/{episode.big_blind}",
    }


def _collect_seed_common_fai(
    *,
    solver: SolverLibrary,
    stage_a: overlay.StageModels,
    stage_b: overlay.StageModels,
    seed: int,
    scenarios: int,
) -> list[dict[str, Any]]:
    sampler = LegacyScenarioSampler(
        seed=int(seed) ^ 0x5CE0A710,
        config=LegacyScenarioConfig(),
    )
    out: list[dict[str, Any]] = []

    for scenario_index in range(int(scenarios)):
        episode = sampler.sample_episode()
        if not episode.game_is_hu:
            continue
        live = [s for s, stack in enumerate(episode.stacks) if int(stack) > 0]
        if len(live) != 2:
            raise RuntimeError("HU episode without exactly two live seats")
        deal_seed = fd._mix64(seed, scenario_index, 0xD34A1)

        for hero in live:
            state_a = solver.create(episode, int(deal_seed))
            state_b = solver.create(episode, int(deal_seed))
            rng_a = random.Random(fd._mix64(seed, scenario_index, hero, 777))
            rng_b = random.Random(fd._mix64(seed, scenario_index, hero, 777))
            action_path: list[int] = []
            last_actor = None
            last_action = None
            try:
                for _ in range(200):
                    if bool(state_a.terminal) != bool(state_b.terminal):
                        raise RuntimeError("paired terminal drift before common FAI")
                    if state_a.terminal:
                        break
                    actor = int(state_a.actor)
                    if actor != int(state_b.actor):
                        raise RuntimeError("paired actor drift before common FAI")

                    if actor != int(hero):
                        ma, sa = _jammer_action(state_a)
                        mb, sb = _jammer_action(state_b)
                        if ma != mb or sa != sb:
                            raise RuntimeError("Jammer drift before common FAI")
                        apply_lean(state_a, ma, sa)
                        apply_lean(state_b, mb, sb)
                        action_path.append(int(sa))
                        last_actor, last_action = actor, int(sa)
                        continue

                    street = int(cond._street(state_a))
                    if (
                        street == 0
                        and last_actor is not None
                        and int(last_actor) != int(hero)
                        and int(last_action) == 9
                    ):
                        out.append(
                            _anchor_from_state(
                                state=state_a,
                                episode=episode,
                                seed=seed,
                                scenario_index=scenario_index,
                                hero=hero,
                                action_path=action_path,
                            )
                        )
                        break

                    ma, la, pa = _behavior_distribution(stage_a, state_a)
                    mb, lb, pb = _behavior_distribution(stage_b, state_b)
                    if ma != mb or la != lb:
                        raise RuntimeError("A/B legal drift before common FAI")
                    sa = _sample_probs(la, pa, rng_a)
                    sb = _sample_probs(lb, pb, rng_b)
                    if int(sa) != int(sb):
                        # Broad FAI sample is intentionally conditioned only on
                        # having a common trajectory up to the FAI anchor.
                        break
                    apply_lean(state_a, ma, sa)
                    apply_lean(state_b, mb, sb)
                    action_path.append(int(sa))
                    last_actor, last_action = actor, int(sa)
                else:
                    raise RuntimeError("common-FAI collection exceeded 200 decisions")
            finally:
                state_a.close()
                state_b.close()

    return out


def _uniform_hand_indices(hands, k, *, seed):
    probs = [1.0 / float(len(hands))] * len(hands)
    return cond._stratified_hand_indices(probs, int(k), seed=int(seed))


def _common_reference(
    *,
    solver: SolverLibrary,
    stage_a: overlay.StageModels,
    stage_b: overlay.StageModels,
    anchor: dict[str, Any],
    hands_n: int,
    boards_n: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    hero_cards = tuple(anchor["hero_cards"])
    hands = cond._ordered_hands(hero_cards)
    legal = torch.tensor(anchor["legal_mask"], dtype=torch.bool)
    indices = _uniform_hand_indices(
        hands,
        int(hands_n),
        seed=fd._mix64(anchor["seed"], anchor["anchor_index"], 0xFA17A),
    )

    canonical_targets = []
    nodes = []
    max_stage_gap_delta = 0.0
    max_raw_offset_abs = 0.0

    for hpos, hidx in enumerate(indices):
        hand = hands[int(hidx)]
        for bpos in range(int(boards_n)):
            board_rng = random.Random(
                fd._mix64(
                    anchor["seed"],
                    anchor["anchor_index"],
                    hpos,
                    bpos,
                    0xB0A2D,
                )
            )
            board = cond._board_for(hero_cards, hand, rng=board_rng)
            rng_seed = fd._mix64(
                anchor["seed"],
                anchor["anchor_index"],
                hpos,
                bpos,
                0x7A2E7,
            )
            ta, na = common._capture_fixed_deal(
                solver=solver,
                stage=stage_a,
                anchor=anchor,
                opponent_hand=hand,
                board=board,
                rng_seed=rng_seed,
            )
            ca = common._canonicalize_target(ta, legal)
            canonical_targets.append(ca)
            nodes.append(int(na))

            # One Stage-B validation per opponent hand is sufficient to retain the
            # previously established stage-invariance contract without doubling
            # the full reference cost.
            if bpos == 0:
                tb, _ = common._capture_fixed_deal(
                    solver=solver,
                    stage=stage_b,
                    anchor=anchor,
                    opponent_hand=hand,
                    board=board,
                    rng_seed=rng_seed,
                )
                _, _, delta, offset = common._assert_same_action_gaps(
                    ta, tb, legal
                )
                max_stage_gap_delta = max(max_stage_gap_delta, float(delta))
                max_raw_offset_abs = max(max_raw_offset_abs, abs(float(offset)))

    q = torch.stack(canonical_targets).mean(dim=0)
    return q, {
        "hands": int(hands_n),
        "boards_per_hand": int(boards_n),
        "deals": len(canonical_targets),
        "mean_nodes": float(statistics.fmean(nodes)),
        "max_stage_a_b_canonical_gap_delta": float(max_stage_gap_delta),
        "max_stage_a_b_raw_common_offset_abs": float(max_raw_offset_abs),
    }


def _stage_metrics(
    *,
    stage: overlay.StageModels,
    anchor: dict[str, Any],
    q: torch.Tensor,
) -> dict[str, Any]:
    legal_tuple = tuple(int(x) for x in anchor["legal"])
    legal = torch.tensor(anchor["legal_mask"], dtype=torch.bool)
    raw = cond._model_raw(
        stage.runtime,
        anchor["observation"],
        tuple(anchor["legal_mask"]),
    ).float()

    sigma = torch.tensor(
        lean_regret_matching_policy(raw.tolist(), legal_tuple),
        dtype=torch.float32,
    )

    v_sigma = float((sigma * q).sum().item())
    true_adv = torch.zeros_like(q)
    true_adv[legal] = q[legal] - v_sigma

    raw_canonical = common._canonicalize_target(raw, legal)
    true_canonical = common._canonicalize_target(true_adv, legal)

    raw_mse = float(torch.mean((raw[legal] - true_adv[legal]) ** 2).item())
    gap_mse = float(
        torch.mean((raw_canonical[legal] - true_canonical[legal]) ** 2).item()
    )

    model_positive = raw[legal] > SIGN_EPS
    target_positive = true_adv[legal] > SIGN_EPS
    false_positive = int((model_positive & ~target_positive).sum().item())
    false_negative = int((~model_positive & target_positive).sum().item())
    positive_support_exact = bool(torch.equal(model_positive, target_positive))

    union = int((model_positive | target_positive).sum().item())
    intersection = int((model_positive & target_positive).sum().item())
    jaccard = 1.0 if union == 0 else float(intersection / union)

    fallback = bool((raw[legal] <= 0.0).all().item())
    target_all_nonpositive = bool((true_adv[legal] <= SIGN_EPS).all().item())

    best_q = float(q[legal].max().item())
    policy_value = float((sigma * q).sum().item())
    regret = float(max(0.0, best_q - policy_value) * CHIP_SCALE)

    best_action = int(q.masked_fill(~legal, float("-inf")).argmax().item())
    model_best = int(raw.masked_fill(~legal, float("-inf")).argmax().item())

    true_negative_mask = legal & (true_adv < -SIGN_EPS)
    true_positive_mask = legal & (true_adv > SIGN_EPS)
    mass_true_negative = float(sigma[true_negative_mask].sum().item())
    mass_true_positive = float(sigma[true_positive_mask].sum().item())

    fold_mass = float(sigma[0].item()) if bool(legal[0]) else 0.0
    continue_mass = float(
        sum(float(sigma[i].item()) for i in legal_tuple if int(i) != 0)
    )

    nonfold = [int(i) for i in legal_tuple if int(i) != 0]
    if nonfold:
        vals = [float(q[i].item()) for i in nonfold]
        continue_spread_chips = float((max(vals) - min(vals)) * CHIP_SCALE)
        best_continue = max(vals)
        fold_value = float(q[0].item()) if 0 in legal_tuple else float("-inf")
        if abs(best_continue - fold_value) * CHIP_SCALE <= 1e-6:
            optimal_class = "TIE"
        elif best_continue > fold_value:
            optimal_class = "CONTINUE"
        else:
            optimal_class = "FOLD"
    else:
        continue_spread_chips = 0.0
        optimal_class = "FOLD"

    class_error_mass = (
        continue_mass
        if optimal_class == "FOLD"
        else fold_mass
        if optimal_class == "CONTINUE"
        else 0.0
    )

    legal_q = q[legal]
    if int(legal_q.numel()) >= 2:
        top2 = torch.topk(legal_q, k=2).values
        best_second_gap_chips = float((top2[0] - top2[1]).item() * CHIP_SCALE)
    else:
        best_second_gap_chips = 0.0

    return {
        "raw_target_mse": raw_mse,
        "canonical_action_gap_mse": gap_mse,
        "fallback_all_nonpositive": fallback,
        "target_all_nonpositive": target_all_nonpositive,
        "model_positive_count": int(model_positive.sum().item()),
        "target_positive_count": int(target_positive.sum().item()),
        "false_positive_count": false_positive,
        "false_negative_count": false_negative,
        "positive_support_exact": positive_support_exact,
        "positive_support_jaccard": jaccard,
        "best_action_match": bool(model_best == best_action),
        "reference_best_action": best_action,
        "model_best_action": model_best,
        "policy_regret_chips": regret,
        "policy_value_normalized": policy_value,
        "mass_on_true_negative_actions": mass_true_negative,
        "mass_on_true_positive_actions": mass_true_positive,
        "fold_mass": fold_mass,
        "continue_mass": continue_mass,
        "optimal_class": optimal_class,
        "class_error_mass": float(class_error_mass),
        "nonfold_reference_value_spread_chips": continue_spread_chips,
        "best_second_reference_gap_chips": best_second_gap_chips,
        "raw_positive_total": float(
            sum(max(0.0, float(raw[i].item())) for i in legal_tuple)
        ),
        "true_positive_total": float(
            sum(max(0.0, float(true_adv[i].item())) for i in legal_tuple)
        ),
        "raw": [float(x) for x in raw.tolist()],
        "true_advantage": [float(x) for x in true_adv.tolist()],
        "sigma": [float(x) for x in sigma.tolist()],
        "q_canonical": [float(x) for x in q.tolist()],
    }


def _cluster_ci(rows, getter):
    by_seed = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), []).append(float(getter(r)))
    seed_means = [float(statistics.fmean(v)) for _, v in sorted(by_seed.items())]
    return _mean_ci(seed_means)


def _anchor_ci(rows, getter):
    return _mean_ci([float(getter(r)) for r in rows])


def _aggregate(rows):
    stage_fields = (
        "raw_target_mse",
        "canonical_action_gap_mse",
        "policy_regret_chips",
        "mass_on_true_negative_actions",
        "mass_on_true_positive_actions",
        "fold_mass",
        "continue_mass",
        "class_error_mass",
        "positive_support_jaccard",
        "best_second_reference_gap_chips",
        "nonfold_reference_value_spread_chips",
    )
    out = {"n": len(rows), "stage_a": {}, "stage_b": {}, "b_minus_a": {}}

    for sk in ("stage_a", "stage_b"):
        for field in stage_fields:
            out[sk][field] = {
                "anchor_ci": _anchor_ci(rows, lambda r, sk=sk, f=field: r[sk][f]),
                "seed_cluster_ci": _cluster_ci(rows, lambda r, sk=sk, f=field: r[sk][f]),
            }
        for field in (
            "fallback_all_nonpositive",
            "positive_support_exact",
            "best_action_match",
        ):
            rate = float(
                statistics.fmean(1.0 if r[sk][field] else 0.0 for r in rows)
            )
            out[sk][field + "_rate"] = rate

    for field in (
        "raw_target_mse",
        "canonical_action_gap_mse",
        "policy_regret_chips",
        "mass_on_true_negative_actions",
        "fold_mass",
        "continue_mass",
        "class_error_mass",
        "positive_support_jaccard",
    ):
        out["b_minus_a"][field] = {
            "anchor_ci": _anchor_ci(
                rows, lambda r, f=field: r["stage_b"][f] - r["stage_a"][f]
            ),
            "seed_cluster_ci": _cluster_ci(
                rows, lambda r, f=field: r["stage_b"][f] - r["stage_a"][f]
            ),
        }

    out["b_minus_a"]["fallback_rate"] = float(
        statistics.fmean(
            (1.0 if r["stage_b"]["fallback_all_nonpositive"] else 0.0)
            - (1.0 if r["stage_a"]["fallback_all_nonpositive"] else 0.0)
            for r in rows
        )
    )
    out["b_minus_a"]["positive_support_exact_rate"] = float(
        statistics.fmean(
            (1.0 if r["stage_b"]["positive_support_exact"] else 0.0)
            - (1.0 if r["stage_a"]["positive_support_exact"] else 0.0)
            for r in rows
        )
    )
    out["b_minus_a"]["best_action_match_rate"] = float(
        statistics.fmean(
            (1.0 if r["stage_b"]["best_action_match"] else 0.0)
            - (1.0 if r["stage_a"]["best_action_match"] else 0.0)
            for r in rows
        )
    )

    classes = {}
    for r in rows:
        classes[r["stage_a"]["optimal_class"]] = classes.get(
            r["stage_a"]["optimal_class"], 0
        ) + 1
    out["reference_optimal_class_counts"] = classes
    return out


def main() -> int:
    args = parse_args()
    if args.scenarios_per_seed <= 0 or args.anchors_per_seed <= 0:
        raise SystemExit("positive scenarios/anchors required")
    if args.reference_hands <= 0 or args.reference_boards_per_hand <= 0:
        raise SystemExit("positive reference budget required")
    if args.threads <= 0:
        raise SystemExit("positive thread count required")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(args.threads))
    solver = SolverLibrary(args.solver.resolve(strict=True))

    snap_a = args.report.parent / "stage_a_hu_models.pt"
    snap_b = args.report.parent / "stage_b_hu_models.pt"
    meta_a = overlay._extract_stage_snapshot(
        args.stage_a.resolve(strict=True), snap_a
    )
    meta_b = overlay._extract_stage_snapshot(
        args.stage_b.resolve(strict=True), snap_b
    )
    stage_a = overlay.StageModels(snap_a, solver)
    stage_b = overlay.StageModels(snap_b, solver)

    selected = []
    candidate_counts = {}
    for seed in FORENSIC_SEEDS:
        pool = _collect_seed_common_fai(
            solver=solver,
            stage_a=stage_a,
            stage_b=stage_b,
            seed=int(seed),
            scenarios=int(args.scenarios_per_seed),
        )
        candidate_counts[str(seed)] = len(pool)
        if len(pool) < int(args.anchors_per_seed):
            raise RuntimeError(
                f"seed {seed}: only {len(pool)} broad common FAI candidates"
            )
        chooser = random.Random(fd._mix64(seed, 0xB20ADF41))
        indices = sorted(
            chooser.sample(range(len(pool)), int(args.anchors_per_seed))
        )
        selected.extend(pool[i] for i in indices)

    for i, anchor in enumerate(selected):
        anchor["anchor_index"] = int(i)

    rows = []
    for i, anchor in enumerate(selected):
        print(
            f"FAI_CALIBRATION anchor={i+1}/{len(selected)} "
            f"seed={anchor['seed']} scenario={anchor['scenario_index']} "
            f"blind={anchor['blind']} path_len={anchor['common_public_action_count']}",
            flush=True,
        )
        q, ref_meta = _common_reference(
            solver=solver,
            stage_a=stage_a,
            stage_b=stage_b,
            anchor=anchor,
            hands_n=int(args.reference_hands),
            boards_n=int(args.reference_boards_per_hand),
        )
        a = _stage_metrics(stage=stage_a, anchor=anchor, q=q)
        b = _stage_metrics(stage=stage_b, anchor=anchor, q=q)
        rows.append({
            "anchor_index": int(i),
            "seed": int(anchor["seed"]),
            "scenario_index": int(anchor["scenario_index"]),
            "blind": str(anchor["blind"]),
            "actor": int(anchor["actor"]),
            "common_public_action_count": int(anchor["common_public_action_count"]),
            "legal": list(anchor["legal"]),
            "reference": ref_meta,
            "stage_a": a,
            "stage_b": b,
        })

    max_gap_delta = max(
        float(r["reference"]["max_stage_a_b_canonical_gap_delta"]) for r in rows
    )
    if max_gap_delta > 1e-7:
        raise RuntimeError(f"stage action-gap invariance failed: {max_gap_delta}")

    summary = _aggregate(rows)
    report = {
        "schema": "SPINCORE_LT2_JAMMER_FAI_BROAD_CALIBRATION_V1",
        "stage_a": {
            "checkpoint": str(args.stage_a.resolve()),
            "completed_iteration": int(meta_a["completed_iteration"]),
        },
        "stage_b": {
            "checkpoint": str(args.stage_b.resolve()),
            "completed_iteration": int(meta_b["completed_iteration"]),
        },
        "method": {
            "read_only": True,
            "new_training_roots": 0,
            "optimizer_steps": 0,
            "training_memory_writes": 0,
            "forensic_seeds": list(FORENSIC_SEEDS),
            "future_holdout_seeds_touched": False,
            "scenarios_per_seed": int(args.scenarios_per_seed),
            "anchors_per_seed": int(args.anchors_per_seed),
            "selection": (
                "uniform deterministic sample from all common Stage-A/B current-"
                "behavior trajectories that reach preflop immediately after a "
                "JAMMER all-in before any earlier A/B sampled-action divergence; "
                "selection does not depend on the FAI action divergence or terminal outcome"
            ),
            "reference": {
                "opponent_hand_posterior": (
                    "uniform compatible hands because JAMMER action is hand-independent"
                ),
                "hands": int(args.reference_hands),
                "boards_per_hand": int(args.reference_boards_per_hand),
                "exact_opponent_levels": 0,
                "canonical_gauge": "Q(a)-mean_legal(Q)",
            },
            "stage_specific_true_advantage": (
                "Q(a)-sum_b sigma_stage(b)Q(b), where sigma_stage is the exact "
                "production lean regret-matching policy from that stage's raw model"
            ),
            "primary_metrics": [
                "policy_regret_chips",
                "canonical_action_gap_mse",
                "raw_target_mse",
                "positive support mistakes",
                "all-nonpositive softmax fallback incidence",
                "mass on truly negative actions",
            ],
        },
        "candidate_counts_by_seed": candidate_counts,
        "max_stage_a_b_canonical_gap_delta": float(max_gap_delta),
        "summary": summary,
        "rows": rows,
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== JAMMER FAI BROAD ACTION-GAP / RM CALIBRATION ===")
    print(f"anchors={summary['n']} candidates_by_seed={candidate_counts}")
    for sk in ("stage_a", "stage_b"):
        x = summary[sk]
        print(
            f"{sk}: "
            f"regret={x['policy_regret_chips']['seed_cluster_ci']['mean']:.2f} "
            f"gapMSE={x['canonical_action_gap_mse']['seed_cluster_ci']['mean']:.6f} "
            f"rawMSE={x['raw_target_mse']['seed_cluster_ci']['mean']:.6f} "
            f"fallback={x['fallback_all_nonpositive_rate']:.3f} "
            f"support_exact={x['positive_support_exact_rate']:.3f} "
            f"best_match={x['best_action_match_rate']:.3f}"
        )
    b = summary["b_minus_a"]
    print(
        "B-A: "
        f"regret={b['policy_regret_chips']['seed_cluster_ci']['mean']:+.2f} "
        f"gapMSE={b['canonical_action_gap_mse']['seed_cluster_ci']['mean']:+.6f} "
        f"rawMSE={b['raw_target_mse']['seed_cluster_ci']['mean']:+.6f} "
        f"fallback_rate={b['fallback_rate']:+.3f} "
        f"support_exact_rate={b['positive_support_exact_rate']:+.3f} "
        f"best_match_rate={b['best_action_match_rate']:+.3f}"
    )
    print("LT2_JAMMER_FAI_BROAD_CALIBRATION_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
