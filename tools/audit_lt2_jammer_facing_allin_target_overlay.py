#!/usr/bin/env python3
from __future__ import annotations

"""Matched Stage-A/Stage-B target overlay on actual Jammer-facing divergences.

This read-only audit closes the causal gap left by the first-divergence
forensic. It reconstructs actual HU Jammer evaluation states where Stage A and
Stage B first sampled different hero actions immediately after an opponent
ALL_IN, then overlays:

  * Stage-A / Stage-B stored AveragePolicy;
  * Stage-A / Stage-B current Advantage-induced regret-matching policy;
  * a stage-specific high-budget information-set conditional target reference;
  * K1 versus K4 board-only target estimators on those exact states.

The forensic seed family is already design data. No holdout seeds are touched.
No training memory is written and no optimizer step is run.
"""

import argparse
import gc
import json
import math
from pathlib import Path
import random
import statistics
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import torch

import audit_lt2_hu_preflop_conditional_resampling as cond
from audit_lt2_checkpoint_fit import _lean_rm_policy_tensor, _selftest_lean_rm_policy_tensor
from audit_lt2_hu_preflop_target_estimator_budget import _mean_ci, _policy_metrics
from audit_lt2_repeated_target_variance import _capture_root_target
from spincore.lean_action_policy import LeanNeuralActionAdvantagePolicy
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import apply_lean, lean_legal_actions
from spincore.lean_training_scope import LeanTrainingScope
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_cfr import legal_mask
from spincore.r7_5_action_training import ActionDeepCFRSession, make_action_bundle
from spincore.solver import Episode, SolverLibrary
from spincore_nn.action_models import (
    collate_action_observations,
    make_advantage_action_model,
    make_policy_action_model,
)

DOMAIN = "TRUE_HEADS_UP"
REPRESENTATION = "C0_V1_FROZEN_CONTROL"
SNAPSHOT_SCHEMA = "SPINCORE_LT2_HU_STAGE_MODELS_V1"
CHIP_SCALE = 1500.0
FORENSIC_SEEDS = (20260920, 20260921, 20260922, 20260923, 20260924, 20260925)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--stage-b", type=Path, required=True)
    p.add_argument("--scenarios-per-seed", type=int, default=5000)
    p.add_argument("--anchors-per-seed", type=int, default=4)
    p.add_argument("--reference-hands", type=int, default=32)
    p.add_argument("--reference-boards-per-hand", type=int, default=8)
    p.add_argument("--candidate-hands", type=int, default=16)
    p.add_argument("--candidate-boards-per-hand", type=int, default=4)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _mix64(*values: int) -> int:
    x = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    for value in values:
        y = int(value) & mask
        x ^= (y + 0x9E3779B97F4A7C15 + ((x << 6) & mask) + (x >> 2)) & mask
        x &= mask
    return x


def _torch_load_mmap(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def _extract_stage_snapshot(checkpoint: Path, destination: Path) -> dict[str, Any]:
    payload = _torch_load_mmap(checkpoint)
    if payload.get("schema") != "SPINCORE_LEAN_FUNCTIONAL_TRAINING_V1":
        raise RuntimeError(f"unexpected checkpoint schema: {checkpoint}")
    if payload.get("representation") != REPRESENTATION:
        raise RuntimeError(f"representation drift: {checkpoint}")
    if not bool(payload.get("finalized")):
        raise RuntimeError(f"source is not finalized: {checkpoint}")
    state = dict(payload["domains"][DOMAIN])
    snapshot = {
        "schema": SNAPSHOT_SCHEMA,
        "source_checkpoint": str(checkpoint.resolve()),
        "completed_iteration": int(payload["completed_iteration"]),
        "domain_seed": int(state["seed"]),
        "advantage_ready": int(state["counters"].get("advantage_ready", 0)),
        "policy": {
            k: v.detach().cpu().clone()
            for k, v in state["policy"].items()
        },
        "advantage": {
            k: v.detach().cpu().clone()
            for k, v in state["advantage"].items()
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot, destination)
    meta = {
        "completed_iteration": int(snapshot["completed_iteration"]),
        "snapshot_bytes": int(destination.stat().st_size),
        "path": str(destination.resolve()),
    }
    del state, payload, snapshot
    gc.collect()
    return meta


class StageModels:
    def __init__(self, snapshot_path: Path, solver: SolverLibrary):
        payload = torch.load(snapshot_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != SNAPSHOT_SCHEMA:
            raise RuntimeError(f"bad stage snapshot: {snapshot_path}")
        self.completed_iteration = int(payload["completed_iteration"])

        _, self.policy_model = make_policy_action_model(
            REPRESENTATION, device="cpu", seed=0
        )
        self.policy_model.load_state_dict(payload["policy"])
        self.policy_model.eval()

        bundle = make_action_bundle(
            int(payload["domain_seed"]),
            domain=DOMAIN,
            selected_representation=REPRESENTATION,
            action_spec=FIRST_RELEASE_ACTION_SPEC,
            device="cpu",
            reservoir_capacity=1,
            advantage_reservoir_capacity=1,
            policy_reservoir_capacity=1,
            lr=1e-3,
        )
        bundle.advantage.load_state_dict(payload["advantage"])
        bundle.policy.load_state_dict(payload["policy"])
        bundle.counters["advantage_ready"] = int(payload["advantage_ready"])
        session = ActionDeepCFRSession(
            solver_library=solver,
            bundle=bundle,
            action_spec=FIRST_RELEASE_ACTION_SPEC,
            terminal_utility=LeanTrainingScope().terminal_utility,
            device="cpu",
        )
        behavior = LeanNeuralActionAdvantagePolicy(
            bundle.advantage,
            selected_representation=REPRESENTATION,
            device="cpu",
            ready=bool(bundle.counters.get("advantage_ready", 0)),
        )
        session.behavior = behavior
        session.collector.policy = behavior
        self.runtime = SimpleNamespace(bundle=bundle, session=session)

    def average_policy(self, state) -> tuple[int, tuple[int, ...], tuple[float, ...]]:
        street = cond._street(state)
        active_mask = FIRST_RELEASE_ACTION_SPEC.active_mask(street)
        legal = lean_legal_actions(state, active_mask)
        obs = state.neural_bytes()
        batch = collate_action_observations(
            REPRESENTATION,
            [obs],
            [legal_mask(legal)],
            device="cpu",
        )
        with torch.no_grad():
            probs = self.policy_model.probabilities(batch)[0].detach().cpu().tolist()
        return int(active_mask), tuple(int(x) for x in legal), tuple(float(x) for x in probs)


def _sample_policy(legal: tuple[int, ...], probs: tuple[float, ...], rng: random.Random) -> int:
    x = rng.random()
    cumulative = 0.0
    for slot in legal:
        cumulative += float(probs[slot])
        if x < cumulative:
            return int(slot)
    return int(legal[-1])


def _jammer_action(state) -> tuple[int, int]:
    street = cond._street(state)
    active_mask = FIRST_RELEASE_ACTION_SPEC.active_mask(street)
    legal = lean_legal_actions(state, active_mask)
    if 9 in legal:
        return int(active_mask), 9
    if 1 in legal:
        return int(active_mask), 1
    if 0 in legal:
        return int(active_mask), 0
    return int(active_mask), int(max(legal))


def _finish_arm(state, *, hero: int, stage: StageModels, rng: random.Random) -> int:
    decisions = 0
    try:
        while not state.terminal:
            if int(state.actor) == int(hero):
                active_mask, legal, probs = stage.average_policy(state)
                slot = _sample_policy(legal, probs, rng)
            else:
                active_mask, slot = _jammer_action(state)
            apply_lean(state, active_mask, slot)
            decisions += 1
            if decisions > 200:
                raise RuntimeError("overlay evaluation arm exceeded 200 decisions")
        delta = state.terminal_chip_delta()
        return int(delta[int(hero)])
    finally:
        state.close()


def _collect_seed_candidates(
    *,
    solver: SolverLibrary,
    stage_a: StageModels,
    stage_b: StageModels,
    seed: int,
    scenarios: int,
) -> list[dict[str, Any]]:
    sampler = LegacyScenarioSampler(
        seed=int(seed) ^ 0x5CE0A710,
        config=LegacyScenarioConfig(),
    )
    out: list[dict[str, Any]] = []
    hu_seen = 0

    for scenario_index in range(int(scenarios)):
        episode = sampler.sample_episode()
        if not episode.game_is_hu:
            continue
        hu_seen += 1
        live = [seat for seat, stack in enumerate(episode.stacks) if int(stack) > 0]
        if len(live) != 2:
            raise RuntimeError("HU episode without exactly two live seats")
        deal_seed = _mix64(int(seed), int(scenario_index), 0xD34A1)

        for hero in live:
            state_a = solver.create(episode, int(deal_seed))
            state_b = solver.create(episode, int(deal_seed))
            rng_a = random.Random(_mix64(seed, scenario_index, hero, 777))
            rng_b = random.Random(_mix64(seed, scenario_index, hero, 777))
            action_path: list[int] = []
            last_actor = None
            last_action = None
            try:
                for _decision in range(200):
                    if bool(state_a.terminal) != bool(state_b.terminal):
                        raise RuntimeError("paired terminal drift before divergence")
                    if state_a.terminal:
                        break
                    actor = int(state_a.actor)
                    if actor != int(state_b.actor):
                        raise RuntimeError("paired actor drift before divergence")

                    if actor != int(hero):
                        mask_a, slot_a = _jammer_action(state_a)
                        mask_b, slot_b = _jammer_action(state_b)
                        if mask_a != mask_b or slot_a != slot_b:
                            raise RuntimeError("Jammer drift before divergence")
                        apply_lean(state_a, mask_a, slot_a)
                        apply_lean(state_b, mask_b, slot_b)
                        action_path.append(int(slot_a))
                        last_actor, last_action = actor, int(slot_a)
                        continue

                    mask_a, legal_a, probs_a = stage_a.average_policy(state_a)
                    mask_b, legal_b, probs_b = stage_b.average_policy(state_b)
                    if mask_a != mask_b or legal_a != legal_b:
                        raise RuntimeError("A/B legal drift on identical state")
                    slot_a = _sample_policy(legal_a, probs_a, rng_a)
                    slot_b = _sample_policy(legal_b, probs_b, rng_b)
                    if slot_a == slot_b:
                        apply_lean(state_a, mask_a, slot_a)
                        apply_lean(state_b, mask_b, slot_b)
                        action_path.append(int(slot_a))
                        last_actor, last_action = actor, int(slot_a)
                        continue

                    street = cond._street(state_a)
                    facing_all_in = (
                        street == 0
                        and last_actor is not None
                        and int(last_actor) != int(hero)
                        and int(last_action) == 9
                    )
                    if facing_all_in:
                        snapshot = state_a.deal_snapshot()
                        opponent = live[1] if int(hero) == int(live[0]) else live[0]
                        observation = state_a.neural_bytes()
                        legal_mask_tuple = cond._legal_mask_tuple(tuple(legal_a))
                        anchor = {
                            "seed": int(seed),
                            "scenario_index": int(scenario_index),
                            "episode": episode,
                            "deal_seed": int(deal_seed),
                            "blind": f"{episode.small_blind}/{episode.big_blind}",
                            "actor": int(hero),
                            "opponent_seat": int(opponent),
                            "hero_cards": tuple(int(x) for x in snapshot.holes[int(hero)]),
                            "action_path": tuple(int(x) for x in action_path),
                            "observation": observation,
                            "legal_mask": legal_mask_tuple,
                            "avg_a_probs": tuple(float(x) for x in probs_a),
                            "avg_b_probs": tuple(float(x) for x in probs_b),
                            "a_slot": int(slot_a),
                            "b_slot": int(slot_b),
                        }

                        # Preserve the forensic terminal contribution for the selected state.
                        apply_lean(state_a, mask_a, slot_a)
                        apply_lean(state_b, mask_b, slot_b)
                        value_a = _finish_arm(
                            state_a,
                            hero=int(hero),
                            stage=stage_a,
                            rng=rng_a,
                        )
                        state_a = None
                        value_b = _finish_arm(
                            state_b,
                            hero=int(hero),
                            stage=stage_b,
                            rng=rng_b,
                        )
                        state_b = None
                        anchor["forensic_delta_b_minus_a"] = int(value_b - value_a)
                        out.append(anchor)
                    break
            finally:
                if state_a is not None:
                    state_a.close()
                if state_b is not None:
                    state_b.close()

    print(
        f"seed={seed} HU_scenarios={hu_seen} "
        f"facing_allin_first_divergences={len(out)}",
        flush=True,
    )
    return out


def _policy_against_reference(
    policy: torch.Tensor,
    reference_target: torch.Tensor,
    legal: torch.Tensor,
) -> dict[str, Any]:
    ref_policy = _lean_rm_policy_tensor(
        reference_target.unsqueeze(0), legal.unsqueeze(0)
    )[0]
    tv = float((0.5 * torch.abs(policy - ref_policy).sum()).item())
    policy_value = float((policy * reference_target).sum().item())
    ref_value = float((ref_policy * reference_target).sum().item())
    best = float(reference_target.masked_fill(~legal, float("-inf")).max().item())
    return {
        "tv_to_reference": tv,
        "argmax_agreement": bool(
            int(policy.masked_fill(~legal, -1.0).argmax().item())
            == int(ref_policy.masked_fill(~legal, -1.0).argmax().item())
        ),
        "branch_mismatch": bool(
            (policy[legal] > 1e-12).any().item()
            != (ref_policy[legal] > 1e-12).any().item()
        ),
        "signed_reference_policy_minus_policy_value_gap_chips": float(
            (ref_value - policy_value) * CHIP_SCALE
        ),
        "policy_regret_to_reference_best_action_chips": float(
            max(0.0, best - policy_value) * CHIP_SCALE
        ),
        "action_mass": [float(x) for x in policy.tolist()],
        "reference_action_mass": [float(x) for x in ref_policy.tolist()],
    }


def _target_for_explicit_deal(
    *,
    solver: SolverLibrary,
    stage: StageModels,
    anchor: dict[str, Any],
    opponent_hand: tuple[int, int],
    board: tuple[int, int, int, int, int],
    rng_seed: int,
) -> tuple[torch.Tensor, int]:
    holes = cond._holes_for(
        anchor["episode"],
        hero_seat=int(anchor["actor"]),
        hero_cards=tuple(anchor["hero_cards"]),
        opponent_seat=int(anchor["opponent_seat"]),
        opponent_hand=tuple(opponent_hand),
    )
    state, _ = cond._replay_to_anchor(
        solver=solver,
        runtime=stage.runtime,
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
        sample, nodes = _capture_root_target(
            stage.runtime,
            state,
            iteration=int(stage.completed_iteration) + 1,
            exact_level=0,
            rng_seed=int(rng_seed),
        )
    finally:
        state.close()
    if sample.observation != anchor["observation"]:
        raise RuntimeError("overlay target observation drift")
    return torch.tensor(sample.target, dtype=torch.float32), int(nodes)


def _conditional_reference_and_candidate(
    *,
    solver: SolverLibrary,
    stage: StageModels,
    anchor: dict[str, Any],
    ref_hands: int,
    ref_boards: int,
    cand_hands: int,
    cand_boards: int,
    seed_tag: int,
) -> dict[str, Any]:
    hands, probs, posterior = cond._posterior_hand_distribution(
        solver=solver,
        runtime=stage.runtime,
        anchor=anchor,
    )
    hero_cards = tuple(anchor["hero_cards"])
    legal = torch.tensor(anchor["legal_mask"], dtype=torch.bool)

    ref_idx = cond._stratified_hand_indices(
        probs,
        int(ref_hands),
        seed=_mix64(seed_tag, anchor["anchor_index"], 0xA11CE),
    )
    ref_targets: list[torch.Tensor] = []
    ref_nodes = 0
    for hpos, hidx in enumerate(ref_idx):
        hand = hands[int(hidx)]
        for bpos in range(int(ref_boards)):
            rng = random.Random(_mix64(seed_tag, anchor["anchor_index"], hpos, bpos, 0xB0A2D))
            board = cond._board_for(hero_cards, hand, rng=rng)
            target, nodes = _target_for_explicit_deal(
                solver=solver,
                stage=stage,
                anchor=anchor,
                opponent_hand=hand,
                board=board,
                rng_seed=_mix64(seed_tag, anchor["anchor_index"], hpos, bpos, 0x7A2E7),
            )
            ref_targets.append(target)
            ref_nodes += int(nodes)
    reference_target = torch.stack(ref_targets).mean(dim=0)
    reference_policy = _lean_rm_policy_tensor(
        reference_target.unsqueeze(0), legal.unsqueeze(0)
    )[0]

    cand_idx = cond._stratified_hand_indices(
        probs,
        int(cand_hands),
        seed=_mix64(seed_tag, anchor["anchor_index"], 0xC4AD),
    )
    k1_rows = []
    k4_rows = []
    cand_nodes = 0
    for hpos, hidx in enumerate(cand_idx):
        hand = hands[int(hidx)]
        board_targets = []
        board_nodes = []
        for bpos in range(int(cand_boards)):
            rng = random.Random(_mix64(seed_tag, anchor["anchor_index"], hpos, bpos, 0xCA4D))
            board = cond._board_for(hero_cards, hand, rng=rng)
            target, nodes = _target_for_explicit_deal(
                solver=solver,
                stage=stage,
                anchor=anchor,
                opponent_hand=hand,
                board=board,
                rng_seed=_mix64(seed_tag, anchor["anchor_index"], hpos, bpos, 0xE571),
            )
            board_targets.append(target)
            board_nodes.append(int(nodes))
            cand_nodes += int(nodes)

        k1 = board_targets[0]
        k4 = torch.stack(board_targets[:4]).mean(dim=0)
        m1 = _policy_metrics(k1, reference_target, legal)
        m4 = _policy_metrics(k4, reference_target, legal)
        m1["nodes"] = int(board_nodes[0])
        m4["nodes"] = int(sum(board_nodes[:4]))
        k1_rows.append(m1)
        k4_rows.append(m4)

    def avg_candidate(rows):
        return {
            "target_mse_to_reference": float(statistics.fmean(r["target_mse_to_reference"] for r in rows)),
            "policy_tv_to_reference": float(statistics.fmean(r["policy_tv_to_reference"] for r in rows)),
            "argmax_agreement_rate": float(statistics.fmean(1.0 if r["argmax_agreement"] else 0.0 for r in rows)),
            "branch_mismatch_rate": float(statistics.fmean(1.0 if r["branch_mismatch"] else 0.0 for r in rows)),
            "signed_reference_policy_minus_candidate_policy_value_gap_chips": float(
                statistics.fmean(r["signed_reference_policy_minus_candidate_policy_value_gap_chips"] for r in rows)
            ),
            "candidate_policy_regret_to_reference_best_action_chips": float(
                statistics.fmean(r["candidate_policy_regret_to_reference_best_action_chips"] for r in rows)
            ),
            "nodes": float(statistics.fmean(r["nodes"] for r in rows)),
        }

    raw = cond._model_raw(stage.runtime, anchor["observation"], tuple(anchor["legal_mask"]))
    adv_metrics = _policy_metrics(raw, reference_target, legal)
    adv_policy = _lean_rm_policy_tensor(raw.unsqueeze(0), legal.unsqueeze(0))[0]

    avg_probs = anchor["avg_a_probs"] if seed_tag == 0xA5A5 else anchor["avg_b_probs"]
    avg_policy = torch.tensor(avg_probs, dtype=torch.float32)
    avg_metrics = _policy_against_reference(avg_policy, reference_target, legal)

    return {
        "posterior": posterior,
        "reference": {
            "hands": int(ref_hands),
            "boards_per_hand": int(ref_boards),
            "deals": int(len(ref_targets)),
            "mean_nodes_per_deal": float(ref_nodes / max(len(ref_targets), 1)),
            "target": [float(x) for x in reference_target.tolist()],
            "policy": [float(x) for x in reference_policy.tolist()],
        },
        "average_policy": avg_metrics,
        "advantage_policy": {
            **adv_metrics,
            "action_mass": [float(x) for x in adv_policy.tolist()],
        },
        "candidate_k1": avg_candidate(k1_rows),
        "candidate_k4": avg_candidate(k4_rows),
        "candidate_total_nodes": int(cand_nodes),
    }


def _paired_ci(rows: list[dict[str, Any]], getter) -> dict[str, float | int]:
    return _mean_ci([float(getter(r)) for r in rows])


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n": len(rows),
        "forensic_delta_b_minus_a": _mean_ci(
            [float(r["forensic_delta_b_minus_a"]) for r in rows]
        ),
        "sampled_transition_counts": {},
        "reference_policy_a_vs_b_tv": _mean_ci([]),
    }
    transitions: dict[str, int] = {}
    for r in rows:
        key = f"{r['a_slot']}->{r['b_slot']}"
        transitions[key] = transitions.get(key, 0) + 1
    out["sampled_transition_counts"] = dict(
        sorted(transitions.items(), key=lambda kv: (-kv[1], kv[0]))
    )

    ref_tvs = []
    for r in rows:
        pa = torch.tensor(r["stage_a"]["reference"]["policy"], dtype=torch.float32)
        pb = torch.tensor(r["stage_b"]["reference"]["policy"], dtype=torch.float32)
        ref_tvs.append(float(0.5 * torch.abs(pa - pb).sum().item()))
    out["reference_policy_a_vs_b_tv"] = _mean_ci(ref_tvs)

    for stage_key in ("stage_a", "stage_b"):
        block: dict[str, Any] = {}
        for model_key in ("average_policy", "advantage_policy"):
            block[model_key] = {
                "tv_to_reference": _paired_ci(
                    rows, lambda r, sk=stage_key, mk=model_key: r[sk][mk]["tv_to_reference"]
                    if mk == "average_policy"
                    else r[sk][mk]["policy_tv_to_reference"]
                ),
                "regret_chips": _paired_ci(
                    rows, lambda r, sk=stage_key, mk=model_key:
                    r[sk][mk]["policy_regret_to_reference_best_action_chips"]
                    if mk == "average_policy"
                    else r[sk][mk]["candidate_policy_regret_to_reference_best_action_chips"]
                ),
            }
        for cand in ("candidate_k1", "candidate_k4"):
            block[cand] = {
                "target_mse_to_reference": _paired_ci(
                    rows, lambda r, sk=stage_key, ck=cand: r[sk][ck]["target_mse_to_reference"]
                ),
                "policy_tv_to_reference": _paired_ci(
                    rows, lambda r, sk=stage_key, ck=cand: r[sk][ck]["policy_tv_to_reference"]
                ),
                "regret_chips": _paired_ci(
                    rows, lambda r, sk=stage_key, ck=cand:
                    r[sk][ck]["candidate_policy_regret_to_reference_best_action_chips"]
                ),
            }
        out[stage_key] = block

    out["stage_b_minus_a"] = {
        "average_policy_tv_to_own_reference": _paired_ci(
            rows,
            lambda r: r["stage_b"]["average_policy"]["tv_to_reference"]
            - r["stage_a"]["average_policy"]["tv_to_reference"],
        ),
        "average_policy_regret_chips": _paired_ci(
            rows,
            lambda r:
            r["stage_b"]["average_policy"]["policy_regret_to_reference_best_action_chips"]
            - r["stage_a"]["average_policy"]["policy_regret_to_reference_best_action_chips"],
        ),
        "advantage_policy_tv_to_own_reference": _paired_ci(
            rows,
            lambda r:
            r["stage_b"]["advantage_policy"]["policy_tv_to_reference"]
            - r["stage_a"]["advantage_policy"]["policy_tv_to_reference"],
        ),
        "advantage_policy_regret_chips": _paired_ci(
            rows,
            lambda r:
            r["stage_b"]["advantage_policy"]["candidate_policy_regret_to_reference_best_action_chips"]
            - r["stage_a"]["advantage_policy"]["candidate_policy_regret_to_reference_best_action_chips"],
        ),
    }

    for stage_key in ("stage_a", "stage_b"):
        out[stage_key]["k4_minus_k1"] = {
            "target_mse_to_reference": _paired_ci(
                rows,
                lambda r, sk=stage_key:
                r[sk]["candidate_k4"]["target_mse_to_reference"]
                - r[sk]["candidate_k1"]["target_mse_to_reference"],
            ),
            "policy_tv_to_reference": _paired_ci(
                rows,
                lambda r, sk=stage_key:
                r[sk]["candidate_k4"]["policy_tv_to_reference"]
                - r[sk]["candidate_k1"]["policy_tv_to_reference"],
            ),
            "regret_chips": _paired_ci(
                rows,
                lambda r, sk=stage_key:
                r[sk]["candidate_k4"]["candidate_policy_regret_to_reference_best_action_chips"]
                - r[sk]["candidate_k1"]["candidate_policy_regret_to_reference_best_action_chips"],
            ),
        }

    def mean_mass(stage_key: str, model_key: str, action: int) -> float:
        vals = []
        for r in rows:
            if model_key == "average_policy":
                vals.append(float(r[stage_key][model_key]["action_mass"][action]))
            else:
                vals.append(float(r[stage_key][model_key]["action_mass"][action]))
        return float(statistics.fmean(vals))

    out["mean_action_mass"] = {}
    for stage_key in ("stage_a", "stage_b"):
        out["mean_action_mass"][stage_key] = {}
        for model_key in ("average_policy", "advantage_policy"):
            out["mean_action_mass"][stage_key][model_key] = {
                "FOLD": mean_mass(stage_key, model_key, 0),
                "CHECK_CALL": mean_mass(stage_key, model_key, 1),
                "ALL_IN": mean_mass(stage_key, model_key, 9),
            }
        out["mean_action_mass"][stage_key]["reference"] = {
            "FOLD": float(statistics.fmean(r[stage_key]["reference"]["policy"][0] for r in rows)),
            "CHECK_CALL": float(statistics.fmean(r[stage_key]["reference"]["policy"][1] for r in rows)),
            "ALL_IN": float(statistics.fmean(r[stage_key]["reference"]["policy"][9] for r in rows)),
        }
    return out


def main() -> int:
    args = parse_args()
    if args.scenarios_per_seed <= 0 or args.anchors_per_seed <= 0:
        raise SystemExit("positive scenario/anchor counts required")
    if args.reference_hands <= 0 or args.reference_boards_per_hand <= 0:
        raise SystemExit("positive reference design required")
    if args.candidate_hands <= 0 or args.candidate_boards_per_hand < 4:
        raise SystemExit("candidate design requires >=4 boards per hand")
    if args.threads <= 0:
        raise SystemExit("positive thread count required")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    solver_path = args.solver.resolve(strict=True)
    torch.set_num_threads(int(args.threads))
    _selftest_lean_rm_policy_tensor()

    stage_a_snapshot = args.report.parent / "stage_a_hu_models.pt"
    stage_b_snapshot = args.report.parent / "stage_b_hu_models.pt"
    print("Extracting lightweight HU AveragePolicy + Advantage snapshots...", flush=True)
    meta_a = _extract_stage_snapshot(args.stage_a.resolve(strict=True), stage_a_snapshot)
    meta_b = _extract_stage_snapshot(args.stage_b.resolve(strict=True), stage_b_snapshot)
    print(
        f"snapshots A={meta_a['snapshot_bytes']/1048576.0:.2f} MiB "
        f"B={meta_b['snapshot_bytes']/1048576.0:.2f} MiB",
        flush=True,
    )

    solver = SolverLibrary(solver_path)
    stage_a = StageModels(stage_a_snapshot, solver)
    stage_b = StageModels(stage_b_snapshot, solver)

    selected: list[dict[str, Any]] = []
    candidate_counts = {}
    for seed in FORENSIC_SEEDS:
        candidates = _collect_seed_candidates(
            solver=solver,
            stage_a=stage_a,
            stage_b=stage_b,
            seed=int(seed),
            scenarios=int(args.scenarios_per_seed),
        )
        candidate_counts[str(seed)] = len(candidates)
        if len(candidates) < int(args.anchors_per_seed):
            raise RuntimeError(
                f"seed {seed} produced only {len(candidates)} facing-all-in divergences"
            )
        chooser = random.Random(_mix64(seed, 0x0A7E21A9))
        indices = sorted(chooser.sample(range(len(candidates)), int(args.anchors_per_seed)))
        for idx in indices:
            selected.append(candidates[idx])

    for index, anchor in enumerate(selected):
        anchor["anchor_index"] = int(index)

    rows = []
    for index, anchor in enumerate(selected):
        print(
            f"TARGET_OVERLAY anchor={index+1}/{len(selected)} "
            f"seed={anchor['seed']} scenario={anchor['scenario_index']} "
            f"blind={anchor['blind']} transition={anchor['a_slot']}->{anchor['b_slot']}",
            flush=True,
        )
        a = _conditional_reference_and_candidate(
            solver=solver,
            stage=stage_a,
            anchor=anchor,
            ref_hands=int(args.reference_hands),
            ref_boards=int(args.reference_boards_per_hand),
            cand_hands=int(args.candidate_hands),
            cand_boards=int(args.candidate_boards_per_hand),
            seed_tag=0xA5A5,
        )
        b = _conditional_reference_and_candidate(
            solver=solver,
            stage=stage_b,
            anchor=anchor,
            ref_hands=int(args.reference_hands),
            ref_boards=int(args.reference_boards_per_hand),
            cand_hands=int(args.candidate_hands),
            cand_boards=int(args.candidate_boards_per_hand),
            seed_tag=0xB6B6,
        )
        rows.append({
            "anchor_index": int(index),
            "seed": int(anchor["seed"]),
            "scenario_index": int(anchor["scenario_index"]),
            "blind": str(anchor["blind"]),
            "actor": int(anchor["actor"]),
            "path_length": len(anchor["action_path"]),
            "action_path": list(anchor["action_path"]),
            "a_slot": int(anchor["a_slot"]),
            "b_slot": int(anchor["b_slot"]),
            "forensic_delta_b_minus_a": int(anchor["forensic_delta_b_minus_a"]),
            "stage_a": a,
            "stage_b": b,
        })

    summary = _aggregate(rows)
    report = {
        "schema": "SPINCORE_LT2_JAMMER_FACING_ALLIN_TARGET_OVERLAY_V1",
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
            "training_memory_writes": 0,
            "optimizer_steps": 0,
            "new_training_roots": 0,
            "forensic_seeds": list(FORENSIC_SEEDS),
            "future_holdout_seeds_touched": False,
            "scenarios_per_seed": int(args.scenarios_per_seed),
            "selection": (
                "deterministic balanced sample of actual HU Jammer seat-runs whose "
                "first sampled Stage-A/Stage-B hero-action divergence occurred preflop "
                "immediately after opponent ALL_IN"
            ),
            "anchors_per_seed": int(args.anchors_per_seed),
            "reference": {
                "stage_specific_self_play_posterior": True,
                "posterior_hand_strata": int(args.reference_hands),
                "future_boards_per_hand": int(args.reference_boards_per_hand),
                "exact_opponent_levels": 0,
                "note": (
                    "opponent is already all-in at the anchor, so there is no future "
                    "opponent-action branch to exactify; remaining target uncertainty "
                    "is hidden hand/future board"
                ),
            },
            "candidate_estimator": {
                "posterior_hand_draws": int(args.candidate_hands),
                "boards_per_fixed_hand": int(args.candidate_boards_per_hand),
                "compare": ["K1", "K4"],
            },
            "warning": (
                "selected states are forensic failure states, not natural-visitation "
                "estimates and not a GTO oracle"
            ),
        },
        "candidate_counts_by_seed": candidate_counts,
        "summary": summary,
        "rows": rows,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== JAMMER FACING-ALL-IN TARGET OVERLAY ===")
    print(
        f"anchors={summary['n']} forensic_B-A="
        f"{summary['forensic_delta_b_minus_a']['mean']:+.2f} chips on selected divergences"
    )
    for stage_key in ("stage_a", "stage_b"):
        x = summary[stage_key]
        print(
            f"{stage_key}: avgTV={x['average_policy']['tv_to_reference']['mean']:.4f} "
            f"advTV={x['advantage_policy']['tv_to_reference']['mean']:.4f} "
            f"K1TV={x['candidate_k1']['policy_tv_to_reference']['mean']:.4f} "
            f"K4TV={x['candidate_k4']['policy_tv_to_reference']['mean']:.4f}"
        )
    d = summary["stage_b_minus_a"]
    print(
        "B-A error deltas: "
        f"avgTV={d['average_policy_tv_to_own_reference']['mean']:+.4f} "
        f"advTV={d['advantage_policy_tv_to_own_reference']['mean']:+.4f} "
        f"avgRegret={d['average_policy_regret_chips']['mean']:+.2f} "
        f"advRegret={d['advantage_policy_regret_chips']['mean']:+.2f}"
    )
    print("LT2_JAMMER_FACING_ALLIN_TARGET_OVERLAY_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
