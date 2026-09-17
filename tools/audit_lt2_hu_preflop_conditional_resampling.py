#!/usr/bin/env python3
from __future__ import annotations

"""Targeted HU-preflop conditional target decomposition for LT2 Stage B.

The stored-reservoir same-input audit found too few exact duplicate inputs to
estimate conditional target variance representatively. This diagnostic creates
fresh conditional resamples while keeping the *observable* HU preflop information
state fixed.

For each selected HU preflop anchor:
  1. preserve hero hole cards, public betting state, and public action path;
  2. enumerate all 2450 ordered opponent two-card assignments consistent with
     the hero cards;
  3. compute the current Stage-B opponent-policy reach likelihood of the observed
     public path for each opponent hand;
  4. draw opponent hands by deterministic randomized stratification from that
     posterior;
  5. draw multiple independent future boards per selected hand;
  6. recompute Advantage targets multiple times per exact hidden deal using
     exact_opponent_levels=1.

Balanced nested resampling yields the exact finite-sample decomposition

  sample-target MSE
    = within-deal opponent-action sampling variance
    + future-board variance within opponent hand
    + opponent-hand posterior variance
    + current-model MSE to the conditional mean target.

This is a diagnostic of the current Stage-B target process. The conditional mean
is not a GTO oracle, and the variance components are not intrinsic game-theoretic
floors.
"""

import argparse
import bisect
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

from audit_lt2_checkpoint_fit import _lean_rm_policy_tensor, _selftest_lean_rm_policy_tensor
from audit_lt2_repeated_target_variance import _capture_root_target
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_functional_training import (
    REPRESENTATION,
    _root_deck_seed,
    load_checkpoint,
)
from spincore.lean_solver_actions import apply_lean, lean_legal_actions
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_cfr import sample_action
from spincore.solver import Episode, SolverLibrary
from spincore_nn.action_models import collate_action_observations
from spincore_nn.codec import decode_spnniv1

DOMAIN = "TRUE_HEADS_UP"
CHIP_SCALE = 1500.0
EXACT_LEVEL = 1
REGION_QUOTAS_DEFAULT = {
    "PREFLOP_ROOT": 16,
    "PREFLOP_CONTINUATION_1": 32,
    "PREFLOP_CONTINUATION_2PLUS": 16,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--hands-per-state", type=int, default=16)
    p.add_argument("--boards-per-hand", type=int, default=4)
    p.add_argument("--repeats-per-deal", type=int, default=4)
    p.add_argument("--root-states", type=int, default=REGION_QUOTAS_DEFAULT["PREFLOP_ROOT"])
    p.add_argument("--cont1-states", type=int, default=REGION_QUOTAS_DEFAULT["PREFLOP_CONTINUATION_1"])
    p.add_argument("--cont2-states", type=int, default=REGION_QUOTAS_DEFAULT["PREFLOP_CONTINUATION_2PLUS"])
    p.add_argument("--max-episodes", type=int, default=50000)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _street(state) -> int:
    value = int(decode_spnniv1(state.neural_bytes()).categorical[1])
    if value not in (0, 1, 2, 3):
        raise RuntimeError(f"bad street {value}")
    return value


def _region(path_length: int) -> str:
    n = int(path_length)
    if n <= 0:
        return "PREFLOP_ROOT"
    if n == 1:
        return "PREFLOP_CONTINUATION_1"
    return "PREFLOP_CONTINUATION_2PLUS"


def _live_seats(episode: Episode) -> tuple[int, int]:
    dead = set(int(x) for x in episode.dead_players)
    if not dead and episode.game_is_hu:
        dead = {i for i, stack in enumerate(episode.stacks) if int(stack) <= 0}
    live = tuple(i for i in range(3) if i not in dead)
    if not episode.game_is_hu or len(live) != 2:
        raise RuntimeError(f"expected exactly two live HU seats, got {live}")
    return int(live[0]), int(live[1])


def _legal_mask_tuple(legal: tuple[int, ...]) -> tuple[int, ...]:
    s = set(int(x) for x in legal)
    return tuple(1 if i in s else 0 for i in range(10))


def _sample_current_action(runtime, state, rng: random.Random) -> int:
    street = _street(state)
    active_mask = FIRST_RELEASE_ACTION_SPEC.active_mask(street)
    legal = lean_legal_actions(state, active_mask)
    if not legal:
        raise RuntimeError("nonterminal state has no lean legal action")
    observation = state.neural_bytes()
    probs = runtime.session.behavior(state, observation, legal)
    slot = int(sample_action(probs, legal, rng))
    apply_lean(state, active_mask, slot)
    return slot


def _ordered_hands(hero_cards: tuple[int, int]) -> list[tuple[int, int]]:
    blocked = {int(hero_cards[0]), int(hero_cards[1])}
    cards = [c for c in range(52) if c not in blocked]
    hands = [(a, b) for a in cards for b in cards if b != a]
    if len(hands) != 2450:
        raise RuntimeError(f"ordered HU opponent-hand count drift: {len(hands)}")
    return hands


def _holes_for(
    episode: Episode,
    *,
    hero_seat: int,
    hero_cards: tuple[int, int],
    opponent_seat: int,
    opponent_hand: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    live = set(_live_seats(episode))
    if hero_seat not in live or opponent_seat not in live or hero_seat == opponent_seat:
        raise RuntimeError("invalid HU hero/opponent seats")
    holes = [(-1, -1), (-1, -1), (-1, -1)]
    holes[int(hero_seat)] = (int(hero_cards[0]), int(hero_cards[1]))
    holes[int(opponent_seat)] = (int(opponent_hand[0]), int(opponent_hand[1]))
    return tuple(holes)  # type: ignore[return-value]


def _board_for(
    hero_cards: tuple[int, int],
    opponent_hand: tuple[int, int],
    *,
    rng: random.Random,
) -> tuple[int, int, int, int, int]:
    used = {int(hero_cards[0]), int(hero_cards[1]), int(opponent_hand[0]), int(opponent_hand[1])}
    if len(used) != 4:
        raise ValueError("hero/opponent cards collide")
    remaining = [c for c in range(52) if c not in used]
    board = tuple(int(x) for x in rng.sample(remaining, 5))
    return board  # type: ignore[return-value]


def _deterministic_board(hero_cards: tuple[int, int], opponent_hand: tuple[int, int]) -> tuple[int, int, int, int, int]:
    used = {int(hero_cards[0]), int(hero_cards[1]), int(opponent_hand[0]), int(opponent_hand[1])}
    remaining = [c for c in range(52) if c not in used]
    return tuple(remaining[:5])  # type: ignore[return-value]


def _replay_to_anchor(
    *,
    solver: SolverLibrary,
    runtime,
    episode: Episode,
    holes,
    board,
    action_path: tuple[int, ...],
    target_actor: int,
    target_observation: bytes,
    target_legal_mask: tuple[int, ...],
    compute_opponent_log_reach: bool,
):
    state = solver.create_with_deal(episode, holes, board)
    log_reach = 0.0
    try:
        for slot in action_path:
            if state.terminal:
                raise RuntimeError("conditional replay reached terminal before anchor")
            if _street(state) != 0:
                raise RuntimeError("conditional replay left preflop before anchor")
            active_mask = FIRST_RELEASE_ACTION_SPEC.active_mask(0)
            legal = lean_legal_actions(state, active_mask)
            if int(slot) not in legal:
                raise RuntimeError("conditional replay action became illegal")
            if compute_opponent_log_reach and int(state.actor) != int(target_actor):
                obs = state.neural_bytes()
                probs = runtime.session.behavior(state, obs, legal)
                p = float(probs[int(slot)])
                if p <= 0.0 or not math.isfinite(p):
                    log_reach = -math.inf
                elif math.isfinite(log_reach):
                    log_reach += math.log(p)
            apply_lean(state, active_mask, int(slot))

        if state.terminal or _street(state) != 0:
            raise RuntimeError("conditional replay anchor is not live preflop")
        if int(state.actor) != int(target_actor):
            raise RuntimeError("conditional replay actor drift")
        obs = state.neural_bytes()
        active_mask = FIRST_RELEASE_ACTION_SPEC.active_mask(0)
        legal = lean_legal_actions(state, active_mask)
        if obs != target_observation:
            raise RuntimeError("conditional replay changed SPNNIV1 target observation")
        if _legal_mask_tuple(legal) != tuple(target_legal_mask):
            raise RuntimeError("conditional replay changed target legal mask")
        return state, float(log_reach)
    except Exception:
        state.close()
        raise


def _posterior_hand_distribution(
    *,
    solver: SolverLibrary,
    runtime,
    anchor: dict[str, Any],
) -> tuple[list[tuple[int, int]], list[float], dict[str, float | int]]:
    hero_cards = tuple(int(x) for x in anchor["hero_cards"])
    hands = _ordered_hands(hero_cards)
    logs: list[float] = []
    for hand in hands:
        holes = _holes_for(
            anchor["episode"],
            hero_seat=int(anchor["actor"]),
            hero_cards=hero_cards,
            opponent_seat=int(anchor["opponent_seat"]),
            opponent_hand=hand,
        )
        board = _deterministic_board(hero_cards, hand)
        state, logw = _replay_to_anchor(
            solver=solver,
            runtime=runtime,
            episode=anchor["episode"],
            holes=holes,
            board=board,
            action_path=tuple(anchor["action_path"]),
            target_actor=int(anchor["actor"]),
            target_observation=anchor["observation"],
            target_legal_mask=tuple(anchor["legal_mask"]),
            compute_opponent_log_reach=True,
        )
        state.close()
        logs.append(float(logw))

    finite = [x for x in logs if math.isfinite(x)]
    if not finite:
        raise RuntimeError("all opponent-hand posterior reach weights are zero")
    maximum = max(finite)
    raw = [math.exp(x - maximum) if math.isfinite(x) else 0.0 for x in logs]
    total = float(sum(raw))
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("invalid opponent-hand posterior normalizer")
    probs = [float(x / total) for x in raw]
    ess = float(1.0 / max(sum(p * p for p in probs), 1e-300))
    positive = int(sum(1 for p in probs if p > 0.0))
    max_p = float(max(probs))
    entropy = float(-sum(p * math.log(p) for p in probs if p > 0.0))
    return hands, probs, {
        "ordered_hand_support": len(hands),
        "positive_weight_hands": positive,
        "effective_sample_size": ess,
        "max_hand_probability": max_p,
        "entropy_nats": entropy,
    }


def _stratified_hand_indices(probs: list[float], k: int, *, seed: int) -> list[int]:
    if k <= 0:
        raise ValueError("hands-per-state must be positive")
    cdf = []
    running = 0.0
    for p in probs:
        running += float(p)
        cdf.append(running)
    cdf[-1] = 1.0
    rng = random.Random(int(seed))
    out = []
    for j in range(int(k)):
        u = (j + rng.random()) / float(k)
        idx = bisect.bisect_left(cdf, u)
        out.append(min(idx, len(cdf) - 1))
    return out


def _model_raw(runtime, observation: bytes, legal_mask: tuple[int, ...]) -> torch.Tensor:
    batch = collate_action_observations(
        REPRESENTATION,
        [observation],
        [legal_mask],
        device="cpu",
    )
    runtime.bundle.advantage.eval()
    with torch.no_grad():
        return runtime.bundle.advantage(batch)[0].detach().cpu().float()


def _mse_over_legal(diff: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    legal_f = legal.float()
    denom = legal_f.sum().clamp_min(1.0)
    return ((diff * diff) * legal_f).sum(dim=-1) / denom


def _anchor_metrics(
    *,
    runtime,
    solver: SolverLibrary,
    anchor: dict[str, Any],
    completed: int,
    hands_per_state: int,
    boards_per_hand: int,
    repeats_per_deal: int,
    seed: int,
) -> dict[str, Any]:
    hands, probs, posterior = _posterior_hand_distribution(
        solver=solver,
        runtime=runtime,
        anchor=anchor,
    )
    selected = _stratified_hand_indices(
        probs,
        int(hands_per_state),
        seed=(int(seed) ^ (int(anchor["anchor_index"]) * 0x9E3779B1) ^ 0xC01DCAFE) & ((1 << 63) - 1),
    )
    hero_cards = tuple(int(x) for x in anchor["hero_cards"])
    pred = _model_raw(runtime, anchor["observation"], tuple(anchor["legal_mask"]))
    legal = torch.tensor(anchor["legal_mask"], dtype=torch.bool)
    if int(legal.sum().item()) <= 0:
        raise RuntimeError("anchor has empty legal mask")

    all_targets: list[list[list[tuple[float, ...]]]] = []
    all_nodes: list[int] = []
    unique_selected = len(set(selected))

    for hpos, hand_idx in enumerate(selected):
        hand = hands[int(hand_idx)]
        boards: list[list[tuple[float, ...]]] = []
        for bpos in range(int(boards_per_hand)):
            board_rng = random.Random(
                (
                    int(seed)
                    ^ 0xB04D5000
                    ^ (int(anchor["anchor_index"]) * 0x45D9F3B)
                    ^ (hpos * 0x13579B)
                    ^ (bpos * 0x2468D)
                )
                & ((1 << 63) - 1)
            )
            board = _board_for(hero_cards, hand, rng=board_rng)
            holes = _holes_for(
                anchor["episode"],
                hero_seat=int(anchor["actor"]),
                hero_cards=hero_cards,
                opponent_seat=int(anchor["opponent_seat"]),
                opponent_hand=hand,
            )
            state, _ = _replay_to_anchor(
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
            repeats: list[tuple[float, ...]] = []
            try:
                for rep in range(int(repeats_per_deal)):
                    rng_seed = (
                        int(seed)
                        ^ 0xA17E1000
                        ^ (int(anchor["anchor_index"]) * 0x7F4A7C15)
                        ^ (hpos * 0x94D049BB)
                        ^ (bpos * 0x369DEA0F)
                        ^ (rep * 0x9E3779B1)
                    ) & ((1 << 63) - 1)
                    sample, nodes = _capture_root_target(
                        runtime,
                        state,
                        iteration=int(completed) + 1,
                        exact_level=EXACT_LEVEL,
                        rng_seed=int(rng_seed),
                    )
                    if sample.observation != anchor["observation"] or tuple(sample.legal) != tuple(anchor["legal_mask"]):
                        raise RuntimeError("conditional target sample identity drift")
                    repeats.append(tuple(float(x) for x in sample.target))
                    all_nodes.append(int(nodes))
            finally:
                state.close()
            boards.append(repeats)
        all_targets.append(boards)

    targets = torch.tensor(all_targets, dtype=torch.float32)
    deal_mean = targets.mean(dim=2)
    hand_mean = deal_mean.mean(dim=1)
    grand_mean = hand_mean.mean(dim=0)

    within_action = float(_mse_over_legal(targets - deal_mean.unsqueeze(2), legal).mean().item())
    future_board = float(_mse_over_legal(deal_mean - hand_mean.unsqueeze(1), legal).mean().item())
    opponent_hand = float(_mse_over_legal(hand_mean - grand_mean.unsqueeze(0), legal).mean().item())
    model_error = float(_mse_over_legal(pred - grand_mean, legal).item())
    sample_mse = float(_mse_over_legal(targets - pred, legal).mean().item())
    component_sum = within_action + future_board + opponent_hand + model_error

    legal2 = legal.unsqueeze(0)
    target_policy = _lean_rm_policy_tensor(grand_mean.unsqueeze(0), legal2)[0]
    model_policy = _lean_rm_policy_tensor(pred.unsqueeze(0), legal2)[0]
    tv = float((0.5 * torch.abs(target_policy - model_policy).sum()).item())
    targ_arg = int(target_policy.masked_fill(~legal, -1.0).argmax().item())
    pred_arg = int(model_policy.masked_fill(~legal, -1.0).argmax().item())
    target_value = float((target_policy * grand_mean).sum().item())
    model_value = float((model_policy * grand_mean).sum().item())
    target_max = float(grand_mean.masked_fill(~legal, float("-inf")).max().item())
    signed_gap = float((target_value - model_value) * CHIP_SCALE)
    model_regret = float(max(0.0, target_max - model_value) * CHIP_SCALE)

    target_positive = bool((torch.clamp(grand_mean, min=0.0) * legal.float()).sum().item() > 0.0)
    pred_positive = bool((torch.clamp(pred, min=0.0) * legal.float()).sum().item() > 0.0)

    return {
        "anchor_index": int(anchor["anchor_index"]),
        "episode_index": int(anchor["episode_index"]),
        "region": str(anchor["region"]),
        "path_length": len(anchor["action_path"]),
        "last_action_slot": None if not anchor["action_path"] else int(anchor["action_path"][-1]),
        "facing_all_in": bool(anchor["action_path"] and int(anchor["action_path"][-1]) == 9),
        "actor": int(anchor["actor"]),
        "opponent_seat": int(anchor["opponent_seat"]),
        "legal_action_count": int(legal.sum().item()),
        "posterior": {
            **posterior,
            "stratified_draws": int(hands_per_state),
            "unique_selected_ordered_hands": int(unique_selected),
        },
        "design": {
            "boards_per_selected_hand": int(boards_per_hand),
            "target_repeats_per_exact_deal": int(repeats_per_deal),
            "exact_opponent_levels": int(EXACT_LEVEL),
        },
        "mse_components": {
            "within_deal_opponent_action": within_action,
            "future_board_within_hand": future_board,
            "opponent_hand_posterior": opponent_hand,
            "model_to_conditional_mean": model_error,
            "sample_target_mse": sample_mse,
            "component_sum": component_sum,
            "closure_abs_error": float(abs(sample_mse - component_sum)),
            "conditional_variance_total": float(within_action + future_board + opponent_hand),
            "conditional_variance_fraction": float(
                (within_action + future_board + opponent_hand) / max(sample_mse, 1e-12)
            ),
            "model_conditional_mean_error_fraction": float(model_error / max(sample_mse, 1e-12)),
        },
        "policy_to_conditional_mean": {
            "tv": tv,
            "argmax_agreement": bool(targ_arg == pred_arg),
            "branch_mismatch": bool(target_positive != pred_positive),
            "signed_conditional_mean_policy_minus_model_policy_value_gap_chips": signed_gap,
            "model_policy_regret_to_conditional_mean_best_action_chips": model_regret,
            "target_action_mass": [float(x) for x in target_policy.tolist()],
            "model_action_mass": [float(x) for x in model_policy.tolist()],
        },
        "node_cost": {
            "target_traversals": int(len(all_nodes)),
            "mean_nodes": float(statistics.fmean(all_nodes)) if all_nodes else 0.0,
            "max_nodes": int(max(all_nodes)) if all_nodes else 0,
            "total_nodes": int(sum(all_nodes)),
        },
    }


def _mean_ci(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sem": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(statistics.fmean(values))
    sem = 0.0 if n == 1 else float(statistics.stdev(values) / math.sqrt(n))
    half = 1.96 * sem
    return {"n": n, "mean": mean, "sem": sem, "ci95_low": mean - half, "ci95_high": mean + half}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    component_names = (
        "within_deal_opponent_action",
        "future_board_within_hand",
        "opponent_hand_posterior",
        "model_to_conditional_mean",
        "sample_target_mse",
        "conditional_variance_total",
    )
    components = {
        name: _mean_ci([float(r["mse_components"][name]) for r in rows])
        for name in component_names
    }
    sample_mean = float(components["sample_target_mse"]["mean"])
    means = {name: float(components[name]["mean"]) for name in component_names}
    variance_total = (
        means["within_deal_opponent_action"]
        + means["future_board_within_hand"]
        + means["opponent_hand_posterior"]
    )
    model_mean = means["model_to_conditional_mean"]

    tv = _mean_ci([float(r["policy_to_conditional_mean"]["tv"]) for r in rows])
    gap = _mean_ci([
        float(r["policy_to_conditional_mean"]["signed_conditional_mean_policy_minus_model_policy_value_gap_chips"])
        for r in rows
    ])
    regret = _mean_ci([
        float(r["policy_to_conditional_mean"]["model_policy_regret_to_conditional_mean_best_action_chips"])
        for r in rows
    ])
    ess = _mean_ci([float(r["posterior"]["effective_sample_size"]) for r in rows])
    node = _mean_ci([float(r["node_cost"]["mean_nodes"]) for r in rows])

    target_mass = [0.0] * 10
    model_mass = [0.0] * 10
    for r in rows:
        for i, value in enumerate(r["policy_to_conditional_mean"]["target_action_mass"]):
            target_mass[i] += float(value)
        for i, value in enumerate(r["policy_to_conditional_mean"]["model_action_mass"]):
            model_mass[i] += float(value)
    target_mass = [x / len(rows) for x in target_mass]
    model_mass = [x / len(rows) for x in model_mass]

    return {
        "n": len(rows),
        "components": components,
        "decomposition_from_component_means": {
            "conditional_variance_fraction": float(variance_total / max(sample_mean, 1e-12)),
            "within_deal_opponent_action_fraction": float(means["within_deal_opponent_action"] / max(sample_mean, 1e-12)),
            "future_board_fraction": float(means["future_board_within_hand"] / max(sample_mean, 1e-12)),
            "opponent_hand_fraction": float(means["opponent_hand_posterior"] / max(sample_mean, 1e-12)),
            "model_conditional_mean_error_fraction": float(model_mean / max(sample_mean, 1e-12)),
            "closure_abs_error": float(abs(sample_mean - (variance_total + model_mean))),
        },
        "policy": {
            "tv": tv,
            "signed_value_gap_chips": gap,
            "model_regret_chips": regret,
            "argmax_agreement_rate": float(statistics.fmean(
                1.0 if bool(r["policy_to_conditional_mean"]["argmax_agreement"]) else 0.0 for r in rows
            )),
            "branch_mismatch_rate": float(statistics.fmean(
                1.0 if bool(r["policy_to_conditional_mean"]["branch_mismatch"]) else 0.0 for r in rows
            )),
            "target_action_mass": target_mass,
            "model_action_mass": model_mass,
        },
        "posterior_effective_sample_size": ess,
        "mean_nodes_per_target_traversal": node,
        "total_target_traversals": int(sum(int(r["node_cost"]["target_traversals"]) for r in rows)),
        "total_nodes": int(sum(int(r["node_cost"]["total_nodes"]) for r in rows)),
    }


def main() -> int:
    args = parse_args()
    if (
        args.hands_per_state <= 0
        or args.boards_per_hand <= 0
        or args.repeats_per_deal < 2
        or args.root_states < 0
        or args.cont1_states < 0
        or args.cont2_states < 0
        or args.max_episodes <= 0
        or args.threads <= 0
    ):
        raise SystemExit("invalid resampling design")

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
    counts = {name: 0 for name in quotas}
    anchors: list[dict[str, Any]] = []

    sampler = LegacyScenarioSampler(
        seed=int(seed) ^ 0xC0D1710A,
        config=LegacyScenarioConfig(heads_up_prob=1.0),
    )

    episodes_seen = 0
    while any(counts[name] < quotas[name] for name in quotas) and episodes_seen < int(args.max_episodes):
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
            while not state.terminal and _street(state) == 0:
                region = _region(len(action_path))
                if (
                    region in quotas
                    and counts[region] < quotas[region]
                    and region not in captured_regions
                ):
                    actor = int(state.actor)
                    live = _live_seats(episode)
                    opponent = live[1] if actor == live[0] else live[0]
                    snapshot = state.deal_snapshot()
                    active_mask = FIRST_RELEASE_ACTION_SPEC.active_mask(0)
                    legal = lean_legal_actions(state, active_mask)
                    anchor = {
                        "anchor_index": len(anchors),
                        "episode_index": int(episode_index),
                        "episode": episode,
                        "region": region,
                        "actor": actor,
                        "opponent_seat": int(opponent),
                        "hero_cards": tuple(int(x) for x in snapshot.holes[actor]),
                        "action_path": tuple(int(x) for x in action_path),
                        "observation": state.neural_bytes(),
                        "legal_mask": _legal_mask_tuple(legal),
                    }
                    anchors.append(anchor)
                    counts[region] += 1
                    captured_regions.add(region)

                slot = _sample_current_action(runtime, state, trajectory_rng)
                action_path.append(int(slot))
                if len(action_path) > 20:
                    raise RuntimeError("HU preflop diagnostic path exceeded 20 decisions")
        finally:
            state.close()

    if any(counts[name] < quotas[name] for name in quotas):
        missing = {name: quotas[name] - counts[name] for name in quotas if counts[name] < quotas[name]}
        raise RuntimeError(
            f"failed to fill HU-preflop anchor quotas after {episodes_seen} episodes: {missing}"
        )

    rows: list[dict[str, Any]] = []
    for i, anchor in enumerate(anchors):
        print(
            f"CONDITIONAL_TARGET anchor={i+1}/{len(anchors)} "
            f"region={anchor['region']} path={len(anchor['action_path'])} "
            f"last={None if not anchor['action_path'] else anchor['action_path'][-1]}",
            flush=True,
        )
        row = _anchor_metrics(
            runtime=runtime,
            solver=solver,
            anchor=anchor,
            completed=int(completed),
            hands_per_state=int(args.hands_per_state),
            boards_per_hand=int(args.boards_per_hand),
            repeats_per_deal=int(args.repeats_per_deal),
            seed=int(seed) ^ 0x51A7C0DE,
        )
        rows.append(row)

    summaries: dict[str, Any] = {
        "ALL_ANCHORS_EQUAL_WEIGHT": _aggregate(rows),
    }
    for region in quotas:
        summaries[region] = _aggregate([r for r in rows if r["region"] == region])
    facing = [r for r in rows if bool(r["facing_all_in"])]
    summaries["FACING_ALL_IN_LAST_ACTION"] = _aggregate(facing)

    report = {
        "schema": "SPINCORE_LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_V1",
        "checkpoint": str(checkpoint),
        "completed_iteration": int(completed),
        "seed": int(seed),
        "method": {
            "read_only": True,
            "no_training_memory_writes": True,
            "no_optimizer_steps": True,
            "new_training_roots": 0,
            "domain": DOMAIN,
            "street": "PREFLOP",
            "anchor_policy": "stored Stage-B Advantage behavior",
            "anchor_region_quotas": quotas,
            "posterior": (
                "all 2450 ordered opponent hands enumerated; weight equals current Stage-B "
                "opponent behavior likelihood of the observed public preflop path; hero-action "
                "factors cancel because hero cards/public observation are fixed"
            ),
            "opponent_hand_sampling": (
                "deterministic seeded randomized-stratified draws from the exact enumerated "
                "opponent-hand posterior"
            ),
            "future_board_sampling": (
                "uniform ordered five-card board samples conditional on hero/opponent cards; "
                "board is still hidden at every audited preflop anchor"
            ),
            "exact_opponent_levels": EXACT_LEVEL,
            "hands_per_state": int(args.hands_per_state),
            "boards_per_hand": int(args.boards_per_hand),
            "repeats_per_exact_deal": int(args.repeats_per_deal),
            "nested_decomposition": (
                "sample-target MSE = within-deal opponent-action variance + future-board variance "
                "+ opponent-hand posterior variance + model MSE to current conditional mean"
            ),
            "interpretation_limit": (
                "conditional mean is the current Stage-B target-process mean under the current "
                "behavior posterior, not a GTO oracle; equal-anchor summaries are diagnostic, "
                "not natural-visitation EV"
            ),
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
    print("=== HU PREFLOP CONDITIONAL RESAMPLING ===")
    print(json.dumps({
        "anchors": overall.get("n", 0),
        "decomposition": overall.get("decomposition_from_component_means"),
        "policy_tv_mean": ((overall.get("policy") or {}).get("tv") or {}).get("mean"),
        "signed_value_gap_chips_mean": ((overall.get("policy") or {}).get("signed_value_gap_chips") or {}).get("mean"),
        "facing_all_in_anchors": summaries["FACING_ALL_IN_LAST_ACTION"].get("n", 0),
    }, indent=2, sort_keys=True))
    print("LT2_HU_PREFLOP_CONDITIONAL_RESAMPLING_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
