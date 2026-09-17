#!/usr/bin/env python3
from __future__ import annotations

"""Repeated-state Advantage target variance audit for LT2 Stage B.

This diagnostic does not update the checkpoint and does not perform optimizer steps.
It samples ordinary current-policy trajectories, selects the first decision reached
on each street, and repeatedly recomputes the traverser's Advantage target from the
exact same solver state/deal. Repetition isolates Monte-Carlo opponent-action noise
inside external sampling. The same states are evaluated with exact_opponent_levels
0 and 1 to measure variance reduction versus node cost.

For each fixed state we decompose held target MSE exactly:
    E[(prediction - sampled_target)^2]
      = Var(sampled_target around repeat mean)
      + (prediction - repeat_mean)^2
over legal actions.

The repeat mean is only a lower-noise diagnostic reference, not a GTO oracle.
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

from audit_lt2_checkpoint_fit import _lean_rm_policy_tensor, _selftest_lean_rm_policy_tensor
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_functional_training import (
    DOMAINS,
    REPRESENTATION,
    _root_deck_seed,
    load_checkpoint,
)
from spincore.lean_solver_actions import apply_lean, lean_legal_actions
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_cfr import sample_action
from spincore.solver import SolverLibrary
from spincore_nn.action_models import collate_action_observations
from spincore_nn.codec import decode_spnniv1

CHIP_SCALE = 1500.0
STREET_NAMES = {0: "PREFLOP", 1: "FLOP", 2: "TURN", 3: "RIVER"}
LEVELS = (0, 1)


class CaptureMemory:
    def __init__(self) -> None:
        self.items = []

    def add(self, item) -> None:
        self.items.append(item)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--states-per-street", type=int, default=64)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-episodes", type=int, default=20000)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _street(state) -> int:
    value = int(decode_spnniv1(state.neural_bytes()).categorical[1])
    if value not in STREET_NAMES:
        raise RuntimeError(f"bad street {value}")
    return value


def _sample_current_action(runtime, state, rng: random.Random) -> None:
    street = _street(state)
    active_mask = FIRST_RELEASE_ACTION_SPEC.active_mask(street)
    legal = lean_legal_actions(state, active_mask)
    if not legal:
        raise RuntimeError("nonterminal state has no lean legal action")
    obs = state.neural_bytes()
    probs = runtime.session.behavior(state, obs, legal)
    slot = sample_action(probs, legal, rng)
    apply_lean(state, active_mask, slot)


def _capture_root_target(
    runtime,
    state,
    *,
    iteration: int,
    exact_level: int,
    rng_seed: int,
):
    collector = runtime.session.collector
    old_memory = collector.advantage_memory
    old_rng = collector.rng
    capture = CaptureMemory()
    root_observation = state.neural_bytes()
    try:
        collector.advantage_memory = capture
        collector.rng = random.Random(int(rng_seed))
        result = collector.collect_advantage_partial_exact(
            state,
            traverser=int(state.actor),
            iteration=int(iteration),
            exact_opponent_levels=int(exact_level),
        )
    finally:
        collector.advantage_memory = old_memory
        collector.rng = old_rng

    matches = [x for x in capture.items if x.observation == root_observation]
    if not matches:
        raise RuntimeError("repeated traversal did not emit root Advantage sample")
    sample = matches[0]
    return sample, int(result.nodes)


def _model_raw(runtime, sample) -> torch.Tensor:
    batch = collate_action_observations(
        REPRESENTATION,
        [sample.observation],
        [sample.legal],
        device="cpu",
    )
    runtime.bundle.advantage.eval()
    with torch.no_grad():
        return runtime.bundle.advantage(batch)[0].detach().cpu().float()


def _state_level_metrics(runtime, samples, nodes: list[int]) -> dict[str, float | int | bool]:
    if len(samples) < 2:
        raise ValueError("need at least two repeats")
    first = samples[0]
    if any(s.observation != first.observation or tuple(s.legal) != tuple(first.legal) for s in samples):
        raise RuntimeError("repeat state/legality drift")

    targets = torch.tensor([s.target for s in samples], dtype=torch.float32)
    legal = torch.tensor(first.legal, dtype=torch.bool)
    legal_count = int(legal.sum().item())
    if legal_count <= 0:
        raise RuntimeError("empty legal mask")
    pred = _model_raw(runtime, first)

    mean_target = targets.mean(dim=0)
    legal_targets = targets[:, legal]
    legal_mean = mean_target[legal]
    legal_pred = pred[legal]

    within_mse = float(((legal_targets - legal_mean) ** 2).mean().item())
    model_mse_samples = float(((legal_targets - legal_pred.unsqueeze(0)) ** 2).mean().item())
    model_mse_mean = float(((legal_pred - legal_mean) ** 2).mean().item())
    decomposition_error = abs(model_mse_samples - (within_mse + model_mse_mean))

    legal2 = legal.unsqueeze(0)
    target_policy = _lean_rm_policy_tensor(mean_target.unsqueeze(0), legal2)[0]
    model_policy = _lean_rm_policy_tensor(pred.unsqueeze(0), legal2)[0]
    repeat_policies = _lean_rm_policy_tensor(
        targets, legal.unsqueeze(0).expand(targets.shape[0], -1)
    )
    sampling_policy_tv = float(
        (0.5 * torch.abs(repeat_policies - target_policy.unsqueeze(0)).sum(dim=1))
        .mean()
        .item()
    )
    model_policy_tv = float((0.5 * torch.abs(target_policy - model_policy).sum()).item())

    target_max = mean_target.masked_fill(~legal, float("-inf")).max()
    target_min = mean_target.masked_fill(~legal, float("inf")).min()
    target_value = (target_policy * mean_target).sum()
    model_value = (model_policy * mean_target).sum()
    signed_gap = float((target_value - model_value).item() * CHIP_SCALE)
    model_regret = float((target_max - model_value).clamp_min(0.0).item() * CHIP_SCALE)

    target_positive = bool((torch.clamp(mean_target, min=0.0) * legal.float()).sum().item() > 0.0)
    pred_positive = bool((torch.clamp(pred, min=0.0) * legal.float()).sum().item() > 0.0)

    repeat_has_positive = (
        (torch.clamp(targets, min=0.0) * legal.float().unsqueeze(0)).sum(dim=1) > 0.0
    ).float()

    return {
        "legal_actions": legal_count,
        "within_target_mse": within_mse,
        "model_mse_to_sampled_targets": model_mse_samples,
        "model_mse_to_repeat_mean": model_mse_mean,
        "mse_decomposition_abs_error": float(decomposition_error),
        "noise_fraction_of_sample_mse": float(within_mse / max(model_mse_samples, 1e-12)),
        "within_target_rmse_chips": float(math.sqrt(max(within_mse, 0.0)) * CHIP_SCALE),
        "model_rmse_to_repeat_mean_chips": float(math.sqrt(max(model_mse_mean, 0.0)) * CHIP_SCALE),
        "model_rmse_to_sampled_targets_chips": float(math.sqrt(max(model_mse_samples, 0.0)) * CHIP_SCALE),
        "repeat_target_policy_tv_to_repeat_mean": sampling_policy_tv,
        "model_policy_tv_to_repeat_mean_target_policy": model_policy_tv,
        "repeat_mean_target_span_chips": float((target_max - target_min).item() * CHIP_SCALE),
        "signed_repeat_mean_target_policy_minus_model_policy_value_gap_chips": signed_gap,
        "model_policy_regret_to_repeat_mean_best_action_chips": model_regret,
        "repeat_mean_target_has_positive": target_positive,
        "model_prediction_has_positive": pred_positive,
        "branch_mismatch_on_repeat_mean": bool(target_positive != pred_positive),
        "repeat_fraction_target_has_positive": float(repeat_has_positive.mean().item()),
        "mean_nodes": float(statistics.fmean(nodes)),
        "max_nodes": int(max(nodes)),
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
    fields = (
        "within_target_mse",
        "model_mse_to_sampled_targets",
        "model_mse_to_repeat_mean",
        "noise_fraction_of_sample_mse",
        "within_target_rmse_chips",
        "model_rmse_to_repeat_mean_chips",
        "model_rmse_to_sampled_targets_chips",
        "repeat_target_policy_tv_to_repeat_mean",
        "model_policy_tv_to_repeat_mean_target_policy",
        "repeat_mean_target_span_chips",
        "signed_repeat_mean_target_policy_minus_model_policy_value_gap_chips",
        "model_policy_regret_to_repeat_mean_best_action_chips",
        "repeat_fraction_target_has_positive",
        "mean_nodes",
    )
    out = {name: _mean_ci([float(r[name]) for r in rows]) for name in fields}
    mean_sample = out["model_mse_to_sampled_targets"]["mean"]
    mean_noise = out["within_target_mse"]["mean"]
    mean_approx = out["model_mse_to_repeat_mean"]["mean"]
    out["decomposition_from_means"] = {
        "noise_fraction": float(mean_noise / max(mean_sample, 1e-12)),
        "approximation_fraction": float(mean_approx / max(mean_sample, 1e-12)),
        "closure_abs_error": float(abs(mean_sample - (mean_noise + mean_approx))),
    }
    out["branch_mismatch_rate_on_repeat_mean"] = float(
        statistics.fmean(1.0 if bool(r["branch_mismatch_on_repeat_mean"]) else 0.0 for r in rows)
    )
    out["repeat_mean_target_positive_rate"] = float(
        statistics.fmean(1.0 if bool(r["repeat_mean_target_has_positive"]) else 0.0 for r in rows)
    )
    out["model_prediction_positive_rate"] = float(
        statistics.fmean(1.0 if bool(r["model_prediction_has_positive"]) else 0.0 for r in rows)
    )
    return out


def _comparison(level0: dict[str, Any], level1: dict[str, Any]) -> dict[str, float]:
    n0 = float(level0["within_target_mse"]["mean"])
    n1 = float(level1["within_target_mse"]["mean"])
    c0 = float(level0["mean_nodes"]["mean"])
    c1 = float(level1["mean_nodes"]["mean"])
    a0 = float(level0["model_mse_to_repeat_mean"]["mean"])
    a1 = float(level1["model_mse_to_repeat_mean"]["mean"])
    return {
        "within_target_mse_level1_over_level0": float(n1 / max(n0, 1e-12)),
        "within_target_mse_reduction_fraction": float(1.0 - n1 / max(n0, 1e-12)),
        "node_cost_level1_over_level0": float(c1 / max(c0, 1e-12)),
        "model_mse_to_repeat_mean_level1_over_level0": float(a1 / max(a0, 1e-12)),
    }


def main() -> int:
    args = parse_args()
    if args.states_per_street <= 0 or args.repeats < 2 or args.threads <= 0 or args.max_episodes <= 0:
        raise SystemExit("invalid states/repeats/threads/max-episodes")

    checkpoint = args.checkpoint.resolve(strict=True)
    solver_path = args.solver.resolve(strict=True)
    torch.set_num_threads(int(args.threads))
    _selftest_lean_rm_policy_tensor()

    solver = SolverLibrary(solver_path)
    seed, config, completed, _sampler0, runtimes, _history, finalized = load_checkpoint(
        checkpoint, solver=solver
    )
    if not finalized or int(completed) != int(config.iterations):
        raise SystemExit("expected finalized Stage-B milestone checkpoint")

    sampler = LegacyScenarioSampler(
        seed=int(seed) ^ 0xA0F5D1A6,
        config=LegacyScenarioConfig(heads_up_prob=float(config.heads_up_prob)),
    )
    quotas = {domain: {street: 0 for street in STREET_NAMES} for domain in DOMAINS}
    rows: list[dict[str, Any]] = []
    episodes_seen = 0

    def complete() -> bool:
        return all(
            quotas[domain][street] >= int(args.states_per_street)
            for domain in DOMAINS
            for street in STREET_NAMES
        )

    while not complete() and episodes_seen < int(args.max_episodes):
        episode_index = episodes_seen
        episodes_seen += 1
        episode = sampler.sample_episode()
        domain = "TRUE_HEADS_UP" if episode.game_is_hu else "THREE_HANDED"
        if all(quotas[domain][street] >= int(args.states_per_street) for street in STREET_NAMES):
            continue

        deck_seed = _root_deck_seed(int(seed) ^ 0x51A7, domain, int(completed) + 1, episode_index)
        trajectory_rng = random.Random(
            (int(seed) ^ 0x7A1EC700 ^ (episode_index * 0x9E3779B1)) & ((1 << 63) - 1)
        )
        state = solver.create(episode, int(deck_seed))
        seen_streets: set[int] = set()
        decisions = 0
        try:
            while not state.terminal:
                street = _street(state)
                if street not in seen_streets and quotas[domain][street] < int(args.states_per_street):
                    state_id = f"{domain}:{STREET_NAMES[street]}:{quotas[domain][street]:03d}"
                    actor = int(state.actor)
                    for level in LEVELS:
                        samples = []
                        nodes = []
                        for rep in range(int(args.repeats)):
                            rng_seed = (
                                int(seed)
                                ^ 0xC0A5E000
                                ^ (episode_index * 0x45D9F3B)
                                ^ (street * 0x13579)
                                ^ (level * 0x2468B)
                                ^ (rep * 0x9E3779B1)
                            ) & ((1 << 63) - 1)
                            sample, node_count = _capture_root_target(
                                runtimes[domain],
                                state,
                                iteration=int(completed) + 1,
                                exact_level=int(level),
                                rng_seed=int(rng_seed),
                            )
                            samples.append(sample)
                            nodes.append(node_count)
                        metrics = _state_level_metrics(runtimes[domain], samples, nodes)
                        rows.append(
                            {
                                "state_id": state_id,
                                "episode_index": int(episode_index),
                                "domain": domain,
                                "street": STREET_NAMES[street],
                                "street_code": int(street),
                                "actor": int(actor),
                                "exact_opponent_levels": int(level),
                                "repeats": int(args.repeats),
                                **metrics,
                            }
                        )
                    quotas[domain][street] += 1
                    seen_streets.add(street)

                _sample_current_action(runtimes[domain], state, trajectory_rng)
                decisions += 1
                if decisions > 200:
                    raise RuntimeError("diagnostic trajectory exceeded 200 decisions")
        finally:
            state.close()

    if not complete():
        missing = {
            domain: {
                STREET_NAMES[s]: int(args.states_per_street) - quotas[domain][s]
                for s in STREET_NAMES
                if quotas[domain][s] < int(args.states_per_street)
            }
            for domain in DOMAINS
        }
        raise RuntimeError(f"failed to fill street quotas after {episodes_seen} episodes: {missing}")

    summaries: dict[str, Any] = {}
    for domain in DOMAINS:
        summaries[domain] = {}
        for street_name in STREET_NAMES.values():
            summaries[domain][street_name] = {}
            level_blocks = {}
            for level in LEVELS:
                subset = [
                    r
                    for r in rows
                    if r["domain"] == domain
                    and r["street"] == street_name
                    and int(r["exact_opponent_levels"]) == int(level)
                ]
                block = _aggregate(subset)
                summaries[domain][street_name][f"exact_level_{level}"] = block
                level_blocks[level] = block
            summaries[domain][street_name]["level1_vs_level0"] = _comparison(level_blocks[0], level_blocks[1])

        summaries[domain]["ALL_STREETS_EQUAL_STATE_WEIGHT"] = {}
        level_blocks = {}
        for level in LEVELS:
            subset = [r for r in rows if r["domain"] == domain and int(r["exact_opponent_levels"]) == int(level)]
            block = _aggregate(subset)
            summaries[domain]["ALL_STREETS_EQUAL_STATE_WEIGHT"][f"exact_level_{level}"] = block
            level_blocks[level] = block
        summaries[domain]["ALL_STREETS_EQUAL_STATE_WEIGHT"]["level1_vs_level0"] = _comparison(level_blocks[0], level_blocks[1])

    report = {
        "schema": "SPINCORE_LT2_REPEATED_TARGET_VARIANCE_V1",
        "checkpoint": str(checkpoint),
        "completed_iteration": int(completed),
        "seed": int(seed),
        "method": {
            "checkpoint_read_only": True,
            "optimizer_steps": 0,
            "training_memory_writes": 0,
            "diagnostic_trajectory_policy": "stored Stage-B Advantage behavior policy",
            "state_selection": "first decision reached on each street of deterministic sampled current-policy trajectories",
            "states_per_domain_per_street": int(args.states_per_street),
            "repeats_per_state_per_exact_level": int(args.repeats),
            "exact_opponent_levels_compared": list(LEVELS),
            "same_state_same_deal_within_repeats": True,
            "variance_isolation": "within-state repeat variance isolates opponent-action external-sampling noise for fixed hidden deal/future board; it does not include across-deal chance/hidden-card variance",
            "mse_decomposition": "sample-target MSE = within-repeat target variance + model MSE to repeat-mean target",
            "repeat_mean_warning": "repeat mean is a lower-noise diagnostic reference, not a GTO oracle",
            "state_weighting_warning": "ALL_STREETS_EQUAL_STATE_WEIGHT intentionally gives each audited street state equal weight; it is not an estimate of natural visitation EV",
            "no_arbitrary_pass_threshold": True,
        },
        "episodes_sampled": int(episodes_seen),
        "quotas": {domain: {STREET_NAMES[s]: int(quotas[domain][s]) for s in STREET_NAMES} for domain in DOMAINS},
        "summaries": summaries,
        "rows": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for domain in DOMAINS:
        print(f"=== {domain} ===")
        for street_name in STREET_NAMES.values():
            c = summaries[domain][street_name]["level1_vs_level0"]
            l0 = summaries[domain][street_name]["exact_level_0"]["decomposition_from_means"]
            print(
                f"{street_name}: level0_noise_fraction={l0['noise_fraction']:.3f} "
                f"level1_noise_reduction={c['within_target_mse_reduction_fraction']:+.3f} "
                f"node_multiplier={c['node_cost_level1_over_level0']:.2f}"
            )
    print("LT2_REPEATED_TARGET_VARIANCE_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
