#!/usr/bin/env python3
from __future__ import annotations

"""Paired offline chip-EV diagnostic for the first functional SpinCore policy.

This is deliberately not an exploitability proof.  It asks a narrower, useful
question before more training is authorized: does the learned AveragePolicy add
measurable chip EV over an untrained uniform-legal control against several fixed,
transparent opponent families under the *full empirical SpinGo sampler*?

For every sampled tournament state/deal we rotate the evaluated hero through all
live seats.  SpinCore and the uniform-hero control see the same scenario and card
deal against the same opponent family, so their difference is a paired result.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler  # noqa: E402
from spincore.solver import Episode  # noqa: E402


BASELINES = ("UNIFORM_LEGAL", "PASSIVE_CALLER", "JAMMER")
POLICIES = ("SPINCORE", "UNIFORM_CONTROL")

_SOLVER = None
_AGENT = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired full-sampler chip-EV strategy diagnostic")
    p.add_argument("--solver", type=Path, default=ROOT / "build" / "libspincore_solver_c.so")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scenarios", type=int, default=1000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--report", type=Path)
    return p.parse_args()


def _mix64(*values: int) -> int:
    x = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    for value in values:
        y = int(value) & mask
        x ^= (y + 0x9E3779B97F4A7C15 + ((x << 6) & mask) + (x >> 2)) & mask
        x &= mask
    return x


def _init_worker(solver_path: str, checkpoint_path: str) -> None:
    global _SOLVER, _AGENT
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["SPINCORE_TORCH_THREADS"] = "1"

    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    from spincore.lean_functional_agent import LeanFunctionalAgent
    from spincore.solver import SolverLibrary

    _SOLVER = SolverLibrary(solver_path)
    _AGENT = LeanFunctionalAgent.from_checkpoint(checkpoint_path, seed=0)


def _legal_context(state):
    from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
    from spincore.lean_functional_agent import _street_from_state
    from spincore.lean_solver_actions import lean_legal_actions

    street = _street_from_state(state)
    active_mask = FIRST_RELEASE_ACTION_SPEC.active_mask(street)
    legal = lean_legal_actions(state, active_mask)
    if not legal:
        raise RuntimeError("nonterminal state has no lean legal action")
    return active_mask, legal


def _sample_probs(legal: tuple[int, ...], probs: tuple[float, ...], rng: random.Random) -> int:
    x = rng.random()
    cumulative = 0.0
    for slot in legal:
        cumulative += float(probs[slot])
        if x < cumulative:
            return int(slot)
    return int(legal[-1])


def _choose_action(policy: str, state, rng: random.Random) -> tuple[int, int]:
    if policy == "SPINCORE":
        active_mask, legal, probs = _AGENT.distribution(state)
        return int(active_mask), _sample_probs(legal, probs, rng)

    active_mask, legal = _legal_context(state)
    if policy in ("UNIFORM_CONTROL", "UNIFORM_LEGAL"):
        return int(active_mask), int(legal[rng.randrange(len(legal))])
    if policy == "PASSIVE_CALLER":
        if 1 in legal:
            return int(active_mask), 1
        if 0 in legal:
            return int(active_mask), 0
        return int(active_mask), int(min(legal))
    if policy == "JAMMER":
        if 9 in legal:
            return int(active_mask), 9
        if 1 in legal:
            return int(active_mask), 1
        if 0 in legal:
            return int(active_mask), 0
        return int(active_mask), int(max(legal))
    raise ValueError(f"unknown policy {policy}")


def _play_one(
    episode: Episode,
    *,
    deal_seed: int,
    hero_seat: int,
    hero_policy: str,
    opponent_policy: str,
    scenario_index: int,
    master_seed: int,
) -> int:
    from spincore.lean_solver_actions import apply_lean

    state = _SOLVER.create(episode, int(deal_seed))
    seat_rng = {
        seat: random.Random(
            _mix64(master_seed, scenario_index, seat, 100 + BASELINES.index(opponent_policy))
        )
        for seat in range(3)
    }
    # Hero random stream is paired across SPINCORE and UNIFORM_CONTROL.
    seat_rng[hero_seat] = random.Random(_mix64(master_seed, scenario_index, hero_seat, 777))
    decisions = 0
    try:
        while not state.terminal:
            actor = int(state.actor)
            policy = hero_policy if actor == hero_seat else opponent_policy
            active_mask, slot = _choose_action(policy, state, seat_rng[actor])
            apply_lean(state, active_mask, slot)
            decisions += 1
            if decisions > 200:
                raise RuntimeError("evaluation hand exceeded 200 decisions")
        delta = state.terminal_chip_delta()
        if sum(delta) != 0:
            raise RuntimeError(f"terminal chip delta not zero-sum: {delta}")
        return int(delta[hero_seat])
    finally:
        state.close()


def _worker(task: tuple[int, Episode, int, int]) -> list[dict[str, Any]]:
    scenario_index, episode, deal_seed, master_seed = task
    live = [seat for seat, stack in enumerate(episode.stacks) if int(stack) > 0]
    domain = "TRUE_HEADS_UP" if episode.game_is_hu else "THREE_HANDED"
    blind = f"{episode.small_blind}/{episode.big_blind}"
    rows: list[dict[str, Any]] = []
    for baseline in BASELINES:
        for hero_seat in live:
            values: dict[str, int] = {}
            for hero_policy in POLICIES:
                values[hero_policy] = _play_one(
                    episode,
                    deal_seed=deal_seed,
                    hero_seat=hero_seat,
                    hero_policy=hero_policy,
                    opponent_policy=baseline,
                    scenario_index=scenario_index,
                    master_seed=master_seed,
                )
            rows.append(
                {
                    "scenario": int(scenario_index),
                    "domain": domain,
                    "blind": blind,
                    "hero_seat": int(hero_seat),
                    "baseline": baseline,
                    "spincore": int(values["SPINCORE"]),
                    "uniform_control": int(values["UNIFORM_CONTROL"]),
                    "paired_gain": int(values["SPINCORE"] - values["UNIFORM_CONTROL"]),
                }
            )
    return rows


def _mean_ci(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sem": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(statistics.fmean(values))
    if n == 1:
        sem = 0.0
    else:
        sem = float(statistics.stdev(values) / math.sqrt(n))
    half = 1.96 * sem
    return {"n": n, "mean": mean, "sem": sem, "ci95_low": mean - half, "ci95_high": mean + half}


def _cluster_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_scenario: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(int(row["scenario"]), []).append(row)
    current = []
    control = []
    gain = []
    for scenario_rows in by_scenario.values():
        current.append(statistics.fmean(float(r["spincore"]) for r in scenario_rows))
        control.append(statistics.fmean(float(r["uniform_control"]) for r in scenario_rows))
        gain.append(statistics.fmean(float(r["paired_gain"]) for r in scenario_rows))
    out = {
        "scenario_clusters": len(by_scenario),
        "seat_runs": len(rows),
        "spincore_chip_ev": _mean_ci(current),
        "uniform_control_chip_ev": _mean_ci(control),
        "paired_gain_chip_ev": _mean_ci(gain),
    }
    for block in ("spincore_chip_ev", "uniform_control_chip_ev", "paired_gain_chip_ev"):
        stat = out[block]
        stat["mean_per_1500"] = float(stat["mean"]) / 1500.0
        stat["ci95_low_per_1500"] = float(stat["ci95_low"]) / 1500.0
        stat["ci95_high_per_1500"] = float(stat["ci95_high"]) / 1500.0
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for baseline in BASELINES:
        b_rows = [r for r in rows if r["baseline"] == baseline]
        summary[baseline] = {"ALL": _cluster_stats(b_rows)}
        for domain in ("THREE_HANDED", "TRUE_HEADS_UP"):
            d_rows = [r for r in b_rows if r["domain"] == domain]
            summary[baseline][domain] = _cluster_stats(d_rows)

    blind_detail: dict[str, Any] = {}
    for baseline in BASELINES:
        blind_detail[baseline] = {}
        combos = sorted({(str(r["domain"]), str(r["blind"])) for r in rows if r["baseline"] == baseline})
        for domain, blind in combos:
            subset = [
                r for r in rows
                if r["baseline"] == baseline and r["domain"] == domain and r["blind"] == blind
            ]
            blind_detail[baseline][f"{domain}:{blind}"] = _cluster_stats(subset)
    return {"summary": summary, "by_blind": blind_detail}


def _print_summary(report: dict[str, Any]) -> None:
    print("=== SpinCore paired chip-EV strategy diagnostic ===")
    print(f"checkpoint={report['checkpoint']}")
    print(f"scenarios={report['scenarios']} workers={report['workers']} seed={report['seed']}")
    print("metric=mean hero chip delta per sampled hand; 95% CI clustered by scenario")
    print("control=uniform legal hero on the exact same scenario/deal/opponent family")
    for baseline in BASELINES:
        print(f"--- opponent={baseline} ---")
        for domain in ("ALL", "THREE_HANDED", "TRUE_HEADS_UP"):
            block = report["summary"][baseline][domain]
            cur = block["spincore_chip_ev"]
            gain = block["paired_gain_chip_ev"]
            print(
                f"{domain} clusters={block['scenario_clusters']} "
                f"spincore={cur['mean']:+.3f} chips/hand "
                f"CI95=[{cur['ci95_low']:+.3f},{cur['ci95_high']:+.3f}] "
                f"paired_vs_uniform={gain['mean']:+.3f} "
                f"CI95=[{gain['ci95_low']:+.3f},{gain['ci95_high']:+.3f}]"
            )
    print("STRATEGY_QUALITY_EVAL_COMPLETE")


def main() -> int:
    args = parse_args()
    if args.scenarios <= 0:
        raise SystemExit("--scenarios must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if not args.solver.is_file():
        raise SystemExit(f"solver not found: {args.solver}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    sampler = LegacyScenarioSampler(seed=args.seed ^ 0x5CE0A710, config=LegacyScenarioConfig())
    tasks: list[tuple[int, Episode, int, int]] = []
    domain_counts = {"THREE_HANDED": 0, "TRUE_HEADS_UP": 0}
    blind_counts: dict[str, int] = {}
    for index in range(args.scenarios):
        episode = sampler.sample_episode()
        domain = "TRUE_HEADS_UP" if episode.game_is_hu else "THREE_HANDED"
        domain_counts[domain] += 1
        blind = f"{episode.small_blind}/{episode.big_blind}"
        blind_counts[f"{domain}:{blind}"] = blind_counts.get(f"{domain}:{blind}", 0) + 1
        deal_seed = _mix64(args.seed, index, 0xD34A1)
        tasks.append((index, episode, deal_seed, int(args.seed)))

    ctx = mp.get_context("spawn")
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=min(args.workers, args.scenarios),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(str(args.solver.resolve()), str(args.checkpoint.resolve())),
    ) as pool:
        for chunk in pool.map(_worker, tasks, chunksize=1):
            rows.extend(chunk)

    blocks = _summarize(rows)
    report = {
        "schema": "SPINCORE_LEAN_STRATEGY_QUALITY_EVAL_V1",
        "checkpoint": str(args.checkpoint.resolve()),
        "scenarios": int(args.scenarios),
        "workers": int(min(args.workers, args.scenarios)),
        "seed": int(args.seed),
        "opponent_families": list(BASELINES),
        "hero_control": "UNIFORM_CONTROL",
        "domain_counts": domain_counts,
        "blind_counts": dict(sorted(blind_counts.items())),
        "method": {
            "scenario_sampler": "legacy empirical full 3H/HU blind-conditioned sampler",
            "deal_pairing": "same scenario and solver deal seed for SpinCore and uniform hero control",
            "seat_rotation": "evaluated hero rotated through every live seat",
            "primary_metric": "raw chip delta per hand; paired SpinCore-minus-uniform-control gain",
            "ci": "normal 95% CI over scenario-cluster means",
            "warning": "diagnostic versus fixed weak baselines; not an exploitability or GTO proof",
        },
        **blocks,
    }

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
