#!/usr/bin/env python3
from __future__ import annotations

"""Read-only Stage-A/Stage-B checkpoint cross-play diagnostic.

This is stronger than the fixed weak-baseline sentinel but is still not an
exploitability/GTO proof. It compares the two finalized AveragePolicies in a
shared contemporary policy ecosystem under the empirical SpinGo sampler.

Primary metric:
  Stage B hero minus Stage A hero chip EV against the exact same deterministic
  50/50 Stage-A/Stage-B opponent mixture, paired on scenario, deal, hero seat
  and all seat RNG streams.

Additional diagnostics:
  * HU direct seat-balanced Stage B vs Stage A.
  * 3H invasion: Stage B singleton vs A/A and Stage A singleton vs B/B.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
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

_SOLVER = None
_A = None
_B = None

A = "LT2A_1P8M"
B = "LT2B_4P5M"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build" / "libspincore_solver_c.so")
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--scenarios", type=int, default=3000)
    p.add_argument("--workers", type=int, default=31)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--rows-report", type=Path, required=True)
    return p.parse_args()


def _mix64(*values: int) -> int:
    x = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    for value in values:
        y = int(value) & mask
        x ^= (y + 0x9E3779B97F4A7C15 + ((x << 6) & mask) + (x >> 2)) & mask
        x &= mask
    return x


def _init_worker(solver_path: str, before_path: str, after_path: str) -> None:
    global _SOLVER, _A, _B
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
    _A = LeanFunctionalAgent.from_checkpoint(before_path, seed=0)
    _B = LeanFunctionalAgent.from_checkpoint(after_path, seed=0)


def _sample_action(agent, state, rng: random.Random) -> tuple[int, int]:
    active_mask, legal, probs = agent.distribution(state)
    x = rng.random()
    cumulative = 0.0
    for slot in legal:
        cumulative += float(probs[slot])
        if x < cumulative:
            return int(active_mask), int(slot)
    return int(active_mask), int(legal[-1])


def _play(
    episode: Episode,
    *,
    deal_seed: int,
    labels: dict[int, str],
    scenario_index: int,
    master_seed: int,
    stream_tag: int,
) -> tuple[int, int, int]:
    from spincore.lean_solver_actions import apply_lean

    state = _SOLVER.create(episode, int(deal_seed))
    rngs = {
        seat: random.Random(_mix64(master_seed, scenario_index, stream_tag, seat))
        for seat in labels
    }
    decisions = 0
    try:
        while not state.terminal:
            actor = int(state.actor)
            label = labels.get(actor)
            if label is None:
                raise RuntimeError(f"active actor {actor} missing policy label")
            agent = _A if label == A else _B
            active_mask, slot = _sample_action(agent, state, rngs[actor])
            apply_lean(state, active_mask, slot)
            decisions += 1
            if decisions > 200:
                raise RuntimeError("cross-play hand exceeded 200 decisions")
        delta = tuple(int(x) for x in state.terminal_chip_delta())
        if sum(delta) != 0:
            raise RuntimeError(f"terminal chip delta not zero-sum: {delta}")
        return delta
    finally:
        state.close()


def _opponent_mix_label(master_seed: int, scenario_index: int, hero_seat: int, seat: int) -> str:
    bit = _mix64(master_seed, scenario_index, hero_seat, seat, 0xC055) & 1
    return B if bit else A


def _worker(task: tuple[int, Episode, int, int]) -> list[dict[str, Any]]:
    index, episode, deal_seed, master_seed = task
    live = [seat for seat, stack in enumerate(episode.stacks) if int(stack) > 0]
    domain = "TRUE_HEADS_UP" if episode.game_is_hu else "THREE_HANDED"
    blind = f"{episode.small_blind}/{episode.big_blind}"
    rows: list[dict[str, Any]] = []

    # Primary paired test: change only the hero checkpoint. Every opponent seat
    # keeps the same deterministic A/B mixture assignment and every seat keeps
    # the same RNG stream across the paired games.
    for hero in live:
        opponents = {
            seat: _opponent_mix_label(master_seed, index, hero, seat)
            for seat in live
            if seat != hero
        }
        values: dict[str, int] = {}
        stream_tag = 1000 + int(hero)
        for hero_label in (A, B):
            labels = dict(opponents)
            labels[hero] = hero_label
            delta = _play(
                episode,
                deal_seed=deal_seed,
                labels=labels,
                scenario_index=index,
                master_seed=master_seed,
                stream_tag=stream_tag,
            )
            values[hero_label] = int(delta[hero])
        rows.append(
            {
                "mode": "MIXTURE_HERO",
                "scenario": int(index),
                "domain": domain,
                "blind": blind,
                "hero_seat": int(hero),
                "before": int(values[A]),
                "after": int(values[B]),
                "delta": int(values[B] - values[A]),
            }
        )

    if domain == "TRUE_HEADS_UP":
        # Seat-balanced direct B-vs-A. B occupies each live seat once.
        for b_seat in live:
            labels = {seat: (B if seat == b_seat else A) for seat in live}
            delta = _play(
                episode,
                deal_seed=deal_seed,
                labels=labels,
                scenario_index=index,
                master_seed=master_seed,
                stream_tag=2000 + int(b_seat),
            )
            rows.append(
                {
                    "mode": "HU_DIRECT",
                    "scenario": int(index),
                    "domain": domain,
                    "blind": blind,
                    "b_seat": int(b_seat),
                    "b_chip_delta": int(delta[b_seat]),
                }
            )
    else:
        # 3H invasion diagnostic. Same hero seat and RNG streams are used for
        # B-vs-A/A and A-vs-B/B, enabling a scenario-clustered difference.
        for hero in live:
            labels_b = {seat: (B if seat == hero else A) for seat in live}
            labels_a = {seat: (A if seat == hero else B) for seat in live}
            stream_tag = 3000 + int(hero)
            delta_b = _play(
                episode,
                deal_seed=deal_seed,
                labels=labels_b,
                scenario_index=index,
                master_seed=master_seed,
                stream_tag=stream_tag,
            )
            delta_a = _play(
                episode,
                deal_seed=deal_seed,
                labels=labels_a,
                scenario_index=index,
                master_seed=master_seed,
                stream_tag=stream_tag,
            )
            b_value = int(delta_b[hero])
            a_value = int(delta_a[hero])
            rows.append(
                {
                    "mode": "THREE_HANDED_INVASION",
                    "scenario": int(index),
                    "domain": domain,
                    "blind": blind,
                    "hero_seat": int(hero),
                    "b_vs_aa": b_value,
                    "a_vs_bb": a_value,
                    "invasion_difference": int(b_value - a_value),
                }
            )
    return rows


def _mean_ci(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "sem": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    mean = float(statistics.fmean(values))
    sem = 0.0 if n == 1 else float(statistics.stdev(values) / math.sqrt(n))
    half = 1.96 * sem
    return {
        "n": n,
        "mean": mean,
        "sem": sem,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _cluster_stat(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    by_scenario: dict[int, list[float]] = {}
    for row in rows:
        by_scenario.setdefault(int(row["scenario"]), []).append(float(row[key]))
    cluster_values = [statistics.fmean(v) for v in by_scenario.values()]
    out = _mean_ci(cluster_values)
    out["scenario_clusters"] = len(by_scenario)
    out["seat_runs"] = len(rows)
    return out


def main() -> int:
    args = parse_args()
    if args.scenarios <= 0 or args.workers <= 0:
        raise SystemExit("scenarios/workers must be positive")
    for path in (args.solver, args.before, args.after):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    sampler = LegacyScenarioSampler(seed=args.seed ^ 0x5CE0A710, config=LegacyScenarioConfig())
    tasks: list[tuple[int, Episode, int, int]] = []
    domain_counts = {"THREE_HANDED": 0, "TRUE_HEADS_UP": 0}
    for index in range(args.scenarios):
        episode = sampler.sample_episode()
        domain = "TRUE_HEADS_UP" if episode.game_is_hu else "THREE_HANDED"
        domain_counts[domain] += 1
        tasks.append((index, episode, _mix64(args.seed, index, 0xD34A1), int(args.seed)))

    ctx = mp.get_context("spawn")
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=min(args.workers, args.scenarios),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(str(args.solver.resolve()), str(args.before.resolve()), str(args.after.resolve())),
    ) as pool:
        for chunk in pool.map(_worker, tasks, chunksize=1):
            rows.extend(chunk)

    mixture_rows = [r for r in rows if r["mode"] == "MIXTURE_HERO"]
    mixture: dict[str, Any] = {}
    for domain in ("ALL", "THREE_HANDED", "TRUE_HEADS_UP"):
        subset = mixture_rows if domain == "ALL" else [r for r in mixture_rows if r["domain"] == domain]
        mixture[domain] = {
            "before_chip_ev": _cluster_stat(subset, "before"),
            "after_chip_ev": _cluster_stat(subset, "after"),
            "paired_delta_after_minus_before": _cluster_stat(subset, "delta"),
        }

    hu_rows = [r for r in rows if r["mode"] == "HU_DIRECT"]
    invasion_rows = [r for r in rows if r["mode"] == "THREE_HANDED_INVASION"]

    report = {
        "schema": "SPINCORE_LT2_CHECKPOINT_CROSSPLAY_V1",
        "before_label": A,
        "after_label": B,
        "before": str(args.before.resolve()),
        "after": str(args.after.resolve()),
        "seed": int(args.seed),
        "scenarios": int(args.scenarios),
        "workers": int(min(args.workers, args.scenarios)),
        "domain_counts": domain_counts,
        "method": {
            "sampler": "legacy empirical full 3H/HU blind-conditioned sampler",
            "pairing": "same scenario, deal, hero seat and per-seat RNG streams",
            "primary": "Stage B hero minus Stage A hero against identical deterministic 50/50 Stage-A/Stage-B opponent mixture",
            "hu_direct": "Stage B vs Stage A with Stage B rotated through every live seat",
            "three_handed_invasion": "Stage B singleton vs A/A and Stage A singleton vs B/B with hero-seat rotation",
            "ci": "normal 95% CI over scenario-cluster means",
            "warning": "relative checkpoint diagnostic only; not exploitability/GTO proof",
        },
        "mixture_hero": mixture,
        "hu_direct_b_vs_a": _cluster_stat(hu_rows, "b_chip_delta"),
        "three_handed_invasion": {
            "b_vs_aa": _cluster_stat(invasion_rows, "b_vs_aa"),
            "a_vs_bb": _cluster_stat(invasion_rows, "a_vs_bb"),
            "difference_b_vs_aa_minus_a_vs_bb": _cluster_stat(invasion_rows, "invasion_difference"),
        },
    }

    rows_payload = {
        "schema": "SPINCORE_LT2_CHECKPOINT_CROSSPLAY_ROWS_V1",
        "seed": int(args.seed),
        "scenarios": int(args.scenarios),
        "rows": rows,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.rows_report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.rows_report.write_text(json.dumps(rows_payload, separators=(",", ":")) + "\n", encoding="utf-8")

    print("=== SpinCore LT2 checkpoint cross-play ===")
    print(f"scenarios={args.scenarios} seed={args.seed} workers={min(args.workers, args.scenarios)}")
    for domain in ("ALL", "THREE_HANDED", "TRUE_HEADS_UP"):
        x = mixture[domain]["paired_delta_after_minus_before"]
        print(
            f"MIXTURE_HERO {domain}: delta_B_minus_A={x['mean']:+.3f} "
            f"CI95=[{x['ci95_low']:+.3f},{x['ci95_high']:+.3f}] "
            f"clusters={x['scenario_clusters']}"
        )
    x = report["hu_direct_b_vs_a"]
    print(
        f"HU_DIRECT B_vs_A: B_chip_ev={x['mean']:+.3f} "
        f"CI95=[{x['ci95_low']:+.3f},{x['ci95_high']:+.3f}]"
    )
    x = report["three_handed_invasion"]["difference_b_vs_aa_minus_a_vs_bb"]
    print(
        f"3H_INVASION difference={x['mean']:+.3f} "
        f"CI95=[{x['ci95_low']:+.3f},{x['ci95_high']:+.3f}]"
    )
    print(f"CROSSPLAY_REPORT={args.report.resolve()}")
    print(f"CROSSPLAY_ROWS={args.rows_report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
