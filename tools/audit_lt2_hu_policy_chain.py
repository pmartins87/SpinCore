#!/usr/bin/env python3
from __future__ import annotations

"""Global HU policy-chain evaluation for preserved LT2 Stage A/B.

Read-only diagnostic on the already-seen forensic seed family.

For identical HU scenarios/deals/RNG streams it evaluates four hero policies:
  AVG_A   Stage-A deployed AveragePolicy
  AVG_B   Stage-B deployed AveragePolicy
  BEH_A   Stage-A current Advantage-induced regret-matching behavior
  BEH_B   Stage-B current Advantage-induced regret-matching behavior

against UNIFORM_LEGAL, PASSIVE_CALLER and JAMMER.

Primary question:
  Did Stage-B current behavior regress too, or did the regression emerge mainly
  in the historical AveragePolicy aggregation/deployment path?

No training roots, optimizer steps, or memory writes.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import gc
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
sys.path.insert(0, str(ROOT / "tools"))

import torch

import audit_lt2_hu_preflop_conditional_resampling as cond
import audit_lt2_jammer_facing_allin_target_overlay as overlay
import audit_lt2_stage_a_b_first_divergence as fd
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import apply_lean, lean_legal_actions
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.solver import Episode, SolverLibrary

BASELINES = ("UNIFORM_LEGAL", "PASSIVE_CALLER", "JAMMER")
POLICIES = ("AVG_A", "AVG_B", "BEH_A", "BEH_B")
FORENSIC_SEEDS = (20260920, 20260921, 20260922, 20260923, 20260924, 20260925)

_SOLVER = None
_STAGE_A = None
_STAGE_B = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--stage-b", type=Path, required=True)
    p.add_argument("--scenarios-per-seed", type=int, default=5000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _mean_ci(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if n <= 0:
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
        "n": int(n),
        "mean": mean,
        "sem": sem,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _init_worker(solver_path: str, stage_a_snapshot: str, stage_b_snapshot: str) -> None:
    global _SOLVER, _STAGE_A, _STAGE_B
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["SPINCORE_TORCH_THREADS"] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    _SOLVER = SolverLibrary(solver_path)
    _STAGE_A = overlay.StageModels(Path(stage_a_snapshot), _SOLVER)
    _STAGE_B = overlay.StageModels(Path(stage_b_snapshot), _SOLVER)


def _legal_context(state):
    street = int(cond._street(state))
    active_mask = FIRST_RELEASE_ACTION_SPEC.active_mask(street)
    legal = tuple(int(x) for x in lean_legal_actions(state, active_mask))
    if not legal:
        raise RuntimeError("nonterminal state has no legal actions")
    return int(active_mask), legal


def _sample_probs(legal: tuple[int, ...], probs: tuple[float, ...], rng: random.Random) -> int:
    x = rng.random()
    cumulative = 0.0
    for slot in legal:
        cumulative += float(probs[slot])
        if x < cumulative:
            return int(slot)
    return int(legal[-1])


def _baseline_action(baseline: str, state, rng: random.Random) -> tuple[int, int]:
    active_mask, legal = _legal_context(state)
    if baseline == "UNIFORM_LEGAL":
        return active_mask, int(legal[rng.randrange(len(legal))])
    if baseline == "PASSIVE_CALLER":
        if 1 in legal:
            return active_mask, 1
        if 0 in legal:
            return active_mask, 0
        return active_mask, int(min(legal))
    if baseline == "JAMMER":
        if 9 in legal:
            return active_mask, 9
        if 1 in legal:
            return active_mask, 1
        if 0 in legal:
            return active_mask, 0
        return active_mask, int(max(legal))
    raise ValueError(f"unknown baseline {baseline}")


def _hero_distribution(policy: str, state):
    if policy.startswith("AVG_"):
        stage = _STAGE_A if policy.endswith("_A") else _STAGE_B
        return stage.average_policy(state)

    stage = _STAGE_A if policy.endswith("_A") else _STAGE_B
    active_mask, legal = _legal_context(state)
    observation = state.neural_bytes()
    probs = stage.runtime.session.behavior(state, observation, legal)
    out = tuple(float(x) for x in probs)
    mass = sum(out[action] for action in legal)
    if not (0.999 <= mass <= 1.001):
        raise RuntimeError(f"behavior probability mass drift: {mass}")
    return int(active_mask), legal, out


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
    state = _SOLVER.create(episode, int(deal_seed))
    seat_rng = {
        seat: random.Random(
            fd._mix64(
                master_seed,
                scenario_index,
                seat,
                100 + BASELINES.index(opponent_policy),
            )
        )
        for seat in range(3)
    }
    # Common random numbers across all four hero policies.
    seat_rng[int(hero_seat)] = random.Random(
        fd._mix64(master_seed, scenario_index, hero_seat, 777)
    )

    decisions = 0
    try:
        while not state.terminal:
            actor = int(state.actor)
            if actor == int(hero_seat):
                active_mask, legal, probs = _hero_distribution(hero_policy, state)
                slot = _sample_probs(legal, probs, seat_rng[actor])
            else:
                active_mask, slot = _baseline_action(
                    opponent_policy, state, seat_rng[actor]
                )
            apply_lean(state, active_mask, slot)
            decisions += 1
            if decisions > 200:
                raise RuntimeError("policy-chain hand exceeded 200 decisions")
        delta = state.terminal_chip_delta()
        if sum(delta) != 0:
            raise RuntimeError(f"non-zero-sum terminal delta: {delta}")
        return int(delta[int(hero_seat)])
    finally:
        state.close()


def _worker(task: tuple[int, int, Episode, int]) -> list[dict[str, Any]]:
    seed, scenario_index, episode, deal_seed = task
    if not episode.game_is_hu:
        return []

    live = [seat for seat, stack in enumerate(episode.stacks) if int(stack) > 0]
    if len(live) != 2:
        raise RuntimeError("HU episode without exactly two live seats")
    blind = f"{episode.small_blind}/{episode.big_blind}"

    rows: list[dict[str, Any]] = []
    for baseline in BASELINES:
        for hero_seat in live:
            values = {}
            for policy in POLICIES:
                values[policy] = _play_one(
                    episode,
                    deal_seed=int(deal_seed),
                    hero_seat=int(hero_seat),
                    hero_policy=policy,
                    opponent_policy=baseline,
                    scenario_index=int(scenario_index),
                    master_seed=int(seed),
                )
            rows.append({
                "seed": int(seed),
                "scenario": int(scenario_index),
                "blind": blind,
                "hero_seat": int(hero_seat),
                "baseline": baseline,
                "avg_a": int(values["AVG_A"]),
                "avg_b": int(values["AVG_B"]),
                "beh_a": int(values["BEH_A"]),
                "beh_b": int(values["BEH_B"]),
                "avg_b_minus_a": int(values["AVG_B"] - values["AVG_A"]),
                "beh_b_minus_a": int(values["BEH_B"] - values["BEH_A"]),
                "avg_minus_beh_a": int(values["AVG_A"] - values["BEH_A"]),
                "avg_minus_beh_b": int(values["AVG_B"] - values["BEH_B"]),
                "chain_delta_b_minus_a": int(
                    (values["AVG_B"] - values["BEH_B"])
                    - (values["AVG_A"] - values["BEH_A"])
                ),
            })
    return rows


def _cluster_values(
    rows: list[dict[str, Any]],
    *,
    baseline: str,
    key: str,
) -> list[float]:
    by_cluster: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        if row["baseline"] != baseline:
            continue
        cluster = (int(row["seed"]), int(row["scenario"]))
        by_cluster.setdefault(cluster, []).append(float(row[key]))
    return [float(statistics.fmean(v)) for v in by_cluster.values()]


def _per_seed(rows: list[dict[str, Any]], baseline: str, key: str) -> dict[str, float]:
    out = {}
    for seed in FORENSIC_SEEDS:
        vals = [
            float(x)
            for x in _cluster_values(
                [r for r in rows if int(r["seed"]) == int(seed)],
                baseline=baseline,
                key=key,
            )
        ]
        out[str(seed)] = float(statistics.fmean(vals)) if vals else float("nan")
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = (
        "avg_a",
        "avg_b",
        "beh_a",
        "beh_b",
        "avg_b_minus_a",
        "beh_b_minus_a",
        "avg_minus_beh_a",
        "avg_minus_beh_b",
        "chain_delta_b_minus_a",
    )
    for baseline in BASELINES:
        block = {
            key: _mean_ci(_cluster_values(rows, baseline=baseline, key=key))
            for key in keys
        }
        block["clusters"] = int(
            block["avg_a"]["n"]
        )
        block["seat_runs"] = int(
            sum(1 for r in rows if r["baseline"] == baseline)
        )
        block["per_seed"] = {
            key: _per_seed(rows, baseline, key)
            for key in (
                "avg_b_minus_a",
                "beh_b_minus_a",
                "chain_delta_b_minus_a",
            )
        }
        out[baseline] = block
    return out


def main() -> int:
    args = parse_args()
    if args.scenarios_per_seed <= 0 or args.workers <= 0:
        raise SystemExit("positive scenarios/workers required")
    solver_path = args.solver.resolve(strict=True)
    stage_a_path = args.stage_a.resolve(strict=True)
    stage_b_path = args.stage_b.resolve(strict=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    snapshot_a = args.report.parent / "stage_a_hu_models.pt"
    snapshot_b = args.report.parent / "stage_b_hu_models.pt"

    print("Extracting lightweight HU AveragePolicy + Advantage snapshots...", flush=True)
    meta_a = overlay._extract_stage_snapshot(stage_a_path, snapshot_a)
    meta_b = overlay._extract_stage_snapshot(stage_b_path, snapshot_b)
    print(
        f"snapshots A={meta_a['snapshot_bytes']/1048576.0:.2f} MiB "
        f"B={meta_b['snapshot_bytes']/1048576.0:.2f} MiB",
        flush=True,
    )

    tasks: list[tuple[int, int, Episode, int]] = []
    hu_counts: dict[str, int] = {}
    for seed in FORENSIC_SEEDS:
        sampler = LegacyScenarioSampler(
            seed=int(seed) ^ 0x5CE0A710,
            config=LegacyScenarioConfig(),
        )
        hu = 0
        for scenario_index in range(int(args.scenarios_per_seed)):
            episode = sampler.sample_episode()
            if episode.game_is_hu:
                hu += 1
            deal_seed = fd._mix64(int(seed), int(scenario_index), 0xD34A1)
            tasks.append((int(seed), int(scenario_index), episode, int(deal_seed)))
        hu_counts[str(seed)] = int(hu)

    rows: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers), len(tasks)),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(
            str(solver_path),
            str(snapshot_a.resolve()),
            str(snapshot_b.resolve()),
        ),
    ) as pool:
        for chunk in pool.map(_worker, tasks, chunksize=1):
            rows.extend(chunk)

    summary = _summarize(rows)
    report = {
        "schema": "SPINCORE_LT2_HU_POLICY_CHAIN_EVAL_V1",
        "stage_a": {
            "checkpoint": str(stage_a_path),
            "completed_iteration": int(meta_a["completed_iteration"]),
        },
        "stage_b": {
            "checkpoint": str(stage_b_path),
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
            "hu_scenarios_by_seed": hu_counts,
            "baselines": list(BASELINES),
            "policies": list(POLICIES),
            "pairing": (
                "same HU scenario, deal, hero seat and RNG seeds across AVG_A, "
                "AVG_B, BEH_A and BEH_B"
            ),
            "warning": (
                "fixed weak-opponent mechanism diagnostic; not exploitability/GTO proof"
            ),
        },
        "summary": summary,
        "rows_count": len(rows),
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== LT2 HU POLICY-CHAIN EVALUATION ===")
    for baseline in BASELINES:
        x = summary[baseline]
        print(
            f"{baseline}: "
            f"AVG B-A={x['avg_b_minus_a']['mean']:+.3f} "
            f"CI95=[{x['avg_b_minus_a']['ci95_low']:+.3f},{x['avg_b_minus_a']['ci95_high']:+.3f}] "
            f"BEH B-A={x['beh_b_minus_a']['mean']:+.3f} "
            f"CI95=[{x['beh_b_minus_a']['ci95_low']:+.3f},{x['beh_b_minus_a']['ci95_high']:+.3f}] "
            f"CHAIN_DELTA={x['chain_delta_b_minus_a']['mean']:+.3f} "
            f"CI95=[{x['chain_delta_b_minus_a']['ci95_low']:+.3f},{x['chain_delta_b_minus_a']['ci95_high']:+.3f}]"
        )
    print("LT2_HU_POLICY_CHAIN_EVAL_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
