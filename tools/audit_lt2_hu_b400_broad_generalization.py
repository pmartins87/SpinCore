#!/usr/bin/env python3
from __future__ import annotations

"""Broad forensic generalization gate for three deterministic Stage-B 400-step refits."""

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

import audit_lt2_hu_policy_chain as chain
import audit_lt2_jammer_facing_allin_target_overlay as overlay
import audit_lt2_stage_a_b_first_divergence as fd
from spincore.lean_functional_training import load_checkpoint
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.solver import Episode, SolverLibrary

BASELINES = ("UNIFORM_LEGAL", "PASSIVE_CALLER", "JAMMER")
POLICIES = ("PROD_A", "PROD_B", "B400_R0", "B400_R1", "B400_R2")
FORENSIC_SEEDS = (20260920, 20260921, 20260922, 20260923, 20260924, 20260925)
DOMAIN = "TRUE_HEADS_UP"

_SOLVER = None
_STAGES = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--stage-b", type=Path, required=True)
    p.add_argument("--controlled-report", type=Path, required=True)
    p.add_argument("--scenarios-per-seed", type=int, default=5000)
    p.add_argument("--workers", type=int, default=31)
    p.add_argument("--threads-fit", type=int, default=8)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args()


def _mean_ci(values):
    xs = [float(x) for x in values]
    n = len(xs)
    if n <= 0:
        return {
            "n": 0, "mean": float("nan"), "sem": float("nan"),
            "ci95_low": float("nan"), "ci95_high": float("nan"),
        }
    mean = float(statistics.fmean(xs))
    sem = 0.0 if n == 1 else float(statistics.stdev(xs) / math.sqrt(n))
    half = 1.96 * sem
    return {
        "n": n, "mean": mean, "sem": sem,
        "ci95_low": mean - half, "ci95_high": mean + half,
    }


def _save_candidate_snapshot(*, runtime, source_checkpoint, iteration, path):
    bundle = runtime.bundle
    payload = {
        "schema": overlay.SNAPSHOT_SCHEMA,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "completed_iteration": int(iteration),
        "domain_seed": int(bundle.seed),
        "advantage_ready": 1,
        "policy": {
            k: v.detach().cpu().clone()
            for k, v in bundle.policy.state_dict().items()
        },
        "advantage": {
            k: v.detach().cpu().clone()
            for k, v in bundle.advantage.state_dict().items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _build_candidates(
    *,
    solver,
    stage_b_path,
    controlled,
    out_dir,
    threads,
):
    torch.set_num_threads(int(threads))
    (
        _seed,
        config,
        iteration,
        _sampler,
        runtimes,
        _history,
        _finalized,
    ) = load_checkpoint(stage_b_path, solver=solver)
    if int(iteration) != 7500:
        raise RuntimeError(f"unexpected Stage-B iteration {iteration}")

    runtime = runtimes[DOMAIN]
    runtime.session.batch_mode = "vectorized"

    trials = sorted(
        [
            trial for trial in controlled["trials"]
            if int(trial["budget"]) == 400
        ],
        key=lambda trial: int(trial["replicate"]),
    )
    if len(trials) != 3:
        raise RuntimeError(f"expected three 400-step trials, got {len(trials)}")

    snapshots = []
    meta = []
    for expected_rep, trial in enumerate(trials):
        rep = int(trial["replicate"])
        if rep != expected_rep:
            raise RuntimeError("controlled-report replicate numbering drift")
        init_seed = int(trial["init_seed"])
        batch_seed = int(trial["batch_seed"])

        runtime.session.reset_advantage_network(
            init_seed=init_seed,
            lr=float(config.learning_rate),
        )
        runtime.bundle.batch_rng.seed(batch_seed)
        losses = runtime.session.train_advantage(
            steps=400,
            batch_size=int(config.batch_size),
        )
        if len(losses) != 400:
            raise RuntimeError("candidate fit step-count drift")

        path = out_dir / f"b400_r{rep}.pt"
        _save_candidate_snapshot(
            runtime=runtime,
            source_checkpoint=stage_b_path,
            iteration=iteration,
            path=path,
        )
        snapshots.append(path)
        meta.append({
            "replicate": rep,
            "init_seed": init_seed,
            "batch_seed": batch_seed,
            "loss_last": float(losses[-1]),
            "snapshot": str(path.resolve()),
            "snapshot_bytes": int(path.stat().st_size),
        })
    return snapshots, meta


def _init_worker(solver_path, names, snapshots):
    global _SOLVER, _STAGES
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
    _STAGES = {
        str(name): overlay.StageModels(Path(path), _SOLVER)
        for name, path in zip(names, snapshots)
    }


def _hero_distribution(policy, state):
    stage = _STAGES[policy]
    active_mask, legal = chain._legal_context(state)
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
    deal_seed,
    hero_seat,
    hero_policy,
    opponent_policy,
    scenario_index,
    master_seed,
):
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
    seat_rng[int(hero_seat)] = random.Random(
        fd._mix64(master_seed, scenario_index, hero_seat, 777)
    )

    decisions = 0
    try:
        while not state.terminal:
            actor = int(state.actor)
            if actor == int(hero_seat):
                active_mask, legal, probs = _hero_distribution(hero_policy, state)
                slot = chain._sample_probs(legal, probs, seat_rng[actor])
            else:
                active_mask, slot = chain._baseline_action(
                    opponent_policy, state, seat_rng[actor]
                )
            chain.apply_lean(state, active_mask, slot)
            decisions += 1
            if decisions > 200:
                raise RuntimeError("B400 broad-gate hand exceeded 200 decisions")
        delta = state.terminal_chip_delta()
        if sum(delta) != 0:
            raise RuntimeError(f"non-zero-sum terminal delta: {delta}")
        return int(delta[int(hero_seat)])
    finally:
        state.close()


def _worker(task):
    seed, scenario_index, episode, deal_seed = task
    if not episode.game_is_hu:
        return []
    live = [
        seat for seat, stack in enumerate(episode.stacks)
        if int(stack) > 0
    ]
    if len(live) != 2:
        raise RuntimeError("HU episode without exactly two live seats")

    rows = []
    for baseline in BASELINES:
        for hero in live:
            values = {
                policy: _play_one(
                    episode,
                    deal_seed=int(deal_seed),
                    hero_seat=int(hero),
                    hero_policy=policy,
                    opponent_policy=baseline,
                    scenario_index=int(scenario_index),
                    master_seed=int(seed),
                )
                for policy in POLICIES
            }
            row = {
                "seed": int(seed),
                "scenario": int(scenario_index),
                "hero_seat": int(hero),
                "baseline": baseline,
                **{policy.lower(): int(value) for policy, value in values.items()},
            }
            for rep in range(3):
                cand = values[f"B400_R{rep}"]
                row[f"b400_r{rep}_minus_prod_b"] = int(cand - values["PROD_B"])
                row[f"b400_r{rep}_minus_prod_a"] = int(cand - values["PROD_A"])
            row["prod_b_minus_a"] = int(values["PROD_B"] - values["PROD_A"])
            rows.append(row)
    return rows


def _cluster_values(rows, *, baseline, key):
    by = {}
    for row in rows:
        if row["baseline"] != baseline:
            continue
        cluster = (int(row["seed"]), int(row["scenario"]))
        by.setdefault(cluster, []).append(float(row[key]))
    return [float(statistics.fmean(values)) for values in by.values()]


def _summary(rows):
    out = {}
    for baseline in BASELINES:
        block = {
            "clusters": len({
                (int(row["seed"]), int(row["scenario"]))
                for row in rows if row["baseline"] == baseline
            }),
            "prod_a": _mean_ci(_cluster_values(rows, baseline=baseline, key="prod_a")),
            "prod_b": _mean_ci(_cluster_values(rows, baseline=baseline, key="prod_b")),
            "prod_b_minus_a": _mean_ci(
                _cluster_values(rows, baseline=baseline, key="prod_b_minus_a")
            ),
            "candidates": {},
        }
        for rep in range(3):
            block["candidates"][f"B400_R{rep}"] = {
                "absolute": _mean_ci(
                    _cluster_values(
                        rows, baseline=baseline, key=f"b400_r{rep}"
                    )
                ),
                "minus_prod_b": _mean_ci(
                    _cluster_values(
                        rows,
                        baseline=baseline,
                        key=f"b400_r{rep}_minus_prod_b",
                    )
                ),
                "minus_prod_a": _mean_ci(
                    _cluster_values(
                        rows,
                        baseline=baseline,
                        key=f"b400_r{rep}_minus_prod_a",
                    )
                ),
            }
        out[baseline] = block
    return out


def main():
    args = parse_args()
    if args.scenarios_per_seed <= 0 or args.workers <= 0:
        raise SystemExit("positive scenarios/workers required")

    solver_path = args.solver.resolve(strict=True)
    stage_a_path = args.stage_a.resolve(strict=True)
    stage_b_path = args.stage_b.resolve(strict=True)
    controlled_path = args.controlled_report.resolve(strict=True)
    controlled = json.loads(controlled_path.read_text(encoding="utf-8"))
    if controlled.get("schema") != "SPINCORE_LT2_JAMMER_FAI_CONTROLLED_REFIT_V1":
        raise SystemExit("wrong controlled-refit schema")
    if controlled["method"]["future_holdout_seeds_touched"] is not False:
        raise RuntimeError("controlled report touched holdout")

    args.report.parent.mkdir(parents=True, exist_ok=True)

    local_solver = SolverLibrary(solver_path)
    prod_a_snapshot = args.report.parent / "prod_a.pt"
    prod_b_snapshot = args.report.parent / "prod_b.pt"
    overlay._extract_stage_snapshot(stage_a_path, prod_a_snapshot)
    overlay._extract_stage_snapshot(stage_b_path, prod_b_snapshot)

    candidate_paths, candidate_meta = _build_candidates(
        solver=local_solver,
        stage_b_path=stage_b_path,
        controlled=controlled,
        out_dir=args.report.parent,
        threads=int(args.threads_fit),
    )
    del local_solver
    gc.collect()

    names = list(POLICIES)
    snapshots = [
        prod_a_snapshot,
        prod_b_snapshot,
        *candidate_paths,
    ]

    tasks = []
    hu_counts = {}
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
        hu_counts[str(seed)] = hu

    rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers), len(tasks)),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(
            str(solver_path),
            names,
            [str(path.resolve()) for path in snapshots],
        ),
    ) as pool:
        for chunk in pool.map(_worker, tasks, chunksize=1):
            rows.extend(chunk)

    summary = _summary(rows)
    report = {
        "schema": "SPINCORE_LT2_HU_B400_BROAD_GENERALIZATION_V1",
        "stage_a": {
            "checkpoint": str(stage_a_path),
            "completed_iteration": 3000,
        },
        "stage_b": {
            "checkpoint": str(stage_b_path),
            "completed_iteration": 7500,
        },
        "controlled_refit_report": str(controlled_path),
        "candidate_meta": candidate_meta,
        "method": {
            "read_only_source_checkpoints": True,
            "new_training_roots": 0,
            "source_training_memory_writes": 0,
            "candidate_optimizer_steps": 1200,
            "future_holdout_seeds_touched": False,
            "forensic_seeds": list(FORENSIC_SEEDS),
            "scenarios_per_seed": int(args.scenarios_per_seed),
            "hu_scenarios_by_seed": hu_counts,
            "baselines": list(BASELINES),
            "policies": list(POLICIES),
            "pairing": (
                "same HU scenario, deal, hero seat and RNG seeds across "
                "PROD_A, PROD_B and all three B400 candidates"
            ),
            "candidate_selection": (
                "all three deterministic 400-step Stage-B refits from the "
                "controlled-refit experiment; no replicate chosen by outcome"
            ),
        },
        "summary": summary,
        "rows_count": len(rows),
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=== LT2 HU B400 BROAD GENERALIZATION ===")
    for baseline in BASELINES:
        block = summary[baseline]
        p = block["prod_b_minus_a"]
        print(
            f"{baseline}: PROD_B-A={p['mean']:+.3f} "
            f"CI95=[{p['ci95_low']:+.3f},{p['ci95_high']:+.3f}]"
        )
        for rep in range(3):
            cand = block["candidates"][f"B400_R{rep}"]
            db = cand["minus_prod_b"]
            da = cand["minus_prod_a"]
            print(
                f"  B400_R{rep}: vs_B={db['mean']:+.3f} "
                f"CI95=[{db['ci95_low']:+.3f},{db['ci95_high']:+.3f}] "
                f"vs_A={da['mean']:+.3f} "
                f"CI95=[{da['ci95_low']:+.3f},{da['ci95_high']:+.3f}]"
            )

    print("LT2_HU_B400_BROAD_GENERALIZATION_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
