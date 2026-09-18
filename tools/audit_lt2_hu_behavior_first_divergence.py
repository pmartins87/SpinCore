#!/usr/bin/env python3
from __future__ import annotations

"""First-divergence attribution for Stage-A vs Stage-B current HU behavior.

This is the current-Advantage analogue of the existing AveragePolicy forensic.
It replays the already-seen forensic seeds on paired HU scenarios. Stage A and
Stage B receive the same scenario, deal, weak opponent, hero seat and random
streams. Until the sampled hero actions first differ, the solver states are
identical.

Every seat-run is assigned exactly one first-divergence group:
  NO_DIVERGENCE
  PREFLOP_ROOT
  PREFLOP_FACING_ALL_IN
  PREFLOP_OTHER
  FLOP
  TURN
  RIVER

The B-minus-A chip-EV contribution of the mutually exclusive groups adds back
to the total paired B-minus-A behavior result.

Read only: no training roots, optimizer steps, or memory writes.
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
GROUPS = (
    "NO_DIVERGENCE",
    "PREFLOP_ROOT",
    "PREFLOP_FACING_ALL_IN",
    "PREFLOP_OTHER",
    "FLOP",
    "TURN",
    "RIVER",
)
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


def _init_worker(solver_path: str, snap_a: str, snap_b: str) -> None:
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
    _STAGE_A = overlay.StageModels(Path(snap_a), _SOLVER)
    _STAGE_B = overlay.StageModels(Path(snap_b), _SOLVER)


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


def _behavior_distribution(stage, state):
    active_mask, legal = _legal_context(state)
    obs = state.neural_bytes()
    probs = stage.runtime.session.behavior(state, obs, legal)
    out = tuple(float(x) for x in probs)
    mass = sum(out[a] for a in legal)
    if not (0.999 <= mass <= 1.001):
        raise RuntimeError(f"behavior probability mass drift: {mass}")
    return active_mask, legal, out


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
    raise ValueError(baseline)


def _finish_arm(state, *, hero_seat: int, stage, baseline: str, seat_rng) -> int:
    decisions = 0
    try:
        while not state.terminal:
            actor = int(state.actor)
            if actor == int(hero_seat):
                active_mask, legal, probs = _behavior_distribution(stage, state)
                slot = _sample_probs(legal, probs, seat_rng[actor])
            else:
                active_mask, slot = _baseline_action(baseline, state, seat_rng[actor])
            apply_lean(state, active_mask, slot)
            decisions += 1
            if decisions > 200:
                raise RuntimeError("behavior forensic arm exceeded 200 decisions")
        delta = state.terminal_chip_delta()
        if sum(delta) != 0:
            raise RuntimeError(f"terminal delta not zero-sum: {delta}")
        return int(delta[int(hero_seat)])
    finally:
        state.close()


def _group_for(*, street: int, common_path, last_actor, last_action, hero_seat: int) -> str:
    if street == 0:
        if not common_path:
            return "PREFLOP_ROOT"
        if (
            last_actor is not None
            and int(last_actor) != int(hero_seat)
            and int(last_action) == 9
        ):
            return "PREFLOP_FACING_ALL_IN"
        return "PREFLOP_OTHER"
    return {1: "FLOP", 2: "TURN", 3: "RIVER"}[int(street)]


def _paired_run(
    episode: Episode,
    *,
    deal_seed: int,
    hero_seat: int,
    baseline: str,
    seed: int,
    scenario_index: int,
) -> dict[str, Any]:
    state_a = _SOLVER.create(episode, int(deal_seed))
    state_b = _SOLVER.create(episode, int(deal_seed))

    rng_a = {
        seat: random.Random(
            fd._mix64(seed, scenario_index, seat, 100 + BASELINES.index(baseline))
        )
        for seat in range(3)
    }
    rng_b = {
        seat: random.Random(
            fd._mix64(seed, scenario_index, seat, 100 + BASELINES.index(baseline))
        )
        for seat in range(3)
    }
    rng_a[int(hero_seat)] = random.Random(fd._mix64(seed, scenario_index, hero_seat, 777))
    rng_b[int(hero_seat)] = random.Random(fd._mix64(seed, scenario_index, hero_seat, 777))

    common_path = []
    last_actor = None
    last_action = None
    common_hero_decisions = 0

    try:
        for _ in range(200):
            if bool(state_a.terminal) != bool(state_b.terminal):
                raise RuntimeError("paired terminal drift before hero divergence")

            if state_a.terminal:
                da = state_a.terminal_chip_delta()
                db = state_b.terminal_chip_delta()
                if tuple(da) != tuple(db):
                    raise RuntimeError("identical path produced different terminal result")
                value = int(da[int(hero_seat)])
                state_a.close()
                state_b.close()
                state_a = None
                state_b = None
                return {
                    "stage_a": value,
                    "stage_b": value,
                    "delta_b_minus_a": 0,
                    "group": "NO_DIVERGENCE",
                    "diverged": False,
                    "street": None,
                    "last_action_slot": None,
                    "facing_all_in": False,
                    "a_slot": None,
                    "b_slot": None,
                    "policy_tv": 0.0,
                    "common_hero_decisions_before_divergence": int(common_hero_decisions),
                    "common_public_action_count": len(common_path),
                    "transition": None,
                }

            actor_a = int(state_a.actor)
            actor_b = int(state_b.actor)
            if actor_a != actor_b:
                raise RuntimeError("paired actor drift before hero divergence")
            actor = actor_a

            if actor != int(hero_seat):
                ma, sa = _baseline_action(baseline, state_a, rng_a[actor])
                mb, sb = _baseline_action(baseline, state_b, rng_b[actor])
                if ma != mb or sa != sb:
                    raise RuntimeError("baseline drift before hero divergence")
                apply_lean(state_a, ma, sa)
                apply_lean(state_b, mb, sb)
                common_path.append((actor, int(sa)))
                last_actor, last_action = actor, int(sa)
                continue

            ma, la, pa = _behavior_distribution(_STAGE_A, state_a)
            mb, lb, pb = _behavior_distribution(_STAGE_B, state_b)
            if ma != mb or la != lb:
                raise RuntimeError("A/B legal drift on identical state")
            common_hero_decisions += 1
            sa = _sample_probs(la, pa, rng_a[actor])
            sb = _sample_probs(lb, pb, rng_b[actor])

            if sa == sb:
                apply_lean(state_a, ma, sa)
                apply_lean(state_b, mb, sb)
                common_path.append((actor, int(sa)))
                last_actor, last_action = actor, int(sa)
                continue

            street = int(cond._street(state_a))
            group = _group_for(
                street=street,
                common_path=common_path,
                last_actor=last_actor,
                last_action=last_action,
                hero_seat=int(hero_seat),
            )
            tv = 0.5 * sum(abs(float(pa[i]) - float(pb[i])) for i in range(10))
            divergence = {
                "group": group,
                "street": street,
                "last_action_slot": None if last_action is None else int(last_action),
                "facing_all_in": bool(group == "PREFLOP_FACING_ALL_IN"),
                "a_slot": int(sa),
                "b_slot": int(sb),
                "policy_tv": float(tv),
                "common_hero_decisions_before_divergence": int(common_hero_decisions - 1),
                "common_public_action_count": len(common_path),
                "transition": f"{int(sa)}->{int(sb)}",
            }

            apply_lean(state_a, ma, sa)
            apply_lean(state_b, mb, sb)

            value_a = _finish_arm(
                state_a,
                hero_seat=int(hero_seat),
                stage=_STAGE_A,
                baseline=baseline,
                seat_rng=rng_a,
            )
            state_a = None
            value_b = _finish_arm(
                state_b,
                hero_seat=int(hero_seat),
                stage=_STAGE_B,
                baseline=baseline,
                seat_rng=rng_b,
            )
            state_b = None
            return {
                "stage_a": int(value_a),
                "stage_b": int(value_b),
                "delta_b_minus_a": int(value_b - value_a),
                "diverged": True,
                **divergence,
            }

        raise RuntimeError("paired behavior forensic exceeded 200 decisions")
    finally:
        if state_a is not None:
            state_a.close()
        if state_b is not None:
            state_b.close()


def _worker(task):
    seed, scenario_index, episode, deal_seed = task
    if not episode.game_is_hu:
        return []
    live = [s for s, stack in enumerate(episode.stacks) if int(stack) > 0]
    if len(live) != 2:
        raise RuntimeError("HU episode without exactly two live seats")
    blind = f"{episode.small_blind}/{episode.big_blind}"

    rows = []
    for baseline in BASELINES:
        for hero in live:
            result = _paired_run(
                episode,
                deal_seed=int(deal_seed),
                hero_seat=int(hero),
                baseline=baseline,
                seed=int(seed),
                scenario_index=int(scenario_index),
            )
            rows.append({
                "seed": int(seed),
                "scenario": int(scenario_index),
                "blind": blind,
                "hero_seat": int(hero),
                "baseline": baseline,
                **result,
            })
    return rows


def _mean_ci(values):
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sem": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    mean = float(statistics.fmean(values))
    sem = 0.0 if n == 1 else float(statistics.stdev(values) / math.sqrt(n))
    half = 1.96 * sem
    return {"n": n, "mean": mean, "sem": sem, "ci95_low": mean - half, "ci95_high": mean + half}


def _cluster_by_scenario(rows, value_key):
    by = {}
    for r in rows:
        by.setdefault((int(r["seed"]), int(r["scenario"])), []).append(float(r[value_key]))
    return [float(statistics.fmean(v)) for v in by.values()]


def _aggregate(rows):
    out = {}
    for baseline in BASELINES:
        b = [r for r in rows if r["baseline"] == baseline]
        total = _mean_ci(_cluster_by_scenario(b, "delta_b_minus_a"))
        by_group = {}
        for group in GROUPS:
            # Contribution is zero in scenario clusters that first diverged elsewhere.
            cluster = {}
            for r in b:
                k = (int(r["seed"]), int(r["scenario"]))
                cluster.setdefault(k, []).append(
                    float(r["delta_b_minus_a"]) if r["group"] == group else 0.0
                )
            vals = [float(statistics.fmean(v)) for v in cluster.values()]
            count = sum(1 for r in b if r["group"] == group)
            by_group[group] = {
                "seat_runs": int(count),
                "seat_run_frequency": float(count / max(len(b), 1)),
                "contribution_b_minus_a": _mean_ci(vals),
            }

        div = [r for r in b if bool(r["diverged"])]
        transitions = {}
        for r in div:
            key = f"{r['group']}:{r['transition']}"
            transitions[key] = transitions.get(key, 0) + 1

        out[baseline] = {
            "total_b_minus_a": total,
            "seat_runs": len(b),
            "divergence_rate": float(len(div) / max(len(b), 1)),
            "groups": by_group,
            "transition_counts": dict(sorted(transitions.items(), key=lambda kv: (-kv[1], kv[0]))),
        }
    return out


def main() -> int:
    args = parse_args()
    if args.scenarios_per_seed <= 0 or args.workers <= 0:
        raise SystemExit("positive scenarios/workers required")
    solver_path = args.solver.resolve(strict=True)
    a_path = args.stage_a.resolve(strict=True)
    b_path = args.stage_b.resolve(strict=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    snap_a = args.report.parent / "stage_a_hu_models.pt"
    snap_b = args.report.parent / "stage_b_hu_models.pt"
    print("Extracting lightweight HU model snapshots...", flush=True)
    meta_a = overlay._extract_stage_snapshot(a_path, snap_a)
    meta_b = overlay._extract_stage_snapshot(b_path, snap_b)

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
        hu_counts[str(seed)] = int(hu)

    rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers), len(tasks)),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(str(solver_path), str(snap_a.resolve()), str(snap_b.resolve())),
    ) as pool:
        for chunk in pool.map(_worker, tasks, chunksize=1):
            rows.extend(chunk)

    summary = _aggregate(rows)
    report = {
        "schema": "SPINCORE_LT2_HU_BEHAVIOR_FIRST_DIVERGENCE_V1",
        "stage_a": {"checkpoint": str(a_path), "completed_iteration": int(meta_a["completed_iteration"])},
        "stage_b": {"checkpoint": str(b_path), "completed_iteration": int(meta_b["completed_iteration"])},
        "method": {
            "read_only": True,
            "new_training_roots": 0,
            "optimizer_steps": 0,
            "training_memory_writes": 0,
            "forensic_seeds": list(FORENSIC_SEEDS),
            "future_holdout_seeds_touched": False,
            "scenarios_per_seed": int(args.scenarios_per_seed),
            "hu_scenarios_by_seed": hu_counts,
            "pairing": "same HU scenario, deal, hero seat, baseline and RNG streams until first sampled current-behavior divergence",
            "groups": list(GROUPS),
        },
        "rows_count": len(rows),
        "summary": summary,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=== LT2 HU CURRENT-BEHAVIOR FIRST-DIVERGENCE ===")
    for baseline in BASELINES:
        x = summary[baseline]
        t = x["total_b_minus_a"]
        print(
            f"{baseline}: B-A={t['mean']:+.3f} "
            f"CI95=[{t['ci95_low']:+.3f},{t['ci95_high']:+.3f}] "
            f"divergence_rate={x['divergence_rate']:.3f}"
        )
        for group in GROUPS:
            g = x["groups"][group]
            c = g["contribution_b_minus_a"]
            if g["seat_runs"]:
                print(
                    f"  {group}: freq={g['seat_run_frequency']:.3f} "
                    f"contrib={c['mean']:+.3f} "
                    f"CI95=[{c['ci95_low']:+.3f},{c['ci95_high']:+.3f}]"
                )
    print("LT2_HU_BEHAVIOR_FIRST_DIVERGENCE_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
