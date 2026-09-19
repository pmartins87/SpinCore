#!/usr/bin/env python3
from __future__ import annotations

"""Full-population reconciliation of the Jammer FAI current-behavior loss.

Purpose
-------
The 48-anchor broad low-noise infoset calibration did NOT show a broad Stage-B
policy-regret deterioration, yet the paired first-divergence forensic attributes
-6.2267 chips/hand to PREFLOP_FACING_ALL_IN.

This audit reconciles those results on the *entire natural evaluation
population* without selecting anchors and without Monte-Carlo action noise.

For every HU Jammer seat-run on the forensic seeds:
  * replay Stage A and B current behavior with the exact same deal/RNG until
    either an earlier hero divergence occurs or a common FAI state is reached;
  * at every common FAI state, before sampling the hero action, compute the
    actual-deal terminal chip value of every legal action by cloning the solver
    state and applying the action;
  * compute the deterministic counterfactual policy-value difference
        sum_a (sigma_B[a] - sigma_A[a]) * Q_actual[a];
  * also sample the FAI action with the exact paired RNG, reproducing the
    previous stochastic first-divergence contribution.

Because the full dealt board/opponent hand are fixed inside the solver, the
per-state Q_actual values use hidden information and are NOT an infoset target.
Averaged over the natural evaluation population, however, they are an unbiased
counterfactual estimate of the policy-value difference and exactly match the
evaluation distribution.

No training roots, optimizer steps, memory writes, or holdout seeds.
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
sys.path.insert(0, str(ROOT / "tools"))

import torch

import audit_lt2_hu_preflop_conditional_resampling as cond
import audit_lt2_jammer_facing_allin_target_overlay as overlay
import audit_lt2_stage_a_b_first_divergence as fd
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_solver_actions import apply_lean, lean_legal_actions
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.solver import Episode, SolverLibrary

FORENSIC_SEEDS = (20260920, 20260921, 20260922, 20260923, 20260924, 20260925)
PRIOR_FAI_CONTRIBUTION = -6.22672064777328

_SOLVER = None
_STAGE_A = None
_STAGE_B = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver", type=Path, default=ROOT / "build/libspincore_solver_c.so")
    p.add_argument("--stage-a", type=Path, required=True)
    p.add_argument("--stage-b", type=Path, required=True)
    p.add_argument("--scenarios-per-seed", type=int, default=5000)
    p.add_argument("--workers", type=int, default=6)
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


def _sample_probs(legal, probs, rng: random.Random) -> int:
    x = rng.random()
    cumulative = 0.0
    for slot in legal:
        cumulative += float(probs[slot])
        if x < cumulative:
            return int(slot)
    return int(legal[-1])


def _behavior(stage, state):
    active_mask, legal = _legal_context(state)
    obs = state.neural_bytes()
    probs = tuple(float(x) for x in stage.runtime.session.behavior(state, obs, legal))
    mass = sum(probs[a] for a in legal)
    if not (0.999 <= mass <= 1.001):
        raise RuntimeError(f"behavior probability mass drift: {mass}")
    return int(active_mask), legal, probs


def _jammer_action(state):
    active_mask, legal = _legal_context(state)
    if 9 in legal:
        return active_mask, 9
    if 1 in legal:
        return active_mask, 1
    if 0 in legal:
        return active_mask, 0
    return active_mask, int(max(legal))


def _actual_action_values(state, *, hero: int, active_mask: int, legal) -> dict[int, int]:
    values: dict[int, int] = {}
    for slot in legal:
        child = state.clone()
        try:
            apply_lean(child, int(active_mask), int(slot))
            if not child.terminal:
                raise RuntimeError(
                    "Jammer FAI action did not terminate hand; direct actual-deal "
                    f"counterfactual unsupported for slot {slot}"
                )
            delta = child.terminal_chip_delta()
            if sum(delta) != 0:
                raise RuntimeError(f"terminal delta not zero-sum: {delta}")
            values[int(slot)] = int(delta[int(hero)])
        finally:
            child.close()
    return values


def _run_seat(
    episode: Episode,
    *,
    deal_seed: int,
    hero: int,
    seed: int,
    scenario_index: int,
) -> dict[str, Any]:
    state_a = _SOLVER.create(episode, int(deal_seed))
    state_b = _SOLVER.create(episode, int(deal_seed))
    rng_a = random.Random(fd._mix64(seed, scenario_index, hero, 777))
    rng_b = random.Random(fd._mix64(seed, scenario_index, hero, 777))

    last_actor = None
    last_action = None
    common_public_action_count = 0

    try:
        for _ in range(200):
            if bool(state_a.terminal) != bool(state_b.terminal):
                raise RuntimeError("paired terminal drift before common FAI")

            if state_a.terminal:
                return {
                    "status": "NO_COMMON_FAI",
                    "expected_fai_delta": 0.0,
                    "sampled_fai_delta": 0.0,
                    "sampled_fai_divergence": False,
                }

            actor = int(state_a.actor)
            if actor != int(state_b.actor):
                raise RuntimeError("paired actor drift before common FAI")

            if actor != int(hero):
                ma, sa = _jammer_action(state_a)
                mb, sb = _jammer_action(state_b)
                if ma != mb or sa != sb:
                    raise RuntimeError("Jammer action drift before common FAI")
                apply_lean(state_a, ma, sa)
                apply_lean(state_b, mb, sb)
                common_public_action_count += 1
                last_actor, last_action = actor, int(sa)
                continue

            street = int(cond._street(state_a))
            if (
                street == 0
                and last_actor is not None
                and int(last_actor) != int(hero)
                and int(last_action) == 9
            ):
                ma, la, pa = _behavior(_STAGE_A, state_a)
                mb, lb, pb = _behavior(_STAGE_B, state_b)
                if ma != mb or la != lb:
                    raise RuntimeError("A/B legal drift at common FAI")

                q = _actual_action_values(
                    state_a, hero=int(hero), active_mask=int(ma), legal=la
                )
                expected = float(
                    sum((float(pb[a]) - float(pa[a])) * float(q[a]) for a in la)
                )

                sa = _sample_probs(la, pa, rng_a)
                sb = _sample_probs(lb, pb, rng_b)
                sampled = float(q[int(sb)] - q[int(sa)]) if int(sa) != int(sb) else 0.0

                fold_a = float(pa[0]) if 0 in la else 0.0
                fold_b = float(pb[0]) if 0 in lb else 0.0
                nonfold = [int(a) for a in la if int(a) != 0]
                spread = (
                    float(max(q[a] for a in nonfold) - min(q[a] for a in nonfold))
                    if len(nonfold) >= 2
                    else 0.0
                )
                if nonfold:
                    continue_value_mean = float(
                        statistics.fmean(float(q[a]) for a in nonfold)
                    )
                    fold_minus_continue_actual = (
                        float(q[0]) - continue_value_mean if 0 in q else float("nan")
                    )
                else:
                    continue_value_mean = float("nan")
                    fold_minus_continue_actual = float("nan")

                tv = 0.5 * sum(abs(float(pa[i]) - float(pb[i])) for i in range(10))
                return {
                    "status": "COMMON_FAI",
                    "expected_fai_delta": expected,
                    "sampled_fai_delta": sampled,
                    "sampled_fai_divergence": bool(int(sa) != int(sb)),
                    "sampled_a_slot": int(sa),
                    "sampled_b_slot": int(sb),
                    "legal": list(la),
                    "actual_q": {str(k): int(v) for k, v in sorted(q.items())},
                    "nonfold_actual_q_spread": spread,
                    "fold_mass_a": fold_a,
                    "fold_mass_b": fold_b,
                    "fold_mass_b_minus_a": float(fold_b - fold_a),
                    "fold_minus_continue_actual_chips": fold_minus_continue_actual,
                    "policy_tv": float(tv),
                    "common_public_action_count": int(common_public_action_count),
                }

            ma, la, pa = _behavior(_STAGE_A, state_a)
            mb, lb, pb = _behavior(_STAGE_B, state_b)
            if ma != mb or la != lb:
                raise RuntimeError("A/B legal drift before common FAI")
            sa = _sample_probs(la, pa, rng_a)
            sb = _sample_probs(lb, pb, rng_b)

            if int(sa) != int(sb):
                return {
                    "status": "EARLIER_HERO_DIVERGENCE",
                    "expected_fai_delta": 0.0,
                    "sampled_fai_delta": 0.0,
                    "sampled_fai_divergence": False,
                    "earlier_street": int(street),
                    "earlier_a_slot": int(sa),
                    "earlier_b_slot": int(sb),
                }

            apply_lean(state_a, ma, sa)
            apply_lean(state_b, mb, sb)
            common_public_action_count += 1
            last_actor, last_action = actor, int(sa)

        raise RuntimeError("population reconciliation exceeded 200 decisions")
    finally:
        state_a.close()
        state_b.close()


def _process_seed(task):
    seed, scenarios = task
    sampler = LegacyScenarioSampler(
        seed=int(seed) ^ 0x5CE0A710,
        config=LegacyScenarioConfig(),
    )
    rows = []
    hu = 0
    for scenario_index in range(int(scenarios)):
        episode = sampler.sample_episode()
        if not episode.game_is_hu:
            continue
        hu += 1
        live = [s for s, stack in enumerate(episode.stacks) if int(stack) > 0]
        if len(live) != 2:
            raise RuntimeError("HU episode without exactly two live seats")
        deal_seed = fd._mix64(seed, scenario_index, 0xD34A1)
        blind = f"{episode.small_blind}/{episode.big_blind}"
        for hero in live:
            result = _run_seat(
                episode,
                deal_seed=int(deal_seed),
                hero=int(hero),
                seed=int(seed),
                scenario_index=int(scenario_index),
            )
            rows.append(
                {
                    "seed": int(seed),
                    "scenario": int(scenario_index),
                    "hero_seat": int(hero),
                    "blind": blind,
                    **result,
                }
            )
    return int(seed), int(hu), rows


def _mean_ci(values):
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


def _cluster_scenario(rows, getter):
    by = {}
    for r in rows:
        key = (int(r["seed"]), int(r["scenario"]))
        by.setdefault(key, []).append(float(getter(r)))
    return [float(statistics.fmean(v)) for v in by.values()]


def _cluster_seed(rows, getter):
    by = {}
    for r in rows:
        by.setdefault(int(r["seed"]), []).append(float(getter(r)))
    return [float(statistics.fmean(v)) for _, v in sorted(by.items())]


def _additive_group_contribution(rows, predicate, field):
    return _mean_ci(
        _cluster_scenario(
            rows,
            lambda r: float(r[field]) if predicate(r) else 0.0,
        )
    )


def _aggregate(rows):
    total_expected = _mean_ci(
        _cluster_scenario(rows, lambda r: r["expected_fai_delta"])
    )
    total_sampled = _mean_ci(
        _cluster_scenario(rows, lambda r: r["sampled_fai_delta"])
    )
    common = [r for r in rows if r["status"] == "COMMON_FAI"]

    status_counts = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    fold_shift = {
        "B_MORE_FOLD": lambda r: r.get("fold_mass_b_minus_a", 0.0) > 1e-12,
        "B_LESS_FOLD": lambda r: r.get("fold_mass_b_minus_a", 0.0) < -1e-12,
        "NO_FOLD_SHIFT": lambda r: abs(r.get("fold_mass_b_minus_a", 0.0)) <= 1e-12,
    }
    by_fold_shift = {
        name: {
            "common_fai_rows": int(sum(1 for r in common if pred(r))),
            "additive_expected_contribution": _additive_group_contribution(
                rows,
                lambda r, p=pred: r["status"] == "COMMON_FAI" and p(r),
                "expected_fai_delta",
            ),
        }
        for name, pred in fold_shift.items()
    }

    signatures = sorted(
        {
            ",".join(str(x) for x in r["legal"])
            for r in common
        }
    )
    by_legal = {}
    for sig in signatures:
        pred = lambda r, s=sig: ",".join(str(x) for x in r.get("legal", [])) == s
        by_legal[sig] = {
            "common_fai_rows": int(sum(1 for r in common if pred(r))),
            "additive_expected_contribution": _additive_group_contribution(
                rows,
                lambda r, p=pred: r["status"] == "COMMON_FAI" and p(r),
                "expected_fai_delta",
            ),
        }

    path_lengths = sorted(
        {int(r["common_public_action_count"]) for r in common}
    )
    by_path_len = {}
    for plen in path_lengths:
        pred = lambda r, p=plen: int(r.get("common_public_action_count", -1)) == p
        by_path_len[str(plen)] = {
            "common_fai_rows": int(sum(1 for r in common if pred(r))),
            "additive_expected_contribution": _additive_group_contribution(
                rows,
                lambda r, p=pred: r["status"] == "COMMON_FAI" and p(r),
                "expected_fai_delta",
            ),
        }

    expected_conditional = _mean_ci(
        _cluster_seed(common, lambda r: r["expected_fai_delta"])
    )
    sampled_conditional = _mean_ci(
        _cluster_seed(common, lambda r: r["sampled_fai_delta"])
    )
    fold_shift_ci = _mean_ci(
        _cluster_seed(common, lambda r: r["fold_mass_b_minus_a"])
    )
    tv_ci = _mean_ci(_cluster_seed(common, lambda r: r["policy_tv"]))
    spread_max = max(
        (float(r["nonfold_actual_q_spread"]) for r in common),
        default=0.0,
    )

    return {
        "seat_runs": len(rows),
        "scenario_clusters": len({(r["seed"], r["scenario"]) for r in rows}),
        "status_counts": status_counts,
        "common_fai_reach_rate": float(len(common) / max(len(rows), 1)),
        "sampled_fai_divergence_rate_given_common_fai": float(
            sum(1 for r in common if r["sampled_fai_divergence"])
            / max(len(common), 1)
        ),
        "overall_additive_expected_fai_contribution": total_expected,
        "overall_additive_sampled_fai_contribution": total_sampled,
        "sampled_minus_prior_first_divergence_contribution": float(
            total_sampled["mean"] - PRIOR_FAI_CONTRIBUTION
        ),
        "conditional_on_common_fai": {
            "expected_delta_seed_cluster_ci": expected_conditional,
            "sampled_delta_seed_cluster_ci": sampled_conditional,
            "fold_mass_b_minus_a_seed_cluster_ci": fold_shift_ci,
            "policy_tv_seed_cluster_ci": tv_ci,
            "fold_mass_a_mean": float(
                statistics.fmean(r["fold_mass_a"] for r in common)
            ),
            "fold_mass_b_mean": float(
                statistics.fmean(r["fold_mass_b"] for r in common)
            ),
            "max_nonfold_actual_q_spread": float(spread_max),
        },
        "by_fold_shift_direction": by_fold_shift,
        "by_legal_signature": by_legal,
        "by_common_public_action_count": by_path_len,
    }


def main() -> int:
    args = parse_args()
    if args.scenarios_per_seed <= 0 or args.workers <= 0:
        raise SystemExit("positive scenarios/workers required")
    args.report.parent.mkdir(parents=True, exist_ok=True)

    solver_path = args.solver.resolve(strict=True)
    a_path = args.stage_a.resolve(strict=True)
    b_path = args.stage_b.resolve(strict=True)

    snap_a = args.report.parent / "stage_a_hu_models.pt"
    snap_b = args.report.parent / "stage_b_hu_models.pt"
    print("Extracting lightweight HU model snapshots...", flush=True)
    meta_a = overlay._extract_stage_snapshot(a_path, snap_a)
    meta_b = overlay._extract_stage_snapshot(b_path, snap_b)

    tasks = [(int(seed), int(args.scenarios_per_seed)) for seed in FORENSIC_SEEDS]
    all_rows = []
    hu_counts = {}

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers), len(tasks)),
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(str(solver_path), str(snap_a.resolve()), str(snap_b.resolve())),
    ) as pool:
        for seed, hu, rows in pool.map(_process_seed, tasks):
            hu_counts[str(seed)] = int(hu)
            all_rows.extend(rows)
            print(
                f"seed={seed} HU_scenarios={hu} seat_runs={len(rows)} "
                f"common_FAI={sum(1 for r in rows if r['status']=='COMMON_FAI')}",
                flush=True,
            )

    summary = _aggregate(all_rows)

    if abs(
        float(summary["sampled_minus_prior_first_divergence_contribution"])
    ) > 1e-9:
        raise RuntimeError(
            "sampled FAI contribution failed to reproduce prior first-divergence "
            "result: "
            f"delta={summary['sampled_minus_prior_first_divergence_contribution']}"
        )

    report = {
        "schema": "SPINCORE_LT2_JAMMER_FAI_POPULATION_RECONCILIATION_V1",
        "stage_a": {
            "checkpoint": str(a_path),
            "completed_iteration": int(meta_a["completed_iteration"]),
        },
        "stage_b": {
            "checkpoint": str(b_path),
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
            "hu_scenarios_by_seed": hu_counts,
            "population": (
                "all HU Jammer seat-runs; no anchor subsampling; common FAI only "
                "when A/B have not previously sampled different hero actions"
            ),
            "actual_deal_counterfactual": (
                "clone common FAI solver state and apply every legal hero action; "
                "terminal chip delta uses the already dealt hidden hand/full board"
            ),
            "interpretation_limit": (
                "actual-deal Q is not an infoset target per state; population "
                "average is an unbiased counterfactual estimate under the exact "
                "evaluation deal distribution"
            ),
            "paired_sample_reproduction": (
                "same hero RNG as current-behavior first-divergence audit; sampled "
                "FAI contribution must reproduce -6.22672064777328 exactly"
            ),
        },
        "summary": summary,
        "rows": all_rows,
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    e = summary["overall_additive_expected_fai_contribution"]
    s = summary["overall_additive_sampled_fai_contribution"]
    c = summary["conditional_on_common_fai"]
    print("=== JAMMER FAI FULL-POPULATION COUNTERFACTUAL RECONCILIATION ===")
    print(
        f"seat_runs={summary['seat_runs']} "
        f"common_fai_rate={summary['common_fai_reach_rate']:.4f} "
        f"sampled_div_rate|FAI={summary['sampled_fai_divergence_rate_given_common_fai']:.4f}"
    )
    print(
        f"sampled additive FAI={s['mean']:+.6f} "
        f"CI95=[{s['ci95_low']:+.6f},{s['ci95_high']:+.6f}] "
        f"prior_delta={summary['sampled_minus_prior_first_divergence_contribution']:+.3e}"
    )
    print(
        f"expected additive FAI={e['mean']:+.6f} "
        f"CI95=[{e['ci95_low']:+.6f},{e['ci95_high']:+.6f}]"
    )
    print(
        f"conditional expected delta={c['expected_delta_seed_cluster_ci']['mean']:+.3f} "
        f"fold_mass A={c['fold_mass_a_mean']:.4f} "
        f"B={c['fold_mass_b_mean']:.4f} "
        f"B-A={c['fold_mass_b_minus_a_seed_cluster_ci']['mean']:+.4f}"
    )
    for name, block in summary["by_fold_shift_direction"].items():
        x = block["additive_expected_contribution"]
        print(
            f"  {name}: n={block['common_fai_rows']} "
            f"contrib={x['mean']:+.6f} "
            f"CI95=[{x['ci95_low']:+.6f},{x['ci95_high']:+.6f}]"
        )
    print("LT2_JAMMER_FAI_POPULATION_RECONCILIATION_COMPLETE")
    print(f"report={args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
