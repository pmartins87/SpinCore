from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility
from spincore.r7 import FROZEN_GATES, audit_model_fit, cross_seed_policy_tv
from spincore.solver import SolverLibrary

from run_r7_3_diagnostic import (
    hu_episode,
    make_bundle,
    shared_cross_seed_observations,
)
from run_r7_3_support_overlap import _aggregate_strategy, _intersection_metrics, _make_keyer
from run_r7_3_variance_decomposition import _advantage_fit_nrmse, _finite


DEFAULT_SEEDS = (20260829, 20260807)
DEFAULT_SHARED_DECK_STREAM_SEED = 0xD3C5EED
PAYOUT = (0.5, 0.3, 0.2)
ADV_RNG_XOR = 0xA9D7A61
STRATEGY_RNG_XOR = 0x57A7E61

MODE_SPECS = {
    "baseline": (1, 1),
    "strategy_x4": (1, 4),
    "advantage_x4": (4, 1),
    "both_x4": (4, 4),
}


def _fit_pass(value: float, gate_name: str) -> bool:
    return _finite(value) and float(value) <= float(FROZEN_GATES[gate_name])


def _safe_ratio(a: float, b: float) -> float:
    return float(a) / max(float(b), 1e-12)


def _iteration_items(items, iteration: int):
    return [item for item in items if int(item.iteration) == int(iteration)]


def _support_metrics(items_a, items_b) -> dict:
    out = {}
    for mode in ("raw", "poker_isomorphic"):
        keyer = _make_keyer(mode)
        agg_a = _aggregate_strategy(items_a, keyer)
        agg_b = _aggregate_strategy(items_b, keyer)
        out[mode] = _intersection_metrics(agg_a, agg_b)
    return out


def _collect_replicated_root(
    *,
    session: DeepCFRDomainSession,
    solver: SolverLibrary,
    episode,
    deck_seed: int,
    iteration: int,
    advantage_replicates: int,
    strategy_replicates: int,
    advantage_rng: random.Random,
    strategy_rng: random.Random,
):
    live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
    nodes = 0
    advantage_added = 0
    strategy_added = 0

    session.collector.rng = advantage_rng
    for player in live:
        for _ in range(int(advantage_replicates)):
            root = solver.create(episode, int(deck_seed))
            try:
                result = session.collector.collect_advantage(
                    root,
                    traverser=player,
                    iteration=int(iteration),
                )
            finally:
                root.close()
            nodes += int(result.nodes)
            advantage_added += int(result.samples_added)

    session.collector.rng = strategy_rng
    for player in live:
        for _ in range(int(strategy_replicates)):
            root = solver.create(episode, int(deck_seed))
            try:
                strategy_added += int(
                    session.collector.collect_strategy_own_reach(
                        root,
                        target_player=player,
                        iteration=int(iteration),
                    )
                )
            finally:
                root.close()

    counters = session.bundle.counters
    counters["iteration"] = max(int(counters["iteration"]), int(iteration))
    # A root here means one unique hidden deal. Replicates are reported separately.
    counters["roots"] += 1
    counters["nodes"] += int(nodes)
    counters["advantage_samples"] += int(advantage_added)
    counters["strategy_samples"] += int(strategy_added)

    return {
        "nodes": int(nodes),
        "advantage_samples": int(advantage_added),
        "strategy_samples": int(strategy_added),
    }


def _fit_average_policy(
    *,
    bundle,
    session: DeepCFRDomainSession,
    seed: int,
    device: str,
    policy_chunk_steps: int,
    policy_max_steps: int,
    policy_fit_target: float,
    batch_size: int,
    audit_size: int,
):
    progress = []
    audit_seed = int(seed) ^ 0x13579BDF
    while int(bundle.counters["policy_optimizer_steps"]) < int(policy_max_steps):
        remaining = int(policy_max_steps) - int(bundle.counters["policy_optimizer_steps"])
        steps = min(int(policy_chunk_steps), remaining)
        session.train_average_policy(steps=steps, batch_size=int(batch_size))
        fit = audit_model_fit(
            bundle,
            sample_size=int(audit_size),
            seed=audit_seed,
            device=device,
        )
        value = float(fit["policy_weighted_mean_tv"])
        row = {
            "optimizer_steps": int(bundle.counters["policy_optimizer_steps"]),
            "weighted_mean_tv": value,
            "frozen_gate_pass": _fit_pass(value, "policy_weighted_mean_tv_max"),
            "fit_target_reached": _finite(value) and value <= float(policy_fit_target),
        }
        progress.append(row)
        if row["fit_target_reached"]:
            break

    final_fit = audit_model_fit(
        bundle,
        sample_size=max(int(audit_size), 2048),
        seed=int(seed) ^ 0x2468ACE0,
        device=device,
    )
    return progress, {
        "advantage_weighted_nrmse": float(final_fit["advantage_weighted_nrmse"]),
        "policy_weighted_mean_tv": float(final_fit["policy_weighted_mean_tv"]),
        "advantage_gate_pass": _fit_pass(
            final_fit["advantage_weighted_nrmse"], "advantage_weighted_nrmse_max"
        ),
        "policy_gate_pass": _fit_pass(
            final_fit["policy_weighted_mean_tv"], "policy_weighted_mean_tv_max"
        ),
    }


def run_mode(
    *,
    mode_name: str,
    advantage_replicates: int,
    strategy_replicates: int,
    seeds: list[int],
    deck_stream_seed: int,
    solver: SolverLibrary,
    device: str,
    iterations: int,
    roots_per_iteration: int,
    advantage_chunk_steps: int,
    advantage_max_steps_per_iteration: int,
    advantage_fit_target: float,
    policy_chunk_steps: int,
    policy_max_steps: int,
    policy_fit_target: float,
    batch_size: int,
    audit_size: int,
    cross_seed_per_seed: int,
    reservoir_capacity: int,
    lr: float,
):
    episode = hu_episode()
    bundles = []
    sessions = []
    reports = []

    for algorithm_seed in seeds:
        bundle = make_bundle(
            int(algorithm_seed),
            device=device,
            reservoir_capacity=int(reservoir_capacity),
            lr=float(lr),
        )
        session = DeepCFRDomainSession(
            solver_library=solver,
            bundle=bundle,
            terminal_utility=icm_delta_utility(PAYOUT),
            device=device,
        )
        advantage_rng = random.Random(int(algorithm_seed) ^ ADV_RNG_XOR)
        strategy_rng = random.Random(int(algorithm_seed) ^ STRATEGY_RNG_XOR)
        global_root = 0
        checkpoints = []

        for iteration in range(1, int(iterations) + 1):
            for _ in range(int(roots_per_iteration)):
                deck_seed = (
                    int(deck_stream_seed) * 1_000_003 + global_root * 97 + iteration
                ) & ((1 << 64) - 1)
                _collect_replicated_root(
                    session=session,
                    solver=solver,
                    episode=episode,
                    deck_seed=deck_seed,
                    iteration=iteration,
                    advantage_replicates=int(advantage_replicates),
                    strategy_replicates=int(strategy_replicates),
                    advantage_rng=advantage_rng,
                    strategy_rng=strategy_rng,
                )
                global_root += 1

            reset_seed = (int(algorithm_seed) ^ (iteration * 0x9E3779B1)) & 0x7FFFFFFF
            session.reset_advantage_network(init_seed=reset_seed, lr=lr)
            local_steps = 0
            advantage_progress = []
            audit_seed = int(algorithm_seed) ^ (iteration * 0x45D9F3B)
            while local_steps < int(advantage_max_steps_per_iteration):
                steps = min(
                    int(advantage_chunk_steps),
                    int(advantage_max_steps_per_iteration) - local_steps,
                )
                session.train_advantage(steps=steps, batch_size=int(batch_size))
                local_steps += steps
                nrmse = _advantage_fit_nrmse(
                    bundle,
                    sample_size=int(audit_size),
                    seed=audit_seed,
                    device=device,
                )
                row = {
                    "optimizer_steps": int(local_steps),
                    "weighted_nrmse": float(nrmse),
                    "frozen_gate_pass": _fit_pass(nrmse, "advantage_weighted_nrmse_max"),
                    "fit_target_reached": _finite(nrmse)
                    and float(nrmse) <= float(advantage_fit_target),
                }
                advantage_progress.append(row)
                print(
                    json.dumps(
                        {
                            "mode": mode_name,
                            "seed": int(algorithm_seed),
                            "iteration": int(iteration),
                            "unique_roots": int(bundle.counters["roots"]),
                            "advantage": row,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                if row["fit_target_reached"]:
                    break

            checkpoints.append(
                {
                    "iteration": int(iteration),
                    "unique_roots": int(bundle.counters["roots"]),
                    "advantage_samples": len(bundle.adv_mem.items),
                    "advantage_seen": int(bundle.adv_mem.seen),
                    "strategy_samples": len(bundle.pol_mem.items),
                    "strategy_seen": int(bundle.pol_mem.seen),
                    "advantage_progress": advantage_progress,
                    "final_advantage_fit": advantage_progress[-1],
                }
            )

        policy_progress, final_fit = _fit_average_policy(
            bundle=bundle,
            session=session,
            seed=int(algorithm_seed),
            device=device,
            policy_chunk_steps=int(policy_chunk_steps),
            policy_max_steps=int(policy_max_steps),
            policy_fit_target=float(policy_fit_target),
            batch_size=int(batch_size),
            audit_size=int(audit_size),
        )

        bundles.append(bundle)
        sessions.append(session)
        reports.append(
            {
                "algorithm_seed": int(algorithm_seed),
                "advantage_replicates_per_unique_root": int(advantage_replicates),
                "strategy_replicates_per_unique_root": int(strategy_replicates),
                "unique_roots": int(bundle.counters["roots"]),
                "advantage_samples": len(bundle.adv_mem.items),
                "advantage_seen": int(bundle.adv_mem.seen),
                "strategy_samples": len(bundle.pol_mem.items),
                "strategy_seen": int(bundle.pol_mem.seen),
                "nodes": int(bundle.counters["nodes"]),
                "advantage_optimizer_steps": int(bundle.counters["adv_optimizer_steps"]),
                "policy_optimizer_steps": int(bundle.counters["policy_optimizer_steps"]),
                "checkpoints": checkpoints,
                "policy_progress": policy_progress,
                "final_fit": final_fit,
            }
        )

    cumulative = _support_metrics(bundles[0].pol_mem.items, bundles[1].pol_mem.items)
    by_iteration = {}
    for iteration in range(1, int(iterations) + 1):
        by_iteration[str(iteration)] = _support_metrics(
            _iteration_items(bundles[0].pol_mem.items, iteration),
            _iteration_items(bundles[1].pol_mem.items, iteration),
        )

    internal_obs = shared_cross_seed_observations(
        bundles,
        per_seed=int(cross_seed_per_seed),
        seed=0x715EED,
    )
    internal_cross = cross_seed_policy_tv(
        bundles[0].policy,
        bundles[1].policy,
        internal_obs,
        device=device,
    )

    return bundles, {
        "mode": mode_name,
        "advantage_replicates_per_unique_root": int(advantage_replicates),
        "strategy_replicates_per_unique_root": int(strategy_replicates),
        "collection": reports,
        "support_cumulative": cumulative,
        "support_by_iteration": by_iteration,
        "cross_seed_internal_corpus": {k: float(v) for k, v in internal_cross.items()},
        "cross_seed_internal_observation_count": len(internal_obs),
        "all_fit_gates_pass": all(
            bool(r["final_fit"]["advantage_gate_pass"])
            and bool(r["final_fit"]["policy_gate_pass"])
            for r in reports
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "R7.3 decomposition of external-sampling path variance by independently "
            "replicating advantage traversals and own-reach average-policy traversals"
        )
    )
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("validation/R7_3_PATH_REPLICATION_SCREEN_256.json"),
    )
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--modes", default=",".join(MODE_SPECS))
    ap.add_argument("--deck-stream-seed", type=int, default=DEFAULT_SHARED_DECK_STREAM_SEED)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--roots-per-iteration", type=int, default=128)
    ap.add_argument("--advantage-chunk-steps", type=int, default=256)
    ap.add_argument("--advantage-max-steps-per-iteration", type=int, default=4096)
    ap.add_argument("--advantage-fit-target", type=float, default=0.50)
    ap.add_argument("--policy-chunk-steps", type=int, default=256)
    ap.add_argument("--policy-max-steps", type=int, default=16384)
    ap.add_argument("--policy-fit-target", type=float, default=0.105)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=512)
    ap.add_argument("--cross-seed-per-seed", type=int, default=1024)
    ap.add_argument("--reservoir-capacity", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if len(seeds) != 2:
        raise SystemExit("path-replication screen requires exactly two algorithm seeds")
    modes = [x.strip() for x in str(args.modes).split(",") if x.strip()]
    if not modes:
        raise SystemExit("at least one mode is required")
    unknown = [x for x in modes if x not in MODE_SPECS]
    if unknown:
        raise SystemExit(f"unknown modes: {unknown}")
    if "baseline" not in modes:
        raise SystemExit("baseline mode is required for controlled ratios")

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()

    mode_bundles = {}
    mode_reports = {}
    for mode_name in modes:
        adv_reps, strat_reps = MODE_SPECS[mode_name]
        bundles, report = run_mode(
            mode_name=mode_name,
            advantage_replicates=adv_reps,
            strategy_replicates=strat_reps,
            seeds=seeds,
            deck_stream_seed=int(args.deck_stream_seed),
            solver=solver,
            device=args.device,
            iterations=int(args.iterations),
            roots_per_iteration=int(args.roots_per_iteration),
            advantage_chunk_steps=int(args.advantage_chunk_steps),
            advantage_max_steps_per_iteration=int(args.advantage_max_steps_per_iteration),
            advantage_fit_target=float(args.advantage_fit_target),
            policy_chunk_steps=int(args.policy_chunk_steps),
            policy_max_steps=int(args.policy_max_steps),
            policy_fit_target=float(args.policy_fit_target),
            batch_size=int(args.batch_size),
            audit_size=int(args.audit_size),
            cross_seed_per_seed=int(args.cross_seed_per_seed),
            reservoir_capacity=int(args.reservoir_capacity),
            lr=float(args.lr),
        )
        mode_bundles[mode_name] = bundles
        mode_reports[mode_name] = report

    baseline_bundles = mode_bundles["baseline"]
    baseline_obs = shared_cross_seed_observations(
        baseline_bundles,
        per_seed=int(args.cross_seed_per_seed),
        seed=0x715EED,
    )
    for mode_name in modes:
        bundles = mode_bundles[mode_name]
        on_baseline = cross_seed_policy_tv(
            bundles[0].policy,
            bundles[1].policy,
            baseline_obs,
            device=args.device,
        )
        mode_reports[mode_name]["cross_seed_on_baseline_corpus"] = {
            k: float(v) for k, v in on_baseline.items()
        }

    baseline = mode_reports["baseline"]
    baseline_support = baseline["support_by_iteration"][str(int(args.iterations))][
        "poker_isomorphic"
    ]
    baseline_cross = baseline["cross_seed_internal_corpus"]
    baseline_j = float(baseline_support["jaccard"])
    baseline_mass = 0.5 * (
        float(baseline_support["lcfr_weight_coverage_A"])
        + float(baseline_support["lcfr_weight_coverage_B"])
    )
    baseline_target_tv = float(baseline_support["shared_target_weighted_mean_tv"])
    baseline_mean_tv = float(baseline_cross["mean_tv"])
    baseline_p95_tv = float(baseline_cross["p95_tv"])

    comparisons = {}
    for mode_name in modes:
        if mode_name == "baseline":
            continue
        report = mode_reports[mode_name]
        support = report["support_by_iteration"][str(int(args.iterations))][
            "poker_isomorphic"
        ]
        cross = report["cross_seed_internal_corpus"]
        mass = 0.5 * (
            float(support["lcfr_weight_coverage_A"])
            + float(support["lcfr_weight_coverage_B"])
        )
        comparisons[mode_name] = {
            "jaccard_ratio_to_baseline": _safe_ratio(float(support["jaccard"]), baseline_j),
            "lcfr_weight_coverage_ratio_to_baseline": _safe_ratio(mass, baseline_mass),
            "shared_target_tv_ratio_to_baseline": _safe_ratio(
                float(support["shared_target_weighted_mean_tv"]), baseline_target_tv
            )
            if math.isfinite(baseline_target_tv)
            else math.inf,
            "cross_seed_mean_tv_ratio_to_baseline": _safe_ratio(
                float(cross["mean_tv"]), baseline_mean_tv
            ),
            "cross_seed_p95_tv_ratio_to_baseline": _safe_ratio(
                float(cross["p95_tv"]), baseline_p95_tv
            ),
        }

    strategy_ratio = comparisons.get("strategy_x4", {}).get(
        "cross_seed_mean_tv_ratio_to_baseline", math.inf
    )
    advantage_ratio = comparisons.get("advantage_x4", {}).get(
        "cross_seed_mean_tv_ratio_to_baseline", math.inf
    )
    both_ratio = comparisons.get("both_x4", {}).get(
        "cross_seed_mean_tv_ratio_to_baseline", math.inf
    )

    if strategy_ratio <= 0.85 and strategy_ratio <= advantage_ratio:
        diagnosis = "AVERAGE_POLICY_OWN_REACH_SAMPLING_VARIANCE_MATERIAL"
    elif advantage_ratio <= 0.85 and advantage_ratio < strategy_ratio:
        diagnosis = "ADVANTAGE_EXTERNAL_SAMPLING_VARIANCE_MATERIAL"
    elif both_ratio <= 0.85:
        diagnosis = "COMBINED_PATH_REPLICATION_VARIANCE_MATERIAL"
    else:
        diagnosis = "PATH_REPLICATION_NOT_DOMINANT_AT_SCREEN_SCALE"

    # Controlled invariant: baseline and strategy_x4 use identical advantage
    # traversals/training. Their final checkpoint NRMSEs should therefore match.
    advantage_control_deltas = []
    if "strategy_x4" in mode_reports:
        for i in range(2):
            base_rows = mode_reports["baseline"]["collection"][i]["checkpoints"]
            strat_rows = mode_reports["strategy_x4"]["collection"][i]["checkpoints"]
            for b, s in zip(base_rows, strat_rows):
                advantage_control_deltas.append(
                    abs(
                        float(b["final_advantage_fit"]["weighted_nrmse"])
                        - float(s["final_advantage_fit"]["weighted_nrmse"])
                    )
                )
    max_advantage_control_delta = max(advantage_control_deltas, default=0.0)

    payload = {
        "schema": "SPINCORE_R7_3_PATH_REPLICATION_SCREEN_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "algorithm_seeds": seeds,
        "shared_deck_stream_seed": int(args.deck_stream_seed),
        "iterations": int(args.iterations),
        "unique_roots_per_iteration": int(args.roots_per_iteration),
        "unique_roots_per_seed": int(args.iterations * args.roots_per_iteration),
        "modes": mode_reports,
        "mode_specs": {
            name: {
                "advantage_replicates_per_unique_root": MODE_SPECS[name][0],
                "strategy_replicates_per_unique_root": MODE_SPECS[name][1],
            }
            for name in modes
        },
        "fit_schedule": {
            "advantage_fit_target": float(args.advantage_fit_target),
            "advantage_max_steps_per_iteration": int(args.advantage_max_steps_per_iteration),
            "policy_fit_target": float(args.policy_fit_target),
            "policy_max_steps": int(args.policy_max_steps),
        },
        "comparisons_to_baseline": comparisons,
        "summary": {
            "baseline_iteration2_poker_isomorphic_jaccard": baseline_j,
            "baseline_iteration2_mean_lcfr_weight_coverage": baseline_mass,
            "baseline_iteration2_shared_target_weighted_mean_tv": baseline_target_tv,
            "baseline_cross_seed_mean_tv": baseline_mean_tv,
            "baseline_cross_seed_p95_tv": baseline_p95_tv,
            "max_baseline_vs_strategy_x4_advantage_checkpoint_nrmse_delta": float(
                max_advantage_control_delta
            ),
            "diagnosis": diagnosis,
        },
        "interpretation_note": (
            "Diagnostic only. All modes use the same unique root-deck stream. Advantage, "
            "strategy and optimizer RNG streams are separated so strategy_x4 can vary only "
            "average-policy own-reach sampling density while keeping CFR advantage dynamics "
            "controlled. Replication preserves stochastic own-reach/external-sampling semantics; "
            "it only increases independent trajectories per unique deal. Production semantics, "
            "checkpoint schema and frozen acceptance gates are unchanged by this screen."
        ),
        "acceptance_gate_changed": False,
        "production_sampling_schedule_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
