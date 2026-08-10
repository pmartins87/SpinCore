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
    HISTORICAL_PARAMS_PER_NETWORK,
    hu_episode,
    make_bundle,
    shared_cross_seed_observations,
)
from run_r7_3_variance_decomposition import _advantage_fit_nrmse, _finite


DEFAULT_SEEDS = (20260829, 20260807)
PAYOUT = (0.5, 0.3, 0.2)
ADV_RNG_XOR = 0x0A9D7A61
STRATEGY_RNG_XOR = 0x057A7E61


def _fit_pass(value: float, gate_name: str) -> bool:
    return _finite(value) and float(value) <= float(FROZEN_GATES[gate_name])


def _unique_deck_seed(algorithm_seed: int, iteration: int, root_index: int) -> int:
    # Match the recovered acceptance runner's independent per-algorithm-seed
    # hidden-deal schedule. Replicates reuse the same deal and differ only in
    # sampled action trajectories.
    return (
        (int(algorithm_seed) << 32) ^ (int(iteration) << 16) ^ int(root_index)
    ) & ((1 << 64) - 1)


def _collect_replicated_root(
    *,
    session: DeepCFRDomainSession,
    solver: SolverLibrary,
    episode,
    deck_seed: int,
    iteration: int,
    advantage_replicates: int,
    strategy_replicates: int,
    advantage_rng,
    strategy_rng,
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
    counters["roots"] += 1  # unique deals, not trajectory count
    counters["nodes"] += int(nodes)
    counters["advantage_samples"] += int(advantage_added)
    counters["strategy_samples"] += int(strategy_added)


def _fit_policy(
    *,
    bundle,
    session,
    seed: int,
    device: str,
    chunk_steps: int,
    max_steps: int,
    fit_target: float,
    batch_size: int,
    audit_size: int,
):
    progress = []
    audit_seed = int(seed) ^ 0x13579BDF
    while int(bundle.counters["policy_optimizer_steps"]) < int(max_steps):
        remaining = int(max_steps) - int(bundle.counters["policy_optimizer_steps"])
        steps = min(int(chunk_steps), remaining)
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
            "fit_target_reached": _finite(value) and value <= float(fit_target),
        }
        progress.append(row)
        print(json.dumps({"policy": row, "seed": int(seed)}, sort_keys=True), flush=True)
        if row["fit_target_reached"]:
            break
    return progress


def run_seed(
    *,
    seed: int,
    solver: SolverLibrary,
    device: str,
    iterations: int,
    roots_per_iteration: int,
    advantage_replicates: int,
    strategy_replicates: int,
    rng_contract: str,
    advantage_chunk_steps: int,
    advantage_max_steps_per_iteration: int,
    advantage_fit_target: float,
    policy_chunk_steps: int,
    policy_max_steps: int,
    policy_fit_target: float,
    batch_size: int,
    audit_size: int,
    reservoir_capacity: int,
    lr: float,
):
    bundle = make_bundle(
        int(seed),
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

    if rng_contract == "separate":
        advantage_rng = random.Random(int(seed) ^ ADV_RNG_XOR)
        strategy_rng = random.Random(int(seed) ^ STRATEGY_RNG_XOR)
    elif rng_contract == "coupled":
        advantage_rng = bundle.batch_rng
        strategy_rng = bundle.batch_rng
    else:
        raise ValueError(rng_contract)

    episode = hu_episode()
    checkpoints = []
    for iteration in range(1, int(iterations) + 1):
        for root_index in range(int(roots_per_iteration)):
            _collect_replicated_root(
                session=session,
                solver=solver,
                episode=episode,
                deck_seed=_unique_deck_seed(seed, iteration, root_index),
                iteration=iteration,
                advantage_replicates=int(advantage_replicates),
                strategy_replicates=int(strategy_replicates),
                advantage_rng=advantage_rng,
                strategy_rng=strategy_rng,
            )

        reset_seed = (int(seed) ^ (iteration * 0x9E3779B1)) & 0x7FFFFFFF
        session.reset_advantage_network(init_seed=reset_seed, lr=lr)
        local_steps = 0
        progress = []
        advantage_audit_seed = int(seed) ^ (iteration * 0x45D9F3B)
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
                seed=advantage_audit_seed,
                device=device,
            )
            row = {
                "optimizer_steps": int(local_steps),
                "weighted_nrmse": float(nrmse),
                "frozen_gate_pass": _fit_pass(nrmse, "advantage_weighted_nrmse_max"),
                "fit_target_reached": _finite(nrmse)
                and float(nrmse) <= float(advantage_fit_target),
            }
            progress.append(row)
            print(
                json.dumps(
                    {
                        "seed": int(seed),
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
                "advantage_progress": progress,
                "final_advantage_fit": progress[-1],
            }
        )

    policy_progress = _fit_policy(
        bundle=bundle,
        session=session,
        seed=int(seed),
        device=device,
        chunk_steps=int(policy_chunk_steps),
        max_steps=int(policy_max_steps),
        fit_target=float(policy_fit_target),
        batch_size=int(batch_size),
        audit_size=int(audit_size),
    )

    final_fit = audit_model_fit(
        bundle,
        sample_size=max(int(audit_size), 2048),
        seed=int(seed) ^ 0x2468ACE0,
        device=device,
    )
    report = {
        "algorithm_seed": int(seed),
        "unique_roots": int(bundle.counters["roots"]),
        "advantage_replicates_per_traverser_per_unique_root": int(advantage_replicates),
        "strategy_replicates_per_target_player_per_unique_root": int(strategy_replicates),
        "advantage_samples": len(bundle.adv_mem.items),
        "advantage_seen": int(bundle.adv_mem.seen),
        "strategy_samples": len(bundle.pol_mem.items),
        "strategy_seen": int(bundle.pol_mem.seen),
        "nodes": int(bundle.counters["nodes"]),
        "advantage_optimizer_steps": int(bundle.counters["adv_optimizer_steps"]),
        "policy_optimizer_steps": int(bundle.counters["policy_optimizer_steps"]),
        "checkpoints": checkpoints,
        "policy_progress": policy_progress,
        "final_fit": {
            "advantage_weighted_nrmse": float(final_fit["advantage_weighted_nrmse"]),
            "policy_weighted_mean_tv": float(final_fit["policy_weighted_mean_tv"]),
            "advantage_gate_pass": _fit_pass(
                final_fit["advantage_weighted_nrmse"], "advantage_weighted_nrmse_max"
            ),
            "policy_gate_pass": _fit_pass(
                final_fit["policy_weighted_mean_tv"], "policy_weighted_mean_tv_max"
            ),
        },
    }
    return bundle, report


def main() -> int:
    ap = argparse.ArgumentParser(description="R7.3 configurable replicated-path 640 candidate")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("validation/R7_3_REPLICATED_640_CANDIDATE.json"),
    )
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--roots-per-iteration", type=int, default=128)
    ap.add_argument("--advantage-replicates", type=int, default=1)
    ap.add_argument("--strategy-replicates", type=int, default=4)
    ap.add_argument("--rng-contract", choices=("separate", "coupled"), default="separate")
    ap.add_argument("--advantage-chunk-steps", type=int, default=256)
    ap.add_argument("--advantage-max-steps-per-iteration", type=int, default=4096)
    ap.add_argument("--advantage-fit-target", type=float, default=0.50)
    ap.add_argument("--policy-chunk-steps", type=int, default=256)
    ap.add_argument("--policy-max-steps", type=int, default=32768)
    ap.add_argument("--policy-fit-target", type=float, default=0.105)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=1024)
    ap.add_argument("--cross-seed-per-seed", type=int, default=1024)
    ap.add_argument("--reservoir-capacity", type=int, default=400000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if len(seeds) != 2:
        raise SystemExit("replicated candidate requires exactly two algorithm seeds")
    if args.advantage_replicates <= 0 or args.strategy_replicates <= 0:
        raise SystemExit("replicate counts must be positive")

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()

    bundles = []
    reports = []
    for seed in seeds:
        bundle, report = run_seed(
            seed=int(seed),
            solver=solver,
            device=args.device,
            iterations=int(args.iterations),
            roots_per_iteration=int(args.roots_per_iteration),
            advantage_replicates=int(args.advantage_replicates),
            strategy_replicates=int(args.strategy_replicates),
            rng_contract=str(args.rng_contract),
            advantage_chunk_steps=int(args.advantage_chunk_steps),
            advantage_max_steps_per_iteration=int(args.advantage_max_steps_per_iteration),
            advantage_fit_target=float(args.advantage_fit_target),
            policy_chunk_steps=int(args.policy_chunk_steps),
            policy_max_steps=int(args.policy_max_steps),
            policy_fit_target=float(args.policy_fit_target),
            batch_size=int(args.batch_size),
            audit_size=int(args.audit_size),
            reservoir_capacity=int(args.reservoir_capacity),
            lr=float(args.lr),
        )
        bundles.append(bundle)
        reports.append(report)

    observations = shared_cross_seed_observations(
        bundles,
        per_seed=int(args.cross_seed_per_seed),
        seed=0x715EED,
    )
    cross = cross_seed_policy_tv(
        bundles[0].policy,
        bundles[1].policy,
        observations,
        device=args.device,
    )
    per_seed_fit_pass = all(
        bool(row["final_fit"]["advantage_gate_pass"])
        and bool(row["final_fit"]["policy_gate_pass"])
        for row in reports
    )
    cross_seed_pass = (
        _finite(cross["mean_tv"])
        and _finite(cross["p95_tv"])
        and float(cross["mean_tv"]) <= FROZEN_GATES["cross_seed_mean_tv_max"]
        and float(cross["p95_tv"]) <= FROZEN_GATES["cross_seed_p95_tv_max"]
    )
    passed = bool(per_seed_fit_pass and cross_seed_pass)

    params = sum(p.numel() for p in bundles[0].advantage.parameters())
    payload = {
        "schema": "SPINCORE_R7_3_REPLICATED_640_CANDIDATE_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "algorithm_seeds": seeds,
        "unique_roots_per_seed": int(args.iterations * args.roots_per_iteration),
        "iterations": int(args.iterations),
        "roots_per_iteration": int(args.roots_per_iteration),
        "sampling_schedule": {
            "advantage_replicates": int(args.advantage_replicates),
            "strategy_replicates": int(args.strategy_replicates),
            "rng_contract": str(args.rng_contract),
            "deck_semantics": "INDEPENDENT_BY_ALGORITHM_SEED_AS_ACCEPTANCE_RUN",
        },
        "fit_schedule": {
            "advantage_fit_target": float(args.advantage_fit_target),
            "advantage_max_steps_per_iteration": int(args.advantage_max_steps_per_iteration),
            "policy_fit_target": float(args.policy_fit_target),
            "policy_max_steps": int(args.policy_max_steps),
            "reservoir_capacity": int(args.reservoir_capacity),
        },
        "network": {
            "trainable_params": int(params),
            "historical_recorded_params": int(HISTORICAL_PARAMS_PER_NETWORK),
            "delta_from_historical": int(params - HISTORICAL_PARAMS_PER_NETWORK),
        },
        "per_seed": reports,
        "cross_seed_observation_count": len(observations),
        "cross_seed": {k: float(v) for k, v in cross.items()},
        "frozen_gates": dict(FROZEN_GATES),
        "per_seed_fit_pass": bool(per_seed_fit_pass),
        "cross_seed_pass": bool(cross_seed_pass),
        "r7_3_pass": bool(passed),
        "acceptance_gate_changed": False,
        "production_contract_changed": bool(
            int(args.advantage_replicates) != 1
            or int(args.strategy_replicates) != 1
            or str(args.rng_contract) != "coupled"
        ),
        "promotion_note": (
            "This is an acceptance-scale experimental candidate. A successful result does not "
            "silently promote the sampling/RNG schedule to production: any changed RNG stream or "
            "replication schedule must be versioned in checkpoint/resume semantics and recertified "
            "before R7.3 can be closed."
        ),
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)

    if args.strict and not passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
