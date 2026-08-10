from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility
from spincore.r7 import stratified_audit_indices, cross_seed_policy_tv
from spincore.solver import SolverLibrary
from spincore_nn.codec import collate_inputs, decode_spnniv1

from run_r7_3_diagnostic import hu_episode, make_bundle, shared_cross_seed_observations
from run_r7_3_path_replication_screen import (
    ADV_RNG_XOR,
    STRATEGY_RNG_XOR,
    PAYOUT,
    _collect_replicated_root,
    _fit_average_policy,
    _fit_pass,
    _support_metrics,
)
from run_r7_3_variance_decomposition import _advantage_fit_nrmse, _finite


DEFAULT_SEEDS = (20260829, 20260807)
DEFAULT_SHARED_DECK_STREAM_SEED = 0xD3C5EED
MODE_REPS = {"baseline": 1, "advantage_x4": 4}


def _behavior_probabilities(model, observations: list[bytes], device: str) -> torch.Tensor:
    if not observations:
        return torch.empty((0, 6), dtype=torch.float32)
    batch = collate_inputs([decode_spnniv1(x) for x in observations], device=device)
    model.eval()
    with torch.no_grad():
        raw = model(batch)
    legal = batch["legal"]
    positive = torch.relu(raw) * legal.float()
    total = positive.sum(dim=1, keepdim=True)
    legal_count = legal.float().sum(dim=1, keepdim=True).clamp_min(1.0)
    uniform = legal.float() / legal_count
    policy = torch.where(total > 0.0, positive / total.clamp_min(1e-12), uniform)
    return policy.detach().cpu()


def _behavior_tv(model_a, model_b, observations: list[bytes], device: str) -> dict[str, float]:
    a = _behavior_probabilities(model_a, observations, device)
    b = _behavior_probabilities(model_b, observations, device)
    if len(a) == 0:
        return {"mean_tv": math.inf, "p50_tv": math.inf, "p95_tv": math.inf, "max_tv": math.inf}
    tv = 0.5 * torch.abs(a - b).sum(dim=1)
    q = torch.quantile(tv, torch.tensor([0.5, 0.95]))
    return {
        "mean_tv": float(tv.mean()),
        "p50_tv": float(q[0]),
        "p95_tv": float(q[1]),
        "max_tv": float(tv.max()),
    }


def _iteration_items(items, iteration: int):
    return [item for item in items if int(item.iteration) == int(iteration)]


def _common_iteration_observations(bundles, iteration: int, per_seed: int, seed: int) -> list[bytes]:
    out: list[bytes] = []
    for j, bundle in enumerate(bundles):
        items = _iteration_items(bundle.pol_mem.items, iteration)
        idx = stratified_audit_indices(len(items), int(per_seed), int(seed) ^ (j * 0x9E3779B9))
        out.extend(items[i].observation for i in idx)
    return out


def _fit_advantage_iteration(
    *,
    session,
    bundle,
    algorithm_seed: int,
    iteration: int,
    lr: float,
    chunk_steps: int,
    max_steps: int,
    fit_target: float,
    batch_size: int,
    audit_size: int,
    device: str,
):
    reset_seed = (int(algorithm_seed) ^ (int(iteration) * 0x9E3779B1)) & 0x7FFFFFFF
    session.reset_advantage_network(init_seed=reset_seed, lr=lr)
    local_steps = 0
    progress = []
    audit_seed = int(algorithm_seed) ^ (int(iteration) * 0x45D9F3B)
    while local_steps < int(max_steps):
        steps = min(int(chunk_steps), int(max_steps) - local_steps)
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
            "fit_target_reached": _finite(nrmse) and float(nrmse) <= float(fit_target),
        }
        progress.append(row)
        if row["fit_target_reached"]:
            break
    return progress


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Measure cross-seed regret-policy divergence after every CFR iteration, "
            "separating short-screen variance reduction from five-iteration compounding"
        )
    )
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=tuple(MODE_REPS), required=True)
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--deck-stream-seed", type=int, default=DEFAULT_SHARED_DECK_STREAM_SEED)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--roots-per-iteration", type=int, default=128)
    ap.add_argument("--advantage-chunk-steps", type=int, default=256)
    ap.add_argument("--advantage-max-steps-per-iteration", type=int, default=4096)
    ap.add_argument("--advantage-fit-target", type=float, default=0.50)
    ap.add_argument("--policy-chunk-steps", type=int, default=256)
    ap.add_argument("--policy-max-steps", type=int, default=16384)
    ap.add_argument("--policy-fit-target", type=float, default=0.105)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=512)
    ap.add_argument("--iteration-cross-per-seed", type=int, default=512)
    ap.add_argument("--final-cross-per-seed", type=int, default=1024)
    ap.add_argument("--reservoir-capacity", type=int, default=400000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if len(seeds) != 2:
        raise SystemExit("iteration-compounding diagnostic requires exactly two algorithm seeds")

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()
    episode = hu_episode()
    reps = int(MODE_REPS[args.mode])

    bundles = []
    sessions = []
    advantage_rngs = []
    strategy_rngs = []
    global_roots = [0, 0]
    seed_reports = []

    for algorithm_seed in seeds:
        bundle = make_bundle(
            int(algorithm_seed),
            device=args.device,
            reservoir_capacity=int(args.reservoir_capacity),
            lr=float(args.lr),
        )
        session = DeepCFRDomainSession(
            solver_library=solver,
            bundle=bundle,
            terminal_utility=icm_delta_utility(PAYOUT),
            device=args.device,
        )
        bundles.append(bundle)
        sessions.append(session)
        advantage_rngs.append(random.Random(int(algorithm_seed) ^ ADV_RNG_XOR))
        strategy_rngs.append(random.Random(int(algorithm_seed) ^ STRATEGY_RNG_XOR))
        seed_reports.append({"algorithm_seed": int(algorithm_seed), "checkpoints": []})

    iteration_reports = []
    for iteration in range(1, int(args.iterations) + 1):
        for seed_index, algorithm_seed in enumerate(seeds):
            bundle = bundles[seed_index]
            session = sessions[seed_index]
            for _ in range(int(args.roots_per_iteration)):
                global_root = global_roots[seed_index]
                deck_seed = (
                    int(args.deck_stream_seed) * 1_000_003 + global_root * 97 + iteration
                ) & ((1 << 64) - 1)
                _collect_replicated_root(
                    session=session,
                    solver=solver,
                    episode=episode,
                    deck_seed=deck_seed,
                    iteration=iteration,
                    advantage_replicates=reps,
                    strategy_replicates=1,
                    advantage_rng=advantage_rngs[seed_index],
                    strategy_rng=strategy_rngs[seed_index],
                )
                global_roots[seed_index] += 1

            progress = _fit_advantage_iteration(
                session=session,
                bundle=bundle,
                algorithm_seed=int(algorithm_seed),
                iteration=iteration,
                lr=float(args.lr),
                chunk_steps=int(args.advantage_chunk_steps),
                max_steps=int(args.advantage_max_steps_per_iteration),
                fit_target=float(args.advantage_fit_target),
                batch_size=int(args.batch_size),
                audit_size=int(args.audit_size),
                device=args.device,
            )
            seed_reports[seed_index]["checkpoints"].append(
                {
                    "iteration": int(iteration),
                    "roots": int(bundle.counters["roots"]),
                    "advantage_samples": len(bundle.adv_mem.items),
                    "advantage_seen": int(bundle.adv_mem.seen),
                    "strategy_samples": len(bundle.pol_mem.items),
                    "strategy_seen": int(bundle.pol_mem.seen),
                    "nodes": int(bundle.counters["nodes"]),
                    "final_advantage_fit": progress[-1],
                }
            )

        observations = _common_iteration_observations(
            bundles,
            iteration,
            int(args.iteration_cross_per_seed),
            0x1A7E0000 ^ iteration,
        )
        behavior_cross = _behavior_tv(
            bundles[0].advantage,
            bundles[1].advantage,
            observations,
            args.device,
        )
        items_a = _iteration_items(bundles[0].pol_mem.items, iteration)
        items_b = _iteration_items(bundles[1].pol_mem.items, iteration)
        support = _support_metrics(items_a, items_b)
        iteration_reports.append(
            {
                "iteration": int(iteration),
                "roots_per_seed_cumulative": int(iteration * args.roots_per_iteration),
                "observation_count": len(observations),
                "advantage_regret_policy_cross_seed": behavior_cross,
                "strategy_support": support,
                "seed_final_advantage_fit": [
                    seed_reports[0]["checkpoints"][-1]["final_advantage_fit"],
                    seed_reports[1]["checkpoints"][-1]["final_advantage_fit"],
                ],
            }
        )
        print(json.dumps(iteration_reports[-1], sort_keys=True), flush=True)

    for seed_index, seed in enumerate(seeds):
        progress, final_fit = _fit_average_policy(
            bundle=bundles[seed_index],
            session=sessions[seed_index],
            seed=int(seed),
            device=args.device,
            policy_chunk_steps=int(args.policy_chunk_steps),
            policy_max_steps=int(args.policy_max_steps),
            policy_fit_target=float(args.policy_fit_target),
            batch_size=int(args.batch_size),
            audit_size=int(args.audit_size),
        )
        seed_reports[seed_index]["policy_optimizer_steps"] = int(
            bundles[seed_index].counters["policy_optimizer_steps"]
        )
        seed_reports[seed_index]["policy_progress"] = progress
        seed_reports[seed_index]["final_fit"] = final_fit

    final_observations = shared_cross_seed_observations(
        bundles,
        per_seed=int(args.final_cross_per_seed),
        seed=0x715EED,
    )
    final_policy_cross = cross_seed_policy_tv(
        bundles[0].policy,
        bundles[1].policy,
        final_observations,
        device=args.device,
    )

    means = [float(x["advantage_regret_policy_cross_seed"]["mean_tv"]) for x in iteration_reports]
    p95s = [float(x["advantage_regret_policy_cross_seed"]["p95_tv"]) for x in iteration_reports]
    first_nontrivial = means[0] if means else math.inf
    final_mean = means[-1] if means else math.inf
    payload = {
        "schema": "SPINCORE_R7_3_ITERATION_COMPOUNDING_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "mode": args.mode,
        "advantage_replicates": reps,
        "strategy_replicates": 1,
        "algorithm_seeds": seeds,
        "shared_deck_stream_seed": int(args.deck_stream_seed),
        "iterations": int(args.iterations),
        "roots_per_iteration": int(args.roots_per_iteration),
        "iteration_reports": iteration_reports,
        "per_seed": seed_reports,
        "final_average_policy_cross_seed": {k: float(v) for k, v in final_policy_cross.items()},
        "final_average_policy_observation_count": len(final_observations),
        "summary": {
            "iteration_mean_tv_series": means,
            "iteration_p95_tv_series": p95s,
            "last_to_first_behavior_mean_tv_ratio": (
                final_mean / max(first_nontrivial, 1e-12) if means else math.inf
            ),
            "peak_behavior_mean_tv": max(means) if means else math.inf,
            "peak_behavior_p95_tv": max(p95s) if p95s else math.inf,
            "final_average_policy_mean_tv": float(final_policy_cross["mean_tv"]),
            "final_average_policy_p95_tv": float(final_policy_cross["p95_tv"]),
        },
        "interpretation_note": (
            "Diagnostic only. Both seeds receive the same hidden-deal schedule and the same "
            "per-mode replication factor. After every CFR iteration, the freshly fitted "
            "AdvantageNets are converted through exact regret matching on a common corpus from "
            "that iteration's strategy memories. This measures where iterative behavior "
            "divergence grows, before the final AveragePolicy fit can obscure the upstream path."
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
