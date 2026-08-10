from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility
from spincore.r7 import FROZEN_GATES, audit_model_fit, cross_seed_policy_tv, stratified_audit_indices
from spincore.solver import Episode, SolverLibrary
from spincore_nn import AdvantageNet, AveragePolicyNet, DomainBundle, NetworkConfig, UniformReservoir


DEFAULT_SEEDS = (20260829, 20260807)
PAYOUT = (0.5, 0.3, 0.2)

# These are training targets, not acceptance gates.  They deliberately leave a
# small safety margin below the frozen R7.3 fit gates so a second stratified
# audit does not oscillate around the threshold because of sample noise.
DEFAULT_ADVANTAGE_FIT_TARGET = 0.70
DEFAULT_POLICY_FIT_TARGET = 0.105


def hu_episode() -> Episode:
    return Episode(
        total_chips=1500,
        game_is_hu=True,
        blind_index=0,
        small_blind=10,
        big_blind=20,
        stacks=(0, 750, 750),
        dealer_id=1,
        dead_players=(0,),
    )


def make_bundle(seed: int, *, device: str, reservoir_capacity: int, lr: float) -> DomainBundle:
    cfg = NetworkConfig()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        advantage = AdvantageNet(cfg).to(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) ^ 0x5DEECE66D)
        policy = AveragePolicyNet(cfg).to(device)
    return DomainBundle(
        "TRUE_HEADS_UP",
        int(seed),
        cfg,
        advantage,
        policy,
        torch.optim.Adam(advantage.parameters(), lr=float(lr)),
        torch.optim.Adam(policy.parameters(), lr=float(lr)),
        UniformReservoir(int(reservoir_capacity), int(seed) ^ 0xA5A5A5A5),
        UniformReservoir(int(reservoir_capacity), int(seed) ^ 0x5A5A5A5A),
        random.Random(int(seed) ^ 0xC0FFEE),
        {},
    )


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def fit_pass(value: float, gate_name: str) -> bool:
    return finite(value) and float(value) <= float(FROZEN_GATES[gate_name])


def run_seed(
    *,
    seed: int,
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
    reservoir_capacity: int,
    lr: float,
) -> tuple[DomainBundle, dict]:
    bundle = make_bundle(seed, device=device, reservoir_capacity=reservoir_capacity, lr=lr)
    session = DeepCFRDomainSession(
        solver_library=solver,
        bundle=bundle,
        terminal_utility=icm_delta_utility(PAYOUT),
        device=device,
    )
    episode = hu_episode()
    checkpoints: list[dict] = []
    global_root = 0

    for iteration in range(1, int(iterations) + 1):
        for _ in range(int(roots_per_iteration)):
            deck_seed = (int(seed) * 1_000_003 + global_root * 97 + iteration) & ((1 << 64) - 1)
            session.collect_root(episode, iteration=iteration, deck_seed=deck_seed)
            global_root += 1

        # Deep CFR fits a fresh advantage network to the accumulated memory.
        # Fixed optimizer steps caused later iterations to receive progressively
        # fewer effective epochs as the reservoir grew.  Train in bounded chunks
        # until a stable fit margin is reached, without changing the frozen gate.
        reset_seed = (int(seed) ^ (iteration * 0x9E3779B1)) & 0x7FFFFFFF
        session.reset_advantage_network(init_seed=reset_seed, lr=lr)
        local_steps = 0
        advantage_progress: list[dict] = []
        audit_seed = int(seed) ^ (iteration * 0x45D9F3B)
        while local_steps < int(advantage_max_steps_per_iteration):
            steps = min(
                int(advantage_chunk_steps),
                int(advantage_max_steps_per_iteration) - local_steps,
            )
            session.train_advantage(steps=steps, batch_size=int(batch_size))
            local_steps += steps
            fit = audit_model_fit(
                bundle,
                sample_size=int(audit_size),
                seed=audit_seed,
                device=device,
            )
            value = float(fit["advantage_weighted_nrmse"])
            row = {
                "local_optimizer_steps": local_steps,
                "advantage_weighted_nrmse": value,
                "frozen_gate_pass": fit_pass(value, "advantage_weighted_nrmse_max"),
                "fit_target_reached": finite(value) and value <= float(advantage_fit_target),
            }
            advantage_progress.append(row)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "iteration": iteration,
                        "roots": bundle.counters["roots"],
                        "advantage": row,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if row["fit_target_reached"]:
                break

        checkpoint_fit = advantage_progress[-1]
        checkpoints.append(
            {
                "iteration": iteration,
                "roots": int(bundle.counters["roots"]),
                "advantage_weighted_nrmse": float(checkpoint_fit["advantage_weighted_nrmse"]),
                "advantage_gate_pass": bool(checkpoint_fit["frozen_gate_pass"]),
                "advantage_fit_target_reached": bool(checkpoint_fit["fit_target_reached"]),
                "advantage_local_optimizer_steps": int(local_steps),
                "advantage_progress": advantage_progress,
                "advantage_samples_in_reservoir": len(bundle.adv_mem.items),
                "strategy_samples_in_reservoir": len(bundle.pol_mem.items),
                "advantage_optimizer_steps_total": int(bundle.counters["adv_optimizer_steps"]),
            }
        )

    # Fit the average-policy approximator only after CFR collection.  The 640
    # baseline showed monotonic improvement but hit the old 3072-step cap at
    # TV ~= 0.16.  The larger bounded cap below separates model-fit failure from
    # genuine cross-seed strategic instability.
    policy_progress: list[dict] = []
    policy_audit_seed = int(seed) ^ 0x13579BDF
    while int(bundle.counters["policy_optimizer_steps"]) < int(policy_max_steps):
        remaining = int(policy_max_steps) - int(bundle.counters["policy_optimizer_steps"])
        steps = min(int(policy_chunk_steps), remaining)
        session.train_average_policy(steps=steps, batch_size=int(batch_size))
        fit = audit_model_fit(
            bundle,
            sample_size=int(audit_size),
            seed=policy_audit_seed,
            device=device,
        )
        value = float(fit["policy_weighted_mean_tv"])
        row = {
            "policy_optimizer_steps": int(bundle.counters["policy_optimizer_steps"]),
            "policy_weighted_mean_tv": value,
            "policy_gate_pass": fit_pass(value, "policy_weighted_mean_tv_max"),
            "fit_target_reached": finite(value) and value <= float(policy_fit_target),
        }
        policy_progress.append(row)
        print(json.dumps({"seed": seed, "policy": row}, sort_keys=True), flush=True)
        if row["fit_target_reached"]:
            break

    final_fit = audit_model_fit(
        bundle,
        sample_size=max(int(audit_size), 2048),
        seed=int(seed) ^ 0x2468ACE0,
        device=device,
    )
    seed_report = {
        "seed": int(seed),
        "network_config": bundle.config.to_dict(),
        "parameter_count_per_model": parameter_count(bundle.advantage),
        "initial_zero_regret_uniform": True,
        "iterations": int(iterations),
        "roots_per_iteration": int(roots_per_iteration),
        "roots_total": int(bundle.counters["roots"]),
        "advantage_chunk_steps": int(advantage_chunk_steps),
        "advantage_max_steps_per_iteration": int(advantage_max_steps_per_iteration),
        "advantage_fit_target": float(advantage_fit_target),
        "policy_chunk_steps": int(policy_chunk_steps),
        "policy_max_steps": int(policy_max_steps),
        "policy_fit_target": float(policy_fit_target),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "checkpoints": checkpoints,
        "policy_progress": policy_progress,
        "final_fit": {k: float(v) for k, v in final_fit.items()},
        "counters": {k: int(v) for k, v in bundle.counters.items()},
        "advantage_gate_pass": fit_pass(
            final_fit["advantage_weighted_nrmse"], "advantage_weighted_nrmse_max"
        ),
        "policy_gate_pass": fit_pass(
            final_fit["policy_weighted_mean_tv"], "policy_weighted_mean_tv_max"
        ),
    }
    return bundle, seed_report


def shared_cross_seed_observations(bundles: list[DomainBundle], *, per_seed: int, seed: int) -> list[bytes]:
    observations: list[bytes] = []
    for index, bundle in enumerate(bundles):
        ids = stratified_audit_indices(
            len(bundle.pol_mem.items), int(per_seed), int(seed) ^ (index * 0x45D9F3B)
        )
        observations.extend(bundle.pol_mem.items[i].observation for i in ids)
    return observations


def main() -> int:
    ap = argparse.ArgumentParser(description="SpinCore R7.3 reproducible two-seed HU diagnostic")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_DIAGNOSTIC.json"))
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--roots-per-iteration", type=int, default=128)
    # --advantage-steps remains the chunk-size spelling for compatibility with
    # the first physical diagnostic and existing run commands.
    ap.add_argument("--advantage-steps", type=int, default=256)
    ap.add_argument("--advantage-max-steps-per-iteration", type=int, default=2048)
    ap.add_argument("--advantage-fit-target", type=float, default=DEFAULT_ADVANTAGE_FIT_TARGET)
    ap.add_argument("--policy-chunk-steps", type=int, default=256)
    ap.add_argument("--policy-max-steps", type=int, default=8192)
    ap.add_argument("--policy-fit-target", type=float, default=DEFAULT_POLICY_FIT_TARGET)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=1024)
    ap.add_argument("--cross-seed-per-seed", type=int, default=1024)
    ap.add_argument("--reservoir-capacity", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if len(seeds) < 2:
        raise SystemExit("R7.3 requires at least two seeds")
    if args.iterations <= 0 or args.roots_per_iteration <= 0:
        raise SystemExit("iterations and roots-per-iteration must be positive")
    if args.advantage_steps <= 0 or args.advantage_max_steps_per_iteration < args.advantage_steps:
        raise SystemExit("invalid advantage fit schedule")
    if args.policy_chunk_steps <= 0 or args.policy_max_steps < args.policy_chunk_steps:
        raise SystemExit("invalid policy fit schedule")

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()
    bundles: list[DomainBundle] = []
    seed_reports: list[dict] = []

    for seed in seeds:
        bundle, report = run_seed(
            seed=seed,
            solver=solver,
            device=args.device,
            iterations=args.iterations,
            roots_per_iteration=args.roots_per_iteration,
            advantage_chunk_steps=args.advantage_steps,
            advantage_max_steps_per_iteration=args.advantage_max_steps_per_iteration,
            advantage_fit_target=args.advantage_fit_target,
            policy_chunk_steps=args.policy_chunk_steps,
            policy_max_steps=args.policy_max_steps,
            policy_fit_target=args.policy_fit_target,
            batch_size=args.batch_size,
            audit_size=args.audit_size,
            reservoir_capacity=args.reservoir_capacity,
            lr=args.lr,
        )
        bundles.append(bundle)
        seed_reports.append(report)

    common_obs = shared_cross_seed_observations(
        bundles,
        per_seed=args.cross_seed_per_seed,
        seed=0x715EED,
    )
    cross = cross_seed_policy_tv(
        bundles[0].policy,
        bundles[1].policy,
        common_obs,
        device=args.device,
    )
    cross_pass = bool(
        finite(cross["mean_tv"])
        and finite(cross["p95_tv"])
        and cross["mean_tv"] <= FROZEN_GATES["cross_seed_mean_tv_max"]
        and cross["p95_tv"] <= FROZEN_GATES["cross_seed_p95_tv_max"]
    )
    per_seed_pass = all(x["advantage_gate_pass"] and x["policy_gate_pass"] for x in seed_reports)
    overall_pass = bool(per_seed_pass and cross_pass)

    payload = {
        "schema": "SPINCORE_R7_3_DIAGNOSTIC_V2",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "seeds": seeds,
        "frozen_gates": dict(FROZEN_GATES),
        "fit_schedule": {
            "initial_zero_regret_uniform": True,
            "advantage_fit_target": float(args.advantage_fit_target),
            "advantage_max_steps_per_iteration": int(args.advantage_max_steps_per_iteration),
            "policy_fit_target": float(args.policy_fit_target),
            "policy_max_steps": int(args.policy_max_steps),
        },
        "historical_parameter_count_recorded": 152434,
        "recovered_parameter_count": parameter_count(bundles[0].advantage),
        "parameter_count_delta": parameter_count(bundles[0].advantage) - 152434,
        "seed_reports": seed_reports,
        "cross_seed_observation_count": len(common_obs),
        "cross_seed": {k: float(v) for k, v in cross.items()},
        "per_seed_fit_pass": per_seed_pass,
        "cross_seed_pass": cross_pass,
        "r7_3_pass": overall_pass,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if args.strict and not overall_pass:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
