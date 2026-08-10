from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Sequence

import torch

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility
from spincore.r7 import (
    FROZEN_GATES,
    cross_seed_policy_tv,
    stratified_audit_indices,
    weighted_mean_tv,
)
from spincore.solver import SolverLibrary
from spincore_nn import AveragePolicyNet, NetworkConfig
from spincore_nn.codec import collate_inputs, decode_spnniv1
from spincore_nn.training import train_step

from run_r7_3_diagnostic import hu_episode, make_bundle


DEFAULT_SEEDS = (20260829, 20260807)
PAYOUT = (0.5, 0.3, 0.2)


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _policy_batch(samples, device: str):
    batch = collate_inputs([decode_spnniv1(x.observation) for x in samples], device=device)
    target = torch.tensor([x.target for x in samples], dtype=torch.float32, device=device)
    weights = torch.tensor([x.weight for x in samples], dtype=torch.float32, device=device)
    return batch, target, weights


def _policy_fit_tv(model, items: Sequence, *, sample_size: int, seed: int, device: str) -> float:
    ids = stratified_audit_indices(len(items), int(sample_size), int(seed))
    if not ids:
        return math.inf
    samples = [items[i] for i in ids]
    batch, target, weights = _policy_batch(samples, device)
    model.eval()
    with torch.no_grad():
        pred = model.probabilities(batch)
    return weighted_mean_tv(pred, target, weights)


def _advantage_fit_nrmse(bundle, *, sample_size: int, seed: int, device: str) -> float:
    from spincore.r7 import audit_model_fit

    return float(
        audit_model_fit(
            bundle,
            sample_size=int(sample_size),
            seed=int(seed),
            device=device,
        )["advantage_weighted_nrmse"]
    )


def collect_strategy_memory(
    *,
    seed: int,
    solver: SolverLibrary,
    device: str,
    iterations: int,
    roots_per_iteration: int,
    advantage_chunk_steps: int,
    advantage_max_steps_per_iteration: int,
    advantage_fit_target: float,
    batch_size: int,
    audit_size: int,
    reservoir_capacity: int,
    lr: float,
) -> tuple[object, dict]:
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
    episode = hu_episode()
    global_root = 0
    checkpoints: list[dict] = []

    for iteration in range(1, int(iterations) + 1):
        for _ in range(int(roots_per_iteration)):
            deck_seed = (
                int(seed) * 1_000_003 + global_root * 97 + iteration
            ) & ((1 << 64) - 1)
            session.collect_root(episode, iteration=iteration, deck_seed=deck_seed)
            global_root += 1

        reset_seed = (int(seed) ^ (iteration * 0x9E3779B1)) & 0x7FFFFFFF
        session.reset_advantage_network(init_seed=reset_seed, lr=lr)

        local_steps = 0
        audit_seed = int(seed) ^ (iteration * 0x45D9F3B)
        progress: list[dict] = []
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
                "frozen_gate_pass": _finite(nrmse)
                and nrmse <= FROZEN_GATES["advantage_weighted_nrmse_max"],
                "fit_target_reached": _finite(nrmse)
                and nrmse <= float(advantage_fit_target),
            }
            progress.append(row)
            print(
                json.dumps(
                    {
                        "phase": "collect",
                        "seed": int(seed),
                        "iteration": int(iteration),
                        "roots": int(bundle.counters["roots"]),
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
                "roots": int(bundle.counters["roots"]),
                "advantage_progress": progress,
                "advantage_samples": len(bundle.adv_mem.items),
                "strategy_samples": len(bundle.pol_mem.items),
            }
        )

    return bundle, {
        "seed": int(seed),
        "roots": int(bundle.counters["roots"]),
        "strategy_samples": len(bundle.pol_mem.items),
        "strategy_seen": int(bundle.pol_mem.seen),
        "advantage_samples": len(bundle.adv_mem.items),
        "advantage_seen": int(bundle.adv_mem.seen),
        "checkpoints": checkpoints,
    }


def fit_policy(
    items: Sequence,
    *,
    config: NetworkConfig,
    init_seed: int,
    batch_seed: int,
    device: str,
    lr: float,
    chunk_steps: int,
    max_steps: int,
    fit_target: float,
    batch_size: int,
    audit_size: int,
    audit_seed: int,
) -> tuple[AveragePolicyNet, dict]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(init_seed))
        model = AveragePolicyNet(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    rng = random.Random(int(batch_seed))
    progress: list[dict] = []
    steps_done = 0

    while steps_done < int(max_steps):
        steps = min(int(chunk_steps), int(max_steps) - steps_done)
        for _ in range(steps):
            samples = rng.sample(items, min(int(batch_size), len(items)))
            batch, target, weights = _policy_batch(samples, device)
            train_step(model, optimizer, batch, target, weights, "strategy")
        steps_done += steps

        fit_tv = _policy_fit_tv(
            model,
            items,
            sample_size=int(audit_size),
            seed=int(audit_seed),
            device=device,
        )
        row = {
            "optimizer_steps": int(steps_done),
            "weighted_mean_tv": float(fit_tv),
            "frozen_gate_pass": _finite(fit_tv)
            and fit_tv <= FROZEN_GATES["policy_weighted_mean_tv_max"],
            "fit_target_reached": _finite(fit_tv) and fit_tv <= float(fit_target),
        }
        progress.append(row)
        print(
            json.dumps(
                {
                    "phase": "policy_fit",
                    "init_seed": int(init_seed),
                    "batch_seed": int(batch_seed),
                    "fit": row,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if row["fit_target_reached"]:
            break

    final_tv = _policy_fit_tv(
        model,
        items,
        sample_size=max(int(audit_size), 2048),
        seed=int(audit_seed) ^ 0x2468ACE0,
        device=device,
    )
    return model, {
        "init_seed": int(init_seed),
        "batch_seed": int(batch_seed),
        "optimizer_steps": int(steps_done),
        "final_weighted_mean_tv": float(final_tv),
        "frozen_gate_pass": _finite(final_tv)
        and final_tv <= FROZEN_GATES["policy_weighted_mean_tv_max"],
        "progress": progress,
    }


def common_observations(memories: Sequence[Sequence], *, per_memory: int, seed: int) -> list[bytes]:
    out: list[bytes] = []
    for index, items in enumerate(memories):
        ids = stratified_audit_indices(
            len(items),
            int(per_memory),
            int(seed) ^ (index * 0x45D9F3B),
        )
        out.extend(items[i].observation for i in ids)
    return out


def _tv(a, b, observations, *, device: str) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in cross_seed_policy_tv(
            a,
            b,
            observations,
            device=device,
        ).items()
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="R7.3 variance decomposition: policy optimizer variance vs CFR-memory variance"
    )
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("validation/R7_3_VARIANCE_DECOMPOSITION_640.json"),
    )
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--roots-per-iteration", type=int, default=128)
    ap.add_argument("--advantage-chunk-steps", type=int, default=256)
    ap.add_argument("--advantage-max-steps-per-iteration", type=int, default=2048)
    ap.add_argument("--advantage-fit-target", type=float, default=0.70)
    ap.add_argument("--policy-chunk-steps", type=int, default=256)
    ap.add_argument("--policy-max-steps", type=int, default=8192)
    ap.add_argument("--policy-fit-target", type=float, default=0.105)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--audit-size", type=int, default=1024)
    ap.add_argument("--cross-seed-per-memory", type=int, default=1024)
    ap.add_argument("--reservoir-capacity", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if len(seeds) != 2:
        raise SystemExit("variance decomposition requires exactly two training seeds")
    if args.iterations <= 0 or args.roots_per_iteration <= 0:
        raise SystemExit("positive collection schedule required")

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()

    bundles = []
    collection_reports = []
    for seed in seeds:
        bundle, report = collect_strategy_memory(
            seed=seed,
            solver=solver,
            device=args.device,
            iterations=args.iterations,
            roots_per_iteration=args.roots_per_iteration,
            advantage_chunk_steps=args.advantage_chunk_steps,
            advantage_max_steps_per_iteration=args.advantage_max_steps_per_iteration,
            advantage_fit_target=args.advantage_fit_target,
            batch_size=args.batch_size,
            audit_size=args.audit_size,
            reservoir_capacity=args.reservoir_capacity,
            lr=args.lr,
        )
        bundles.append(bundle)
        collection_reports.append(report)

    memories = [bundle.pol_mem.items for bundle in bundles]
    observations = common_observations(
        memories,
        per_memory=args.cross_seed_per_memory,
        seed=0x715EED,
    )

    # Two controlled optimizer seeds. Train the same pair of initializations on
    # each CFR memory. Within-memory disagreement measures optimizer/init
    # variance; across-memory disagreement with the same optimizer seed measures
    # variance induced by CFR/chance data.
    train_specs = (
        (0x12345, 0xABCDE),
        (0x6789A, 0xFEDCB),
    )
    models: dict[tuple[int, int], AveragePolicyNet] = {}
    fit_reports: dict[str, dict] = {}

    for memory_index, bundle in enumerate(bundles):
        for train_index, (init_seed, batch_seed) in enumerate(train_specs):
            model, report = fit_policy(
                bundle.pol_mem.items,
                config=bundle.config,
                init_seed=init_seed,
                batch_seed=batch_seed,
                device=args.device,
                lr=args.lr,
                chunk_steps=args.policy_chunk_steps,
                max_steps=args.policy_max_steps,
                fit_target=args.policy_fit_target,
                batch_size=args.batch_size,
                audit_size=args.audit_size,
                audit_seed=seeds[memory_index] ^ (train_index * 0x13579BDF),
            )
            models[(memory_index, train_index)] = model
            fit_reports[f"memory_{memory_index}_train_{train_index}"] = report

    within_a = _tv(models[(0, 0)], models[(0, 1)], observations, device=args.device)
    within_b = _tv(models[(1, 0)], models[(1, 1)], observations, device=args.device)
    across_t0 = _tv(models[(0, 0)], models[(1, 0)], observations, device=args.device)
    across_t1 = _tv(models[(0, 1)], models[(1, 1)], observations, device=args.device)

    within_mean = 0.5 * (within_a["mean_tv"] + within_b["mean_tv"])
    across_mean = 0.5 * (across_t0["mean_tv"] + across_t1["mean_tv"])
    ratio = across_mean / max(within_mean, 1e-12)

    all_fit_pass = all(x["frozen_gate_pass"] for x in fit_reports.values())
    if not all_fit_pass:
        diagnosis = "POLICY_FIT_INSUFFICIENT_FOR_CLEAN_DECOMPOSITION"
    elif across_mean > within_mean * 1.5:
        diagnosis = "CFR_MEMORY_VARIANCE_DOMINANT"
    elif within_mean > across_mean * 0.75:
        diagnosis = "POLICY_OPTIMIZATION_VARIANCE_MATERIAL"
    else:
        diagnosis = "MIXED_VARIANCE"

    payload = {
        "schema": "SPINCORE_R7_3_VARIANCE_DECOMPOSITION_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "training_seeds": seeds,
        "roots_per_seed": int(args.iterations * args.roots_per_iteration),
        "frozen_gates": dict(FROZEN_GATES),
        "policy_fit_target": float(args.policy_fit_target),
        "collection": collection_reports,
        "policy_fits": fit_reports,
        "common_observation_count": len(observations),
        "comparisons": {
            "same_memory_A_different_optimizer_seed": within_a,
            "same_memory_B_different_optimizer_seed": within_b,
            "different_memory_same_optimizer_seed_0": across_t0,
            "different_memory_same_optimizer_seed_1": across_t1,
        },
        "summary": {
            "all_policy_fit_gates_pass": bool(all_fit_pass),
            "within_memory_mean_tv_average": float(within_mean),
            "across_memory_mean_tv_average": float(across_mean),
            "across_to_within_mean_tv_ratio": float(ratio),
            "diagnosis": diagnosis,
        },
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
