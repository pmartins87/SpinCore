from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

# Recovery metadata is evidence, not part of the generic diagnostic API.
import run_r7_3_diagnostic as diagnostic

diagnostic.HISTORICAL_PARAMS_PER_NETWORK = 152_434

import run_r7_3_replicated_640_candidate as legacy
from spincore.r7 import FROZEN_GATES, cross_seed_policy_tv
from spincore.solver import SolverLibrary


DEFAULT_SEEDS = (20260829, 20260807)


def authoritative_deck_seed(
    algorithm_seed: int,
    iteration: int,
    root_index_within_iteration: int,
    roots_per_iteration: int,
) -> int:
    """Exactly match tools/run_r7_3_diagnostic.py deck scheduling.

    Authoritative generation-2 acceptance runner:
      deck_seed = seed * 1_000_003 + global_root * 97 + iteration
    where global_root is monotonic across CFR iterations.
    """
    global_root = (
        (int(iteration) - 1) * int(roots_per_iteration)
        + int(root_index_within_iteration)
    )
    return (
        int(algorithm_seed) * 1_000_003
        + global_root * 97
        + int(iteration)
    ) & ((1 << 64) - 1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "R7.3 replicated-path candidate V2 with byte-exact generation-2 "
            "acceptance hidden-deal schedule"
        )
    )
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("validation/R7_3_REPLICATED_640_CANDIDATE_V2.json"),
    )
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--roots-per-iteration", type=int, default=128)
    ap.add_argument("--advantage-replicates", type=int, default=4)
    ap.add_argument("--strategy-replicates", type=int, default=1)
    ap.add_argument("--rng-contract", choices=("separate", "coupled"), default="coupled")
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
        raise SystemExit("V2 replicated candidate requires exactly two algorithm seeds")
    if args.iterations <= 0 or args.roots_per_iteration <= 0:
        raise SystemExit("positive iterations and roots_per_iteration required")
    if args.advantage_replicates <= 0 or args.strategy_replicates <= 0:
        raise SystemExit("positive replicate counts required")

    # Fail closed if the documented formula ever drifts from the authoritative
    # generation-2 runner. These explicit points cover cross-iteration global-root
    # continuity, which V1 accidentally failed to preserve.
    for seed in seeds:
        probes = [
            (1, 0),
            (1, int(args.roots_per_iteration) - 1),
            (2, 0),
            (int(args.iterations), int(args.roots_per_iteration) - 1),
        ]
        for iteration, root_index in probes:
            global_root = (iteration - 1) * int(args.roots_per_iteration) + root_index
            expected = (
                int(seed) * 1_000_003 + global_root * 97 + iteration
            ) & ((1 << 64) - 1)
            observed = authoritative_deck_seed(
                seed, iteration, root_index, int(args.roots_per_iteration)
            )
            if observed != expected:
                raise RuntimeError("authoritative deck schedule self-check failed")

    # Reuse the already-smoke-certified collection/fitting implementation while
    # replacing only its V1 deck function with the exact authoritative formula.
    legacy._unique_deck_seed = lambda seed, iteration, root_index: authoritative_deck_seed(
        seed,
        iteration,
        root_index,
        int(args.roots_per_iteration),
    )

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()
    bundles = []
    reports = []
    for seed in seeds:
        bundle, report = legacy.run_seed(
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

    observations = diagnostic.shared_cross_seed_observations(
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
        legacy._finite(cross["mean_tv"])
        and legacy._finite(cross["p95_tv"])
        and float(cross["mean_tv"]) <= FROZEN_GATES["cross_seed_mean_tv_max"]
        and float(cross["p95_tv"]) <= FROZEN_GATES["cross_seed_p95_tv_max"]
    )
    passed = bool(per_seed_fit_pass and cross_seed_pass)
    params = sum(p.numel() for p in bundles[0].advantage.parameters())

    payload = {
        "schema": "SPINCORE_R7_3_REPLICATED_640_CANDIDATE_V2",
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
            "deck_semantics": "GENERATION2_AUTHORITATIVE_GLOBAL_ROOT_FORMULA_EXACT",
            "deck_formula": "seed*1000003 + global_root*97 + iteration",
            "global_root_continuous_across_iterations": True,
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
            "historical_recorded_params": 152434,
            "delta_from_historical": int(params - 152434),
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
        "v1_evidence_comparability_note": (
            "Replicated-candidate V1 used a different hidden-deal formula and reset root_index "
            "inside each CFR iteration. Its physical results remain valid experiments, but they "
            "must not be described as deck-identical to the corrected generation-2 acceptance "
            "baseline. V2 fixes only that experimental-control error."
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
