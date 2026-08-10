from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import run_r7_3_diagnostic as diagnostic
import run_r7_3_partial_exact_ensemble_paired as paired
from run_r7_3_partial_exact_policy_mixture_paired import PolicyMixtureEnsembleAdvantagePolicy
from run_r7_3_partial_exact_policy_ensemble import (
    AveragePolicyEnsemble,
    _cross_policy_tv,
    _policy_fit_tv,
    _train_side_policy_member,
)
from run_r7_3_variance_decomposition import _finite
from spincore.r7 import FROZEN_GATES
from spincore.solver import SolverLibrary


DEFAULT_SEEDS = (20260829, 20260807)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Paired partial-exact Advantage policy mixture plus final AveragePolicy ensemble"
    )
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--advantage-ensemble-size", type=int, choices=(4, 8), default=4)
    ap.add_argument("--policy-ensemble-sizes", default="1,2,4")
    ap.add_argument("--exact-opponent-levels", type=int, default=2)
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
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
    policy_sizes = sorted({int(x.strip()) for x in str(args.policy_ensemble_sizes).split(",") if x.strip()})
    if len(seeds) != 2:
        raise SystemExit("requires exactly two algorithm seeds")
    if not policy_sizes or policy_sizes[0] != 1 or any(x not in (1, 2, 4) for x in policy_sizes):
        raise SystemExit("policy ensemble sizes must be a subset of 1,2,4 and include 1")

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()

    # Replace only the multi-model Advantage behavior map. The paired collector,
    # authoritative deck formula, primary coupled RNG stream, partial-exact
    # estimator and primary Advantage training path remain unchanged.
    paired.EnsembleAdvantagePolicy = PolicyMixtureEnsembleAdvantagePolicy

    bundles = []
    upstream_reports = []
    policy_sets = []
    policy_member_reports = []

    for seed in seeds:
        bundle, upstream = paired.run_seed(
            seed=int(seed),
            ensemble_size=int(args.advantage_ensemble_size),
            solver=solver,
            args=args,
        )
        bundles.append(bundle)
        upstream_reports.append(upstream)

        primary = bundle.policy
        members = [primary]
        member_reports = [
            {
                "member": 0,
                "role": "PRIMARY_POLICY_FROM_PAIRED_LIVE_RNG",
                "optimizer_steps": int(bundle.counters["policy_optimizer_steps"]),
            }
        ]
        memory_state = bundle.pol_mem.state_dict()
        for member in range(1, max(policy_sizes)):
            model, report = _train_side_policy_member(
                memory_state=memory_state,
                algorithm_seed=int(seed),
                member=int(member),
                config=bundle.config,
                args=args,
            )
            members.append(model)
            member_reports.append(report)

        policy_sets.append(
            {size: AveragePolicyEnsemble(members[:size]) for size in policy_sizes}
        )
        policy_member_reports.append(member_reports)

    observations = diagnostic.shared_cross_seed_observations(
        bundles,
        per_seed=int(args.cross_seed_per_seed),
        seed=0x715EED,
    )

    cross = {}
    fit = {}
    pass_by_size = {}
    for size in policy_sizes:
        k = str(size)
        cross[k] = _cross_policy_tv(
            policy_sets[0][size],
            policy_sets[1][size],
            observations,
            device=args.device,
        )
        fit[k] = []
        for seed_index, bundle in enumerate(bundles):
            tv = _policy_fit_tv(
                policy_sets[seed_index][size],
                bundle.pol_mem,
                sample_size=max(int(args.audit_size), 2048),
                seed=int(seeds[seed_index]) ^ 0x2468ACE0,
                device=args.device,
            )
            fit[k].append(float(tv))

        upstream_adv_fit_pass = all(
            row["final_fit"]["advantage_gate_pass"] for row in upstream_reports
        )
        policy_fit_pass = all(
            _finite(v) and float(v) <= FROZEN_GATES["policy_weighted_mean_tv_max"]
            for v in fit[k]
        )
        cross_pass = bool(
            _finite(cross[k]["mean_tv"])
            and _finite(cross[k]["p95_tv"])
            and float(cross[k]["mean_tv"]) <= FROZEN_GATES["cross_seed_mean_tv_max"]
            and float(cross[k]["p95_tv"]) <= FROZEN_GATES["cross_seed_p95_tv_max"]
        )
        pass_by_size[k] = {
            "advantage_fit_pass": bool(upstream_adv_fit_pass),
            "policy_fit_pass": bool(policy_fit_pass),
            "cross_seed_pass": bool(cross_pass),
            "r7_3_pass": bool(upstream_adv_fit_pass and policy_fit_pass and cross_pass),
        }

    baseline = cross["1"]
    ratios = {
        k: {
            "mean_tv_ratio_to_policy_size1": float(cross[k]["mean_tv"] / max(float(baseline["mean_tv"]), 1e-12)),
            "p95_tv_ratio_to_policy_size1": float(cross[k]["p95_tv"] / max(float(baseline["p95_tv"]), 1e-12)),
        }
        for k in cross
    }

    payload = {
        "schema": "SPINCORE_R7_3_POLICY_MIXTURE_FINAL_POLICY_ENSEMBLE_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "algorithm_seeds": seeds,
        "advantage_behavior": "PARTIAL_EXACT_LEVEL2_POLICY_MIXTURE",
        "advantage_ensemble_size": int(args.advantage_ensemble_size),
        "final_policy_ensemble_sizes": policy_sizes,
        "iterations": int(args.iterations),
        "roots_per_iteration": int(args.roots_per_iteration),
        "roots_per_seed": int(args.iterations * args.roots_per_iteration),
        "deck_formula": "seed*1000003 + global_root*97 + iteration",
        "primary_rng_contract": "RECOVERED_SINGLE_COUPLED_BATCH_RNG",
        "extra_advantage_members_perturb_primary_rng": False,
        "extra_policy_members_perturb_primary_rng": False,
        "upstream_per_seed": upstream_reports,
        "policy_member_reports": policy_member_reports,
        "policy_ensemble_fit_tv": fit,
        "cross_seed_by_policy_ensemble_size": cross,
        "ratios_to_policy_size1": ratios,
        "pass_by_policy_ensemble_size": pass_by_size,
        "frozen_gates": dict(FROZEN_GATES),
        "acceptance_gate_changed": False,
        "production_advantage_mapping_changed": False,
        "production_final_policy_ensemble_changed": False,
        "interpretation_note": (
            "Factorial diagnostic only. CFR uses the already material size-N Advantage policy-mixture "
            "mapping under the authoritative deck and primary RNG contract. After CFR collection is "
            "frozen, extra AveragePolicy members are side fits on the same strategy memory and do not "
            "advance the primary RNG. This directly measures whether final-policy ensembling supplies "
            "a useful residual tail reduction on top of upstream behavior stabilization."
        ),
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
