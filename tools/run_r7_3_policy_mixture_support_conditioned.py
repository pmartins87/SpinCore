from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch

import run_r7_3_diagnostic as diagnostic
import run_r7_3_partial_exact_ensemble_paired as paired
from run_r7_3_partial_exact_policy_mixture_paired import PolicyMixtureEnsembleAdvantagePolicy
from spincore.r7 import stratified_audit_indices
from spincore.solver import SolverLibrary
from spincore_nn.codec import collate_inputs, decode_spnniv1


DEFAULT_SEEDS = (20260829, 20260807)


def _policy_tv(model_a, model_b, observations, device):
    if not observations:
        return {"count": 0, "mean_tv": math.inf, "p50_tv": math.inf, "p95_tv": math.inf, "max_tv": math.inf}
    batch = collate_inputs([decode_spnniv1(x) for x in observations], device=device)
    model_a.eval(); model_b.eval()
    with torch.no_grad():
        a = model_a.probabilities(batch).detach().cpu()
        b = model_b.probabilities(batch).detach().cpu()
    tv = 0.5 * torch.abs(a - b).sum(1)
    return {
        "count": len(observations),
        "mean_tv": float(tv.mean()),
        "p50_tv": float(torch.quantile(tv, torch.tensor(0.50))),
        "p95_tv": float(torch.quantile(tv, torch.tensor(0.95))),
        "max_tv": float(tv.max()),
    }


def _sample(items, *, n, seed, predicate=None):
    candidates = [i for i, x in enumerate(items) if predicate is None or predicate(x)]
    if not candidates:
        return []
    local = stratified_audit_indices(len(candidates), min(int(n), len(candidates)), int(seed))
    return [items[candidates[i]].observation for i in local]


def main() -> int:
    ap = argparse.ArgumentParser(description="Support-conditioned final-policy disagreement after paired Advantage policy mixture")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ensemble-size", type=int, choices=(4, 8), default=4)
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--exact-opponent-levels", type=int, default=2)
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
        raise SystemExit("requires exactly two seeds")
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()

    paired.EnsembleAdvantagePolicy = PolicyMixtureEnsembleAdvantagePolicy
    bundles = []
    reports = []
    for seed in seeds:
        bundle, report = paired.run_seed(
            seed=int(seed), ensemble_size=int(args.ensemble_size), solver=solver, args=args
        )
        bundles.append(bundle)
        reports.append(report)

    items_a = bundles[0].pol_mem.items
    items_b = bundles[1].pol_mem.items
    counts_a = Counter(x.observation for x in items_a)
    counts_b = Counter(x.observation for x in items_b)
    keys_a = set(counts_a)
    keys_b = set(counts_b)
    shared = keys_a & keys_b
    unique_a = keys_a - keys_b
    unique_b = keys_b - keys_a

    n = int(args.cross_seed_per_seed)
    corpora = {
        "support_A_all": _sample(items_a, n=n, seed=0xA11A),
        "support_B_all": _sample(items_b, n=n, seed=0xB11B),
        "support_A_shared_exact": _sample(items_a, n=n, seed=0xA22A, predicate=lambda x: x.observation in shared),
        "support_B_shared_exact": _sample(items_b, n=n, seed=0xB22B, predicate=lambda x: x.observation in shared),
        "support_A_unique_exact": _sample(items_a, n=n, seed=0xA33A, predicate=lambda x: x.observation in unique_a),
        "support_B_unique_exact": _sample(items_b, n=n, seed=0xB33B, predicate=lambda x: x.observation in unique_b),
    }
    # Exact shared observations have byte-identical NeuralInputV1 keys, so one
    # canonical list is sufficient and avoids duplicate weighting by seed.
    shared_exact_canonical = sorted(shared)
    if len(shared_exact_canonical) > n:
        ids = stratified_audit_indices(len(shared_exact_canonical), n, 0x5A4ED)
        shared_exact_canonical = [shared_exact_canonical[i] for i in ids]
    corpora["shared_exact_canonical"] = shared_exact_canonical

    metrics = {
        name: _policy_tv(bundles[0].policy, bundles[1].policy, obs, args.device)
        for name, obs in corpora.items()
    }
    union_obs = corpora["support_A_all"] + corpora["support_B_all"]
    metrics["union_A_B"] = _policy_tv(bundles[0].policy, bundles[1].policy, union_obs, args.device)

    payload = {
        "schema": "SPINCORE_R7_3_POLICY_MIXTURE_SUPPORT_CONDITIONED_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "algorithm_seeds": seeds,
        "advantage_policy_mixture_size": int(args.ensemble_size),
        "iterations": int(args.iterations),
        "roots_per_iteration": int(args.roots_per_iteration),
        "roots_per_seed": int(args.iterations * args.roots_per_iteration),
        "deck_formula": "seed*1000003 + global_root*97 + iteration",
        "primary_rng_contract": "RECOVERED_SINGLE_COUPLED_BATCH_RNG",
        "per_seed": reports,
        "support_counts": {
            "unique_observations_A": len(keys_a),
            "unique_observations_B": len(keys_b),
            "shared_exact": len(shared),
            "union_exact": len(keys_a | keys_b),
            "jaccard_exact": len(shared) / max(len(keys_a | keys_b), 1),
            "shared_occurrence_mass_A": sum(counts_a[k] for k in shared) / max(len(items_a), 1),
            "shared_occurrence_mass_B": sum(counts_b[k] for k in shared) / max(len(items_b), 1),
        },
        "cross_seed_by_support": metrics,
        "interpretation_note": (
            "Diagnostic only. The authoritative paired policy-mixture CFR candidate is trained normally, "
            "then the two final AveragePolicy models are evaluated separately on seed-A support, seed-B "
            "support, exact shared observations, and exact one-sided observations. This quantifies how "
            "much of the residual cross-seed tail is true shared-state disagreement versus off-support "
            "generalization/extrapolation. Exact-shared means byte-identical SPNNIV1 observations."
        ),
        "acceptance_gate_changed": False,
        "production_semantics_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
