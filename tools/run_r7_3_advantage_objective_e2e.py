from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import DeepCFRDomainSession, _batch, icm_delta_utility
from spincore.r7 import FROZEN_GATES, audit_model_fit, cross_seed_policy_tv
from spincore.solver import SolverLibrary

from run_r7_3_advantage_objective_stability import _train_custom_step
from run_r7_3_diagnostic import hu_episode, make_bundle, shared_cross_seed_observations
from run_r7_3_path_replication_screen import ADV_RNG_XOR, STRATEGY_RNG_XOR, PAYOUT, _fit_average_policy
from run_r7_3_variance_decomposition import _advantage_fit_nrmse, _finite


DEFAULT_SEEDS = (20260829, 20260807)
DEFAULT_SHARED_DECK_STREAM_SEED = 0xD3C5EED


def _fit_advantage(session, bundle, *, seed: int, iteration: int, objective: str, aux_weight: float, args):
    reset_seed = (int(seed) ^ (int(iteration) * 0x9E3779B1)) & 0x7FFFFFFF
    session.reset_advantage_network(init_seed=reset_seed, lr=float(args.lr))
    local_steps = 0
    progress = []
    audit_seed = int(seed) ^ (int(iteration) * 0x45D9F3B)
    while local_steps < int(args.advantage_max_steps_per_iteration):
        chunk = min(
            int(args.advantage_chunk_steps),
            int(args.advantage_max_steps_per_iteration) - local_steps,
        )
        losses = []
        for _ in range(chunk):
            samples = bundle.adv_mem.sample(
                min(int(args.batch_size), len(bundle.adv_mem.items)), bundle.batch_rng
            )
            batch, target, weights = _batch(samples, args.device)
            losses.append(
                _train_custom_step(
                    bundle.advantage,
                    bundle.adv_opt,
                    batch,
                    target,
                    weights,
                    objective,
                    float(aux_weight),
                )
            )
        local_steps += chunk
        # Custom training bypasses DeepCFRDomainSession.train_advantage(), so V1
        # accidentally left NeuralAdvantagePolicy.ready=False after every fit.
        # That forced every later CFR iteration to use the zero-regret uniform
        # bootstrap and made baseline/candidate cross-seed policies spuriously
        # almost identical.  Mirror the authoritative session state transition
        # explicitly and fail closed if it ever regresses again.
        bundle.counters["adv_optimizer_steps"] += int(chunk)
        session.behavior.ready = True
        bundle.counters["advantage_ready"] = 1
        if not session.behavior.ready or int(bundle.counters["advantage_ready"]) != 1:
            raise RuntimeError("custom Advantage fit did not activate neural CFR behavior")

        nrmse = _advantage_fit_nrmse(
            bundle,
            sample_size=int(args.audit_size),
            seed=audit_seed,
            device=args.device,
        )
        progress.append(
            {
                "optimizer_steps": int(local_steps),
                "weighted_nrmse": float(nrmse),
                "mean_training_loss": sum(losses) / max(len(losses), 1),
                "fit_target_reached": _finite(nrmse)
                and float(nrmse) <= float(args.advantage_fit_target),
                "frozen_gate_pass": _finite(nrmse)
                and float(nrmse) <= FROZEN_GATES["advantage_weighted_nrmse_max"],
                "behavior_ready_after_fit": bool(session.behavior.ready),
            }
        )
        if progress[-1]["fit_target_reached"]:
            break
    return progress


def run_mode(*, objective: str, aux_weight: float, seeds: list[int], solver: SolverLibrary, args):
    bundles = []
    reports = []
    episode = hu_episode()

    for seed in seeds:
        bundle = make_bundle(
            int(seed),
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
        advantage_rng = random.Random(int(seed) ^ ADV_RNG_XOR)
        strategy_rng = random.Random(int(seed) ^ STRATEGY_RNG_XOR)
        live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
        global_root = 0
        checkpoints = []

        for iteration in range(1, int(args.iterations) + 1):
            for _ in range(int(args.roots_per_iteration)):
                deck_seed = (
                    int(args.deck_stream_seed) * 1_000_003 + global_root * 97 + iteration
                ) & ((1 << 64) - 1)
                nodes = advantage_added = strategy_added = 0
                session.collector.rng = advantage_rng
                for traverser in live:
                    root = solver.create(episode, int(deck_seed))
                    try:
                        result = session.collector.collect_advantage(
                            root,
                            traverser=int(traverser),
                            iteration=int(iteration),
                        )
                    finally:
                        root.close()
                    nodes += int(result.nodes)
                    advantage_added += int(result.samples_added)
                session.collector.rng = strategy_rng
                for target_player in live:
                    root = solver.create(episode, int(deck_seed))
                    try:
                        strategy_added += int(
                            session.collector.collect_strategy_own_reach(
                                root,
                                target_player=int(target_player),
                                iteration=int(iteration),
                            )
                        )
                    finally:
                        root.close()
                c = bundle.counters
                c["iteration"] = max(int(c["iteration"]), int(iteration))
                c["roots"] += 1
                c["nodes"] += int(nodes)
                c["advantage_samples"] += int(advantage_added)
                c["strategy_samples"] += int(strategy_added)
                global_root += 1

            progress = _fit_advantage(
                session,
                bundle,
                seed=int(seed),
                iteration=int(iteration),
                objective=objective,
                aux_weight=float(aux_weight),
                args=args,
            )
            if not progress[-1]["behavior_ready_after_fit"]:
                raise RuntimeError("Advantage fit readiness invariant failed")
            checkpoints.append(
                {
                    "iteration": int(iteration),
                    "roots": int(bundle.counters["roots"]),
                    "advantage_samples": len(bundle.adv_mem.items),
                    "advantage_seen": int(bundle.adv_mem.seen),
                    "strategy_samples": len(bundle.pol_mem.items),
                    "strategy_seen": int(bundle.pol_mem.seen),
                    "advantage_optimizer_steps_total": int(bundle.counters["adv_optimizer_steps"]),
                    "final_advantage_fit": progress[-1],
                }
            )

        bundle.batch_rng = random.Random(int(seed) ^ 0xA9E12C7)
        policy_progress, _ = _fit_average_policy(
            bundle=bundle,
            session=session,
            seed=int(seed),
            device=args.device,
            policy_chunk_steps=int(args.policy_chunk_steps),
            policy_max_steps=int(args.policy_max_steps),
            policy_fit_target=float(args.policy_fit_target),
            batch_size=int(args.batch_size),
            audit_size=int(args.audit_size),
        )
        final_fit = audit_model_fit(
            bundle,
            sample_size=max(int(args.audit_size), 2048),
            seed=int(seed) ^ 0x2468ACE0,
            device=args.device,
        )
        reports.append(
            {
                "algorithm_seed": int(seed),
                "roots": int(bundle.counters["roots"]),
                "nodes": int(bundle.counters["nodes"]),
                "advantage_samples": len(bundle.adv_mem.items),
                "advantage_seen": int(bundle.adv_mem.seen),
                "strategy_samples": len(bundle.pol_mem.items),
                "strategy_seen": int(bundle.pol_mem.seen),
                "advantage_optimizer_steps": int(bundle.counters["adv_optimizer_steps"]),
                "checkpoints": checkpoints,
                "policy_progress": policy_progress,
                "final_fit": {
                    "advantage_weighted_nrmse": float(final_fit["advantage_weighted_nrmse"]),
                    "policy_weighted_mean_tv": float(final_fit["policy_weighted_mean_tv"]),
                    "advantage_gate_pass": float(final_fit["advantage_weighted_nrmse"])
                    <= FROZEN_GATES["advantage_weighted_nrmse_max"],
                    "policy_gate_pass": float(final_fit["policy_weighted_mean_tv"])
                    <= FROZEN_GATES["policy_weighted_mean_tv_max"],
                },
            }
        )
        bundles.append(bundle)

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
    return {
        "objective": objective,
        "aux_weight": float(aux_weight),
        "per_seed": reports,
        "cross_seed": {k: float(v) for k, v in cross.items()},
        "observation_count": len(observations),
        "all_fit_gates_pass": all(
            x["final_fit"]["advantage_gate_pass"] and x["final_fit"]["policy_gate_pass"]
            for x in reports
        ),
        "all_behavior_ready_after_fit": all(
            c["final_advantage_fit"]["behavior_ready_after_fit"]
            for x in reports for c in x["checkpoints"]
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="End-to-end behavior-aware Advantage objective screen")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument("--out", type=Path, default=Path("validation/R7_3_ADVANTAGE_OBJECTIVE_E2E_V2_256.json"))
    ap.add_argument("--aux-weight", type=float, default=0.10)
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
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
        raise SystemExit("requires exactly two seeds")
    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()
    baseline = run_mode(objective="mse", aux_weight=0.0, seeds=seeds, solver=solver, args=args)
    candidate = run_mode(
        objective="mse_policy",
        aux_weight=float(args.aux_weight),
        seeds=seeds,
        solver=solver,
        args=args,
    )
    if not baseline["all_behavior_ready_after_fit"] or not candidate["all_behavior_ready_after_fit"]:
        raise RuntimeError("behavior readiness invariant failed in end-to-end objective screen")
    mean_ratio = float(candidate["cross_seed"]["mean_tv"]) / max(
        float(baseline["cross_seed"]["mean_tv"]), 1e-12
    )
    p95_ratio = float(candidate["cross_seed"]["p95_tv"]) / max(
        float(baseline["cross_seed"]["p95_tv"]), 1e-12
    )
    payload = {
        "schema": "SPINCORE_R7_3_ADVANTAGE_OBJECTIVE_E2E_V2",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "baseline_mse": baseline,
        "mse_policy_candidate": candidate,
        "summary": {
            "aux_weight": float(args.aux_weight),
            "candidate_to_baseline_mean_tv_ratio": float(mean_ratio),
            "candidate_to_baseline_p95_tv_ratio": float(p95_ratio),
            "diagnosis": (
                "BEHAVIOR_AWARE_ADVANTAGE_OBJECTIVE_MATERIAL_END_TO_END"
                if candidate["all_fit_gates_pass"] and min(mean_ratio, p95_ratio) <= 0.85
                else "BEHAVIOR_AWARE_ADVANTAGE_OBJECTIVE_NOT_MATERIAL_END_TO_END"
            ),
        },
        "v1_evidence_invalidated": True,
        "v1_invalid_reason": (
            "V1 custom Advantage training bypassed DeepCFRDomainSession.train_advantage and failed "
            "to set NeuralAdvantagePolicy.ready=True; subsequent CFR iterations therefore remained "
            "on zero-regret uniform behavior."
        ),
        "interpretation_note": (
            "Diagnostic only. V2 explicitly activates neural CFR behavior after each custom "
            "Advantage fit. Baseline uses recovered weighted MSE; the candidate adds the smooth "
            "regret-policy auxiliary objective. Shared decks and split traversal RNGs remain a "
            "controlled diagnostic design, not the authoritative acceptance RNG/deck contract."
        ),
        "acceptance_gate_changed": False,
        "production_advantage_objective_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
