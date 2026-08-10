from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import (
    DeepCFRDomainSession,
    ExternalSamplingCollector,
    TraversalResult,
    icm_delta_utility,
    sample_action,
)
from spincore.r7 import FROZEN_GATES, audit_model_fit, cross_seed_policy_tv
from spincore.solver import SolverLibrary
from spincore_nn.reservoir import AdvantageSample

from run_r7_3_diagnostic import hu_episode, make_bundle, shared_cross_seed_observations
from run_r7_3_variance_decomposition import _advantage_fit_nrmse, _finite


DEFAULT_SEEDS = (20260829, 20260807)
DEFAULT_SHARED_DECK_STREAM_SEED = 0xD3C5EED
PAYOUT = (0.5, 0.3, 0.2)
ADV_RNG_XOR = 0x0A9D7A61
STRATEGY_RNG_XOR = 0x057A7E61


class PartialExactAdvantageCollector(ExternalSamplingCollector):
    """External-sampling Advantage collector with bounded opponent enumeration.

    `exact_opponent_levels=0` is intentionally identical to the recovered
    external-sampling collector. For positive levels, the next N opponent
    decisions on each path are enumerated exactly. Samples below an enumerated
    opponent branch receive an explicit probability-mass multiplier; later
    sampled opponent decisions remain represented by Monte-Carlo occurrence
    probability. This preserves the expected Advantage training distribution.
    """

    def collect_advantage_partial_exact(
        self,
        root,
        *,
        traverser: int,
        iteration: int,
        exact_opponent_levels: int,
    ):
        if iteration <= 0:
            raise ValueError("positive iteration required")
        if exact_opponent_levels < 0:
            raise ValueError("nonnegative exact_opponent_levels required")
        utility, nodes, added = self._adv_partial(
            root,
            int(traverser),
            int(iteration),
            int(exact_opponent_levels),
            1.0,
        )
        return TraversalResult(float(utility), int(nodes), int(added))

    def _adv_partial(
        self,
        state,
        traverser: int,
        iteration: int,
        exact_levels: int,
        explicit_opponent_mass: float,
    ):
        if state.terminal:
            return float(self.terminal_utility(state)[traverser]), 1, 0

        actor = state.actor
        legal = state.legal_actions()
        observation = state.neural_bytes()
        sigma = self._p(state, observation, legal)

        if actor == traverser:
            values = [0.0] * 6
            nodes = 1
            added = 0
            for action in legal:
                child = state.child(action)
                try:
                    value, child_nodes, child_added = self._adv_partial(
                        child,
                        traverser,
                        iteration,
                        exact_levels,
                        explicit_opponent_mass,
                    )
                finally:
                    child.close()
                values[action] = float(value)
                nodes += int(child_nodes)
                added += int(child_added)

            node_value = sum(float(sigma[a]) * float(values[a]) for a in legal)
            target = [0.0] * 6
            for action in legal:
                target[action] = float(values[action]) - float(node_value)
            if explicit_opponent_mass > 0.0:
                self.advantage_memory.add(
                    AdvantageSample(
                        observation,
                        tuple(1 if action in legal else 0 for action in range(6)),
                        tuple(target),
                        float(iteration) * float(explicit_opponent_mass),
                        int(iteration),
                    )
                )
                added += 1
            return float(node_value), int(nodes), int(added)

        if exact_levels > 0:
            value = 0.0
            nodes = 1
            added = 0
            for action in legal:
                probability = float(sigma[action])
                if probability <= 0.0:
                    continue
                child = state.child(action)
                try:
                    child_value, child_nodes, child_added = self._adv_partial(
                        child,
                        traverser,
                        iteration,
                        exact_levels - 1,
                        float(explicit_opponent_mass) * probability,
                    )
                finally:
                    child.close()
                value += probability * float(child_value)
                nodes += int(child_nodes)
                added += int(child_added)
            return float(value), int(nodes), int(added)

        action = sample_action(sigma, legal, self.rng)
        child = state.child(action)
        try:
            value, nodes, added = self._adv_partial(
                child,
                traverser,
                iteration,
                0,
                explicit_opponent_mass,
            )
        finally:
            child.close()
        return float(value), int(nodes) + 1, int(added)


def _fit_pass(value: float, gate_name: str) -> bool:
    return _finite(value) and float(value) <= float(FROZEN_GATES[gate_name])


def _safe_ratio(a: float, b: float) -> float:
    return float(a) / max(float(b), 1e-12)


def run_mode(
    *,
    exact_opponent_levels: int,
    seeds: list[int],
    solver: SolverLibrary,
    device: str,
    deck_stream_seed: int,
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
    bundles = []
    reports = []
    episode = hu_episode()

    for seed in seeds:
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
        advantage_rng = random.Random(int(seed) ^ ADV_RNG_XOR)
        strategy_rng = random.Random(int(seed) ^ STRATEGY_RNG_XOR)
        advantage_collector = PartialExactAdvantageCollector(
            policy=session.behavior,
            terminal_utility=session.terminal_utility,
            rng=advantage_rng,
            advantage_memory=bundle.adv_mem,
            strategy_memory=bundle.pol_mem,
        )
        session.collector.rng = strategy_rng
        live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
        global_root = 0
        checkpoints = []

        for iteration in range(1, int(iterations) + 1):
            for _ in range(int(roots_per_iteration)):
                deck_seed = (
                    int(deck_stream_seed) * 1_000_003 + global_root * 97 + iteration
                ) & ((1 << 64) - 1)
                nodes = 0
                advantage_added = 0
                strategy_added = 0
                for player in live:
                    root = solver.create(episode, int(deck_seed))
                    try:
                        result = advantage_collector.collect_advantage_partial_exact(
                            root,
                            traverser=int(player),
                            iteration=int(iteration),
                            exact_opponent_levels=int(exact_opponent_levels),
                        )
                    finally:
                        root.close()
                    nodes += int(result.nodes)
                    advantage_added += int(result.samples_added)
                for player in live:
                    root = solver.create(episode, int(deck_seed))
                    try:
                        strategy_added += int(
                            session.collector.collect_strategy_own_reach(
                                root,
                                target_player=int(player),
                                iteration=int(iteration),
                            )
                        )
                    finally:
                        root.close()

                counters = bundle.counters
                counters["iteration"] = max(int(counters["iteration"]), int(iteration))
                counters["roots"] += 1
                counters["nodes"] += int(nodes)
                counters["advantage_samples"] += int(advantage_added)
                counters["strategy_samples"] += int(strategy_added)
                global_root += 1

            reset_seed = (int(seed) ^ (iteration * 0x9E3779B1)) & 0x7FFFFFFF
            session.reset_advantage_network(init_seed=reset_seed, lr=lr)
            local_steps = 0
            progress = []
            audit_seed = int(seed) ^ (iteration * 0x45D9F3B)
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
                progress.append(row)
                print(
                    json.dumps(
                        {
                            "exact_opponent_levels": int(exact_opponent_levels),
                            "seed": int(seed),
                            "iteration": int(iteration),
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
                    "advantage_samples": len(bundle.adv_mem.items),
                    "advantage_seen": int(bundle.adv_mem.seen),
                    "strategy_samples": len(bundle.pol_mem.items),
                    "strategy_seen": int(bundle.pol_mem.seen),
                    "nodes": int(bundle.counters["nodes"]),
                    "advantage_progress": progress,
                    "final_advantage_fit": progress[-1],
                }
            )

        policy_progress = []
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
            tv = float(fit["policy_weighted_mean_tv"])
            row = {
                "optimizer_steps": int(bundle.counters["policy_optimizer_steps"]),
                "weighted_mean_tv": tv,
                "frozen_gate_pass": _fit_pass(tv, "policy_weighted_mean_tv_max"),
                "fit_target_reached": _finite(tv) and tv <= float(policy_fit_target),
            }
            policy_progress.append(row)
            if row["fit_target_reached"]:
                break

        final_fit = audit_model_fit(
            bundle,
            sample_size=max(int(audit_size), 2048),
            seed=int(seed) ^ 0x2468ACE0,
            device=device,
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
        )
        bundles.append(bundle)

    observations = shared_cross_seed_observations(
        bundles,
        per_seed=int(cross_seed_per_seed),
        seed=0x715EED,
    )
    cross = cross_seed_policy_tv(
        bundles[0].policy,
        bundles[1].policy,
        observations,
        device=device,
    )
    return {
        "exact_opponent_levels": int(exact_opponent_levels),
        "collection": reports,
        "cross_seed": {k: float(v) for k, v in cross.items()},
        "cross_seed_observation_count": len(observations),
        "all_fit_gates_pass": all(
            bool(row["final_fit"]["advantage_gate_pass"])
            and bool(row["final_fit"]["policy_gate_pass"])
            for row in reports
        ),
        "total_nodes": sum(int(row["nodes"]) for row in reports),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="R7.3 partial-exact opponent Advantage screen")
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("validation/R7_3_PARTIAL_EXACT_ADVANTAGE_SCREEN_256.json"),
    )
    ap.add_argument(
        "--reference",
        type=Path,
        default=Path("validation/R7_3_PATH_REPLICATION_SCREEN_256.json"),
    )
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--levels", default="0,1,2")
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
    ap.add_argument("--reservoir-capacity", type=int, default=250000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if len(seeds) != 2:
        raise SystemExit("partial-exact screen requires exactly two seeds")
    levels = [int(x.strip()) for x in str(args.levels).split(",") if x.strip()]
    if not levels or 0 not in levels or any(x < 0 for x in levels):
        raise SystemExit("levels must be nonnegative and include 0 baseline")

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()

    modes = {}
    for level in levels:
        modes[str(level)] = run_mode(
            exact_opponent_levels=int(level),
            seeds=seeds,
            solver=solver,
            device=args.device,
            deck_stream_seed=int(args.deck_stream_seed),
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

    baseline = modes["0"]
    baseline_mean = float(baseline["cross_seed"]["mean_tv"])
    baseline_p95 = float(baseline["cross_seed"]["p95_tv"])
    baseline_nodes = float(baseline["total_nodes"])

    reference_check = None
    if args.reference.exists() and int(args.iterations) == 2 and int(args.roots_per_iteration) == 128:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        reference_mean = float(reference["summary"]["baseline_cross_seed_mean_tv"])
        reference_p95 = float(reference["summary"]["baseline_cross_seed_p95_tv"])
        reference_check = {
            "reference_mean_tv": reference_mean,
            "reference_p95_tv": reference_p95,
            "mean_abs_delta": abs(baseline_mean - reference_mean),
            "p95_abs_delta": abs(baseline_p95 - reference_p95),
        }
        if reference_check["mean_abs_delta"] > 1e-9 or reference_check["p95_abs_delta"] > 1e-9:
            raise RuntimeError("level-0 custom collector failed exact baseline reproduction")

    comparisons = {}
    for level in levels:
        if level == 0:
            continue
        report = modes[str(level)]
        comparisons[str(level)] = {
            "mean_tv_ratio_to_baseline": _safe_ratio(
                float(report["cross_seed"]["mean_tv"]), baseline_mean
            ),
            "p95_tv_ratio_to_baseline": _safe_ratio(
                float(report["cross_seed"]["p95_tv"]), baseline_p95
            ),
            "node_ratio_to_baseline": _safe_ratio(float(report["total_nodes"]), baseline_nodes),
        }

    # Reference independent x4 result from the already completed controlled path screen.
    independent_x4_reference = None
    if args.reference.exists():
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        adv_x4 = reference["modes"]["advantage_x4"]["cross_seed_internal_corpus"]
        independent_x4_reference = {
            "mean_tv": float(adv_x4["mean_tv"]),
            "p95_tv": float(adv_x4["p95_tv"]),
        }

    best_level = min(
        (level for level in levels if level > 0),
        key=lambda level: float(modes[str(level)]["cross_seed"]["mean_tv"]),
        default=0,
    )
    best = modes[str(best_level)] if best_level else baseline
    best_mean_ratio = _safe_ratio(float(best["cross_seed"]["mean_tv"]), baseline_mean)

    payload = {
        "schema": "SPINCORE_R7_3_PARTIAL_EXACT_ADVANTAGE_SCREEN_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "algorithm_seeds": seeds,
        "shared_deck_stream_seed": int(args.deck_stream_seed),
        "levels": levels,
        "iterations": int(args.iterations),
        "roots_per_iteration": int(args.roots_per_iteration),
        "modes": modes,
        "comparisons_to_level0": comparisons,
        "level0_reference_reproduction": reference_check,
        "independent_advantage_x4_reference": independent_x4_reference,
        "summary": {
            "baseline_mean_tv": baseline_mean,
            "baseline_p95_tv": baseline_p95,
            "best_partial_exact_level_by_mean_tv": int(best_level),
            "best_partial_exact_mean_tv": float(best["cross_seed"]["mean_tv"]),
            "best_partial_exact_p95_tv": float(best["cross_seed"]["p95_tv"]),
            "best_partial_exact_mean_ratio_to_baseline": float(best_mean_ratio),
            "diagnosis": (
                "PARTIAL_EXACT_OPPONENT_EXPECTATION_MATERIAL"
                if best_level > 0 and best_mean_ratio <= 0.85
                else "PARTIAL_EXACT_OPPONENT_EXPECTATION_NOT_MATERIAL_AT_SCREEN_SCALE"
            ),
        },
        "interpretation_note": (
            "Versioned-estimator diagnostic only. Level 0 is the recovered external-sampling "
            "Advantage estimator and is required to reproduce the prior controlled baseline "
            "exactly. Positive levels enumerate the next N opponent decisions on each path, "
            "weighting downstream Advantage samples by explicit opponent probability mass while "
            "later opponent decisions remain sampled. This preserves the external-sampling "
            "training-distribution expectation and trades additional tree work for lower variance."
        ),
        "acceptance_gate_changed": False,
        "production_estimator_changed": False,
        "ready_for_tables": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
