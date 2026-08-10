from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility, regret_matching_policy, uniform_policy
from spincore.solver import SolverLibrary
from spincore_nn.reservoir import AdvantageSample

from run_r7_3_advantage_target_curve import _aggregate_advantage
from run_r7_3_diagnostic import hu_episode, make_bundle
from run_r7_3_support_overlap import _make_keyer, _quantile


DEFAULT_DECK_STREAM_SEED = 0xD3C5EED
PAYOUT = (0.5, 0.3, 0.2)
SAMPLE_RNG_XOR = 0x0EAA71CE


class ExactTraversalCapExceeded(RuntimeError):
    pass


def _exact_adv(
    state,
    *,
    traverser: int,
    opponent_reach: float,
    samples: list[AdvantageSample],
    stats: dict,
    terminal_utility,
    node_cap: int,
    depth: int,
    depth_cap: int,
):
    stats["nodes"] += 1
    stats["max_depth"] = max(int(stats["max_depth"]), int(depth))
    if stats["nodes"] > int(node_cap):
        raise ExactTraversalCapExceeded("node cap exceeded")
    if depth > int(depth_cap):
        raise ExactTraversalCapExceeded("depth cap exceeded")
    if state.terminal:
        stats["terminal_nodes"] += 1
        return float(terminal_utility(state)[int(traverser)])

    actor = state.actor
    legal = state.legal_actions()
    observation = state.neural_bytes()
    sigma = uniform_policy(state, observation, legal)

    if actor == int(traverser):
        values = [0.0] * 6
        for action in legal:
            child = state.child(action)
            try:
                values[action] = _exact_adv(
                    child,
                    traverser=traverser,
                    opponent_reach=opponent_reach,
                    samples=samples,
                    stats=stats,
                    terminal_utility=terminal_utility,
                    node_cap=node_cap,
                    depth=depth + 1,
                    depth_cap=depth_cap,
                )
            finally:
                child.close()
        node_value = sum(float(sigma[a]) * float(values[a]) for a in legal)
        target = [0.0] * 6
        for action in legal:
            target[action] = float(values[action]) - float(node_value)
        if opponent_reach > 0.0:
            samples.append(
                AdvantageSample(
                    observation,
                    tuple(1 if action in legal else 0 for action in range(6)),
                    tuple(target),
                    float(opponent_reach),
                    1,
                )
            )
            stats["advantage_samples"] += 1
            stats["advantage_weight_sum"] += float(opponent_reach)
        return float(node_value)

    # Exact expectation of the external-sampling opponent node. The sampled
    # estimator chooses one action ~sigma. Here we enumerate every action and
    # weight both returned utility and downstream training-distribution mass by
    # opponent reach * sigma[action].
    value = 0.0
    for action in legal:
        probability = float(sigma[action])
        if probability <= 0.0:
            continue
        child = state.child(action)
        try:
            child_value = _exact_adv(
                child,
                traverser=traverser,
                opponent_reach=float(opponent_reach) * probability,
                samples=samples,
                stats=stats,
                terminal_utility=terminal_utility,
                node_cap=node_cap,
                depth=depth + 1,
                depth_cap=depth_cap,
            )
        finally:
            child.close()
        value += probability * float(child_value)
    return float(value)


def exact_advantage_samples(
    *,
    solver: SolverLibrary,
    unique_roots: int,
    deck_stream_seed: int,
    node_cap_per_target_root: int,
    depth_cap: int,
):
    episode = hu_episode()
    live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
    terminal_utility = icm_delta_utility(PAYOUT)
    samples: list[AdvantageSample] = []
    rows = []
    completed = True
    started = time.time()

    for root_index in range(int(unique_roots)):
        deck_seed = (
            int(deck_stream_seed) * 1_000_003 + root_index * 97 + 1
        ) & ((1 << 64) - 1)
        for traverser in live:
            stats = {
                "root_index": int(root_index),
                "traverser": int(traverser),
                "nodes": 0,
                "terminal_nodes": 0,
                "advantage_samples": 0,
                "advantage_weight_sum": 0.0,
                "max_depth": 0,
                "completed": True,
                "failure": None,
            }
            root = solver.create(episode, int(deck_seed))
            try:
                try:
                    _exact_adv(
                        root,
                        traverser=int(traverser),
                        opponent_reach=1.0,
                        samples=samples,
                        stats=stats,
                        terminal_utility=terminal_utility,
                        node_cap=int(node_cap_per_target_root),
                        depth=0,
                        depth_cap=int(depth_cap),
                    )
                except ExactTraversalCapExceeded as exc:
                    stats["completed"] = False
                    stats["failure"] = str(exc)
                    completed = False
            finally:
                root.close()
            rows.append(stats)
            if not completed:
                break
        if not completed:
            break

    return samples, {
        "completed": bool(completed),
        "requested_unique_roots": int(unique_roots),
        "completed_target_roots": sum(1 for row in rows if row["completed"]),
        "node_cap_per_target_root": int(node_cap_per_target_root),
        "depth_cap": int(depth_cap),
        "duration_seconds": time.time() - started,
        "total_nodes": sum(int(row["nodes"]) for row in rows),
        "total_advantage_samples": len(samples),
        "total_advantage_weight": sum(float(x.weight) for x in samples),
        "max_depth": max((int(row["max_depth"]) for row in rows), default=0),
        "rows": rows,
    }


def sampled_advantage_samples(
    *,
    solver: SolverLibrary,
    unique_roots: int,
    deck_stream_seed: int,
    replicates: int,
    algorithm_seed: int,
    reservoir_capacity: int,
    device: str,
):
    bundle = make_bundle(
        int(algorithm_seed),
        device=device,
        reservoir_capacity=int(reservoir_capacity),
        lr=1e-3,
    )
    session = DeepCFRDomainSession(
        solver_library=solver,
        bundle=bundle,
        terminal_utility=icm_delta_utility(PAYOUT),
        device=device,
    )
    if session.behavior.ready:
        raise RuntimeError("uniform bootstrap unexpectedly ready")
    session.collector.rng = random.Random(int(algorithm_seed) ^ SAMPLE_RNG_XOR)
    episode = hu_episode()
    live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
    nodes = 0

    for root_index in range(int(unique_roots)):
        deck_seed = (
            int(deck_stream_seed) * 1_000_003 + root_index * 97 + 1
        ) & ((1 << 64) - 1)
        for traverser in live:
            for _ in range(int(replicates)):
                root = solver.create(episode, int(deck_seed))
                try:
                    result = session.collector.collect_advantage(
                        root,
                        traverser=int(traverser),
                        iteration=1,
                    )
                finally:
                    root.close()
                nodes += int(result.nodes)

    return list(bundle.adv_mem.items), {
        "replicates": int(replicates),
        "advantage_samples": len(bundle.adv_mem.items),
        "advantage_seen": int(bundle.adv_mem.seen),
        "nodes": int(nodes),
    }


def compare_to_exact(exact_items, sampled_items, key_mode: str) -> dict:
    keyer = _make_keyer(key_mode)
    exact = _aggregate_advantage(exact_items, keyer)
    sampled = _aggregate_advantage(sampled_items, keyer)
    exact_keys = set(exact)
    sampled_keys = set(sampled)
    shared = exact_keys & sampled_keys

    exact_total_weight = sum(float(row["weight"]) for row in exact.values())
    exact_shared_weight = sum(float(exact[key]["weight"]) for key in shared)
    sampled_outside = sampled_keys - exact_keys

    weights = []
    tvs = []
    squared_error = 0.0
    exact_energy = 0.0
    greedy = []
    for key in shared:
        e = exact[key]
        s = sampled[key]
        if e["legal"] != s["legal"]:
            raise RuntimeError("legal mask mismatch against exact expectation")
        legal = tuple(i for i, yes in enumerate(e["legal"]) if yes)
        weight = float(e["weight"])
        weights.append(weight)
        for action in legal:
            exact_value = float(e["target"][action])
            sampled_value = float(s["target"][action])
            squared_error += weight * (sampled_value - exact_value) ** 2
            exact_energy += weight * exact_value * exact_value
        pe = regret_matching_policy(e["target"], legal)
        ps = regret_matching_policy(s["target"], legal)
        tvs.append(0.5 * sum(abs(a - b) for a, b in zip(pe, ps)))
        greedy_e = max(legal, key=lambda a: float(e["target"][a]))
        greedy_s = max(legal, key=lambda a: float(s["target"][a]))
        greedy.append(1.0 if greedy_e == greedy_s else 0.0)

    total = sum(weights)
    weighted_tv = (
        sum(w * tv for w, tv in zip(weights, tvs)) / total if total > 0.0 else math.inf
    )
    weighted_greedy = (
        sum(w * x for w, x in zip(weights, greedy)) / total if total > 0.0 else 0.0
    )
    relative_rmse = math.sqrt(squared_error / max(exact_energy, 1e-12)) if total > 0.0 else math.inf

    return {
        "exact_unique": len(exact_keys),
        "sampled_unique": len(sampled_keys),
        "shared_unique": len(shared),
        "sampled_outside_exact": len(sampled_outside),
        "exact_weight_coverage": exact_shared_weight / max(exact_total_weight, 1e-12),
        "sampled_precision_against_exact": len(shared) / max(len(sampled_keys), 1),
        "target_relative_rmse_on_shared": float(relative_rmse),
        "regret_matching_weighted_mean_tv": float(weighted_tv),
        "regret_matching_p50_tv": _quantile(tvs, 0.50),
        "regret_matching_p95_tv": _quantile(tvs, 0.95),
        "weighted_greedy_action_agreement": float(weighted_greedy),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark exact opponent expectation for Advantage external sampling"
    )
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("validation/R7_3_EXACT_ADVANTAGE_FEASIBILITY.json"),
    )
    ap.add_argument("--unique-roots", type=int, default=4)
    ap.add_argument("--deck-stream-seed", type=int, default=DEFAULT_DECK_STREAM_SEED)
    ap.add_argument("--node-cap-per-target-root", type=int, default=1000000)
    ap.add_argument("--depth-cap", type=int, default=128)
    ap.add_argument("--sample-replicates", default="1,4,8")
    ap.add_argument("--algorithm-seed", type=int, default=20260829)
    ap.add_argument("--reservoir-capacity", type=int, default=250000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    reps = [int(x.strip()) for x in str(args.sample_replicates).split(",") if x.strip()]
    if not reps or any(x <= 0 for x in reps):
        raise SystemExit("sample replicate counts must be positive")

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()

    exact, exact_report = exact_advantage_samples(
        solver=solver,
        unique_roots=int(args.unique_roots),
        deck_stream_seed=int(args.deck_stream_seed),
        node_cap_per_target_root=int(args.node_cap_per_target_root),
        depth_cap=int(args.depth_cap),
    )

    sampled_rows = []
    if exact_report["completed"]:
        for reps_count in reps:
            sampled, report = sampled_advantage_samples(
                solver=solver,
                unique_roots=int(args.unique_roots),
                deck_stream_seed=int(args.deck_stream_seed),
                replicates=int(reps_count),
                algorithm_seed=int(args.algorithm_seed),
                reservoir_capacity=int(args.reservoir_capacity),
                device=args.device,
            )
            report["comparison_to_exact"] = {
                mode: compare_to_exact(exact, sampled, mode)
                for mode in ("raw", "poker_isomorphic")
            }
            sampled_rows.append(report)

    payload = {
        "schema": "SPINCORE_R7_3_EXACT_ADVANTAGE_FEASIBILITY_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "shared_deck_stream_seed": int(args.deck_stream_seed),
        "uniform_zero_regret_policy": True,
        "exact_opponent_expectation": exact_report,
        "sampled_comparison": sampled_rows,
        "summary": {
            "exact_completed": bool(exact_report["completed"]),
            "exact_total_nodes": int(exact_report["total_nodes"]),
            "exact_total_advantage_samples": int(exact_report["total_advantage_samples"]),
            "exact_total_advantage_weight": float(exact_report["total_advantage_weight"]),
            "diagnosis": (
                "EXACT_ADVANTAGE_OPPONENT_EXPECTATION_FEASIBLE_AT_BENCHMARK_SCALE"
                if exact_report["completed"]
                else "EXACT_ADVANTAGE_OPPONENT_EXPECTATION_EXCEEDS_BENCHMARK_CAP"
            ),
        },
        "interpretation_note": (
            "Exact control for the recovered external-sampling Advantage estimator. Traverser "
            "actions remain enumerated as before; opponent actions are also enumerated, with "
            "downstream training-distribution mass weighted by opponent reach. This is the exact "
            "expectation of sampling one opponent action ~sigma. Sampled 1/4/8-path memories are "
            "compared directly to the exact target vectors and regret-matching policies on the "
            "same hidden deals. Diagnostic only; no production estimator or gate is changed."
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
