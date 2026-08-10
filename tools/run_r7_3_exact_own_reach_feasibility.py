from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from spincore.deep_cfr import DeepCFRDomainSession, icm_delta_utility, uniform_policy
from spincore.solver import SolverLibrary

from run_r7_3_diagnostic import hu_episode, make_bundle
from run_r7_3_support_overlap import _make_keyer


DEFAULT_DECK_STREAM_SEED = 0xD3C5EED
PAYOUT = (0.5, 0.3, 0.2)
SAMPLE_RNG_XOR = 0x0EAC7A11


class ExactTraversalCapExceeded(RuntimeError):
    pass


def _legal_mask(legal: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(1 if action in legal else 0 for action in range(6))


def _exact_walk(
    state,
    *,
    target_player: int,
    iteration: int,
    own_reach: float,
    observations: set[bytes],
    stats: dict,
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
        return

    actor = state.actor
    legal = state.legal_actions()
    observation = state.neural_bytes()
    sigma = uniform_policy(state, observation, legal)

    if actor == int(target_player):
        if own_reach > 0.0:
            observations.add(observation)
            stats["strategy_samples"] += 1
            stats["strategy_weight_sum"] += float(iteration) * float(own_reach)
        for action in legal:
            probability = float(sigma[action])
            if probability <= 0.0:
                continue
            child = state.child(action)
            try:
                _exact_walk(
                    child,
                    target_player=target_player,
                    iteration=iteration,
                    own_reach=float(own_reach) * probability,
                    observations=observations,
                    stats=stats,
                    node_cap=node_cap,
                    depth=depth + 1,
                    depth_cap=depth_cap,
                )
            finally:
                child.close()
        return

    # Recovered own-reach semantics enumerate non-target actions. Their policy
    # probability is deliberately NOT folded into target-player reach.
    for action in legal:
        child = state.child(action)
        try:
            _exact_walk(
                child,
                target_player=target_player,
                iteration=iteration,
                own_reach=own_reach,
                observations=observations,
                stats=stats,
                node_cap=node_cap,
                depth=depth + 1,
                depth_cap=depth_cap,
            )
        finally:
            child.close()


def exact_support_for_deals(
    *,
    solver: SolverLibrary,
    unique_roots: int,
    deck_stream_seed: int,
    node_cap_per_target_root: int,
    depth_cap: int,
):
    episode = hu_episode()
    live = [i for i, stack in enumerate(episode.stacks) if stack > 0]
    observations: set[bytes] = set()
    rows = []
    completed = True
    started = time.time()

    for root_index in range(int(unique_roots)):
        deck_seed = (
            int(deck_stream_seed) * 1_000_003 + root_index * 97 + 1
        ) & ((1 << 64) - 1)
        for player in live:
            stats = {
                "root_index": int(root_index),
                "target_player": int(player),
                "nodes": 0,
                "terminal_nodes": 0,
                "strategy_samples": 0,
                "strategy_weight_sum": 0.0,
                "max_depth": 0,
                "completed": True,
                "failure": None,
            }
            root = solver.create(episode, int(deck_seed))
            try:
                try:
                    _exact_walk(
                        root,
                        target_player=int(player),
                        iteration=1,
                        own_reach=1.0,
                        observations=observations,
                        stats=stats,
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
            if not stats["completed"]:
                break
        if not completed:
            break

    return observations, {
        "completed": bool(completed),
        "requested_unique_roots": int(unique_roots),
        "completed_target_roots": sum(1 for row in rows if row["completed"]),
        "node_cap_per_target_root": int(node_cap_per_target_root),
        "depth_cap": int(depth_cap),
        "duration_seconds": time.time() - started,
        "unique_raw_observations": len(observations),
        "total_nodes": sum(int(row["nodes"]) for row in rows),
        "total_strategy_samples": sum(int(row["strategy_samples"]) for row in rows),
        "max_depth": max((int(row["max_depth"]) for row in rows), default=0),
        "rows": rows,
    }


def sampled_support_for_deals(
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

    for root_index in range(int(unique_roots)):
        deck_seed = (
            int(deck_stream_seed) * 1_000_003 + root_index * 97 + 1
        ) & ((1 << 64) - 1)
        for player in live:
            for _ in range(int(replicates)):
                root = solver.create(episode, int(deck_seed))
                try:
                    session.collector.collect_strategy_own_reach(
                        root,
                        target_player=int(player),
                        iteration=1,
                    )
                finally:
                    root.close()

    return {item.observation for item in bundle.pol_mem.items}, {
        "replicates": int(replicates),
        "strategy_samples": len(bundle.pol_mem.items),
        "strategy_seen": int(bundle.pol_mem.seen),
        "unique_raw_observations": len({item.observation for item in bundle.pol_mem.items}),
    }


def _coverage(sampled: set[bytes], exact: set[bytes], key_mode: str) -> dict:
    keyer = _make_keyer(key_mode)
    sampled_keys = {keyer(x) for x in sampled}
    exact_keys = {keyer(x) for x in exact}
    shared = sampled_keys & exact_keys
    outside = sampled_keys - exact_keys
    return {
        "sampled_unique": len(sampled_keys),
        "exact_unique": len(exact_keys),
        "shared_unique": len(shared),
        "sampled_outside_exact": len(outside),
        "exact_support_coverage": len(shared) / max(len(exact_keys), 1),
        "sampled_precision_against_exact": len(shared) / max(len(sampled_keys), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Benchmark exact own-reach expectation enumeration against sampled own-reach "
            "collection under the uniform zero-regret bootstrap"
        )
    )
    ap.add_argument("--solver", type=Path, default=Path("build/libspincore_solver_c.so"))
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("validation/R7_3_EXACT_OWN_REACH_FEASIBILITY.json"),
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

    if args.unique_roots <= 0 or args.node_cap_per_target_root <= 0:
        raise SystemExit("roots and node cap must be positive")
    replicates = [int(x.strip()) for x in str(args.sample_replicates).split(",") if x.strip()]
    if not replicates or any(x <= 0 for x in replicates):
        raise SystemExit("sample replicate counts must be positive")

    torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
    solver = SolverLibrary(args.solver)
    started = time.time()

    exact, exact_report = exact_support_for_deals(
        solver=solver,
        unique_roots=int(args.unique_roots),
        deck_stream_seed=int(args.deck_stream_seed),
        node_cap_per_target_root=int(args.node_cap_per_target_root),
        depth_cap=int(args.depth_cap),
    )

    sampled_rows = []
    if exact_report["completed"]:
        for reps in replicates:
            sampled, sample_report = sampled_support_for_deals(
                solver=solver,
                unique_roots=int(args.unique_roots),
                deck_stream_seed=int(args.deck_stream_seed),
                replicates=int(reps),
                algorithm_seed=int(args.algorithm_seed),
                reservoir_capacity=int(args.reservoir_capacity),
                device=args.device,
            )
            sample_report["coverage"] = {
                mode: _coverage(sampled, exact, mode)
                for mode in ("raw", "poker_isomorphic")
            }
            sampled_rows.append(sample_report)

    payload = {
        "schema": "SPINCORE_R7_3_EXACT_OWN_REACH_FEASIBILITY_V1",
        "generated_at_unix": time.time(),
        "duration_seconds": time.time() - started,
        "solver": str(args.solver),
        "shared_deck_stream_seed": int(args.deck_stream_seed),
        "uniform_zero_regret_policy": True,
        "exact_enumeration": exact_report,
        "sampled_comparison": sampled_rows,
        "summary": {
            "exact_completed": bool(exact_report["completed"]),
            "exact_total_nodes": int(exact_report["total_nodes"]),
            "exact_total_strategy_samples": int(exact_report["total_strategy_samples"]),
            "exact_unique_raw_observations": int(exact_report["unique_raw_observations"]),
            "diagnosis": (
                "EXACT_OWN_REACH_ENUMERATION_FEASIBLE_AT_BENCHMARK_SCALE"
                if exact_report["completed"]
                else "EXACT_OWN_REACH_ENUMERATION_EXCEEDS_BENCHMARK_CAP"
            ),
        },
        "interpretation_note": (
            "Diagnostic only. Exact enumeration records every target-player infoset with its "
            "mathematically correct own-reach expectation weight by recursively enumerating "
            "target actions and multiplying reach by sigma. Non-target actions remain enumerated "
            "without entering target-player reach, matching the recovered own-reach estimator. "
            "The benchmark tests computational feasibility and measures how much of that exact "
            "support is covered by 1/4/8 sampled own-reach trajectories. No production contract "
            "or acceptance gate is changed."
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
