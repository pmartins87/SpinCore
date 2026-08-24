from __future__ import annotations

import argparse
from pathlib import Path

import r7_5_arch_reset_v1plus_phase2b6_preflop_damping_training_pilot as b6
import r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition as b10
import r7_5_arch_reset_v1plus_phase2b13_root_iid64_target_training as b13
import r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance as b15
import r7_5_arch_reset_v1plus_phase2c2_range_reach_target_kernel_causal_pilot as c2
import spincore.r7_5_representation_v3_stage as stage

from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_representation_v3_stage_contract import (
    EXACT_OPPONENT_LEVELS,
    TRAINING_SEEDS,
    deck_seed,
)
from spincore.solver import SolverLibrary


def run_one(repo_root: Path, solver_path: Path, b13_root: Path, seed: int) -> None:
    checkpoint = b13_root / b13.CANDIDATE_ARM / f"seed_{seed}" / "resume_checkpoint.pt"
    source_states, _identity = b15._load_behavior_states(checkpoint, int(seed))
    b10._worker_init(str(repo_root), str(solver_path), int(seed), source_states)

    task = {
        "training_seed": int(seed),
        "scenario_index": 0,
        "global_root": 0,
        "iteration": 1,
        "anchor_deck_seed": int(deck_seed(int(seed), 0, 1)),
    }
    aux = c2._combined_aux_task(task)
    root_row = aux["root"]
    cont_row = aux["continuation"]

    solver = SolverLibrary(solver_path)
    config = stage.frozen_config()
    bundle, _unused_session, _unused_behavior, spec, _state = stage.new_phase2_v3_runtime(
        repo_root,
        solver=solver,
        representation=c2.REPRESENTATION,
        domain=c2.DOMAIN,
        training_seed=int(seed),
        config=config,
    )
    behavior = b6._make_behavior_from_states(source_states, config=config)
    session = stage._make_session(solver, bundle, spec, behavior)
    floor_policy = b6.PreflopContinuationFloorPolicy(behavior, floor=c2.FLOOR)
    session.collector.policy = floor_policy

    proxy = c2.MultiReplacingAdvantageMemory(
        bundle.adv_mem,
        iteration=1,
        replacements=[
            {
                "label": "ROOT_IID64_MEAN",
                "observation": bytes(root_row["root_observation"]),
                "target": root_row["mean_target"],
                "legal_mask": root_row["legal_mask"],
            },
            {
                "label": "DEPTH2_RANGE_TARGET",
                "observation": bytes(cont_row["observation"]),
                "target": cont_row["mean_target"],
                "legal_mask": cont_row["legal_mask"],
            },
        ],
    )
    original = session.collector.advantage_memory
    session.collector.advantage_memory = proxy
    try:
        session.collect_root(
            action_scenario_cycle(c2.DOMAIN)[0],
            iteration=1,
            exact_opponent_levels=EXACT_OPPONENT_LEVELS,
            deck_seed=int(task["anchor_deck_seed"]),
        )
    finally:
        session.collector.advantage_memory = original
    proxy.assert_complete()
    if int(root_row["aux_traversals"]) != c2.K:
        raise RuntimeError("Phase2C2 live preflight root K drift")
    if int(cont_row["target_traversals"]) != c2.K:
        raise RuntimeError("Phase2C2 live preflight continuation K drift")
    print(
        f"Phase2C2 live replacement seed={seed} PASS "
        f"root_obs={root_row['root_observation_sha256'][:12]} "
        f"cont_obs={cont_row['observation_sha256'][:12]} "
        f"adv_seen={bundle.adv_mem.seen}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase2C2 live exact replacement preflight")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--phase2b13-root", type=Path, required=True)
    args = parser.parse_args()
    for seed in map(int, TRAINING_SEEDS):
        run_one(args.repo_root.resolve(), args.solver.resolve(), args.phase2b13_root.resolve(), seed)
    print("Phase2C2 live root+continuation replacement preflight PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
