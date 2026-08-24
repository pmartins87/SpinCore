from __future__ import annotations

"""Windows replay runtime correction for Phase2B15.

The frozen heldout V3 corpus was generated on Linux.  Its historical deck_seed
cannot be replayed portably on Windows because HandEngine's seeded constructor
uses std::shuffle, whose permutation algorithm is not specified across C++
standard-library implementations.  Phase2B15 only needs the current actor's
preflop private cards as the conditioning variable for its conditional-IID
proposal distribution.  SPNNIV3 carries those cards losslessly up to suit
symmetry (ordered ranks plus the same-suit relation), so this correction builds
a canonical suit-isomorphic explicit deal, replays the stored public action
path, and requires byte-identical SPNNIV3/actor/legal identity before any target
traversal.

Scientific design, K, anchors, blocks, behavior policies, gates, and target
semantics are unchanged.
"""

import argparse
import json
import math
import random
from pathlib import Path

import r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance as b15
import r7_5_arch_reset_v1plus_phase2b10_private_public_chance_decomposition as b10
import r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator as b11

from spincore.r7_5_action_scenarios import action_scenario_cycle
from spincore.r7_5_representation_v3_referee_states import effective_pf0
from spincore.r7_5_representation_v3_stage_contract import validate_phase2_v3_contract
from spincore.solver import SolverLibrary
from spincore.solver_v3 import neural_bytes_v3

FIX_SCHEMA = "SPINCORE_R7_5_ARCH_RESET_V1PLUS_PHASE2B15_WINDOWS_HELDOUT_REPLAY_RUNTIMEFIX_V1"


def _canonical_actor_cards_from_observation(observation: bytes) -> tuple[int, int]:
    obs = bytes(observation)
    if len(obs) < 120 or not obs.startswith(b"SPNNIV3\x00"):
        raise RuntimeError("Phase2B15 runtimefix requires authoritative SPNNIV3 bytes")
    # Wire layout: magic[8], categorical[10], rank_tokens[7], same_suit[21].
    rank0 = int(obs[18])
    rank1 = int(obs[19])
    same_suit = int(obs[25])  # first lexicographic relation is card slots (0,1)
    if not (2 <= rank0 <= 14 and 2 <= rank1 <= 14):
        raise RuntimeError("Phase2B15 runtimefix invalid preflop actor rank token")
    if same_suit not in (0, 1):
        raise RuntimeError("Phase2B15 runtimefix invalid hole-card suit relation")
    if same_suit and rank0 == rank1:
        raise RuntimeError("Phase2B15 runtimefix impossible suited pocket pair")

    suit0 = 0
    suit1 = 0 if same_suit else 1
    card0 = (rank0 - 2) * 4 + suit0
    card1 = (rank1 - 2) * 4 + suit1
    if card0 == card1:
        raise RuntimeError("Phase2B15 runtimefix canonical actor cards collide")
    return int(card0), int(card1)


def _canonical_explicit_deal(task: dict):
    actor = int(task["actor"])
    if actor not in (0, 1, 2):
        raise RuntimeError("Phase2B15 runtimefix invalid actor")
    actor_cards = _canonical_actor_cards_from_observation(bytes(task["observation"]))
    remaining = [card for card in range(52) if card not in set(actor_cards)]
    holes = [[-1, -1] for _ in range(3)]
    holes[actor] = [actor_cards[0], actor_cards[1]]
    cursor = 0
    for seat in range(3):
        if seat == actor:
            continue
        holes[seat] = [remaining[cursor], remaining[cursor + 1]]
        cursor += 2
    board = remaining[cursor:cursor + 5]
    return tuple(tuple(int(x) for x in row) for row in holes), tuple(int(x) for x in board)


def _canonical_snapshot_with(solver, action_spec, task: dict):
    scenarios = action_scenario_cycle(b15.DOMAIN)
    holes, board = _canonical_explicit_deal(task)
    state = solver.create_with_deal(
        scenarios[int(task["scenario_index"])], holes, board
    )
    try:
        for action in task["action_path"]:
            if state.terminal:
                raise RuntimeError("Phase2B15 runtimefix canonical path reaches terminal early")
            active_mask, legal, _exact = effective_pf0(state, action_spec)
            if int(action) not in legal:
                raise RuntimeError("Phase2B15 runtimefix canonical path action is illegal")
            state.apply_universal(active_mask, int(action))
        if state.terminal:
            raise RuntimeError("Phase2B15 runtimefix canonical continuation is terminal")
        observation = neural_bytes_v3(state)
        active_mask, legal, _exact = effective_pf0(state, action_spec)
        if observation != bytes(task["observation"]):
            raise RuntimeError("Phase2B15 runtimefix canonical explicit-deal observation drift")
        if int(state.actor) != int(task["actor"]):
            raise RuntimeError("Phase2B15 runtimefix canonical actor drift")
        if int(active_mask) != int(task["active_mask"]) or tuple(legal) != tuple(task["legal_slots"]):
            raise RuntimeError("Phase2B15 runtimefix canonical legal identity drift")
        snapshot = state.deal_snapshot()
    finally:
        state.close()
    if snapshot.visible_board_count != 0:
        raise RuntimeError("Phase2B15 runtimefix anchors must remain preflop")
    # B11 conditional-IID generation consumes only snapshot.holes[actor] from
    # this canonical snapshot; all opponent holes and board are resampled.
    if tuple(int(x) for x in snapshot.holes[int(task["actor"])]) != _canonical_actor_cards_from_observation(bytes(task["observation"])):
        raise RuntimeError("Phase2B15 runtimefix canonical actor-card identity drift")
    return snapshot


def _canonical_snapshot(task: dict):
    if b10._WORKER_SOLVER is None or b10._WORKER_ACTION_SPEC is None:
        raise RuntimeError("Phase2B15 runtimefix worker not initialized")
    return _canonical_snapshot_with(b10._WORKER_SOLVER, b10._WORKER_ACTION_SPEC, task)


def _worker_task(task: dict) -> dict:
    snapshot = _canonical_snapshot(task)
    actor = int(task["actor"])
    block = int(task["block"])
    traversal_seed = b15._traversal_seed(int(task["evaluation_seed"]), int(task["state_index"]))
    targets = []
    log_weights = []
    nodes = 0
    started = b15.time.perf_counter()
    for sample_index in range(b15.K):
        private_seed, public_seed = b15._chance_seeds(
            int(task["evaluation_seed"]), int(task["state_index"]), block, sample_index
        )
        deal = b11._deal_from_factors(snapshot, actor, private_seed, public_seed)
        target, log_weight, node_count = b15._variant_likelihood_and_target(
            task, deal, traversal_seed
        )
        targets.append(target)
        log_weights.append(float(log_weight))
        nodes += int(node_count)

    unweighted = b15._mean_targets(targets)
    posterior, weight_stats = b15._self_normalized_mean(targets, log_weights)
    return {
        "schema": b15.PARTIAL_SCHEMA,
        "runtimefix_schema": FIX_SCHEMA,
        "behavior_seed": int(task["behavior_seed"]),
        "evaluation_seed": int(task["evaluation_seed"]),
        "state_index": int(task["state_index"]),
        "scenario_index": int(task["scenario_index"]),
        "region": str(task["region"]),
        "block": block,
        "k": b15.K,
        "actor": actor,
        "action_path_length": len(task["action_path"]),
        "legal_slots": [int(x) for x in task["legal_slots"]],
        "observation_sha256": str(task["observation_sha256"]),
        "unweighted_target": [float(x) for x in unweighted],
        "posterior_target": [float(x) for x in posterior],
        "weight_stats": weight_stats,
        "target_nodes": int(nodes),
        "target_traversals": b15.K,
        "seconds": float(b15.time.perf_counter() - started),
    }


def _valid_partial(payload: dict, task: dict) -> bool:
    return bool(
        payload.get("runtimefix_schema") == FIX_SCHEMA
        and b15._valid_partial_original(payload, task)
    )


def _activate_runtimefix() -> None:
    if not hasattr(b15, "_valid_partial_original"):
        b15._valid_partial_original = b15._valid_partial
    b15._worker_task = _worker_task
    b15._valid_partial = _valid_partial


def _preflight(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--solver", required=True)
    parser.add_argument("--heldout-root", required=True)
    parser.add_argument("--phase2b14-result", required=True)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    solver = SolverLibrary(Path(args.solver).resolve())
    if not solver.explicit_deal_available:
        raise RuntimeError("Phase2B15 runtimefix preflight requires explicit-deal solver API")
    b14_result = json.loads(Path(args.phase2b14_result).read_text(encoding="utf-8"))
    anchors, _ = b15._select_anchors(Path(args.heldout_root).resolve(), b14_result)
    contract = validate_phase2_v3_contract(
        repo_root,
        representation=b15.REPRESENTATION,
        domain=b15.DOMAIN,
        training_seed=int(b15.TRAINING_SEEDS[0]),
    )
    action_spec = contract["action_spec"]
    for anchor in anchors:
        snapshot = _canonical_snapshot_with(solver, action_spec, anchor)
        actor = int(anchor["actor"])
        expected = _canonical_actor_cards_from_observation(bytes(anchor["observation"]))
        if tuple(int(x) for x in snapshot.holes[actor]) != expected:
            raise RuntimeError("Phase2B15 runtimefix preflight actor-card mismatch")
    print(f"Phase2B15 Windows heldout replay runtimefix preflight PASS anchors={len(anchors)}")
    return 0


def main() -> int:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--preflight-only":
        return _preflight(sys.argv[2:])
    _activate_runtimefix()
    return int(b15.main())


if __name__ == "__main__":
    raise SystemExit(main())
