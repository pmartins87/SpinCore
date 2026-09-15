#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC  # noqa: E402
from spincore.lean_functional_agent import LeanFunctionalAgent, _street_from_state  # noqa: E402
from spincore.lean_solver_actions import apply_lean  # noqa: E402
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler  # noqa: E402
from spincore.solver import SolverLibrary  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline self-play smoke for a finalized lean SpinCore checkpoint")
    p.add_argument("--solver", type=Path, default=ROOT / "build" / "libspincore_solver_c.so")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--hands", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--greedy", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.hands <= 0:
        raise SystemExit("--hands must be positive")
    solver = SolverLibrary(args.solver)
    agent = LeanFunctionalAgent.from_checkpoint(args.checkpoint, seed=args.seed ^ 0x51F15EED)
    sampler = LegacyScenarioSampler(
        seed=args.seed ^ 0xA0F5A0F5,
        config=LegacyScenarioConfig(),
    )

    domains = {"THREE_HANDED": 0, "TRUE_HEADS_UP": 0}
    blind_counts: dict[str, int] = {}
    street_decisions = [0, 0, 0, 0]
    action_counts = [0] * 10
    decisions = 0
    max_decisions = 0

    for hand_index in range(args.hands):
        episode = sampler.sample_episode()
        domain = "TRUE_HEADS_UP" if episode.game_is_hu else "THREE_HANDED"
        domains[domain] += 1
        blind_key = f"{episode.small_blind}/{episode.big_blind}"
        blind_counts[blind_key] = blind_counts.get(blind_key, 0) + 1
        state = solver.create(episode, (args.seed << 16) ^ hand_index)
        hand_decisions = 0
        try:
            while not state.terminal:
                street = _street_from_state(state)
                active_mask, legal, _probs = agent.distribution(state)
                slot = agent.choose_slot(state, greedy=args.greedy)
                if slot not in legal:
                    raise RuntimeError("agent selected illegal lean action")
                apply_lean(state, active_mask, slot)
                street_decisions[street] += 1
                action_counts[slot] += 1
                decisions += 1
                hand_decisions += 1
                if hand_decisions > 200:
                    raise RuntimeError("self-play hand exceeded 200 decisions")
            delta = state.terminal_chip_delta()
            if sum(delta) != 0:
                raise RuntimeError(f"terminal chip delta is not zero-sum: {delta}")
        finally:
            state.close()
        max_decisions = max(max_decisions, hand_decisions)

    result = {
        "schema": "SPINCORE_LEAN_SELFPLAY_SMOKE_V1",
        "hands": int(args.hands),
        "decisions": int(decisions),
        "max_decisions_in_hand": int(max_decisions),
        "domains": domains,
        "blind_counts": dict(sorted(blind_counts.items())),
        "street_decisions": street_decisions,
        "action_counts": action_counts,
        "greedy": bool(args.greedy),
        "action_candidate": FIRST_RELEASE_ACTION_SPEC.candidate_id,
        "status": "PASS",
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
