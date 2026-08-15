from __future__ import annotations

from pathlib import Path

import math

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_representation_v3_local_deviation import evaluate_local_deviation_state
from spincore.r7_5_representation_v3_referee_states import HeldoutV3State, effective_pf0, uniform_policy
from spincore.solver import Episode, SolverLibrary
from spincore.solver_v3 import neural_bytes_v3


def test_local_deviation_proxy_is_nonnegative_and_deterministic() -> None:
    library = Path("build/libspincore_solver_c.so")
    assert library.exists()
    solver = SolverLibrary(library)
    spec = postflop_candidate_specs(Path("."))["PF0_CONTROL_33_75_AI"]
    episode = Episode(
        total_chips=80,
        game_is_hu=True,
        blind_index=0,
        small_blind=10,
        big_blind=20,
        stacks=(0, 40, 40),
        dead_players=(0,),
        dealer_id=1,
    )
    state = solver.create(episode, 0x753C)
    try:
        active_mask, legal, exact = effective_pf0(state, spec)
        descriptor = HeldoutV3State(
            domain="TRUE_HEADS_UP",
            evaluation_seed=2029384436,
            state_index=0,
            hand_index=0,
            scenario_index=0,
            deck_seed=0x753C,
            action_path=(),
            actor=int(state.actor),
            observation_v3=neural_bytes_v3(state),
            active_mask=active_mask,
            legal_slots=legal,
            exact_actions=exact,
        )

        def policy(_state, _observation, legal_actions):
            return uniform_policy(tuple(legal_actions))

        a = evaluate_local_deviation_state(
            state,
            descriptor=descriptor,
            action_spec=spec,
            candidate_policy=policy,
            exact_opponent_levels=0,
        )
        b = evaluate_local_deviation_state(
            state,
            descriptor=descriptor,
            action_spec=spec,
            candidate_policy=policy,
            exact_opponent_levels=0,
        )
        assert a == b
        assert math.isfinite(a.policy_value)
        assert math.isfinite(a.best_local_value)
        assert math.isfinite(a.gain)
        assert a.gain >= 0.0
        assert abs(a.gain - (a.best_local_value - a.policy_value)) < 1e-12
        q = dict(a.q_values)
        assert a.best_local_value == max(q.values())
        assert abs(sum(a.policy_probabilities) - 1.0) < 1e-12
    finally:
        state.close()
