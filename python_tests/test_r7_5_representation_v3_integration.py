from __future__ import annotations

from pathlib import Path

from spincore.r7_5_action_contract import postflop_candidate_specs
from spincore.r7_5_representation_v3 import (
    H2_FINAL,
    RepresentationV3DeepCFRSession,
    make_representation_v3_bundle,
)
from spincore.solver import Episode, SolverLibrary


def _chip_utility(state):
    return tuple(float(value) for value in state.terminal_chip_delta())


def test_h2_v3_runs_through_authoritative_universal_cfr_and_training() -> None:
    library = Path("build/libspincore_solver_c.so")
    if not library.exists():
        raise AssertionError("main regression must build solver before Python tests")
    solver = SolverLibrary(library)
    action_spec = postflop_candidate_specs(Path("."))["PF0_CONTROL_33_75_AI"]
    bundle = make_representation_v3_bundle(
        H2_FINAL,
        7532026,
        reservoir_capacity=2048,
        lr=0.001,
    )
    session = RepresentationV3DeepCFRSession(
        solver_library=solver,
        bundle=bundle,
        action_spec=action_spec,
        terminal_utility=_chip_utility,
    )
    # Mechanical integration state only: 2 BB each keeps the unit test tiny.
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
    report = session.collect_root(
        episode,
        iteration=1,
        exact_opponent_levels=0,
        deck_seed=0x753C,
    )
    assert report["nodes"] > 0
    assert report["advantage_samples"] > 0
    assert report["strategy_samples"] > 0
    assert bundle.adv_mem.items
    assert bundle.pol_mem.items
    assert all(sample.observation.startswith(b"SPNNIV3\0") for sample in bundle.adv_mem.items)
    assert all(len(sample.legal) == 10 for sample in bundle.adv_mem.items)

    adv_loss = session.train_advantage(steps=1, batch_size=16)
    pol_loss = session.train_average_policy(steps=1, batch_size=16)
    assert len(adv_loss) == 1 and adv_loss[0] >= 0.0
    assert len(pol_loss) == 1 and pol_loss[0] >= 0.0
    assert bundle.counters["advantage_ready"] == 1
