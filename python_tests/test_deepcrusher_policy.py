from types import SimpleNamespace
from unittest.mock import patch

from spincore.deepcrusher_policy import DeepCrusherR8Policy
from spincore.deepcrusher_state import (
    DeepCrusherStateView,
    STREET_PREFLOP,
)
from spincore.openppl_program import OpenPPLProgram


def simple_view():
    return DeepCrusherStateView(
        domain=1,
        street=STREET_PREFLOP,
        dealer_rel=0,
        small_blind_rel=0,
        big_blind_rel=1,
        live_count=2,
        visible_board=0,
        statuses=(0, 0, 2),
        ranks=(14, 14, 0, 0, 0, 0, 0),
        same_suit=(0,) * 21,
        pot_bb=1.5,
        to_call_bb=0.5,
        current_bet_bb=1.0,
        stacks_bb=(24.5, 24.0, 0.0),
        street_commitments_bb=(0.5, 1.0, 0.0),
        total_commitments_bb=(0.5, 1.0, 0.0),
        small_blind_bb=0.5,
        blind_index=0,
        min_raise_to_bb=2.0,
        max_raise_to_bb=25.0,
        primitive_legal=(True, False, True, False, True, True),
        history=(),
        exact_suits=(3, 0, -1, -1, -1, -1, -1),
    )


def fake_policy():
    p = DeepCrusherR8Policy.__new__(DeepCrusherR8Policy)
    p.source_path = None
    p.library_paths = ()
    p.source_metadata = {}
    p.program = OpenPPLProgram.from_text("""
##f$ini_function_on_startup##
me_st_X_1
##f$ini_function_on_new_round##
me_inc_X
##f$ini_function_on_my_turn##
When Others Set user_seen
##f$preflop##
When user_seen && me_re_X = 2 Return 3 Force
When Others Fold Force
##f$flop##
When Others Fold Force
##f$turn##
When Others Fold Force
##f$river##
When Others Fold Force
""")
    p.environment = {}
    p._sessions = {}
    p._startup_done = set()
    p._handreset_done = set()
    p._last_street = {}
    p._big_blind_chips = None
    p._hand_serial = 0
    return p


def public():
    return SimpleNamespace(
        legal_fold=True,
        legal_check=False,
        legal_call=True,
        legal_bet=False,
        legal_raise=True,
        legal_all_in=True,
        min_raise_to=40,
        max_raise_to=500,
        current_bet=20,
        pot=30,
        to_call=10,
        street_commitments=(10, 20, 0),
    )


def test_policy_replays_startup_newround_myturn_before_primary_callback():
    p = fake_policy()
    episode = SimpleNamespace(big_blind=20)
    lineup = SimpleNamespace(seats=("DEEPCRUSHER", "SPINCORE", "DEAD"))
    p.begin_hand(
        episode=episode,
        lineup=lineup,
        scenario_index=1,
        lineup_index=0,
        deal_seed=123,
    )
    state = SimpleNamespace(
        owner=SimpleNamespace(explicit_deal_available=True),
        public_snapshot=public,
    )

    with patch("spincore.deepcrusher_policy.state_view", return_value=simple_view()):
        action = p.choose_exact(state, seat=0, rng=None)
        again = p.choose_exact(state, seat=0, rng=None)

    assert (action.action_type, action.amount_to) == (4, 60)
    assert again == action
    # Startup sets X=1, first new-round increments once to X=2. The second
    # decision on the same street must not run new-round again.
    assert p._sessions[0].memory_symbols["x"] == 2
    assert "user_seen" in p._sessions[0].user_variables


def test_begin_hand_isolates_openppl_memory_between_paired_replays():
    p = fake_policy()
    episode = SimpleNamespace(big_blind=20)
    lineup = SimpleNamespace(seats=("DEEPCRUSHER", "SPINCORE", "DEAD"))
    p.begin_hand(
        episode=episode,
        lineup=lineup,
        scenario_index=1,
        lineup_index=0,
        deal_seed=123,
    )
    p._sessions[0].memory_symbols["leak"] = 99
    p.begin_hand(
        episode=episode,
        lineup=lineup,
        scenario_index=1,
        lineup_index=1,
        deal_seed=123,
    )
    assert p._sessions[0].memory_symbols == {}
