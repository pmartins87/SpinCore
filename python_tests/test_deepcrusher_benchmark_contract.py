from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.deepcrusher_benchmark import (  # noqa: E402
    DEEPC_RUSHER_POLICY_ID,
    SPINCORE_POLICY_ID,
    ExternalExactAction,
    Lineup,
    MatchObservation,
    aggregate_policy_chip_delta,
    balanced_hu_lineups,
    balanced_three_handed_lineups,
    benchmark_contract,
    validate_three_handed_balance,
)


def test_hu_lineups_swap_policies_and_keep_dead_seat():
    for dead in range(3):
        first, second = balanced_hu_lineups(dead)
        assert first.seats[dead] == "DEAD"
        assert second.seats[dead] == "DEAD"
        live = [seat for seat in range(3) if seat != dead]
        assert first.seats[live[0]] == second.seats[live[1]] == SPINCORE_POLICY_ID
        assert first.seats[live[1]] == second.seats[live[0]] == DEEPC_RUSHER_POLICY_ID


def test_three_handed_block_is_exactly_balanced():
    lineups = balanced_three_handed_lineups()
    validate_three_handed_balance(lineups)
    assert len(lineups) == 6
    for policy in (SPINCORE_POLICY_ID, DEEPC_RUSHER_POLICY_ID):
        assert sum(row.count(policy) for row in lineups) == 9
        assert [sum(1 for row in lineups if row.seats[seat] == policy) for seat in range(3)] == [3, 3, 3]


def test_policy_aggregate_is_zero_sum():
    rows = [
        MatchObservation(0, "THREE_HANDED", "10/20", Lineup((SPINCORE_POLICY_ID, DEEPC_RUSHER_POLICY_ID, DEEPC_RUSHER_POLICY_ID)), (10, -4, -6)),
        MatchObservation(0, "THREE_HANDED", "10/20", Lineup((DEEPC_RUSHER_POLICY_ID, SPINCORE_POLICY_ID, DEEPC_RUSHER_POLICY_ID)), (-5, 7, -2)),
    ]
    total = aggregate_policy_chip_delta(rows)
    assert total[SPINCORE_POLICY_ID] == 17
    assert total[DEEPC_RUSHER_POLICY_ID] == -17


def test_external_exact_action_rejects_invalid_amount_semantics():
    assert ExternalExactAction(4, 120).name == "RAISE_TO"
    assert ExternalExactAction(2, 0).name == "CALL"
    try:
        ExternalExactAction(2, 120)
    except ValueError:
        pass
    else:
        raise AssertionError("CALL with amount_to must fail")


def test_contract_pins_frozen_deepcrusher_and_no_quantization():
    contract = benchmark_contract()
    assert contract["deepcrusher"]["branch"] == "r8-v22-stable-20260914"
    assert contract["no_size_quantization"] is True
    assert contract["offline_only"] is True
