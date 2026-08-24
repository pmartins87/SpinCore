from __future__ import annotations

import r7_5_arch_reset_v1plus_phase2b11_factorized_chance_estimator as p
from spincore.solver import DealSnapshot


def _metric(tv: float, sign: float, mad: float, p95: float, mismatch: float) -> dict:
    return {
        "pair_count": 10,
        "target_mean_abs_diff": {"count": 10, "mean": mad, "p50": mad, "p95": mad, "max": mad},
        "legal_sign_disagreement_fraction": {"count": 10, "mean": sign, "p50": sign, "p95": sign, "max": sign},
        "regret_matching_policy_tv": {"count": 10, "mean": tv, "p50": tv, "p95": p95, "max": p95},
        "dominant_legal_action_mismatch_rate": mismatch,
    }


def main() -> int:
    snapshot = DealSnapshot(
        holes=((0, 1), (2, 3), (4, 5)),
        board=(6, 7, 8, 9, 10),
        visible_board_count=0,
    )
    actor = 0

    d1 = p._deal_from_factors(snapshot, actor, 12345, 67890)
    d2 = p._deal_from_factors(snapshot, actor, 12345, 67890)
    assert d1 == d2
    assert d1.holes[actor] == snapshot.holes[actor]
    cards = [card for row in d1.holes for card in row] + list(d1.board)
    assert len(cards) == 11 and len(set(cards)) == 11

    private_a = p._private_holes(snapshot, actor, 111)
    private_b = p._private_holes(snapshot, actor, 222)
    assert private_a[actor] == snapshot.holes[actor]
    assert private_b[actor] == snapshot.holes[actor]
    board_a = p._public_board(snapshot.holes[actor], private_a, 333)
    board_b = p._public_board(snapshot.holes[actor], private_b, 333)
    assert len(board_a) == len(board_b) == 5
    assert not (set(board_a) & {card for seat, row in enumerate(private_a) if seat != actor for card in row})
    assert not (set(board_b) & {card for seat, row in enumerate(private_b) if seat != actor for card in row})

    iid4 = p._iid_deals(snapshot, actor, scenario_index=0, anchor_index=0, block=0, count=4, namespace=101)
    f22 = p._factorized_deals(snapshot, actor, scenario_index=0, anchor_index=0, block=0, side=2, namespace=201)
    iid16 = p._iid_deals(snapshot, actor, scenario_index=0, anchor_index=0, block=0, count=16, namespace=301)
    f44 = p._factorized_deals(snapshot, actor, scenario_index=0, anchor_index=0, block=0, side=4, namespace=401)
    assert len(iid4) == len(f22) == 4
    assert len(iid16) == len(f44) == 16
    for deal in iid4 + f22 + iid16 + f44:
        assert deal.holes[actor] == snapshot.holes[actor]
        flat = [card for row in deal.holes for card in row] + list(deal.board)
        assert len(flat) == 11 and len(set(flat)) == 11

    targets = [tuple(float(i + j) for j in range(10)) for i in range(4)]
    mean = p._mean_targets(targets)
    assert mean == tuple(float(1.5 + j) for j in range(10))

    zero = p._pair_metric((1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1, 1, 0, 0, 0, 0, 0, 0, 0, 0))
    assert zero["target_mean_abs_diff"] == 0.0
    assert zero["regret_matching_policy_tv"] == 0.0
    assert zero["dominant_legal_action_mismatch"] == 0

    pooled = {
        "IID4": _metric(0.40, 0.30, 0.020, 0.70, 0.40),
        "FACTOR2X2": _metric(0.30, 0.24, 0.016, 0.60, 0.34),
        "IID16": _metric(0.35, 0.28, 0.018, 0.65, 0.38),
        "FACTOR4X4": _metric(0.24, 0.20, 0.013, 0.55, 0.30),
    }
    by_seed = {
        str(seed): {
            "IID4": _metric(0.40, 0.30, 0.020, 0.70, 0.40),
            "FACTOR2X2": _metric(0.31, 0.24, 0.016, 0.60, 0.34),
            "IID16": _metric(0.36, 0.28, 0.018, 0.65, 0.38),
            "FACTOR4X4": _metric(0.25, 0.20, 0.013, 0.55, 0.30),
        }
        for seed in p.TRAINING_SEEDS
    }
    decision = p._decision(by_seed, pooled)
    assert decision["screen_pass"] is True
    assert decision["both_source_behavior_seeds_directionally_improve"] is True
    assert decision["next_route"] == "PRECOMMIT_SMALL_FACTORIZED_CHANCE_TARGET_TRAINING_PILOT_WITH_EQUAL_COMPUTE_CONTROL"

    print("R7.5 architecture-reset Phase2B11 factorized chance estimator tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
