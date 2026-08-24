from __future__ import annotations

import r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance as b15
import r7_5_arch_reset_v1plus_phase2b15_posterior_weighted_continuation_chance_runtimefix as fix


def _observation(rank0: int, rank1: int, same_suit: int) -> bytes:
    obs = bytearray(120)
    obs[:8] = b"SPNNIV3\x00"
    obs[9] = 0
    obs[18] = int(rank0)
    obs[19] = int(rank1)
    obs[25] = int(same_suit)
    return bytes(obs)


def main() -> int:
    # Ordered ranks are retained; physical suit labels are intentionally
    # canonicalized because SPNNIV3 carries only pairwise suit relations.
    assert fix._canonical_actor_cards_from_observation(_observation(14, 13, 1)) == (48, 44)
    assert fix._canonical_actor_cards_from_observation(_observation(14, 13, 0)) == (48, 45)
    assert fix._canonical_actor_cards_from_observation(_observation(9, 9, 0)) == (28, 29)

    for ranks in ((2, 14), (14, 2), (7, 7), (12, 11)):
        for suited in (0, 1):
            if suited and ranks[0] == ranks[1]:
                continue
            cards = fix._canonical_actor_cards_from_observation(
                _observation(ranks[0], ranks[1], suited)
            )
            assert len(cards) == 2 and cards[0] != cards[1]
            decoded_ranks = (2 + cards[0] // 4, 2 + cards[1] // 4)
            decoded_suited = int(cards[0] % 4 == cards[1] % 4)
            assert decoded_ranks == ranks
            assert decoded_suited == suited

    try:
        fix._canonical_actor_cards_from_observation(_observation(8, 8, 1))
    except RuntimeError as exc:
        assert "suited pocket pair" in str(exc)
    else:
        raise AssertionError("impossible suited pair was accepted")

    # Runtimefix partials are explicitly distinguishable from any partial that
    # may have been produced by the failed pre-correction Windows execution.
    task = {
        "behavior_seed": 1,
        "evaluation_seed": 2,
        "state_index": 3,
        "block": 0,
        "observation_sha256": "abc",
    }
    payload = {
        "schema": b15.PARTIAL_SCHEMA,
        "behavior_seed": 1,
        "evaluation_seed": 2,
        "state_index": 3,
        "block": 0,
        "k": b15.K,
        "observation_sha256": "abc",
    }
    original = b15._valid_partial
    try:
        fix._activate_runtimefix()
        assert b15._valid_partial(payload, task) is False
        payload["runtimefix_schema"] = fix.FIX_SCHEMA
        assert b15._valid_partial(payload, task) is True
    finally:
        b15._valid_partial = original
        if hasattr(b15, "_valid_partial_original"):
            delattr(b15, "_valid_partial_original")

    print("R7.5 Phase2B15 Windows replay runtimefix synthetic tests PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
