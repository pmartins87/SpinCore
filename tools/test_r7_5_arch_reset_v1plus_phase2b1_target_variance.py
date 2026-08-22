from __future__ import annotations

import math

import r7_5_arch_reset_v1plus_phase2b1_target_variance as p


def _mask(*slots: int) -> tuple[int, ...]:
    return tuple(1 if i in slots else 0 for i in range(10))


def test_identical_targets_are_zero_variance() -> None:
    target = (1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = [target for _ in range(p.REPLICATES)]
    for k in p.K_VALUES:
        metrics = p.k_pair_metrics(rows, _mask(0, 1), k)
        assert len(metrics) == p.REPLICATES // (2 * k)
        for row in metrics:
            assert row["target_mean_abs_diff"] == 0.0
            assert row["legal_sign_disagreement_fraction"] == 0.0
            assert row["regret_matching_policy_tv"] == 0.0
            assert row["dominant_legal_action_mismatch"] == 0


def test_k_aggregation_can_cancel_alternating_noise() -> None:
    # Consecutive pairs average to the same target. K1 sees maximal sign/policy
    # disagreement; K2 and above remove this synthetic zero-mean noise exactly.
    rows = []
    for i in range(p.REPLICATES):
        if i % 2 == 0:
            rows.append((2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        else:
            rows.append((-2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    k1 = p.k_pair_metrics(rows, _mask(0, 1), 1)
    k2 = p.k_pair_metrics(rows, _mask(0, 1), 2)
    assert all(math.isclose(row["regret_matching_policy_tv"], 1.0) for row in k1)
    assert all(math.isclose(row["legal_sign_disagreement_fraction"], 1.0) for row in k1)
    assert all(math.isclose(row["regret_matching_policy_tv"], 0.0) for row in k2)
    assert all(math.isclose(row["legal_sign_disagreement_fraction"], 0.0) for row in k2)


def test_partition_counts_are_frozen() -> None:
    rows = [(float(i), -float(i), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) for i in range(p.REPLICATES)]
    expected = {1: 8, 2: 4, 4: 2, 8: 1}
    for k, count in expected.items():
        assert len(p.k_pair_metrics(rows, _mask(0, 1), k)) == count


def test_seed_namespaces_are_deterministic_and_distinct() -> None:
    deck = [p._deck_seed(3, i) for i in range(32)]
    traversal = [p._traversal_seed(3, i) for i in range(p.REPLICATES)]
    assert len(set(deck)) == len(deck)
    assert len(set(traversal)) == len(traversal)
    assert deck == [p._deck_seed(3, i) for i in range(32)]
    assert traversal == [p._traversal_seed(3, i) for i in range(p.REPLICATES)]


def test_frozen_contract_constants() -> None:
    assert p.DOMAIN == "THREE_HANDED"
    assert p.SOURCE_EXECUTION_SHA == "4bfa55d69029cd69536fa6dbfcadd162719cb887"
    assert p.REPLICATES == 16
    assert p.K_VALUES == (1, 2, 4, 8)
    assert p.ARMS == ("TRAVERSAL_ONLY", "CHANCE_ONLY", "COMBINED")
    assert p.COLLISION_SEARCH_BUDGET == 50_000
    assert p.TARGET_ITERATION == 3
    assert p.MAX_WORKERS == 12


def main() -> int:
    test_identical_targets_are_zero_variance()
    test_k_aggregation_can_cancel_alternating_noise()
    test_partition_counts_are_frozen()
    test_seed_namespaces_are_deterministic_and_distinct()
    test_frozen_contract_constants()
    print("R7.5 architecture-reset Phase2B1 target-variance synthetic tests PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
