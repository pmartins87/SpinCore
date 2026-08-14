from __future__ import annotations

from pathlib import Path

import pytest

from spincore.r7_5_action_stage_contract import (
    ADVANTAGE_STEPS,
    AUDIT_SIZE,
    BATCH_SIZE,
    CROSS_SEED_PER_SEED,
    ENSEMBLE_SIZE,
    ITERATIONS,
    POLICY_STEPS,
    POSTFLOP_TRAINING_SEEDS,
    ROOTS_PER_ITERATION,
    SELECTED_REPRESENTATION,
    TORCH_THREADS,
    deck_seed,
    primary_reset_seed,
    side_member_seeds,
    validate_action_stage_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_action_stage_contract_matches_durable_freezes() -> None:
    out = validate_action_stage_contract(
        ROOT,
        candidate_id="PF0_CONTROL_33_75_AI",
        training_seed=POSTFLOP_TRAINING_SEEDS[0],
    )
    assert out["preflight"]["ready_to_start"] is True
    assert out["representation"]["selected_candidate"] == SELECTED_REPRESENTATION
    assert ITERATIONS == 5
    assert ROOTS_PER_ITERATION == 32
    assert ADVANTAGE_STEPS == 4096
    assert POLICY_STEPS == 16384
    assert BATCH_SIZE == 256
    assert ENSEMBLE_SIZE == 4
    assert AUDIT_SIZE == 2048
    assert CROSS_SEED_PER_SEED == 1024
    assert TORCH_THREADS == 2


def test_seed_formulas_reproduce_accepted_ensemble_semantics() -> None:
    seed = POSTFLOP_TRAINING_SEEDS[0]
    assert primary_reset_seed(seed, 1) == (seed ^ 0x9E3779B1) & 0x7FFFFFFF
    for member in (1, 2, 3):
        init_seed, batch_seed = side_member_seeds(seed, 2, member)
        expected_init = (
            seed ^ 0x0E115EED ^ (2 * 0x9E3779B1) ^ (member * 0x045D9F3B)
        ) & 0x7FFFFFFF
        expected_batch = (
            seed ^ 0xBA7C8A11 ^ (2 * 0x85EBCA77) ^ (member * 0xC2B2AE3D)
        ) & ((1 << 64) - 1)
        assert init_seed == expected_init
        assert batch_seed == expected_batch


def test_deck_seed_uses_continuous_global_root_formula() -> None:
    seed = POSTFLOP_TRAINING_SEEDS[1]
    assert deck_seed(seed, 0, 1) == (seed * 1_000_003 + 1) & ((1 << 64) - 1)
    assert deck_seed(seed, 32, 2) == (seed * 1_000_003 + 32 * 97 + 2) & ((1 << 64) - 1)


def test_non_frozen_seed_and_unknown_candidate_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-frozen"):
        validate_action_stage_contract(
            ROOT,
            candidate_id="PF0_CONTROL_33_75_AI",
            training_seed=123,
        )
    with pytest.raises(ValueError, match="unknown"):
        validate_action_stage_contract(
            ROOT,
            candidate_id="NOT_A_CANDIDATE",
            training_seed=POSTFLOP_TRAINING_SEEDS[0],
        )


def test_side_member_index_outside_frozen_ensemble_fails() -> None:
    with pytest.raises(ValueError):
        side_member_seeds(POSTFLOP_TRAINING_SEEDS[0], 1, 0)
    with pytest.raises(ValueError):
        side_member_seeds(POSTFLOP_TRAINING_SEEDS[0], 1, 4)
