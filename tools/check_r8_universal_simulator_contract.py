#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from spincore.simulator_profile import (  # noqa: E402
    HandsBlindLevel,
    OfficialRuleReference,
    PRODUCT_TARGET,
    SIMULATOR_PROFILE_SCHEMA,
    SimulatorPresentation,
    UniversalThreeMaxSimulatorProfile,
    require_stake_invariant_policy_identity,
)


CONTRACT = ROOT / "validation/R8_0_UNIVERSAL_3MAX_SIMULATOR_CONTRACT_20260817.json"


def _profile(stake: int, reference: OfficialRuleReference) -> UniversalThreeMaxSimulatorProfile:
    # Mechanics-only fixture.  These structural values are not an official
    # multiplier row and must never be promoted into training configuration.
    return UniversalThreeMaxSimulatorProfile(
        starting_chips_per_player=500,
        blind_levels=(HandsBlindLevel(10, 20, 0, 8),),
        payout_share_by_place=(1.0, 0.0, 0.0),
        ruleset_id="SPINRULESET-4",
        action_abstraction_id="STAKE_INVARIANCE_TEST_FIXTURE",
        utility_model_id="ICM_EXACT_V1_EXPLICIT_PAYOUT_DELTA",
        learning_profile_id="STAKE_INVARIANCE_TEST_FIXTURE",
        presentation=SimulatorPresentation(
            currency="USD",
            nominal_buy_in_minor_units=stake,
            displayed_multiplier=2,
            tournament_fee_fraction=0.07,
        ),
        official_references=(reference,),
    )


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "PASS_UNIVERSAL_CONTRACT_STAKE_INVARIANT"
    assert contract["product_target"] == PRODUCT_TARGET
    assert contract["active_profile_schema"] == SIMULATOR_PROFILE_SCHEMA
    assert contract["table_size"] == 3
    assert contract["user_capture_required"] is False
    assert contract["external_user_data_dependency"] is False
    assert contract["real_money_client_integration_authorized"] is False
    assert contract["production_training_authorized"] is False
    assert contract["ready_for_simulator_tables"] is False
    assert contract["blind_progression"]["simulator_basis"] == "COMPLETED_HANDS"
    assert contract["blind_progression"]["wall_clock_seconds_are_strategy_input"] is False
    assert contract["variant_matrix"]["pilot_constants_allowed_as_substitute"] is False

    source = contract["official_sources"][0]
    reference = OfficialRuleReference(
        locator=source["locator"],
        observed_at_utc=source["observed_at_utc"],
        proven_fields=tuple(source["proven_fields"]),
    )
    stakes = tuple(int(x) for x in contract["official_nominal_buy_in_minor_units"])
    profiles = tuple(_profile(stake, reference) for stake in stakes)
    policy_id = require_stake_invariant_policy_identity(profiles, "THREE_HANDED")
    assert len({profile.profile_id for profile in profiles}) == 1
    assert profiles[0].policy_id("TRUE_HEADS_UP") != policy_id

    print(
        json.dumps(
            {
                "status": "PASS",
                "profile_schema": SIMULATOR_PROFILE_SCHEMA,
                "nominal_stakes_checked": list(stakes),
                "three_handed_policy_id": policy_id,
                "user_capture_required": False,
                "real_money_client_integration_authorized": False,
                "production_training_authorized": False,
                "ready_for_simulator_tables": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
