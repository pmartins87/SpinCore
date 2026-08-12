from __future__ import annotations

from dataclasses import dataclass

from .production_evidence_packet import SelectedStateEvidencePacket
from .production_profile import ProductionEvidence, ProductionProfile


@dataclass(frozen=True)
class ProductionStrategyIdentity:
    game_family: str
    ruleset_id: str
    action_abstraction_id: str
    utility_model_id: str
    learning_profile_id: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.game_family, "game_family"),
            (self.ruleset_id, "ruleset_id"),
            (self.action_abstraction_id, "action_abstraction_id"),
            (self.utility_model_id, "utility_model_id"),
            (self.learning_profile_id, "learning_profile_id"),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} is required")


def build_profile_from_bound_evidence(
    *,
    selected_state: SelectedStateEvidencePacket,
    tournament_fee_fraction: float,
    tournament_fee_evidence: ProductionEvidence,
    strategy: ProductionStrategyIdentity,
) -> ProductionProfile:
    """Build one exact production profile without importing pilot assumptions.

    State-dependent stack/blind/payout values come exclusively from a selected
    client/rule-document packet.  The tournament fee comes from separate
    first-party GLOBAL_GAME evidence and is never inferred from payouts or the
    Spin multiplier.
    """

    if tournament_fee_evidence.scope != "GLOBAL_GAME":
        raise ValueError("tournament fee evidence must be GLOBAL_GAME")
    if tuple(tournament_fee_evidence.proven_fields) != ("tournament_fee_fraction",):
        raise ValueError("tournament fee evidence may prove only tournament_fee_fraction")
    if not (0.0 <= float(tournament_fee_fraction) < 1.0):
        raise ValueError("tournament fee fraction must be in [0,1)")

    return ProductionProfile(
        platform="GGPOKER",
        game_family=strategy.game_family,
        table_size=selected_state.table_size,
        currency=selected_state.currency,
        buy_in_minor_units=selected_state.buy_in_minor_units,
        multiplier=selected_state.multiplier,
        starting_chips_per_player=selected_state.starting_chips_per_player,
        blind_levels=selected_state.blind_levels,
        payout_share_by_place=selected_state.payouts.normalized_shares,
        tournament_fee_fraction=float(tournament_fee_fraction),
        ruleset_id=strategy.ruleset_id,
        action_abstraction_id=strategy.action_abstraction_id,
        utility_model_id=strategy.utility_model_id,
        learning_profile_id=strategy.learning_profile_id,
        evidence=(selected_state.to_production_evidence(), tournament_fee_evidence),
    )
