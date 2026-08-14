from __future__ import annotations

from spincore_nn.hybrid_v3 import HybridNetV3, HybridNetworkConfigV3


class HybridCandidateNetV3(HybridNetV3):
    """R7.5.3C candidate model with no unused candidate-only modules.

    `HybridNetV3` was intentionally implemented as one superset prototype to
    exercise all branches quickly.  That makes functional outputs correct but
    inflates state-dict/parameter accounting because modules belonging to other
    candidates are still registered.  Scientific candidate fits must use this
    pruned class so parameter/RAM comparisons count only modules reachable by
    that candidate's forward path.
    """

    def __init__(self, candidate: str, cfg: HybridNetworkConfigV3 | None = None):
        super().__init__(candidate, cfg)

        if candidate == "H0_FIXED_V1":
            del self.relational_cards
            del self.structured_history_emb
            del self.structured_history_gru
            del self.preflop_emb
            del self.semantic_cat_emb
        elif candidate == "H1_RELATIONAL_EXACT":
            del self.raw_card_emb
            del self.structured_history_emb
            del self.structured_history_gru
            del self.preflop_emb
            del self.semantic_cat_emb
        elif candidate == "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY":
            del self.raw_card_emb
            del self.legacy_history_emb
            del self.legacy_history_gru
            del self.preflop_emb
            del self.semantic_cat_emb
        elif candidate in {"H3_HYBRID_EXACT_SEMANTIC", "H4_HYBRID_CAPACITY"}:
            del self.raw_card_emb
            del self.legacy_history_emb
            del self.legacy_history_gru
        else:  # superclass already validates; keep fail-closed if this drifts.
            raise ValueError(f"unhandled hybrid candidate: {candidate}")
