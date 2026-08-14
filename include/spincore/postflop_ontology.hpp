#pragma once

#include "spincore/betting_engine.hpp"

#include <cstdint>

namespace spincore {

// Semantic reason for the first aggressive action on a postflop street.
// These values describe public game history; they never prescribe a response.
enum class PostflopLineType : std::uint8_t {
    None = 0,
    CBet = 1,
    DonkBet = 2,
    ProbeBet = 3,
    FloatBet = 4,
    DelayedFloatBet = 5,
    DelayedCBet = 6,
    DoubleDelayedCBet = 7,
    GenericBet = 8,
    Raise = 9,
};

struct PostflopOntologyContext {
    Street street{Street::Preflop};
    std::int32_t actor{-1};

    // Most recent aggressive actor strictly before the current street.
    bool has_lineage_aggressor{false};
    std::int32_t lineage_aggressor{-1};
    std::int8_t lineage_aggression_street{-1};
    std::uint8_t skipped_streets_since_lineage{0};

    // Semantic identity of the opening bet on this street. If there is more
    // than one aggressive action, facing_line becomes Raise while opening_line
    // preserves whether the street began as a c-bet, donk, probe, etc.
    PostflopLineType opening_line{PostflopLineType::None};
    PostflopLineType facing_line{PostflopLineType::None};
    PostflopLineType attack_opportunity{PostflopLineType::None};
    std::uint8_t current_street_aggression_count{0};
    std::uint8_t raise_depth{0};

    // Exact cheap facts retained for NeuralInputV2. No legacy sizing threshold
    // is frozen here; normalized ratios can be derived without information loss.
    std::int32_t to_call{0};
    std::int32_t pot{0};

    bool lineage_checked_current_street{false};
    bool actor_called_lineage_aggression{false};
};

[[nodiscard]] PostflopOntologyContext derive_postflop_ontology(const BettingEngine& betting);
[[nodiscard]] const char* postflop_line_name(PostflopLineType line) noexcept;

}  // namespace spincore
