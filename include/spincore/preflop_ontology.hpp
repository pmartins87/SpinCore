#pragma once

#include "spincore/betting_engine.hpp"

#include <cstdint>

namespace spincore {

// Objective lineage of voluntary preflop action. These categories describe how
// the pot was built; they never prescribe a response or inherit legacy sizing
// thresholds from Crusher Framework.
enum class PreflopLineageType : std::uint8_t {
    Unopened = 0,
    Limped = 1,
    OpenRaised = 2,
    RaiseOverLimp = 3,
    Reraised = 4,
    LimpReraised = 5,
};

struct PreflopOntologyContext {
    PreflopLineageType lineage{PreflopLineageType::Unopened};
    std::int32_t actor{-1};

    std::uint8_t voluntary_action_count{0};
    std::uint8_t limp_count{0};
    std::uint8_t voluntary_call_count{0};
    std::uint8_t calls_after_aggression{0};
    std::uint8_t aggression_count{0};

    std::int32_t first_aggressor{-1};
    std::int32_t last_aggressor{-1};
    std::int32_t last_voluntary_actor{-1};

    std::int32_t first_raise_to{0};
    std::int32_t last_raise_to{0};
    std::int32_t first_raise_increment{0};
    std::int32_t last_raise_increment{0};

    std::int32_t pot{0};
    std::int32_t to_call{0};
    bool had_limp_before_first_aggression{false};
    bool limper_became_reraiser{false};
};

[[nodiscard]] PreflopOntologyContext derive_preflop_ontology(const BettingEngine& betting);
[[nodiscard]] const char* preflop_lineage_name(PreflopLineageType lineage) noexcept;

}  // namespace spincore
