#pragma once

#include "spincore/betting_engine.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace spincore {

// R7.5.4 universal action vocabulary. This module is deliberately parallel to
// the frozen six-slot R7.3/R7.4 abstraction; merely compiling it does not change
// any existing traversal semantics.
enum class UniversalActionSlotV2 : std::uint8_t {
    Fold = 0,
    CheckCall = 1,
    MinRaise = 2,
    Pot33 = 3,
    Pot40 = 4,
    Pot50 = 5,
    Pot66 = 6,
    Pot75 = 7,
    Pot100 = 8,
    AllIn = 9,
};

constexpr std::size_t kUniversalActionCountV2 = 10;
using UniversalActionMaskV2 = std::array<std::uint8_t, kUniversalActionCountV2>;

struct ResolvedUniversalActionV2 {
    UniversalActionSlotV2 slot{UniversalActionSlotV2::CheckCall};
    ExactAction exact{};
    std::int32_t unclamped_target{0};
    std::int32_t realized_target{0};
    bool clamped_to_min{false};
    bool clamped_to_allin{false};

    friend bool operator==(const ResolvedUniversalActionV2&, const ResolvedUniversalActionV2&) = default;
};

[[nodiscard]] const char* universal_action_name_v2(UniversalActionSlotV2 slot) noexcept;
[[nodiscard]] double universal_pot_fraction_v2(UniversalActionSlotV2 slot) noexcept;
[[nodiscard]] bool universal_is_fractional_v2(UniversalActionSlotV2 slot) noexcept;

// Resolve an active universal mask against the exact current betting state and
// suppress state-local aliases. The returned vector is ordered by universal
// slot id and contains at most one neural branch for each exact action.
[[nodiscard]] std::vector<ResolvedUniversalActionV2> resolve_universal_actions_v2(
    const BettingEngine& betting,
    const UniversalActionMaskV2& active_mask);

} // namespace spincore
