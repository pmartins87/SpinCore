#pragma once

#include "spincore/card.hpp"

#include <array>
#include <cstdint>

namespace spincore {

// Lossless Hold'em flop identity modulo card order and a single global
// permutation of the four suit names. Output is rank,suit repeated three times.
[[nodiscard]] std::array<std::uint8_t, 6> canonical_flop_signature(
    const std::array<Card, 3>& flop
);

}  // namespace spincore
