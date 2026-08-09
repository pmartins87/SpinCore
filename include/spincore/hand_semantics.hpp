#pragma once
#include "spincore/card.hpp"
#include <array>
#include <string>
namespace spincore {
[[nodiscard]] std::array<Card,2> canonical_hole(std::array<Card,2> hole);
[[nodiscard]] std::string hand_class(std::array<Card,2> hole);
}
