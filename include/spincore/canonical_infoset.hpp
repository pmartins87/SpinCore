#pragma once
#include "spincore/betting_engine.hpp"
#include "spincore/card.hpp"
#include <array>
#include <cstdint>
#include <vector>
namespace spincore {
struct CanonicalInfoset {
 std::array<Card,2> hole{}; std::array<Card,5> board{}; std::uint8_t visible_board{0}; StrategyDomain domain{StrategyDomain::ThreeHanded}; Street street{Street::Preflop}; std::uint8_t dealer_rel{0}; std::uint8_t live_count{0}; std::array<std::int32_t,3> stacks{}; std::array<std::int32_t,3> street_commitments{}; std::array<std::int32_t,3> total_commitments{}; std::array<std::uint8_t,3> statuses{}; std::int32_t pot{0}; std::int32_t to_call{0}; std::int32_t current_bet{0}; std::int32_t small_blind{0}; std::int32_t big_blind{0}; std::int32_t blind_index{0}; std::array<std::uint8_t,6> legal_action_mask{}; std::vector<std::uint8_t> public_history;
};
}
