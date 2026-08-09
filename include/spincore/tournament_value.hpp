#pragma once
#include "spincore/ruleset_contract.hpp"
#include <array>
namespace spincore {
struct PayoutProfile { std::array<double,3> by_place{1.0,0.0,0.0}; };
[[nodiscard]] std::array<double,3> icm_values(const TournamentState& state,const PayoutProfile& payout);
[[nodiscard]] std::array<double,3> terminal_continuation_delta(const TournamentState& before,const std::array<std::int32_t,3>& final_stacks,const PayoutProfile& payout);
}
