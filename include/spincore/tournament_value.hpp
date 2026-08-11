#pragma once
#include "spincore/ruleset_contract.hpp"
#include <array>
namespace spincore {
struct PayoutProfile { std::array<double,3> by_place{1.0,0.0,0.0}; };
[[nodiscard]] std::array<double,3> icm_values(const TournamentState& state,const PayoutProfile& payout);
// Conservative context-free overload: equal-stack simultaneous eliminations
// under unequal payouts remain fail-closed because table-position tie-break
// information is unavailable.
[[nodiscard]] std::array<double,3> terminal_continuation_delta(const TournamentState& before,const std::array<std::int32_t,3>& final_stacks,const PayoutProfile& payout);
// Dealer-aware overload used by SpinTraversalState. This can resolve equal-stack
// same-hand eliminations according to the tournament table-position rule.
[[nodiscard]] std::array<double,3> terminal_continuation_delta(const TournamentState& before,const std::array<std::int32_t,3>& final_stacks,const PayoutProfile& payout,std::int32_t dealer_id);
}
