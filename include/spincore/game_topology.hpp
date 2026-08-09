#pragma once
#include "spincore/ruleset_contract.hpp"
#include <array>
#include <cstdint>
namespace spincore {
enum class StrategyDomain : std::uint8_t { ThreeHanded=0, TrueHeadsUp=1 };
struct GameTopology {
    StrategyDomain domain{StrategyDomain::ThreeHanded};
    std::int32_t dealer{0};
    std::int32_t small_blind_seat{1};
    std::int32_t big_blind_seat{2};
    std::int32_t first_preflop{0};
    std::int32_t first_postflop{1};
    std::array<std::int32_t,3> live{0,1,2};
    std::int32_t live_count{3};
    friend bool operator==(const GameTopology&, const GameTopology&) = default;
};
[[nodiscard]] GameTopology make_game_topology(const EpisodeScenario& s);
[[nodiscard]] StrategyDomain strategy_domain(const GameTopology& t) noexcept;
[[nodiscard]] int next_live_clockwise(const GameTopology& t, int seat);
}
