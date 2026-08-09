#pragma once
#include <array>
#include <cstdint>
namespace spincore {
struct TournamentState {
    std::int32_t total_chips{1500};
    bool game_is_hu{false};
    std::int32_t blind_index{0};
    std::int32_t small_blind{10};
    std::int32_t big_blind{20};
    std::array<std::int32_t,3> stacks{500,500,500};
    std::array<std::int32_t,3> dead_players{-1,-1,-1};
    std::int32_t dead_player_count{0};
    friend bool operator==(const TournamentState&, const TournamentState&) = default;
};
struct EpisodeScenario {
    TournamentState state{};
    std::int32_t dealer_id{0};
    friend bool operator==(const EpisodeScenario&, const EpisodeScenario&) = default;
};
void validate_episode_scenario(const EpisodeScenario& s);
}
