#pragma once
#include "spincore/action_abstraction.hpp"
#include "spincore/game_topology.hpp"
#include <array>
#include <cstdint>
#include <vector>
namespace spincore {
enum class Street : std::uint8_t { Preflop=0, Flop=1, Turn=2, River=3 };
struct PlayerBetState { std::int32_t stack{0}; std::int32_t street_commitment{0}; std::int32_t total_commitment{0}; bool folded{false}; bool all_in{false}; bool acted_since_full_raise{false}; friend bool operator==(const PlayerBetState&,const PlayerBetState&)=default; };
struct LegalActions { bool fold{false}; bool check{false}; bool call{false}; bool bet{false}; bool raise{false}; bool all_in{false}; std::int32_t to_call{0}; std::int32_t min_raise_to{0}; std::int32_t max_raise_to{0}; };
// Exact public betting-history event. Existing fields retain their historical
// order so V1 behavior and aggregate users remain source-compatible. The new
// fields are observation metadata only: pot_before makes sizing reconstruction
// lossless, while forced distinguishes blind posts from voluntary aggression.
struct ActionEvent { std::int32_t actor{-1}; Street street{Street::Preflop}; ExactAction action{}; std::int32_t paid{0}; std::int32_t resulting_commitment{0}; std::int32_t pot_after{0}; std::int32_t pot_before{0}; bool forced{false}; friend bool operator==(const ActionEvent&,const ActionEvent&)=default; };
class BettingEngine {
public:
 BettingEngine(const EpisodeScenario& s, GameTopology topology);
 [[nodiscard]] const GameTopology& topology() const noexcept{return topology_;}
 [[nodiscard]] const std::array<PlayerBetState,3>& players() const noexcept{return players_;}
 [[nodiscard]] std::array<PlayerBetState,3>& players() noexcept{return players_;}
 [[nodiscard]] Street street() const noexcept{return street_;}
 [[nodiscard]] std::int32_t actor() const noexcept{return actor_;}
 [[nodiscard]] std::int32_t current_bet() const noexcept{return current_bet_;}
 [[nodiscard]] std::int32_t pot() const noexcept;
 [[nodiscard]] const std::vector<ActionEvent>& history() const noexcept{return history_;}
 [[nodiscard]] bool street_complete() const noexcept{return street_complete_;}
 [[nodiscard]] bool hand_over_by_fold() const noexcept{return hand_over_by_fold_;}
 [[nodiscard]] int nonfolded_count() const noexcept;
 [[nodiscard]] int actionable_count() const noexcept;
 [[nodiscard]] LegalActions legal_actions(int seat) const;
 ActionEvent apply(int seat, ExactAction action);
 void advance_street();
private:
 EpisodeScenario scenario_{}; GameTopology topology_{}; std::array<PlayerBetState,3> players_{}; Street street_{Street::Preflop}; int actor_{-1}; int current_bet_{0}; int last_full_raise_size_{0}; bool street_complete_{false}; bool hand_over_by_fold_{false}; std::vector<ActionEvent> history_;
 void post_blind(int seat,int amount); [[nodiscard]] bool needs_action(int seat) const; [[nodiscard]] int next_actor_from(int seat) const; void update_completion_after_action(int acted_seat); void pay_to(int seat,int target);
};
}
