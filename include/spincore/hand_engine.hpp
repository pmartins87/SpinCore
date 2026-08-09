#pragma once
#include "spincore/betting_engine.hpp"
#include "spincore/card.hpp"
#include "spincore/ruleset_contract.hpp"
#include <array>
#include <cstdint>
namespace spincore {
struct HandSettlement { std::array<std::int32_t,3> final_stacks{}; friend bool operator==(const HandSettlement&,const HandSettlement&)=default; };
class HandEngine {
public:
 HandEngine(const EpisodeScenario& s,std::uint64_t deck_seed);
 [[nodiscard]] const EpisodeScenario& scenario() const noexcept{return scenario_;}
 [[nodiscard]] const BettingEngine& betting() const noexcept{return betting_;}
 [[nodiscard]] BettingEngine& betting() noexcept{return betting_;}
 [[nodiscard]] const std::array<std::array<Card,2>,3>& hole_cards() const noexcept{return hole_;}
 [[nodiscard]] const std::array<Card,5>& board() const noexcept{return board_;}
 [[nodiscard]] int visible_board_count() const noexcept{return visible_board_count_;}
 [[nodiscard]] bool terminal() const noexcept{return terminal_;}
 ActionEvent apply(int seat,ExactAction action);
 [[nodiscard]] HandSettlement settle() const;
private:
 EpisodeScenario scenario_{}; BettingEngine betting_; std::array<std::array<Card,2>,3> hole_{}; std::array<bool,3> has_hole_{}; std::array<Card,5> board_{}; int visible_board_count_{0}; bool terminal_{false};
 void maybe_advance();
};
}
