#pragma once
#include "spincore/action_abstraction.hpp"
#include "spincore/hand_engine.hpp"
#include "spincore/hand_infoset_adapter.hpp"
#include "spincore/neural_encoder.hpp"
#include "spincore/tournament_value.hpp"
#include <array>
#include <cstdint>
#include <vector>
namespace spincore {
class SpinTraversalState {
public:
 SpinTraversalState(const EpisodeScenario& scenario,std::uint64_t deck_seed);
 SpinTraversalState(const EpisodeScenario& scenario,const std::array<std::array<Card,2>,3>& hole_cards,const std::array<Card,5>& board_cards);
 [[nodiscard]] const EpisodeScenario& scenario() const noexcept{return scenario_;}
 [[nodiscard]] const HandEngine& hand() const noexcept{return hand_;}
 [[nodiscard]] HandEngine& hand() noexcept{return hand_;}
 [[nodiscard]] std::int32_t blind_index() const noexcept{return scenario_.state.blind_index;}
 [[nodiscard]] bool terminal() const noexcept{return hand_.terminal();}
 [[nodiscard]] std::int32_t actor() const noexcept{return terminal()?-1:hand_.betting().actor();}
 [[nodiscard]] CanonicalInfoset infoset() const;
 [[nodiscard]] NeuralInputV1 neural_input() const;
 [[nodiscard]] std::vector<AbstractActionSlot> legal_abstract_actions() const;
 [[nodiscard]] ExactAction resolve(AbstractActionSlot action) const;
 ActionEvent apply(AbstractActionSlot action); ActionEvent apply_exact(ExactAction action); [[nodiscard]] SpinTraversalState child(AbstractActionSlot action) const;
 [[nodiscard]] std::array<std::int32_t,3> terminal_chip_delta() const;
 [[nodiscard]] std::array<double,3> terminal_icm_delta(const PayoutProfile& payout) const;
private: EpisodeScenario scenario_{}; HandEngine hand_;
};
}
