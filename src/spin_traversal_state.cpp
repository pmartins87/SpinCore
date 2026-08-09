#include "spincore/spin_traversal_state.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spincore {
namespace {

std::int32_t clamp_raise_target(std::int32_t target, const LegalActions& legal) {
    target = std::max(target, legal.min_raise_to);
    target = std::min(target, legal.max_raise_to);
    return target;
}

} // namespace

SpinTraversalState::SpinTraversalState(const EpisodeScenario& scenario, std::uint64_t deck_seed)
    : scenario_(scenario), hand_(scenario, deck_seed) {}

CanonicalInfoset SpinTraversalState::infoset() const {
    if (terminal()) throw std::logic_error("terminal state has no infoset");
    return build_current_actor_infoset(hand_, blind_index());
}

NeuralInputV1 SpinTraversalState::neural_input() const {
    return encode_neural_input_v1(infoset());
}

std::vector<AbstractActionSlot> SpinTraversalState::legal_abstract_actions() const {
    if (terminal()) return {};
    const auto info = infoset();
    std::vector<AbstractActionSlot> out;
    for (std::uint8_t i=0; i<6; ++i) {
        if (info.legal_action_mask[static_cast<std::size_t>(i)]) {
            out.push_back(static_cast<AbstractActionSlot>(i));
        }
    }
    return out;
}

ExactAction SpinTraversalState::resolve(AbstractActionSlot action) const {
    if (terminal()) throw std::logic_error("cannot resolve action in terminal state");
    const auto seat = actor();
    const auto& b = hand_.betting();
    const auto legal = b.legal_actions(seat);
    const auto& p = b.players()[static_cast<std::size_t>(seat)];

    switch (action) {
        case AbstractActionSlot::Fold:
            if (!legal.fold) throw std::invalid_argument("abstract fold is illegal");
            return {ExactActionType::Fold, 0};
        case AbstractActionSlot::CheckCall:
            if (legal.check) return {ExactActionType::Check, 0};
            if (legal.call) return {ExactActionType::Call, 0};
            throw std::invalid_argument("abstract check/call is illegal");
        case AbstractActionSlot::AllIn:
            if (!legal.all_in) throw std::invalid_argument("abstract all-in is illegal");
            return {ExactActionType::AllIn, 0};
        case AbstractActionSlot::ContextRaise: {
            if (b.street() != Street::Preflop || !(legal.bet || legal.raise)) {
                throw std::invalid_argument("context raise is illegal");
            }
            const auto target = clamp_raise_target(legal.min_raise_to, legal);
            if (target >= legal.max_raise_to) return {ExactActionType::AllIn, 0};
            return {b.current_bet()==0 ? ExactActionType::BetTo : ExactActionType::RaiseTo, target};
        }
        case AbstractActionSlot::SmallPot:
        case AbstractActionSlot::LargePot: {
            if (b.street() == Street::Preflop || !(legal.bet || legal.raise)) {
                throw std::invalid_argument("postflop pot action is illegal");
            }
            const double fraction = action == AbstractActionSlot::SmallPot ? 0.33 : 0.75;
            const auto pot_after_call = b.pot() + legal.to_call;
            const auto raw_increment = static_cast<std::int32_t>(std::llround(fraction * static_cast<double>(pot_after_call)));
            const auto call_target = p.street_commitment + legal.to_call;
            auto target = clamp_raise_target(call_target + std::max<std::int32_t>(1, raw_increment), legal);
            if (target >= legal.max_raise_to) return {ExactActionType::AllIn, 0};
            return {b.current_bet()==0 ? ExactActionType::BetTo : ExactActionType::RaiseTo, target};
        }
    }
    throw std::invalid_argument("unknown abstract action");
}

ActionEvent SpinTraversalState::apply(AbstractActionSlot action) {
    return hand_.apply(actor(), resolve(action));
}

ActionEvent SpinTraversalState::apply_exact(ExactAction action) {
    if (terminal()) throw std::logic_error("cannot act in terminal state");
    return hand_.apply(actor(), action);
}

SpinTraversalState SpinTraversalState::child(AbstractActionSlot action) const {
    SpinTraversalState out = *this;
    out.apply(action);
    return out;
}

std::array<std::int32_t,3> SpinTraversalState::terminal_chip_delta() const {
    if (!terminal()) throw std::logic_error("terminal utility requested before terminal");
    auto copy = hand_;
    const auto settlement = copy.settle();
    std::array<std::int32_t,3> delta{};
    for (std::size_t i=0; i<3; ++i) {
        delta[i] = settlement.final_stacks[i] - scenario_.state.stacks[i];
    }
    return delta;
}

} // namespace spincore
