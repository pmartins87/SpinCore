#include "spincore/neural_encoder_v2.hpp"
#include "spincore/ruleset_contract.hpp"

#include <iomanip>
#include <iostream>

using namespace spincore;

int main() {
    EpisodeScenario scenario{};
    scenario.state.total_chips = 1500;
    scenario.state.game_is_hu = false;
    scenario.state.small_blind = 10;
    scenario.state.big_blind = 20;
    scenario.state.stacks = {500, 500, 500};
    scenario.dealer_id = 0;

    HandEngine hand(scenario, 0x1234ABCDEFULL);
    hand.apply(0, {ExactActionType::Call, 0});
    hand.apply(1, {ExactActionType::Call, 0});
    hand.apply(2, {ExactActionType::Check, 0});
    const int actor = hand.betting().actor();
    hand.apply(actor, {ExactActionType::BetTo, 30});

    const auto bytes = serialize_neural_input_v2(encode_neural_input_v2(hand, 0));
    for (std::uint8_t byte : bytes) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
    }
    std::cout << '\n';
    return 0;
}
