#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "spincore/neural_encoder_v2.hpp"
#include "spincore/neural_encoder_v3.hpp"
#include "spincore/solver_c_api.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace spincore;

SPIN_TEST(neural_v3_initial_three_handed_wire_has_exact_lossless_carrier) {
    HandEngine hand(sc3(0), 4242);
    const auto v3 = encode_neural_input_v3(hand, 7);

    REQUIRE(v3.categorical[0] == static_cast<std::uint8_t>(StrategyDomain::ThreeHanded));
    REQUIRE(v3.categorical[1] == static_cast<std::uint8_t>(Street::Preflop));
    REQUIRE(v3.categorical[5] == 3U);
    REQUIRE(v3.categorical[6] == 0U);
    REQUIRE(v3.rank_tokens[0] >= 2U && v3.rank_tokens[0] <= 14U);
    REQUIRE(v3.rank_tokens[1] >= 2U && v3.rank_tokens[1] <= 14U);
    for (std::size_t index = 2; index < v3.rank_tokens.size(); ++index) {
        REQUIRE(v3.rank_tokens[index] == 0U);
    }
    REQUIRE(v3.history.size() == 2U); // exact forced blind events
    REQUIRE(v3.history[0].categorical[3] == 1U);
    REQUIRE(v3.history[1].categorical[3] == 1U);
    REQUIRE(v3.numeric[13] == 7.0F);

    const auto bytes = serialize_neural_input_v3(v3);
    REQUIRE(bytes.size() == kNeuralV3FixedSerializedBytes + 2U * kNeuralV3HistoryEventSerializedBytes);
    REQUIRE(bytes.size() == 160U);
    REQUIRE(bytes[0] == 'S' && bytes[1] == 'P' && bytes[2] == 'N');
    REQUIRE(bytes[6] == '3');
}

SPIN_TEST(neural_v3_true_hu_always_canonicalizes_dead_chair_to_rel2) {
    // Same two 750-chip live players, but dealer/actor sits at physical seat 2.
    // The physical dead seat 0 is actor-relative rel1 before V3 canonicalization.
    HandEngine hand(schu(2), 919191);
    const auto v3 = encode_neural_input_v3(hand, 0);

    REQUIRE(v3.categorical[0] == static_cast<std::uint8_t>(StrategyDomain::TrueHeadsUp));
    REQUIRE(v3.categorical[5] == 2U);
    REQUIRE(v3.categorical[7] == 0U); // Hero active
    REQUIRE(v3.categorical[8] == 0U); // live opponent active
    REQUIRE(v3.categorical[9] == 2U); // absent chair marker
    REQUIRE(v3.numeric[5] == 0.0F);   // rel2 stack
    REQUIRE(v3.numeric[8] == 0.0F);   // rel2 street commitment
    REQUIRE(v3.numeric[11] == 0.0F);  // rel2 total commitment

    // Dealer is the HU small blind and the live opponent is the BB after the
    // canonical [Hero, live opponent, absent] remap.
    REQUIRE(v3.categorical[2] == 0U);
    REQUIRE(v3.categorical[3] == 0U);
    REQUIRE(v3.categorical[4] == 1U);
    for (const auto& event : v3.history) {
        REQUIRE(event.categorical[0] <= 1U);
    }
}

SPIN_TEST(neural_v3_keeps_complete_33_event_legal_history_while_v2_truncates) {
    HandEngine hand(schu(1), 424242);
    std::size_t raises = 0;
    while (hand.betting().history().size() <= 32U) {
        REQUIRE(!hand.terminal());
        REQUIRE(hand.betting().street() == Street::Preflop);
        const int actor = hand.betting().actor();
        const auto legal = hand.betting().legal_actions(actor);
        REQUIRE(legal.raise);
        REQUIRE(legal.min_raise_to < legal.max_raise_to);
        hand.apply(actor, {ExactActionType::RaiseTo, legal.min_raise_to});
        REQUIRE(++raises < 64U);
    }
    REQUIRE(hand.betting().history().size() == 33U);

    const auto v2 = encode_neural_input_v2(hand, 0);
    const auto v3 = encode_neural_input_v3(hand, 0);
    REQUIRE(v2.history_len == 32U);
    REQUIRE(v3.history.size() == 33U);
    REQUIRE(v3.history[0].categorical[3] == 1U); // first blind still present
    REQUIRE(v3.history[1].categorical[3] == 1U); // second blind still present

    const auto bytes = serialize_neural_input_v3(v3);
    REQUIRE(bytes.size() == kNeuralV3FixedSerializedBytes + 33U * kNeuralV3HistoryEventSerializedBytes);
    REQUIRE(bytes.size() == 780U);
}

SPIN_TEST(neural_v3_solver_c_api_supports_exact_size_query_and_copy) {
    spincore_solver_scenario_v2 scenario{};
    scenario.total_chips = 1500;
    scenario.game_is_hu = 0;
    scenario.blind_index = 3;
    scenario.small_blind = 10;
    scenario.big_blind = 20;
    scenario.stack_0 = 500;
    scenario.stack_1 = 500;
    scenario.stack_2 = 500;
    scenario.dead_player_0 = -1;
    scenario.dead_player_1 = -1;
    scenario.dead_player_count = 0;
    scenario.dealer_id = 0;

    auto* state = spincore_solver_state_create_v2(&scenario, 123456U);
    REQUIRE(state != nullptr);
    const std::size_t required = spincore_solver_state_neural_input_v3(state, nullptr, 0U);
    REQUIRE(required == 160U);

    std::vector<std::uint8_t> bytes(required);
    const std::size_t written = spincore_solver_state_neural_input_v3(state, bytes.data(), bytes.size());
    REQUIRE(written == required);
    REQUIRE(bytes[0] == 'S' && bytes[1] == 'P' && bytes[2] == 'N');
    REQUIRE(bytes[6] == '3');

    std::vector<std::uint8_t> too_small(required - 1U);
    REQUIRE(spincore_solver_state_neural_input_v3(state, too_small.data(), too_small.size()) == 0U);
    REQUIRE(std::string(spincore_solver_last_error()).find("buffer too small") != std::string::npos);
    spincore_solver_state_destroy(state);
}
