#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "spincore/hand_infoset_adapter.hpp"
#include "spincore/neural_encoder.hpp"

using namespace spincore;

SPIN_TEST(neural_preflop_has_two_cards_only) {
    HandEngine hand(sc3(), 42);
    const auto input = encode_neural_input_v1(build_current_actor_infoset(hand, 0));
    int non_zero = 0;
    for (auto token : input.card_tokens) {
        if (token) {
            ++non_zero;
        }
    }
    REQUIRE(non_zero == 2);
}

SPIN_TEST(neural_serialization_has_magic) {
    HandEngine hand(sc3(), 42);
    const auto bytes = serialize_neural_input_v1(
        encode_neural_input_v1(build_current_actor_infoset(hand, 0))
    );
    REQUIRE(bytes.size() > 100);
    REQUIRE(bytes[0] == 'S' && bytes[1] == 'P' && bytes[2] == 'N');
}

SPIN_TEST(neural_encoder_rejects_visible_board_overflow) {
    HandEngine hand(sc3(), 42);
    auto infoset = build_current_actor_infoset(hand, 0);
    infoset.visible_board = 6;
    REQUIRE_THROWS(encode_neural_input_v1(infoset));
}

SPIN_TEST(neural_encoder_rejects_nonpositive_big_blind) {
    HandEngine hand(sc3(), 42);
    auto infoset = build_current_actor_infoset(hand, 0);
    infoset.big_blind = 0;
    REQUIRE_THROWS(encode_neural_input_v1(infoset));
}
