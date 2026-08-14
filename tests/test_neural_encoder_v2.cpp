#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "spincore/hand_infoset_adapter.hpp"
#include "spincore/neural_encoder.hpp"
#include "spincore/neural_encoder_v2.hpp"
#include "spincore/preflop_ontology.hpp"

using namespace spincore;

namespace {
Card nc(int rank, int suit) { return Card{static_cast<std::uint8_t>(rank), static_cast<std::uint8_t>(suit)}; }
}

SPIN_TEST(neural_v2_preflop_class_id_is_complete_169_scheme) {
    REQUIRE(preflop_class_id_v2({nc(14,0), nc(14,1)}) == 0);   // AA
    REQUIRE(preflop_class_id_v2({nc(2,0), nc(2,1)}) == 12);    // 22
    REQUIRE(preflop_class_id_v2({nc(14,0), nc(13,0)}) == 13);  // AKs
    REQUIRE(preflop_class_id_v2({nc(14,0), nc(13,1)}) == 14);  // AKo
    REQUIRE(preflop_class_id_v2({nc(3,0), nc(2,1)}) == 168);   // 32o
}

SPIN_TEST(neural_v2_initial_state_preserves_exact_geometry_and_forced_history) {
    HandEngine hand(sc3(0), 42);
    const auto v2 = encode_neural_input_v2(hand, 0);
    REQUIRE(v2.preflop_class_id <= 168);
    for (auto value : v2.canonical_flop_signature) {
        REQUIRE(value == 0);
    }
    REQUIRE(v2.numeric[0] == 1.5F); // 30-chip pot / 20-chip BB
    REQUIRE(v2.categorical[10] == static_cast<std::uint8_t>(PreflopLineageType::Unopened));
    REQUIRE(v2.history_len == 2);
    REQUIRE(v2.history[0].categorical[3] == 1);
    REQUIRE(v2.history[1].categorical[3] == 1);
    REQUIRE(v2.history[0].numeric[2] == 0.0F);
    REQUIRE(v2.history[0].numeric[3] == 0.5F);
    REQUIRE(v2.history[1].numeric[2] == 0.5F);
    REQUIRE(v2.history[1].numeric[3] == 1.5F);
}

SPIN_TEST(neural_v2_serialization_is_separate_from_frozen_v1) {
    HandEngine hand(sc3(0), 42);
    const auto v1_bytes = serialize_neural_input_v1(
        encode_neural_input_v1(build_current_actor_infoset(hand, 0))
    );
    const auto v2_bytes = serialize_neural_input_v2(encode_neural_input_v2(hand, 0));

    REQUIRE(v1_bytes.size() == 126);
    REQUIRE(v1_bytes[0] == 'S' && v1_bytes[6] == '1');
    REQUIRE(v2_bytes.size() == 830);
    REQUIRE(v2_bytes[0] == 'S' && v2_bytes[1] == 'P' && v2_bytes[2] == 'N');
    REQUIRE(v2_bytes[6] == '2');
    REQUIRE(v1_bytes != v2_bytes);
}

SPIN_TEST(neural_v2_reaches_flop_without_absolute_suit_flop_ids) {
    HandEngine hand(sc3(0), 42);
    hand.apply(0, {ExactActionType::Call, 0});
    hand.apply(1, {ExactActionType::Call, 0});
    hand.apply(2, {ExactActionType::Check, 0});
    REQUIRE(hand.betting().street() == Street::Flop);
    REQUIRE(hand.visible_board_count() == 3);

    const auto v2 = encode_neural_input_v2(hand, 0);
    REQUIRE(v2.categorical[6] == 3);
    REQUIRE(v2.canonical_flop_signature[0] >= 2);
    REQUIRE(v2.canonical_flop_signature[0] <= 14);
    REQUIRE(v2.canonical_flop_signature[1] <= 3);
    REQUIRE(v2.history_len == 5); // 2 forced blinds + 3 voluntary actions
    REQUIRE(v2.history[2].categorical[3] == 0);
}
