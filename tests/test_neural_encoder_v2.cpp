#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "spincore/hand_infoset_adapter.hpp"
#include "spincore/neural_encoder.hpp"
#include "spincore/neural_encoder_v2.hpp"
#include "spincore/preflop_ontology.hpp"

using namespace spincore;

namespace {
Card nc(int rank, int suit) { return Card{static_cast<std::uint8_t>(rank), static_cast<std::uint8_t>(suit)}; }
void reach_limped_flop(HandEngine& hand) {
    hand.apply(0, {ExactActionType::Call, 0});
    hand.apply(1, {ExactActionType::Call, 0});
    hand.apply(2, {ExactActionType::Check, 0});
    REQUIRE(hand.betting().street() == Street::Flop);
}
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
    reach_limped_flop(hand);
    REQUIRE(hand.visible_board_count() == 3);

    const auto v2 = encode_neural_input_v2(hand, 0);
    REQUIRE(v2.categorical[6] == 3);
    REQUIRE(v2.canonical_flop_signature[0] >= 2);
    REQUIRE(v2.canonical_flop_signature[0] <= 14);
    REQUIRE(v2.canonical_flop_signature[1] <= 3);
    REQUIRE(v2.history_len == 5); // 2 forced blinds + 3 voluntary actions
    REQUIRE(v2.history[2].categorical[3] == 0);
}

SPIN_TEST(neural_v2_same_state_serializes_identically) {
    HandEngine hand(sc3(0), 4242);
    reach_limped_flop(hand);
    const auto first = serialize_neural_input_v2(encode_neural_input_v2(hand, 0));
    const auto second = serialize_neural_input_v2(encode_neural_input_v2(hand, 0));
    REQUIRE(first == second);
}

SPIN_TEST(neural_v2_distinguishes_one_third_and_one_half_pot_bets) {
    HandEngine one_third(sc3(0), 777);
    HandEngine one_half(sc3(0), 777);
    reach_limped_flop(one_third);
    reach_limped_flop(one_half);

    REQUIRE(one_third.betting().pot() == 60);
    REQUIRE(one_half.betting().pot() == 60);
    const int actor = one_third.betting().actor();
    REQUIRE(actor == one_half.betting().actor());
    REQUIRE(one_third.betting().legal_actions(actor).min_raise_to <= 20);
    REQUIRE(one_half.betting().legal_actions(actor).max_raise_to >= 30);

    one_third.apply(actor, {ExactActionType::BetTo, 20}); // 20 / 60 = 33.3%
    one_half.apply(actor, {ExactActionType::BetTo, 30});  // 30 / 60 = 50%

    const auto a = encode_neural_input_v2(one_third, 0);
    const auto b = encode_neural_input_v2(one_half, 0);
    REQUIRE(a.history_len == b.history_len);
    const auto last = static_cast<std::size_t>(a.history_len - 1U);
    REQUIRE(a.history[last].categorical == b.history[last].categorical);
    REQUIRE(a.history[last].numeric[0] == 1.0F);
    REQUIRE(b.history[last].numeric[0] == 1.5F);
    REQUIRE(a.history[last].numeric[2] == 3.0F);
    REQUIRE(b.history[last].numeric[2] == 3.0F);
    REQUIRE(a.history[last].numeric[3] == 4.0F);
    REQUIRE(b.history[last].numeric[3] == 4.5F);
    REQUIRE(serialize_neural_input_v2(a) != serialize_neural_input_v2(b));
}

SPIN_TEST(neural_v2_encoding_is_side_effect_free_for_terminal_settlement) {
    HandEngine control(schu(1), 998877);
    HandEngine observed(schu(1), 998877);
    for (int i = 0; i < 32; ++i) {
        const auto bytes = serialize_neural_input_v2(encode_neural_input_v2(observed, 0));
        REQUIRE(bytes.size() == 830);
    }
    REQUIRE(control.betting().actor() == observed.betting().actor());
    const int actor = control.betting().actor();
    control.apply(actor, {ExactActionType::Fold, 0});
    observed.apply(actor, {ExactActionType::Fold, 0});
    REQUIRE(control.terminal());
    REQUIRE(observed.terminal());
    REQUIRE(control.settle() == observed.settle());
}
