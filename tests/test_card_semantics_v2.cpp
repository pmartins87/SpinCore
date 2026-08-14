#include "test_framework.hpp"
#include "spincore/card_semantics_v2.hpp"

using namespace spincore;

namespace {
Card c(int rank, int suit) { return Card{static_cast<std::uint8_t>(rank), static_cast<std::uint8_t>(suit)}; }
std::array<Card,5> board3(Card a, Card b, Card d) { return {a,b,d,Card{},Card{}}; }
}

SPIN_TEST(card_semantics_v2_detects_top_pair_and_overpair) {
    auto flop = board3(c(14,1), c(7,2), c(2,3));
    auto top = derive_private_hand_semantics_v2({c(14,0), c(13,2)}, flop, 3);
    REQUIRE(top.preflop_class == "AKo");
    REQUIRE(top.made_category == HandCategory::Pair);
    REQUIRE(top.pair_relation == PairRelationV2::TopPair);

    auto low_flop = board3(c(11,1), c(7,2), c(2,3));
    auto over = derive_private_hand_semantics_v2({c(12,0), c(12,2)}, low_flop, 3);
    REQUIRE(over.pocket_pair);
    REQUIRE(over.made_category == HandCategory::Pair);
    REQUIRE(over.pair_relation == PairRelationV2::PocketOverpair);
}

SPIN_TEST(card_semantics_v2_detects_combo_flush_gutshot) {
    // Ah Kh on Qh Jh 2c: nut-contributor heart draw plus Broadway gutshot.
    auto flop = board3(c(12,1), c(11,1), c(2,3));
    auto sem = derive_private_hand_semantics_v2({c(14,1), c(13,1)}, flop, 3);
    REQUIRE(sem.flush_draw);
    REQUIRE(sem.flush_draw_suit == 1);
    REQUIRE(sem.flush_draw_higher_unseen_count == 0);
    REQUIRE(sem.straight_draw);
    REQUIRE(sem.gutshot);
    REQUIRE(!sem.open_ended_straight_draw);
    REQUIRE(sem.straight_draw_missing_rank_count == 1);
}

SPIN_TEST(card_semantics_v2_detects_open_ended_draw) {
    // 87 on 652 has 4/9 as distinct straight-completion ranks.
    auto flop = board3(c(6,3), c(5,0), c(2,1));
    auto sem = derive_private_hand_semantics_v2({c(8,1), c(7,2)}, flop, 3);
    REQUIRE(sem.straight_draw);
    REQUIRE(sem.open_ended_straight_draw);
    REQUIRE(!sem.gutshot);
    REQUIRE(sem.straight_draw_missing_rank_count >= 2);
}

SPIN_TEST(board_semantics_v2_tracks_turn_texture_delta) {
    std::array<Card,5> board{c(12,1), c(11,1), c(2,3), c(14,1), Card{}};
    auto sem = derive_board_semantics_v2(board, 4);
    REQUIRE(sem.visible_count == 4);
    REQUIRE(sem.max_suit_count == 3);
    REQUIRE(sem.broadway_count == 3);
    REQUIRE(sem.new_card_over_prior_high);
    REQUIRE(sem.new_card_creates_three_suit_board);
    REQUIRE(sem.new_card_increases_straight_window_occupancy);
}

SPIN_TEST(board_semantics_v2_tracks_pairing_turn) {
    std::array<Card,5> board{c(12,1), c(11,1), c(2,3), c(12,2), Card{}};
    auto sem = derive_board_semantics_v2(board, 4);
    REQUIRE(sem.paired);
    REQUIRE(sem.new_card_pairs_prior_rank);
    REQUIRE(!sem.new_card_over_prior_high);
    REQUIRE(!sem.new_card_under_prior_low);
}
