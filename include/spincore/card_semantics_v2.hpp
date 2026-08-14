#pragma once

#include "spincore/card.hpp"
#include "spincore/hand_evaluator.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace spincore {

enum class PairRelationV2 : std::uint8_t {
    None = 0,
    PocketOverpair = 1,
    PocketBelowTop = 2,
    PocketPairHitSet = 3,
    TopPair = 4,
    SecondPair = 5,
    LowerPair = 6,
    BothHoleRanksPaired = 7,
    BoardOnlyPair = 8,
};

struct BoardSemanticsV2 {
    std::uint8_t visible_count{0};
    std::uint8_t distinct_rank_count{0};
    std::uint8_t distinct_suit_count{0};
    std::uint8_t max_suit_count{0};
    std::uint8_t broadway_count{0};
    std::uint8_t high_rank{0};
    std::uint8_t low_rank{0};
    std::uint8_t rank_span{0};
    std::uint8_t max_straight_window_occupancy{0};
    std::uint8_t straight_windows_with_3plus{0};
    std::uint8_t straight_windows_with_4plus{0};

    bool paired{false};
    bool two_pair_on_board{false};
    bool trips_on_board{false};
    bool quads_on_board{false};
    bool board_has_straight{false};

    // Turn/river delta relative to the previously visible board.
    bool new_card_pairs_prior_rank{false};
    bool new_card_over_prior_high{false};
    bool new_card_under_prior_low{false};
    bool new_card_creates_three_suit_board{false};
    bool new_card_creates_four_suit_board{false};
    bool new_card_increases_straight_window_occupancy{false};
    bool new_card_creates_four_to_straight_window{false};
    bool new_card_completes_board_straight{false};
};

struct PrivateHandSemanticsV2 {
    std::string preflop_class;
    bool has_postflop{false};
    HandCategory made_category{HandCategory::HighCard};
    PairRelationV2 pair_relation{PairRelationV2::None};

    bool pocket_pair{false};
    std::uint8_t hole_rank_matches{0};
    std::uint8_t overcard_count{0};

    bool flush_draw{false};
    std::uint8_t flush_draw_suit{4}; // 0..3, 4 = none
    std::uint8_t flush_draw_higher_unseen_count{0};
    bool backdoor_flush{false};

    bool straight_draw{false};
    bool open_ended_straight_draw{false};
    bool gutshot{false};
    bool double_gutshot{false};
    std::uint8_t straight_draw_missing_rank_count{0};
    bool backdoor_straight{false};
};

[[nodiscard]] BoardSemanticsV2 derive_board_semantics_v2(
    const std::array<Card, 5>& board,
    std::uint8_t visible_board
);

[[nodiscard]] PrivateHandSemanticsV2 derive_private_hand_semantics_v2(
    const std::array<Card, 2>& hole,
    const std::array<Card, 5>& board,
    std::uint8_t visible_board
);

[[nodiscard]] const char* pair_relation_v2_name(PairRelationV2 relation) noexcept;

}  // namespace spincore
