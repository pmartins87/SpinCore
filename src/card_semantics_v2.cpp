#include "spincore/card_semantics_v2.hpp"

#include "spincore/hand_semantics.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace spincore {
namespace {

constexpr std::array<std::array<int, 5>, 10> kStraightWindows{{
    {{14, 2, 3, 4, 5}},
    {{2, 3, 4, 5, 6}},
    {{3, 4, 5, 6, 7}},
    {{4, 5, 6, 7, 8}},
    {{5, 6, 7, 8, 9}},
    {{6, 7, 8, 9, 10}},
    {{7, 8, 9, 10, 11}},
    {{8, 9, 10, 11, 12}},
    {{9, 10, 11, 12, 13}},
    {{10, 11, 12, 13, 14}},
}};

struct StraightSummary {
    int maximum{0};
    int three_plus{0};
    int four_plus{0};
};

[[nodiscard]] StraightSummary summarize_straight_windows(const std::array<int, 15>& rank_counts) noexcept {
    StraightSummary out{};
    for (const auto& window : kStraightWindows) {
        int occupied = 0;
        for (int rank : window) {
            if (rank_counts[static_cast<std::size_t>(rank)] > 0) {
                ++occupied;
            }
        }
        out.maximum = std::max(out.maximum, occupied);
        if (occupied >= 3) {
            ++out.three_plus;
        }
        if (occupied >= 4) {
            ++out.four_plus;
        }
    }
    return out;
}

void validate_cards(
    const std::array<Card, 2>* hole,
    const std::array<Card, 5>& board,
    std::uint8_t visible_board
) {
    if (visible_board > board.size()) {
        throw std::invalid_argument("visible_board exceeds five community cards");
    }
    std::array<bool, 52> used{};
    auto add = [&](const Card& card) {
        if (!card.valid()) {
            throw std::invalid_argument("invalid visible card in semantic encoder");
        }
        const auto id = static_cast<std::size_t>(card.id());
        if (used[id]) {
            throw std::invalid_argument("duplicate visible card in semantic encoder");
        }
        used[id] = true;
    };
    if (hole != nullptr) {
        add((*hole)[0]);
        add((*hole)[1]);
    }
    for (std::size_t index = 0; index < visible_board; ++index) {
        add(board[index]);
    }
}

[[nodiscard]] std::array<int, 15> board_rank_counts(
    const std::array<Card, 5>& board,
    std::uint8_t visible_board
) noexcept {
    std::array<int, 15> counts{};
    for (std::size_t index = 0; index < visible_board; ++index) {
        ++counts[board[index].rank];
    }
    return counts;
}

[[nodiscard]] bool hero_adds_distinct_rank_to_window(
    const std::array<Card, 2>& hole,
    const std::array<int, 15>& board_counts,
    const std::array<int, 5>& window
) noexcept {
    for (const auto& card : hole) {
        if (board_counts[card.rank] > 0) {
            continue;
        }
        if (std::find(window.begin(), window.end(), static_cast<int>(card.rank)) != window.end()) {
            return true;
        }
    }
    return false;
}

}  // namespace

BoardSemanticsV2 derive_board_semantics_v2(
    const std::array<Card, 5>& board,
    std::uint8_t visible_board
) {
    validate_cards(nullptr, board, visible_board);
    BoardSemanticsV2 out{};
    out.visible_count = visible_board;
    if (visible_board == 0) {
        return out;
    }

    std::array<int, 15> rank_counts{};
    std::array<int, 4> suit_counts{};
    int high = 0;
    int low = 15;
    for (std::size_t index = 0; index < visible_board; ++index) {
        const auto& card = board[index];
        ++rank_counts[card.rank];
        ++suit_counts[card.suit];
        high = std::max(high, static_cast<int>(card.rank));
        low = std::min(low, static_cast<int>(card.rank));
        if (card.rank >= 10) {
            ++out.broadway_count;
        }
    }

    int rank_pairs_or_better = 0;
    for (int rank = 2; rank <= 14; ++rank) {
        const int count = rank_counts[static_cast<std::size_t>(rank)];
        if (count > 0) {
            ++out.distinct_rank_count;
        }
        if (count >= 2) {
            ++rank_pairs_or_better;
        }
        if (count >= 3) {
            out.trips_on_board = true;
        }
        if (count >= 4) {
            out.quads_on_board = true;
        }
    }
    out.paired = rank_pairs_or_better > 0;
    out.two_pair_on_board = rank_pairs_or_better >= 2;

    for (int suit = 0; suit < 4; ++suit) {
        const int count = suit_counts[static_cast<std::size_t>(suit)];
        if (count > 0) {
            ++out.distinct_suit_count;
        }
        out.max_suit_count = static_cast<std::uint8_t>(
            std::max<int>(out.max_suit_count, count)
        );
    }

    out.high_rank = static_cast<std::uint8_t>(high);
    out.low_rank = static_cast<std::uint8_t>(low);
    out.rank_span = static_cast<std::uint8_t>(high - low);
    const auto straight = summarize_straight_windows(rank_counts);
    out.max_straight_window_occupancy = static_cast<std::uint8_t>(straight.maximum);
    out.straight_windows_with_3plus = static_cast<std::uint8_t>(straight.three_plus);
    out.straight_windows_with_4plus = static_cast<std::uint8_t>(straight.four_plus);
    out.board_has_straight = straight.maximum >= 5;

    if (visible_board >= 4) {
        const std::uint8_t prior_visible = static_cast<std::uint8_t>(visible_board - 1U);
        std::array<int, 15> prior_ranks{};
        std::array<int, 4> prior_suits{};
        int prior_high = 0;
        int prior_low = 15;
        for (std::size_t index = 0; index < prior_visible; ++index) {
            const auto& card = board[index];
            ++prior_ranks[card.rank];
            ++prior_suits[card.suit];
            prior_high = std::max(prior_high, static_cast<int>(card.rank));
            prior_low = std::min(prior_low, static_cast<int>(card.rank));
        }
        const auto& new_card = board[visible_board - 1U];
        out.new_card_pairs_prior_rank = prior_ranks[new_card.rank] > 0;
        out.new_card_over_prior_high = new_card.rank > prior_high;
        out.new_card_under_prior_low = new_card.rank < prior_low;

        int prior_max_suit = 0;
        for (int count : prior_suits) {
            prior_max_suit = std::max(prior_max_suit, count);
        }
        out.new_card_creates_three_suit_board = prior_max_suit < 3 && out.max_suit_count >= 3;
        out.new_card_creates_four_suit_board = prior_max_suit < 4 && out.max_suit_count >= 4;

        const auto prior_straight = summarize_straight_windows(prior_ranks);
        out.new_card_increases_straight_window_occupancy = straight.maximum > prior_straight.maximum;
        out.new_card_creates_four_to_straight_window = prior_straight.maximum < 4 && straight.maximum >= 4;
        out.new_card_completes_board_straight = prior_straight.maximum < 5 && straight.maximum >= 5;
    }

    return out;
}

PrivateHandSemanticsV2 derive_private_hand_semantics_v2(
    const std::array<Card, 2>& hole,
    const std::array<Card, 5>& board,
    std::uint8_t visible_board
) {
    validate_cards(&hole, board, visible_board);
    PrivateHandSemanticsV2 out{};
    out.preflop_class = hand_class(hole);
    out.pocket_pair = hole[0].rank == hole[1].rank;
    if (visible_board < 3) {
        return out;
    }
    out.has_postflop = true;

    std::array<Card, 7> cards{};
    cards[0] = hole[0];
    cards[1] = hole[1];
    for (std::size_t index = 0; index < visible_board; ++index) {
        cards[index + 2U] = board[index];
    }
    out.made_category = evaluate_best(
        std::span<const Card>(cards.data(), static_cast<std::size_t>(visible_board) + 2U)
    ).category;

    const auto bcounts = board_rank_counts(board, visible_board);
    int board_high = 0;
    std::vector<int> distinct_board_ranks;
    for (int rank = 14; rank >= 2; --rank) {
        if (bcounts[static_cast<std::size_t>(rank)] > 0) {
            board_high = std::max(board_high, rank);
            distinct_board_ranks.push_back(rank);
        }
    }
    const bool board_paired = std::any_of(
        bcounts.begin() + 2,
        bcounts.end(),
        [](int count) { return count >= 2; }
    );

    bool match0 = bcounts[hole[0].rank] > 0;
    bool match1 = bcounts[hole[1].rank] > 0;
    out.hole_rank_matches = static_cast<std::uint8_t>(static_cast<int>(match0) + static_cast<int>(match1));
    out.overcard_count = static_cast<std::uint8_t>(
        static_cast<int>(hole[0].rank > board_high) + static_cast<int>(hole[1].rank > board_high)
    );

    if (out.pocket_pair) {
        if (match0) {
            out.pair_relation = PairRelationV2::PocketPairHitSet;
        } else if (hole[0].rank > board_high) {
            out.pair_relation = PairRelationV2::PocketOverpair;
        } else {
            out.pair_relation = PairRelationV2::PocketBelowTop;
        }
    } else if (match0 && match1) {
        out.pair_relation = PairRelationV2::BothHoleRanksPaired;
    } else if (match0 || match1) {
        const int matched_rank = match0 ? hole[0].rank : hole[1].rank;
        auto found = std::find(distinct_board_ranks.begin(), distinct_board_ranks.end(), matched_rank);
        const auto position = static_cast<std::size_t>(std::distance(distinct_board_ranks.begin(), found));
        if (position == 0U) {
            out.pair_relation = PairRelationV2::TopPair;
        } else if (position == 1U) {
            out.pair_relation = PairRelationV2::SecondPair;
        } else {
            out.pair_relation = PairRelationV2::LowerPair;
        }
    } else if (board_paired) {
        out.pair_relation = PairRelationV2::BoardOnlyPair;
    }

    std::array<int, 4> combined_suits{};
    std::array<std::array<bool, 15>, 4> rank_present_in_suit{};
    auto add_suit = [&](const Card& card) {
        ++combined_suits[card.suit];
        rank_present_in_suit[card.suit][card.rank] = true;
    };
    add_suit(hole[0]);
    add_suit(hole[1]);
    for (std::size_t index = 0; index < visible_board; ++index) {
        add_suit(board[index]);
    }

    if (visible_board < 5) {
        for (int suit = 0; suit < 4; ++suit) {
            const bool hero_has_suit = hole[0].suit == suit || hole[1].suit == suit;
            if (combined_suits[static_cast<std::size_t>(suit)] == 4 && hero_has_suit) {
                out.flush_draw = true;
                out.flush_draw_suit = static_cast<std::uint8_t>(suit);
                int hero_high = 0;
                if (hole[0].suit == suit) {
                    hero_high = std::max(hero_high, static_cast<int>(hole[0].rank));
                }
                if (hole[1].suit == suit) {
                    hero_high = std::max(hero_high, static_cast<int>(hole[1].rank));
                }
                int higher_unseen = 0;
                for (int rank = hero_high + 1; rank <= 14; ++rank) {
                    if (!rank_present_in_suit[static_cast<std::size_t>(suit)][static_cast<std::size_t>(rank)]) {
                        ++higher_unseen;
                    }
                }
                out.flush_draw_higher_unseen_count = static_cast<std::uint8_t>(higher_unseen);
                break;
            }
        }
    }

    if (visible_board == 3 && !out.flush_draw) {
        for (int suit = 0; suit < 4; ++suit) {
            const bool hero_has_suit = hole[0].suit == suit || hole[1].suit == suit;
            if (combined_suits[static_cast<std::size_t>(suit)] == 3 && hero_has_suit) {
                out.backdoor_flush = true;
                break;
            }
        }
    }

    std::array<int, 15> combined_ranks{};
    ++combined_ranks[hole[0].rank];
    ++combined_ranks[hole[1].rank];
    for (std::size_t index = 0; index < visible_board; ++index) {
        ++combined_ranks[board[index].rank];
    }
    const auto combined_straight = summarize_straight_windows(combined_ranks);
    const bool already_has_straight = combined_straight.maximum >= 5;

    std::array<bool, 15> missing_ranks{};
    if (!already_has_straight && visible_board < 5) {
        for (const auto& window : kStraightWindows) {
            int occupied = 0;
            int missing = -1;
            for (int rank : window) {
                if (combined_ranks[static_cast<std::size_t>(rank)] > 0) {
                    ++occupied;
                } else {
                    missing = rank;
                }
            }
            if (occupied == 4 && missing >= 2 && hero_adds_distinct_rank_to_window(hole, bcounts, window)) {
                missing_ranks[static_cast<std::size_t>(missing)] = true;
            }
        }

        int missing_count = 0;
        for (int rank = 2; rank <= 14; ++rank) {
            if (missing_ranks[static_cast<std::size_t>(rank)]) {
                ++missing_count;
            }
        }
        out.straight_draw_missing_rank_count = static_cast<std::uint8_t>(missing_count);
        out.straight_draw = missing_count > 0;

        // Four consecutive ranks with two legal end completions: 3-6 through T-K.
        for (int low = 3; low <= 10 && !out.open_ended_straight_draw; ++low) {
            bool all_present = true;
            bool hero_contributes = false;
            for (int rank = low; rank <= low + 3; ++rank) {
                if (combined_ranks[static_cast<std::size_t>(rank)] == 0) {
                    all_present = false;
                    break;
                }
                if ((hole[0].rank == rank && bcounts[hole[0].rank] == 0) ||
                    (hole[1].rank == rank && bcounts[hole[1].rank] == 0)) {
                    hero_contributes = true;
                }
            }
            if (all_present && hero_contributes) {
                out.open_ended_straight_draw = true;
            }
        }
        out.gutshot = out.straight_draw && !out.open_ended_straight_draw && missing_count == 1;
        out.double_gutshot = out.straight_draw && !out.open_ended_straight_draw && missing_count >= 2;
    }

    if (visible_board == 3 && !already_has_straight && !out.straight_draw) {
        for (const auto& window : kStraightWindows) {
            int occupied = 0;
            for (int rank : window) {
                if (combined_ranks[static_cast<std::size_t>(rank)] > 0) {
                    ++occupied;
                }
            }
            if (occupied == 3 && hero_adds_distinct_rank_to_window(hole, bcounts, window)) {
                out.backdoor_straight = true;
                break;
            }
        }
    }

    return out;
}

const char* pair_relation_v2_name(PairRelationV2 relation) noexcept {
    switch (relation) {
        case PairRelationV2::None: return "none";
        case PairRelationV2::PocketOverpair: return "pocket_overpair";
        case PairRelationV2::PocketBelowTop: return "pocket_below_top";
        case PairRelationV2::PocketPairHitSet: return "pocket_pair_hit_set";
        case PairRelationV2::TopPair: return "top_pair";
        case PairRelationV2::SecondPair: return "second_pair";
        case PairRelationV2::LowerPair: return "lower_pair";
        case PairRelationV2::BothHoleRanksPaired: return "both_hole_ranks_paired";
        case PairRelationV2::BoardOnlyPair: return "board_only_pair";
    }
    return "unknown";
}

}  // namespace spincore
