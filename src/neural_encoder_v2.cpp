#include "spincore/neural_encoder_v2.hpp"

#include "spincore/card_semantics_v2.hpp"
#include "spincore/flop_canonicalization.hpp"
#include "spincore/hand_infoset_adapter.hpp"
#include "spincore/postflop_ontology.hpp"
#include "spincore/preflop_ontology.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace spincore {
namespace {

[[nodiscard]] std::uint8_t rank_index(std::uint8_t rank) {
    if (rank < 2 || rank > 14) {
        throw std::invalid_argument("invalid hole-card rank for preflop class");
    }
    return static_cast<std::uint8_t>(14U - rank);
}

[[nodiscard]] std::uint8_t relative_seat_plus_one(std::int32_t seat, std::int32_t actor) noexcept {
    if (seat < 0 || actor < 0) {
        return 0;
    }
    return static_cast<std::uint8_t>(((seat - actor + 3) % 3) + 1);
}

[[nodiscard]] float safe_ratio(float numerator, float denominator) noexcept {
    return denominator > 0.0F ? numerator / denominator : 0.0F;
}

void append_float_le(std::vector<std::uint8_t>& bytes, float value) {
    const std::uint32_t raw = std::bit_cast<std::uint32_t>(value);
    for (int k = 0; k < 4; ++k) {
        bytes.push_back(static_cast<std::uint8_t>((raw >> (8 * k)) & 0xffU));
    }
}

}  // namespace

std::uint8_t preflop_class_id_v2(const std::array<Card, 2>& hole) {
    if (!hole[0].valid() || !hole[1].valid() || hole[0] == hole[1]) {
        throw std::invalid_argument("invalid hole cards for preflop class");
    }

    const std::uint8_t high_rank = std::max(hole[0].rank, hole[1].rank);
    const std::uint8_t low_rank = std::min(hole[0].rank, hole[1].rank);
    const std::uint8_t high_index = rank_index(high_rank);
    const std::uint8_t low_index = rank_index(low_rank);

    if (high_rank == low_rank) {
        return high_index; // AA=0 ... 22=12
    }

    // Enumerate unordered non-pair rank combinations in descending high-rank
    // order, then suited before offsuit. 13 pair classes + 2*78 = 169.
    std::uint16_t ordinal = 0;
    for (std::uint8_t h = 0; h < high_index; ++h) {
        ordinal = static_cast<std::uint16_t>(ordinal + (12U - h));
    }
    ordinal = static_cast<std::uint16_t>(ordinal + (low_index - high_index - 1U));
    const bool suited = hole[0].suit == hole[1].suit;
    const std::uint16_t result = static_cast<std::uint16_t>(13U + 2U * ordinal + (suited ? 0U : 1U));
    if (result > 168U) {
        throw std::logic_error("preflop class id overflow");
    }
    return static_cast<std::uint8_t>(result);
}

NeuralInputV2 encode_neural_input_v2(const HandEngine& hand, std::int32_t blind_index) {
    if (hand.terminal()) {
        throw std::logic_error("terminal hand has no neural input");
    }

    const auto infoset = build_current_actor_infoset(hand, blind_index);
    if (infoset.big_blind <= 0) {
        throw std::invalid_argument("big_blind must be positive for neural V2 encoding");
    }

    NeuralInputV2 out{};
    out.preflop_class_id = preflop_class_id_v2(infoset.hole);
    if (infoset.visible_board >= 3) {
        out.canonical_flop_signature = canonical_flop_signature(
            {infoset.board[0], infoset.board[1], infoset.board[2]}
        );
    }

    const auto board_sem = derive_board_semantics_v2(infoset.board, infoset.visible_board);
    const auto private_sem = derive_private_hand_semantics_v2(
        infoset.hole,
        infoset.board,
        infoset.visible_board
    );
    const auto pre = derive_preflop_ontology(hand.betting());
    const auto post = derive_postflop_ontology(hand.betting());

    const float bb = static_cast<float>(infoset.big_blind);
    out.numeric[0] = static_cast<float>(infoset.pot) / bb;
    out.numeric[1] = static_cast<float>(infoset.to_call) / bb;
    out.numeric[2] = static_cast<float>(infoset.current_bet) / bb;
    for (std::size_t index = 0; index < 3; ++index) {
        out.numeric[3 + index] = static_cast<float>(infoset.stacks[index]) / bb;
        out.numeric[6 + index] = static_cast<float>(infoset.street_commitments[index]) / bb;
        out.numeric[9 + index] = static_cast<float>(infoset.total_commitments[index]) / bb;
    }
    out.numeric[12] = static_cast<float>(infoset.small_blind) / bb;

    std::int32_t min_live_opponent_stack = std::numeric_limits<std::int32_t>::max();
    for (std::size_t rel = 1; rel < 3; ++rel) {
        if (infoset.statuses[rel] != 1U) { // not folded
            min_live_opponent_stack = std::min(min_live_opponent_stack, infoset.stacks[rel]);
        }
    }
    if (min_live_opponent_stack == std::numeric_limits<std::int32_t>::max()) {
        min_live_opponent_stack = 0;
    }
    const std::int32_t effective_remaining = std::min(infoset.stacks[0], min_live_opponent_stack);
    out.numeric[13] = static_cast<float>(effective_remaining) / bb;
    out.numeric[14] = static_cast<float>(min_live_opponent_stack) / bb;
    out.numeric[15] = safe_ratio(
        static_cast<float>(effective_remaining),
        static_cast<float>(infoset.pot)
    );
    out.numeric[16] = safe_ratio(
        static_cast<float>(infoset.to_call),
        static_cast<float>(infoset.pot + infoset.to_call)
    );
    out.numeric[17] = static_cast<float>(pre.first_raise_to) / bb;
    out.numeric[18] = static_cast<float>(pre.last_raise_to) / bb;
    out.numeric[19] = static_cast<float>(pre.first_raise_increment) / bb;
    out.numeric[20] = static_cast<float>(pre.last_raise_increment) / bb;
    out.numeric[21] = safe_ratio(
        static_cast<float>(infoset.to_call),
        static_cast<float>(infoset.pot)
    );
    out.numeric[22] = safe_ratio(
        static_cast<float>(infoset.current_bet),
        static_cast<float>(infoset.pot)
    );
    out.numeric[23] = safe_ratio(
        static_cast<float>(infoset.stacks[0]),
        static_cast<float>(infoset.pot)
    );

    const auto& topology = hand.betting().topology();
    const std::int32_t actor = hand.betting().actor();
    out.categorical[0] = static_cast<std::uint8_t>(infoset.domain);
    out.categorical[1] = static_cast<std::uint8_t>(infoset.street);
    out.categorical[2] = infoset.dealer_rel;
    out.categorical[3] = relative_seat_plus_one(topology.small_blind_seat, actor);
    out.categorical[4] = relative_seat_plus_one(topology.big_blind_seat, actor);
    out.categorical[5] = infoset.live_count;
    out.categorical[6] = infoset.visible_board;
    for (std::size_t index = 0; index < 3; ++index) {
        out.categorical[7 + index] = infoset.statuses[index];
    }

    out.categorical[10] = static_cast<std::uint8_t>(pre.lineage);
    out.categorical[11] = pre.voluntary_action_count;
    out.categorical[12] = pre.limp_count;
    out.categorical[13] = pre.calls_after_aggression;
    out.categorical[14] = pre.aggression_count;
    out.categorical[15] = relative_seat_plus_one(pre.first_aggressor, actor);
    out.categorical[16] = relative_seat_plus_one(pre.last_aggressor, actor);

    out.categorical[17] = static_cast<std::uint8_t>(post.opening_line);
    out.categorical[18] = static_cast<std::uint8_t>(post.facing_line);
    out.categorical[19] = static_cast<std::uint8_t>(post.attack_opportunity);
    out.categorical[20] = post.current_street_aggression_count;
    out.categorical[21] = post.raise_depth;
    out.categorical[22] = relative_seat_plus_one(post.lineage_aggressor, actor);
    out.categorical[23] = post.has_lineage_aggressor
        ? static_cast<std::uint8_t>(post.lineage_aggression_street + 1)
        : 0U;
    out.categorical[24] = post.skipped_streets_since_lineage;
    out.categorical[25] = post.lineage_checked_current_street ? 1U : 0U;
    out.categorical[26] = post.actor_called_lineage_aggression ? 1U : 0U;

    out.categorical[27] = static_cast<std::uint8_t>(private_sem.made_category);
    out.categorical[28] = static_cast<std::uint8_t>(private_sem.pair_relation);
    out.categorical[29] = private_sem.pocket_pair ? 1U : 0U;
    out.categorical[30] = private_sem.hole_rank_matches;
    out.categorical[31] = private_sem.overcard_count;
    out.categorical[32] = private_sem.flush_draw ? 1U : 0U;
    out.categorical[33] = private_sem.flush_draw
        ? static_cast<std::uint8_t>(private_sem.flush_draw_suit + 1U)
        : 0U;
    out.categorical[34] = private_sem.flush_draw_higher_unseen_count;
    out.categorical[35] = private_sem.backdoor_flush ? 1U : 0U;
    out.categorical[36] = private_sem.straight_draw ? 1U : 0U;
    out.categorical[37] = private_sem.open_ended_straight_draw ? 1U : 0U;
    out.categorical[38] = private_sem.gutshot ? 1U : 0U;
    out.categorical[39] = private_sem.double_gutshot ? 1U : 0U;
    out.categorical[40] = private_sem.straight_draw_missing_rank_count;
    out.categorical[41] = private_sem.backdoor_straight ? 1U : 0U;

    out.categorical[42] = board_sem.distinct_rank_count;
    out.categorical[43] = board_sem.distinct_suit_count;
    out.categorical[44] = board_sem.max_suit_count;
    out.categorical[45] = board_sem.broadway_count;
    out.categorical[46] = board_sem.high_rank;
    out.categorical[47] = board_sem.low_rank;
    out.categorical[48] = board_sem.rank_span;
    out.categorical[49] = board_sem.max_straight_window_occupancy;
    out.categorical[50] = board_sem.straight_windows_with_3plus;
    out.categorical[51] = board_sem.straight_windows_with_4plus;
    out.categorical[52] = board_sem.paired ? 1U : 0U;
    out.categorical[53] = board_sem.two_pair_on_board ? 1U : 0U;
    out.categorical[54] = board_sem.trips_on_board ? 1U : 0U;
    out.categorical[55] = board_sem.quads_on_board ? 1U : 0U;
    out.categorical[56] = board_sem.board_has_straight ? 1U : 0U;
    out.categorical[57] = board_sem.new_card_pairs_prior_rank ? 1U : 0U;
    out.categorical[58] = board_sem.new_card_over_prior_high ? 1U : 0U;
    out.categorical[59] = board_sem.new_card_under_prior_low ? 1U : 0U;
    out.categorical[60] = board_sem.new_card_creates_three_suit_board ? 1U : 0U;
    out.categorical[61] = board_sem.new_card_creates_four_suit_board ? 1U : 0U;
    out.categorical[62] = board_sem.new_card_increases_straight_window_occupancy ? 1U : 0U;
    out.categorical[63] = board_sem.new_card_creates_four_to_straight_window ? 1U : 0U;
    out.categorical[64] = board_sem.new_card_completes_board_straight ? 1U : 0U;

    std::uint8_t hole0_board_suit_count = 0;
    std::uint8_t hole1_board_suit_count = 0;
    for (std::size_t index = 0; index < infoset.visible_board; ++index) {
        if (infoset.board[index].suit == infoset.hole[0].suit) {
            ++hole0_board_suit_count;
        }
        if (infoset.board[index].suit == infoset.hole[1].suit) {
            ++hole1_board_suit_count;
        }
    }
    out.categorical[65] = hole0_board_suit_count;
    out.categorical[66] = hole1_board_suit_count;
    out.categorical[67] = infoset.hole[0].suit == infoset.hole[1].suit ? 1U : 0U;
    out.categorical[68] = private_sem.has_postflop ? 1U : 0U;
    out.categorical[69] = pre.had_limp_before_first_aggression ? 1U : 0U;
    out.categorical[70] = pre.limper_became_reraiser ? 1U : 0U;
    out.categorical[71] = 0U; // reserved by schema V1 for a future precommitted field

    out.legal_action_mask = infoset.legal_action_mask;

    const std::size_t count = std::min<std::size_t>(
        kNeuralV2HistoryCapacity,
        infoset.public_events.size()
    );
    out.history_len = static_cast<std::uint8_t>(count);
    const std::size_t first = infoset.public_events.size() - count;
    for (std::size_t index = 0; index < count; ++index) {
        const auto& source = infoset.public_events[first + index];
        auto& target = out.history[index];
        target.categorical[0] = source.actor_rel;
        target.categorical[1] = static_cast<std::uint8_t>(source.street);
        target.categorical[2] = static_cast<std::uint8_t>(source.action_type);
        target.categorical[3] = source.forced ? 1U : 0U;
        target.numeric[0] = static_cast<float>(source.paid) / bb;
        target.numeric[1] = static_cast<float>(source.resulting_commitment) / bb;
        target.numeric[2] = static_cast<float>(source.pot_before) / bb;
        target.numeric[3] = static_cast<float>(source.pot_after) / bb;
    }

    return out;
}

std::vector<std::uint8_t> serialize_neural_input_v2(const NeuralInputV2& input) {
    std::vector<std::uint8_t> bytes;
    const char magic[8] = {'S', 'P', 'N', 'N', 'I', 'V', '2', '\0'};
    bytes.insert(bytes.end(), magic, magic + 8);
    bytes.push_back(input.preflop_class_id);
    bytes.insert(
        bytes.end(),
        input.canonical_flop_signature.begin(),
        input.canonical_flop_signature.end()
    );
    for (float value : input.numeric) {
        append_float_le(bytes, value);
    }
    bytes.insert(bytes.end(), input.categorical.begin(), input.categorical.end());
    bytes.insert(bytes.end(), input.legal_action_mask.begin(), input.legal_action_mask.end());
    bytes.push_back(input.history_len);
    for (const auto& event : input.history) {
        bytes.insert(bytes.end(), event.categorical.begin(), event.categorical.end());
        for (float value : event.numeric) {
            append_float_le(bytes, value);
        }
    }
    return bytes;
}

}  // namespace spincore
