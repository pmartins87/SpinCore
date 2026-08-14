#include "spincore/neural_encoder_v3.hpp"

#include "spincore/hand_infoset_adapter.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace spincore {
namespace {

struct RelativeSeatMapV3 {
    // new relative slot -> old actor-relative slot
    std::array<std::uint8_t, 3> new_to_old{0U, 1U, 2U};
    // old actor-relative slot -> new canonical relative slot
    std::array<std::uint8_t, 3> old_to_new{0U, 1U, 2U};
};

[[nodiscard]] RelativeSeatMapV3 seat_map_v3(const CanonicalInfoset& infoset) {
    RelativeSeatMapV3 map{};
    if (infoset.domain != StrategyDomain::TrueHeadsUp) {
        return map;
    }

    int dead_rel = -1;
    for (int rel = 1; rel < 3; ++rel) {
        const auto index = static_cast<std::size_t>(rel);
        const bool absent =
            infoset.statuses[index] == 2U &&
            infoset.stacks[index] == 0 &&
            infoset.street_commitments[index] == 0 &&
            infoset.total_commitments[index] == 0;
        if (absent) {
            if (dead_rel >= 0) {
                throw std::logic_error("true-HU V3 observation has multiple absent relative seats");
            }
            dead_rel = rel;
        }
    }
    if (dead_rel < 0) {
        throw std::logic_error("true-HU V3 observation cannot identify absent relative seat");
    }
    const int live_rel = dead_rel == 1 ? 2 : 1;
    map.new_to_old = {
        0U,
        static_cast<std::uint8_t>(live_rel),
        static_cast<std::uint8_t>(dead_rel),
    };
    for (std::size_t new_rel = 0; new_rel < 3; ++new_rel) {
        const auto old_rel = map.new_to_old[new_rel];
        map.old_to_new[old_rel] = static_cast<std::uint8_t>(new_rel);
    }
    return map;
}

[[nodiscard]] std::uint8_t physical_to_old_rel(int seat, int actor) {
    if (seat < 0 || seat >= 3 || actor < 0 || actor >= 3) {
        throw std::logic_error("invalid physical seat for V3 relative mapping");
    }
    return static_cast<std::uint8_t>((seat - actor + 3) % 3);
}

[[nodiscard]] float bb_ratio(std::int32_t chips, float bb) noexcept {
    return static_cast<float>(chips) / bb;
}

void append_float_le(std::vector<std::uint8_t>& bytes, float value) {
    const std::uint32_t raw = std::bit_cast<std::uint32_t>(value);
    for (int shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<std::uint8_t>((raw >> shift) & 0xffU));
    }
}

void append_u32_le(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
    }
}

}  // namespace

NeuralInputV3 encode_neural_input_v3(const HandEngine& hand, std::int32_t blind_index) {
    if (hand.terminal()) {
        throw std::logic_error("terminal hand has no neural V3 input");
    }

    const auto infoset = build_current_actor_infoset(hand, blind_index);
    if (infoset.big_blind <= 0) {
        throw std::invalid_argument("big_blind must be positive for neural V3 encoding");
    }
    const auto& betting = hand.betting();
    const int actor = betting.actor();
    if (actor < 0 || actor >= 3) {
        throw std::logic_error("nonterminal hand has invalid actor for neural V3 encoding");
    }
    const auto mapping = seat_map_v3(infoset);
    const float bb = static_cast<float>(infoset.big_blind);

    NeuralInputV3 out{};

    std::array<Card, kNeuralV3CardSlotCount> card_slots{};
    std::array<bool, kNeuralV3CardSlotCount> visible{};
    card_slots[0] = infoset.hole[0];
    card_slots[1] = infoset.hole[1];
    visible[0] = true;
    visible[1] = true;
    for (std::size_t index = 0; index < infoset.visible_board; ++index) {
        card_slots[index + 2U] = infoset.board[index];
        visible[index + 2U] = true;
    }
    for (std::size_t index = 0; index < card_slots.size(); ++index) {
        if (visible[index]) {
            if (!card_slots[index].valid()) {
                throw std::logic_error("visible V3 card slot contains invalid card");
            }
            out.rank_tokens[index] = card_slots[index].rank;
        }
    }

    std::size_t relation_index = 0;
    for (std::size_t left = 0; left < card_slots.size(); ++left) {
        for (std::size_t right = left + 1U; right < card_slots.size(); ++right) {
            if (relation_index >= out.same_suit.size()) {
                throw std::logic_error("V3 suit-relation overflow");
            }
            out.same_suit[relation_index++] =
                visible[left] && visible[right] && card_slots[left].suit == card_slots[right].suit
                    ? 1U
                    : 0U;
        }
    }
    if (relation_index != out.same_suit.size()) {
        throw std::logic_error("V3 suit-relation count mismatch");
    }

    const auto old_dealer_rel = infoset.dealer_rel;
    const auto old_sb_rel = physical_to_old_rel(betting.topology().small_blind_seat, actor);
    const auto old_bb_rel = physical_to_old_rel(betting.topology().big_blind_seat, actor);
    out.categorical[0] = static_cast<std::uint8_t>(infoset.domain);
    out.categorical[1] = static_cast<std::uint8_t>(infoset.street);
    out.categorical[2] = mapping.old_to_new[old_dealer_rel];
    out.categorical[3] = mapping.old_to_new[old_sb_rel];
    out.categorical[4] = mapping.old_to_new[old_bb_rel];
    out.categorical[5] = infoset.live_count;
    out.categorical[6] = infoset.visible_board;
    for (std::size_t new_rel = 0; new_rel < 3; ++new_rel) {
        const auto old_rel = mapping.new_to_old[new_rel];
        out.categorical[7 + new_rel] = infoset.statuses[old_rel];
    }

    out.numeric[0] = bb_ratio(infoset.pot, bb);
    out.numeric[1] = bb_ratio(infoset.to_call, bb);
    out.numeric[2] = bb_ratio(infoset.current_bet, bb);
    for (std::size_t new_rel = 0; new_rel < 3; ++new_rel) {
        const auto old_rel = mapping.new_to_old[new_rel];
        out.numeric[3 + new_rel] = bb_ratio(infoset.stacks[old_rel], bb);
        out.numeric[6 + new_rel] = bb_ratio(infoset.street_commitments[old_rel], bb);
        out.numeric[9 + new_rel] = bb_ratio(infoset.total_commitments[old_rel], bb);
    }
    out.numeric[12] = bb_ratio(infoset.small_blind, bb);
    out.numeric[13] = static_cast<float>(blind_index);

    const auto legal = betting.legal_actions(actor);
    out.numeric[14] = bb_ratio(legal.min_raise_to, bb);
    out.numeric[15] = bb_ratio(legal.max_raise_to, bb);
    out.primitive_legal = {
        legal.fold ? 1U : 0U,
        legal.check ? 1U : 0U,
        legal.call ? 1U : 0U,
        legal.bet ? 1U : 0U,
        legal.raise ? 1U : 0U,
        legal.all_in ? 1U : 0U,
    };

    out.history.reserve(infoset.public_events.size());
    for (const auto& source : infoset.public_events) {
        NeuralHistoryEventV3 target{};
        if (source.actor_rel > 2U) {
            throw std::logic_error("V3 public event actor_rel outside [0,2]");
        }
        target.categorical[0] = mapping.old_to_new[source.actor_rel];
        target.categorical[1] = static_cast<std::uint8_t>(source.street);
        target.categorical[2] = static_cast<std::uint8_t>(source.action_type);
        target.categorical[3] = source.forced ? 1U : 0U;
        target.numeric[0] = bb_ratio(source.paid, bb);
        target.numeric[1] = bb_ratio(source.resulting_commitment, bb);
        target.numeric[2] = bb_ratio(source.pot_before, bb);
        target.numeric[3] = bb_ratio(source.pot_after, bb);
        out.history.push_back(target);
    }

    return out;
}

std::vector<std::uint8_t> serialize_neural_input_v3(const NeuralInputV3& input) {
    if (input.history.size() > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::length_error("V3 history exceeds uint32 wire length");
    }

    const std::size_t history_bytes = input.history.size() * kNeuralV3HistoryEventSerializedBytes;
    if (history_bytes > std::numeric_limits<std::size_t>::max() - kNeuralV3FixedSerializedBytes) {
        throw std::length_error("V3 serialized size overflow");
    }

    std::vector<std::uint8_t> bytes;
    bytes.reserve(kNeuralV3FixedSerializedBytes + history_bytes);
    const char magic[8] = {'S', 'P', 'N', 'N', 'I', 'V', '3', '\0'};
    bytes.insert(bytes.end(), magic, magic + 8);
    bytes.insert(bytes.end(), input.categorical.begin(), input.categorical.end());
    bytes.insert(bytes.end(), input.rank_tokens.begin(), input.rank_tokens.end());
    bytes.insert(bytes.end(), input.same_suit.begin(), input.same_suit.end());
    for (float value : input.numeric) {
        append_float_le(bytes, value);
    }
    bytes.insert(bytes.end(), input.primitive_legal.begin(), input.primitive_legal.end());
    append_u32_le(bytes, static_cast<std::uint32_t>(input.history.size()));
    for (const auto& event : input.history) {
        bytes.insert(bytes.end(), event.categorical.begin(), event.categorical.end());
        for (float value : event.numeric) {
            append_float_le(bytes, value);
        }
    }

    const std::size_t expected = kNeuralV3FixedSerializedBytes + history_bytes;
    if (bytes.size() != expected) {
        throw std::logic_error("V3 serialized byte count mismatch");
    }
    return bytes;
}

}  // namespace spincore
