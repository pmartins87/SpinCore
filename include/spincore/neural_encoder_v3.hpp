#pragma once

#include "spincore/hand_engine.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace spincore {

// SPNNIV3 is the production-candidate observation carrier for R7.5.3C.
// It deliberately transports a lossless observable poker state (up to true
// game symmetries) rather than baking subjective poker buckets into the wire.
// Semantic H3 features are derived from this exact carrier by a separately
// tested layer so an ontology bug cannot erase the source state.
inline constexpr std::size_t kNeuralV3CardSlotCount = 7;
inline constexpr std::size_t kNeuralV3SuitRelationCount = 21; // C(7,2)
inline constexpr std::size_t kNeuralV3NumericCount = 16;
inline constexpr std::size_t kNeuralV3CategoricalCount = 10;
inline constexpr std::size_t kNeuralV3PrimitiveLegalCount = 6;
inline constexpr std::size_t kNeuralV3FixedSerializedBytes = 120;
inline constexpr std::size_t kNeuralV3HistoryEventSerializedBytes = 20;

struct NeuralHistoryEventV3 {
    // actor_rel, street, ExactActionType, forced
    std::array<std::uint8_t, 4> categorical{};
    // paid_bb, resulting_commitment_bb, pot_before_bb, pot_after_bb
    std::array<float, 4> numeric{};
    friend bool operator==(const NeuralHistoryEventV3&, const NeuralHistoryEventV3&) = default;
};

struct NeuralInputV3 {
    // Fixed role slots:
    // [hero_hole_0, hero_hole_1, flop_0, flop_1, flop_2, turn, river].
    // Rank is 2..14 for visible cards, 0 for unrevealed public slots.
    std::array<std::uint8_t, kNeuralV3CardSlotCount> rank_tokens{};

    // Upper-triangle pairwise same-suit relation in lexicographic (i,j) order,
    // i<j. 1 iff both slots are visible and share a suit; otherwise 0.
    // Physical suit labels are never serialized.
    std::array<std::uint8_t, kNeuralV3SuitRelationCount> same_suit{};

    // domain, street, dealer_rel, small_blind_rel, big_blind_rel, live_count,
    // visible_board, status_rel0, status_rel1, status_rel2.
    // In true HU relative seats are canonicalized to [Hero, live opponent,
    // absent], so the physical identity of the eliminated 3-max chair is gone.
    std::array<std::uint8_t, kNeuralV3CategoricalCount> categorical{};

    // BB-normalized exact current geometry:
    //  0 pot
    //  1 to_call
    //  2 current_bet
    //  3..5 stacks rel0..2
    //  6..8 street commitments rel0..2
    //  9..11 total commitments rel0..2
    // 12 small blind
    // 13 blind_index (raw scalar, retained for frozen profile context)
    // 14 min_raise_to
    // 15 max_raise_to
    std::array<float, kNeuralV3NumericCount> numeric{};

    // Primitive exact legality, independent of any chosen sizing abstraction:
    // fold, check, call, bet, raise, all-in.
    std::array<std::uint8_t, kNeuralV3PrimitiveLegalCount> primitive_legal{};

    // Complete public history. There is intentionally no strategic last-N cap.
    std::vector<NeuralHistoryEventV3> history;
};

[[nodiscard]] NeuralInputV3 encode_neural_input_v3(
    const HandEngine& hand,
    std::int32_t blind_index
);

[[nodiscard]] std::vector<std::uint8_t> serialize_neural_input_v3(
    const NeuralInputV3& input
);

}  // namespace spincore
