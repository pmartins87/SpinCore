#pragma once

#include "spincore/hand_engine.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace spincore {

inline constexpr std::size_t kNeuralV2NumericCount = 24;
inline constexpr std::size_t kNeuralV2CategoricalCount = 72;
inline constexpr std::size_t kNeuralV2HistoryCapacity = 32;

struct NeuralHistoryEventV2 {
    // actor_rel, street, ExactActionType, forced
    std::array<std::uint8_t, 4> categorical{};
    // paid_bb, resulting_commitment_bb, pot_before_bb, pot_after_bb
    std::array<float, 4> numeric{};
};

// R7.5 semantic candidate wire. This is deliberately separate from
// NeuralInputV1/SPNNIV1 so all R7.3/R7.4 evidence remains frozen.
//
// Flop abstraction is NOT selected here. canonical_flop_signature carries the
// exact 1,755-class suit-isomorphic language; H1/H2/H3/H4 are applied at the
// neural-observation/model boundary during the frozen R7.5.3 ablation.
struct NeuralInputV2 {
    std::uint8_t preflop_class_id{0}; // 0..168
    std::array<std::uint8_t, 6> canonical_flop_signature{};
    std::array<float, kNeuralV2NumericCount> numeric{};
    std::array<std::uint8_t, kNeuralV2CategoricalCount> categorical{};
    std::array<std::uint8_t, 6> legal_action_mask{};
    std::array<NeuralHistoryEventV2, kNeuralV2HistoryCapacity> history{};
    std::uint8_t history_len{0};
};

[[nodiscard]] std::uint8_t preflop_class_id_v2(const std::array<Card, 2>& hole);
[[nodiscard]] NeuralInputV2 encode_neural_input_v2(const HandEngine& hand, std::int32_t blind_index);
[[nodiscard]] std::vector<std::uint8_t> serialize_neural_input_v2(const NeuralInputV2& input);

}  // namespace spincore
