#pragma once
#include "spincore/canonical_infoset.hpp"
#include <array>
#include <cstdint>
#include <vector>
namespace spincore {
struct NeuralInputV1 { std::array<std::uint8_t,7> card_tokens{}; std::array<float,16> numeric{}; std::array<std::uint8_t,8> categorical{}; std::array<std::uint8_t,6> legal_action_mask{}; std::array<std::uint8_t,32> history_tokens{}; std::uint8_t history_len{0}; };
[[nodiscard]] NeuralInputV1 encode_neural_input_v1(const CanonicalInfoset& i);
[[nodiscard]] std::vector<std::uint8_t> serialize_neural_input_v1(const NeuralInputV1& i);
}
