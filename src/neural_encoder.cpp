#include "spincore/neural_encoder.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>

namespace spincore {

NeuralInputV1 encode_neural_input_v1(const CanonicalInfoset& i) {
    if (i.visible_board > i.board.size()) {
        throw std::invalid_argument("visible_board exceeds five community cards");
    }
    if (i.big_blind <= 0) {
        throw std::invalid_argument("big_blind must be positive for neural encoding");
    }

    NeuralInputV1 out{};
    out.card_tokens[0] = static_cast<std::uint8_t>(i.hole[0].id() + 1);
    out.card_tokens[1] = static_cast<std::uint8_t>(i.hole[1].id() + 1);

    const std::size_t visible = std::min<std::size_t>(i.visible_board, i.board.size());
    for (std::size_t j = 0; j < visible; ++j) {
        out.card_tokens[j + 2] = static_cast<std::uint8_t>(i.board[j].id() + 1);
    }

    const float bb = static_cast<float>(i.big_blind);
    out.numeric[0] = static_cast<float>(i.pot) / bb;
    out.numeric[1] = static_cast<float>(i.to_call) / bb;
    out.numeric[2] = static_cast<float>(i.current_bet) / bb;
    for (std::size_t j = 0; j < 3; ++j) {
        out.numeric[3 + j] = static_cast<float>(i.stacks[j]) / bb;
        out.numeric[6 + j] = static_cast<float>(i.street_commitments[j]) / bb;
        out.numeric[9 + j] = static_cast<float>(i.total_commitments[j]) / bb;
    }
    out.numeric[12] = static_cast<float>(i.small_blind) / bb;
    out.numeric[13] = 1.0F;
    out.numeric[14] = static_cast<float>(i.blind_index);
    out.numeric[15] = static_cast<float>(i.live_count);

    out.categorical[0] = static_cast<std::uint8_t>(i.domain);
    out.categorical[1] = static_cast<std::uint8_t>(i.street);
    out.categorical[2] = i.dealer_rel;
    out.categorical[3] = i.live_count;
    for (std::size_t j = 0; j < 3; ++j) {
        out.categorical[4 + j] = i.statuses[j];
    }
    out.categorical[7] = i.visible_board;

    out.legal_action_mask = i.legal_action_mask;
    out.history_len = static_cast<std::uint8_t>(
        std::min<std::size_t>(out.history_tokens.size(), i.public_history.size())
    );
    for (std::size_t j = 0; j < out.history_len; ++j) {
        out.history_tokens[j] = i.public_history[i.public_history.size() - out.history_len + j];
    }
    return out;
}

std::vector<std::uint8_t> serialize_neural_input_v1(const NeuralInputV1& i) {
    std::vector<std::uint8_t> bytes;
    const char magic[8] = {'S', 'P', 'N', 'N', 'I', 'V', '1', '\0'};
    bytes.insert(bytes.end(), magic, magic + 8);
    bytes.insert(bytes.end(), i.card_tokens.begin(), i.card_tokens.end());
    for (float value : i.numeric) {
        const std::uint32_t raw = std::bit_cast<std::uint32_t>(value);
        for (int k = 0; k < 4; ++k) {
            bytes.push_back(static_cast<std::uint8_t>((raw >> (8 * k)) & 0xffU));
        }
    }
    bytes.insert(bytes.end(), i.categorical.begin(), i.categorical.end());
    bytes.insert(bytes.end(), i.legal_action_mask.begin(), i.legal_action_mask.end());
    bytes.push_back(i.history_len);
    bytes.insert(bytes.end(), i.history_tokens.begin(), i.history_tokens.end());
    return bytes;
}

}  // namespace spincore
