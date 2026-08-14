#include "spincore/flop_canonicalization.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <utility>

namespace spincore {

std::array<std::uint8_t, 6> canonical_flop_signature(const std::array<Card, 3>& flop) {
    for (const auto& card : flop) {
        if (!card.valid()) {
            throw std::invalid_argument("invalid flop card");
        }
    }
    if (flop[0] == flop[1] || flop[0] == flop[2] || flop[1] == flop[2]) {
        throw std::invalid_argument("duplicate flop card");
    }

    std::array<int, 4> permutation{0, 1, 2, 3};
    std::array<std::uint8_t, 6> best{};
    bool have_best = false;
    do {
        std::array<std::pair<std::uint8_t, std::uint8_t>, 3> candidate_cards{};
        for (std::size_t index = 0; index < flop.size(); ++index) {
            candidate_cards[index] = {
                flop[index].rank,
                static_cast<std::uint8_t>(permutation[flop[index].suit]),
            };
        }
        std::sort(candidate_cards.begin(), candidate_cards.end());

        std::array<std::uint8_t, 6> candidate{};
        for (std::size_t index = 0; index < candidate_cards.size(); ++index) {
            candidate[index * 2U] = candidate_cards[index].first;
            candidate[index * 2U + 1U] = candidate_cards[index].second;
        }
        if (!have_best || candidate < best) {
            best = candidate;
            have_best = true;
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));

    return best;
}

}  // namespace spincore
