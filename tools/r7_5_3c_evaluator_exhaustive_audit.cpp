#include "spincore/card.hpp"
#include "spincore/hand_evaluator.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>

using namespace spincore;

int main() {
    // Exact combinatorial census of all C(52,5) poker hands. Counts exclude
    // higher categories exactly as HandCategory does (e.g. ordinary Flush does
    // not include straight flushes).
    constexpr std::array<std::uint64_t, 9> expected{
        1302540ULL, // HighCard
        1098240ULL, // Pair
        123552ULL,  // TwoPair
        54912ULL,   // Trips
        10200ULL,   // Straight
        5108ULL,    // Flush
        3744ULL,    // FullHouse
        624ULL,     // Quads
        40ULL,      // StraightFlush
    };
    constexpr std::uint64_t expected_total = 2598960ULL;
    constexpr std::size_t expected_distinct_ranks = 7462U;

    std::array<Card, 52> deck{};
    for (std::uint8_t id = 0; id < 52; ++id) {
        deck[id] = card_from_id(id);
    }

    std::array<std::uint64_t, 9> counts{};
    std::set<HandRank> distinct;
    std::array<Card, 5> hand{};
    std::uint64_t total = 0;

    for (int a = 0; a < 48; ++a) {
        hand[0] = deck[static_cast<std::size_t>(a)];
        for (int b = a + 1; b < 49; ++b) {
            hand[1] = deck[static_cast<std::size_t>(b)];
            for (int c = b + 1; c < 50; ++c) {
                hand[2] = deck[static_cast<std::size_t>(c)];
                for (int d = c + 1; d < 51; ++d) {
                    hand[3] = deck[static_cast<std::size_t>(d)];
                    for (int e = d + 1; e < 52; ++e) {
                        hand[4] = deck[static_cast<std::size_t>(e)];
                        const auto rank = evaluate_five(hand);
                        const auto category = static_cast<std::size_t>(rank.category);
                        if (category >= counts.size()) {
                            throw std::runtime_error("evaluator emitted unknown category");
                        }
                        ++counts[category];
                        ++total;
                        distinct.insert(rank);
                    }
                }
            }
        }
    }

    std::cout << "total=" << total << " distinct_ranks=" << distinct.size() << '\n';
    for (std::size_t category = 0; category < counts.size(); ++category) {
        std::cout << "category[" << category << "]=" << counts[category]
                  << " expected=" << expected[category] << '\n';
    }

    if (total != expected_total) {
        std::cerr << "FAIL total five-card combinations\n";
        return 1;
    }
    if (counts != expected) {
        std::cerr << "FAIL evaluator category census\n";
        return 2;
    }
    if (distinct.size() != expected_distinct_ranks) {
        std::cerr << "FAIL distinct five-card HandRank census\n";
        return 3;
    }
    std::cout << "R7.5.3C exhaustive evaluator census PASS\n";
    return 0;
}
