#pragma once
#include "spincore/card.hpp"
#include <array>
#include <compare>
#include <cstdint>
#include <span>
namespace spincore {
enum class HandCategory : std::uint8_t { HighCard=0, Pair=1, TwoPair=2, Trips=3, Straight=4, Flush=5, FullHouse=6, Quads=7, StraightFlush=8 };
struct HandRank {
    HandCategory category{HandCategory::HighCard};
    std::array<std::uint8_t,5> kickers{};
    friend auto operator<=>(const HandRank&, const HandRank&) = default;
    friend bool operator==(const HandRank&, const HandRank&) = default;
};
[[nodiscard]] HandRank evaluate_five(const std::array<Card,5>& cards);
[[nodiscard]] HandRank evaluate_best(std::span<const Card> cards); // 5..7 cards
}
