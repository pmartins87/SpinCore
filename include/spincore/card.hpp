#pragma once
#include <cstdint>
#include <string>

namespace spincore {
struct Card {
    std::uint8_t rank{0}; // 2..14
    std::uint8_t suit{0}; // 0..3
    friend bool operator==(const Card&, const Card&) = default;
    [[nodiscard]] bool valid() const noexcept { return rank >= 2 && rank <= 14 && suit < 4; }
    [[nodiscard]] std::uint8_t id() const;
    [[nodiscard]] std::string str() const;
};
[[nodiscard]] Card card_from_id(std::uint8_t id);
}
