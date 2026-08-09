#include "spincore/card.hpp"
#include <stdexcept>
namespace spincore {
std::uint8_t Card::id() const {
    if (!valid()) throw std::invalid_argument("invalid card");
    return static_cast<std::uint8_t>((rank - 2u) * 4u + suit);
}
std::string Card::str() const {
    if (!valid()) return "??";
    static constexpr char ranks[] = "23456789TJQKA";
    static constexpr char suits[] = "shdc";
    std::string out; out += ranks[rank-2]; out += suits[suit]; return out;
}
Card card_from_id(std::uint8_t id) {
    if (id >= 52) throw std::out_of_range("card id");
    return Card{static_cast<std::uint8_t>(2 + id/4), static_cast<std::uint8_t>(id%4)};
}
}
