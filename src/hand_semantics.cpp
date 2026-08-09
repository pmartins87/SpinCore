#include "spincore/hand_semantics.hpp"
#include <algorithm>
#include <stdexcept>
namespace spincore {
std::array<Card,2> canonical_hole(std::array<Card,2> h) {
    if(!h[0].valid()||!h[1].valid()||h[0]==h[1]) throw std::invalid_argument("bad hole cards");
    if (h[1].rank > h[0].rank || (h[1].rank==h[0].rank && h[1].suit > h[0].suit)) std::swap(h[0],h[1]);
    return h;
}
std::string hand_class(std::array<Card,2> h) {
    h=canonical_hole(h); static constexpr char ranks[]="23456789TJQKA";
    std::string s; s+=ranks[h[0].rank-2]; s+=ranks[h[1].rank-2];
    if(h[0].rank!=h[1].rank) s += (h[0].suit==h[1].suit ? 's':'o');
    return s;
}
}
