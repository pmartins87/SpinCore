#include "test_framework.hpp"
#include "spincore/flop_canonicalization.hpp"

using namespace spincore;

namespace {
Card fc(int rank, int suit) { return Card{static_cast<std::uint8_t>(rank), static_cast<std::uint8_t>(suit)}; }
}

SPIN_TEST(flop_canonicalization_matches_python_reference_example) {
    // Qs Jh 2h -> Python R7.5 exact reference key 2s Js Qh.
    const auto sig = canonical_flop_signature({fc(12,0), fc(11,1), fc(2,1)});
    REQUIRE(sig[0] == 2); REQUIRE(sig[1] == 0);
    REQUIRE(sig[2] == 11); REQUIRE(sig[3] == 0);
    REQUIRE(sig[4] == 12); REQUIRE(sig[5] == 1);
}

SPIN_TEST(flop_canonicalization_ignores_global_suit_renaming_and_card_order) {
    const auto a = canonical_flop_signature({fc(12,0), fc(11,1), fc(2,1)});
    const auto b = canonical_flop_signature({fc(2,3), fc(12,2), fc(11,3)});
    const auto c = canonical_flop_signature({fc(2,1), fc(11,1), fc(12,0)});
    REQUIRE(a == b);
    REQUIRE(a == c);
}

SPIN_TEST(flop_canonicalization_keeps_structurally_distinct_flops_distinct) {
    const auto a = canonical_flop_signature({fc(12,0), fc(11,1), fc(2,1)});
    const auto paired = canonical_flop_signature({fc(12,1), fc(11,1), fc(11,2)});
    REQUIRE(a != paired);
}
