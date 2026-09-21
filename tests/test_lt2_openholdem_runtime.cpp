#include "test_framework.hpp"
#include "spincore/card.hpp"
#include "spincore/lt2_openholdem_runtime.hpp"

#include <array>

using namespace spincore;

SPIN_TEST(openholdem_card_mapping_all_52) {
    // OpenHoldem suit constants: hearts=0, diamonds=1, clubs=2, spades=3.
    // SpinCore Card suit order: spades=0, hearts=1, diamonds=2, clubs=3.
    constexpr std::array<int,4> expected_spin_suit{{1,2,3,0}};
    std::array<bool,52> seen{};
    for (int rank=2;rank<=14;++rank) {
        for (int oh_suit=0;oh_suit<4;++oh_suit) {
            const int id=lt2oh::openholdem_card_id_from_rank_suit(rank,oh_suit);
            REQUIRE(id>=0);
            REQUIRE(id<52);
            REQUIRE(!seen[static_cast<std::size_t>(id)]);
            seen[static_cast<std::size_t>(id)]=true;
            const auto card=card_from_id(static_cast<std::uint8_t>(id));
            REQUIRE(card.rank==rank);
            REQUIRE(card.suit==expected_spin_suit[static_cast<std::size_t>(oh_suit)]);
        }
    }
    for (bool value:seen) REQUIRE(value);
    REQUIRE(lt2oh::openholdem_card_id_from_rank_suit(14,3)==48); // As
    REQUIRE(lt2oh::openholdem_card_id_from_rank_suit(14,0)==49); // Ah
    REQUIRE(lt2oh::openholdem_card_id_from_rank_suit(14,1)==50); // Ad
    REQUIRE(lt2oh::openholdem_card_id_from_rank_suit(14,2)==51); // Ac
}

SPIN_TEST(openholdem_card_mapping_rejects_bad_values) {
    REQUIRE_THROWS(lt2oh::openholdem_card_id_from_rank_suit(1,0));
    REQUIRE_THROWS(lt2oh::openholdem_card_id_from_rank_suit(15,0));
    REQUIRE_THROWS(lt2oh::openholdem_card_id_from_rank_suit(14,-1));
    REQUIRE_THROWS(lt2oh::openholdem_card_id_from_rank_suit(14,4));
}

SPIN_TEST(openholdem_betround_is_card_derived) {
    REQUIRE(lt2oh::openholdem_betround_from_visible_count(0)==1);
    REQUIRE(lt2oh::openholdem_betround_from_visible_count(3)==2);
    REQUIRE(lt2oh::openholdem_betround_from_visible_count(4)==3);
    REQUIRE(lt2oh::openholdem_betround_from_visible_count(5)==4);
    REQUIRE_THROWS(lt2oh::openholdem_betround_from_visible_count(1));
    REQUIRE_THROWS(lt2oh::openholdem_betround_from_visible_count(2));
}
