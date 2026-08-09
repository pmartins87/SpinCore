#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "spincore/game_topology.hpp"
using namespace spincore;
SPIN_TEST(topology_3h_roles){auto t=make_game_topology(sc3(0));REQUIRE(t.domain==StrategyDomain::ThreeHanded);REQUIRE(t.small_blind_seat==1);REQUIRE(t.big_blind_seat==2);REQUIRE(t.first_preflop==0);REQUIRE(t.first_postflop==1);}
SPIN_TEST(topology_hu_button_is_sb){auto t=make_game_topology(schu(1));REQUIRE(t.domain==StrategyDomain::TrueHeadsUp);REQUIRE(t.small_blind_seat==1);REQUIRE(t.big_blind_seat==2);REQUIRE(t.first_preflop==1);REQUIRE(t.first_postflop==2);}
