#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "spincore/ruleset_contract.hpp"
using namespace spincore;
SPIN_TEST(rules_accept_3h){validate_episode_scenario(sc3());}
SPIN_TEST(rules_accept_true_hu){validate_episode_scenario(schu());}
SPIN_TEST(rules_reject_chip_nonconservation){auto s=sc3();s.state.stacks[0]++;REQUIRE_THROWS(validate_episode_scenario(s));}
SPIN_TEST(rules_reject_hu_without_locked_third){auto s=schu();s.state.dead_player_count=0;REQUIRE_THROWS(validate_episode_scenario(s));}
SPIN_TEST(rules_reject_dead_dealer){auto s=schu();s.dealer_id=0;REQUIRE_THROWS(validate_episode_scenario(s));}
