#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "spincore/hand_infoset_adapter.hpp"
using namespace spincore;
SPIN_TEST(infoset_exposes_only_actor_hole){HandEngine h(sc3(),42);auto i=build_current_actor_infoset(h,0);REQUIRE(i.hole[0].valid()&&i.hole[1].valid());REQUIRE(i.visible_board==0);}
SPIN_TEST(infoset_legal_mask_matches_engine){HandEngine h(sc3(),42);auto i=build_current_actor_infoset(h,0);auto l=h.betting().legal_actions(h.betting().actor());REQUIRE(i.legal_action_mask[0]==(uint8_t)l.fold);REQUIRE(i.legal_action_mask[1]==1);}
