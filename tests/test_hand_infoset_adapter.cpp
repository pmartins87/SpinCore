#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "spincore/hand_infoset_adapter.hpp"
using namespace spincore;
SPIN_TEST(infoset_exposes_only_actor_hole){HandEngine h(sc3(),42);auto i=build_current_actor_infoset(h,0);REQUIRE(i.hole[0].valid()&&i.hole[1].valid());REQUIRE(i.visible_board==0);}
SPIN_TEST(infoset_legal_mask_matches_engine){HandEngine h(sc3(),42);auto i=build_current_actor_infoset(h,0);auto l=h.betting().legal_actions(h.betting().actor());REQUIRE(i.legal_action_mask[0]==(uint8_t)l.fold);REQUIRE(i.legal_action_mask[1]==1);}
SPIN_TEST(infoset_preserves_exact_public_events_without_changing_v1_tokens){HandEngine h(sc3(),42);auto i=build_current_actor_infoset(h,0);REQUIRE(i.public_history.size()==2);REQUIRE(i.public_events.size()==2);REQUIRE(i.public_events[0].forced);REQUIRE(i.public_events[1].forced);REQUIRE(i.public_events[0].pot_before==0);REQUIRE(i.public_events[0].pot_after==10);REQUIRE(i.public_events[1].pot_before==10);REQUIRE(i.public_events[1].pot_after==30);REQUIRE(i.public_events[0].actor_rel==1);REQUIRE(i.public_events[1].actor_rel==2);REQUIRE(i.public_history[0]==(uint8_t)(1+(int)ExactActionType::BetTo));}
