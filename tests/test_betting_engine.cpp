#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "spincore/betting_engine.hpp"
using namespace spincore;
SPIN_TEST(betting_posts_3h_blinds){auto s=sc3(0);BettingEngine b(s,make_game_topology(s));REQUIRE(b.pot()==30);REQUIRE(b.actor()==0);REQUIRE(b.players()[1].street_commitment==10);REQUIRE(b.players()[2].street_commitment==20);}
SPIN_TEST(betting_hu_button_acts_first){auto s=schu(1);BettingEngine b(s,make_game_topology(s));REQUIRE(b.actor()==1);REQUIRE(b.legal_actions(1).to_call==10);}
SPIN_TEST(betting_fold_ends_hu){auto s=schu(1);BettingEngine b(s,make_game_topology(s));b.apply(1,{ExactActionType::Fold,0});REQUIRE(b.hand_over_by_fold());REQUIRE(b.nonfolded_count()==1);}
SPIN_TEST(betting_call_check_closes_preflop){auto s=schu(1);BettingEngine b(s,make_game_topology(s));b.apply(1,{ExactActionType::Call,0});REQUIRE(b.actor()==2);b.apply(2,{ExactActionType::Check,0});REQUIRE(b.street_complete());}
SPIN_TEST(betting_full_raise_reopens){auto s=schu(1);BettingEngine b(s,make_game_topology(s));auto l=b.legal_actions(1);b.apply(1,{ExactActionType::RaiseTo,l.min_raise_to});REQUIRE(b.actor()==2);REQUIRE(b.legal_actions(2).raise);}
SPIN_TEST(betting_allin_is_legal){auto s=schu(1);BettingEngine b(s,make_game_topology(s));b.apply(1,{ExactActionType::AllIn,0});REQUIRE(b.players()[1].all_in);REQUIRE(b.players()[1].stack==0);}
