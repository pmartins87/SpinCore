#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "spincore/hand_engine.hpp"
#include <numeric>
using namespace spincore;
SPIN_TEST(hand_deal_is_seed_deterministic){HandEngine a(sc3(),123),b(sc3(),123);REQUIRE(a.hole_cards()==b.hole_cards());REQUIRE(a.board()==b.board());}
SPIN_TEST(hand_different_seed_changes_deal){HandEngine a(sc3(),123),b(sc3(),124);REQUIRE(!(a.hole_cards()==b.hole_cards()&&a.board()==b.board()));}
SPIN_TEST(hand_fold_settlement_conserves_chips){auto s=schu(1);HandEngine h(s,7);h.apply(1,{ExactActionType::Fold,0});REQUIRE(h.terminal());auto x=h.settle();REQUIRE(std::accumulate(x.final_stacks.begin(),x.final_stacks.end(),0)==1500);REQUIRE(x.final_stacks[2]>s.state.stacks[2]-20);}
SPIN_TEST(hand_checkdown_reaches_terminal){auto s=schu(1);HandEngine h(s,9);int g=0;while(!h.terminal()&&g++<20){int a=h.betting().actor();auto l=h.betting().legal_actions(a);if(l.check)h.apply(a,{ExactActionType::Check,0});else if(l.call)h.apply(a,{ExactActionType::Call,0});else h.apply(a,{ExactActionType::Fold,0});}REQUIRE(h.terminal());REQUIRE(h.visible_board_count()==5);}
