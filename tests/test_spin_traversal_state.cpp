#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "spincore/spin_traversal_state.hpp"
using namespace spincore;
SPIN_TEST(traversal_clone_preserves_hidden_deal_and_parent){auto sc=sc3();SpinTraversalState root(sc,1234567);auto bh=root.hand().hole_cards();auto bb=root.hand().board();auto hist=root.hand().betting().history();auto legal=root.legal_abstract_actions();REQUIRE(!legal.empty());auto child=root.child(legal.front());REQUIRE(root.hand().hole_cards()==bh);REQUIRE(root.hand().board()==bb);REQUIRE(root.hand().betting().history()==hist);REQUIRE(child.hand().hole_cards()==bh);REQUIRE(child.hand().board()==bb);}
SPIN_TEST(traversal_infoset_never_exposes_opponent_private_cards){SpinTraversalState s(sc3(1),99);auto n=s.neural_input();int nz=0;for(auto x:n.card_tokens)if(x)++nz;REQUIRE(nz==2);}
SPIN_TEST(traversal_can_reach_terminal_zero_sum_chips){SpinTraversalState s(schu(),777);int g=0;while(!s.terminal()&&g++<64){auto acts=s.legal_abstract_actions();REQUIRE(!acts.empty());auto a=acts.front();for(auto x:acts)if(x==AbstractActionSlot::CheckCall){a=x;break;}s.apply(a);}REQUIRE(s.terminal());auto d=s.terminal_chip_delta();REQUIRE(d[0]+d[1]+d[2]==0);}
SPIN_TEST(traversal_icm_delta_available_only_terminal){SpinTraversalState s(schu(),777);REQUIRE_THROWS(s.terminal_icm_delta(PayoutProfile{{.5,.3,.2}}));}
