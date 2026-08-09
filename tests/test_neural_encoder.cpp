#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "spincore/hand_infoset_adapter.hpp"
#include "spincore/neural_encoder.hpp"
using namespace spincore;
SPIN_TEST(neural_preflop_has_two_cards_only){HandEngine h(sc3(),42);auto n=encode_neural_input_v1(build_current_actor_infoset(h,0));int nz=0;for(auto x:n.card_tokens)if(x)++nz;REQUIRE(nz==2);}
SPIN_TEST(neural_serialization_has_magic){HandEngine h(sc3(),42);auto b=serialize_neural_input_v1(encode_neural_input_v1(build_current_actor_infoset(h,0)));REQUIRE(b.size()>100);REQUIRE(b[0]=='S'&&b[1]=='P'&&b[2]=='N');}
