#include "test_framework.hpp"
#include "spincore/action_abstraction.hpp"
using namespace spincore;
SPIN_TEST(action_slots_are_stable_0_to_5){REQUIRE((int)AbstractActionSlot::Fold==0);REQUIRE((int)AbstractActionSlot::AllIn==5);}
SPIN_TEST(action_names_nonempty){for(int i=0;i<6;++i)REQUIRE(action_name((AbstractActionSlot)i)[0]);}
