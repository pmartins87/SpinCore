#include "test_framework.hpp"
#include "spincore/card.hpp"
using namespace spincore;
SPIN_TEST(card_roundtrip_all_52){for(int i=0;i<52;++i){auto c=card_from_id((uint8_t)i);REQUIRE(c.id()==i);REQUIRE(c.str().size()==2);}}
SPIN_TEST(card_rejects_bad_id){REQUIRE_THROWS(card_from_id(52));}
