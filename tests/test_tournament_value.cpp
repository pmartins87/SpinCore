#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "spincore/tournament_value.hpp"
#include <cmath>
using namespace spincore;
SPIN_TEST(icm_wta_equals_chip_fraction_3h){auto s=sc3().state;PayoutProfile p{{1,0,0}};auto v=icm_values(s,p);for(int i=0;i<3;++i)REQUIRE(std::abs(v[i]-1.0/3.0)<1e-12);}
SPIN_TEST(icm_hu_preserves_locked_third){auto s=schu().state;PayoutProfile p{{0.5,0.3,0.2}};auto v=icm_values(s,p);REQUIRE(std::abs(v[0]-0.2)<1e-12);REQUIRE(std::abs(v[1]-0.4)<1e-12);REQUIRE(std::abs(v[2]-0.4)<1e-12);}
SPIN_TEST(icm_values_sum_payouts){auto s=sc3().state;PayoutProfile p{{0.6,0.3,0.1}};auto v=icm_values(s,p);REQUIRE(std::abs(v[0]+v[1]+v[2]-1.0)<1e-12);}
SPIN_TEST(continuation_delta_sums_zero){auto s=sc3().state;PayoutProfile p{{0.6,0.3,0.1}};auto d=terminal_continuation_delta(s,{1000,500,0},p);REQUIRE(std::abs(d[0]+d[1]+d[2])<1e-12);}
SPIN_TEST(simultaneous_equal_stack_elimination_without_dealer_context_still_fails_closed){auto s=sc3().state;PayoutProfile p{{0.6,0.3,0.1}};REQUIRE_THROWS(terminal_continuation_delta(s,{1500,0,0},p));}
SPIN_TEST(simultaneous_equal_stack_elimination_dealer0_orders_left_of_button_ahead){auto s=sc3().state;PayoutProfile p{{0.6,0.3,0.1}};auto d=terminal_continuation_delta(s,{1500,0,0},p,0);REQUIRE(std::abs(d[0]-(0.6-1.0/3.0))<1e-12);REQUIRE(std::abs(d[1]-(0.3-1.0/3.0))<1e-12);REQUIRE(std::abs(d[2]-(0.1-1.0/3.0))<1e-12);REQUIRE(std::abs(d[0]+d[1]+d[2])<1e-12);}
SPIN_TEST(simultaneous_equal_stack_elimination_dealer1_puts_seat2_ahead_of_button){auto s=sc3().state;PayoutProfile p{{0.6,0.3,0.1}};auto d=terminal_continuation_delta(s,{1500,0,0},p,1);REQUIRE(std::abs(d[1]-(0.1-1.0/3.0))<1e-12);REQUIRE(std::abs(d[2]-(0.3-1.0/3.0))<1e-12);REQUIRE(std::abs(d[0]+d[1]+d[2])<1e-12);}
SPIN_TEST(simultaneous_equal_stack_elimination_dealer2_orders_seat1_ahead_of_button){auto s=sc3().state;PayoutProfile p{{0.6,0.3,0.1}};auto d=terminal_continuation_delta(s,{1500,0,0},p,2);REQUIRE(std::abs(d[1]-(0.3-1.0/3.0))<1e-12);REQUIRE(std::abs(d[2]-(0.1-1.0/3.0))<1e-12);REQUIRE(std::abs(d[0]+d[1]+d[2])<1e-12);}
SPIN_TEST(dealer_aware_terminal_continuation_rejects_bad_dealer){auto s=sc3().state;PayoutProfile p{{0.6,0.3,0.1}};REQUIRE_THROWS(terminal_continuation_delta(s,{1500,0,0},p,3));}
SPIN_TEST(simultaneous_unequal_stack_elimination_is_ordered){auto s=sc3().state;s.stacks={800,400,300};PayoutProfile p{{0.6,0.3,0.1}};auto d=terminal_continuation_delta(s,{1500,0,0},p);REQUIRE(std::abs(d[0]+d[1]+d[2])<1e-12);}
