#include "test_framework.hpp"
#include "spincore/solver_c_api.h"
#include <vector>
SPIN_TEST(abi_reports_v2){REQUIRE(spincore_solver_c_abi_version()==2);}
SPIN_TEST(frontier_stops_at_target_or_terminal){spincore_solver_scenario_v2 s{1500,1,0,10,20,0,750,750,0,-1,1,1};auto*r=spincore_solver_state_create_v2(&s,123);REQUIRE(r);auto*f=spincore_solver_frontier_create_until_actor(r,2,10000,64);REQUIRE(f);REQUIRE(spincore_solver_frontier_size(f)>0);for(size_t i=0;i<spincore_solver_frontier_size(f);++i){auto*c=spincore_solver_frontier_clone_state(f,i);REQUIRE(c);REQUIRE(spincore_solver_frontier_is_terminal(f,i)==1||spincore_solver_state_actor(c)==2);spincore_solver_state_destroy(c);}spincore_solver_frontier_destroy(f);spincore_solver_state_destroy(r);}
SPIN_TEST(frontier_caps_fail_closed){spincore_solver_scenario_v2 s{1500,1,0,10,20,0,750,750,0,-1,1,1};auto*r=spincore_solver_state_create_v2(&s,123);auto*f=spincore_solver_frontier_create_until_actor(r,2,1,64);REQUIRE(f==nullptr);REQUIRE(spincore_solver_last_error()[0]);spincore_solver_state_destroy(r);}
