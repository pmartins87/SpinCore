#include "test_framework.hpp"
#include "spincore/scenario_sampler.hpp"
#include "spincore/ruleset_contract.hpp"
using namespace spincore;
SPIN_TEST(sampler_deterministic){ScenarioSampler a(123),b(123);for(int i=0;i<20;++i)REQUIRE(a.sample(i%2)==b.sample(i%2));}
SPIN_TEST(sampler_emits_valid_states){ScenarioSampler a(987);for(int i=0;i<100;++i)validate_episode_scenario(a.sample(i%3==0));}
