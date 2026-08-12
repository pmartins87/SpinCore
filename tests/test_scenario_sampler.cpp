#include "test_framework.hpp"
#include "spincore/scenario_sampler.hpp"
#include "spincore/ruleset_contract.hpp"
#include <stdexcept>
using namespace spincore;
SPIN_TEST(sampler_deterministic){ScenarioSampler a(123),b(123);for(int i=0;i<20;++i)REQUIRE(a.sample(i%2)==b.sample(i%2));}
SPIN_TEST(sampler_emits_valid_states){ScenarioSampler a(987);for(int i=0;i<100;++i)validate_episode_scenario(a.sample(i%3==0));}
SPIN_TEST(sampler_preserves_configured_blind_level_identity){
 ScenarioSamplerConfig cfg{};cfg.total_chips=2100;cfg.small_blind=25;cfg.big_blind=50;cfg.blind_index=4;
 ScenarioSampler a(20260812,cfg);
 for(int i=0;i<30;++i){auto s=a.sample(i%2==0);REQUIRE(s.state.total_chips==2100);REQUIRE(s.state.small_blind==25);REQUIRE(s.state.big_blind==50);REQUIRE(s.state.blind_index==4);}
}
SPIN_TEST(sampler_rejects_invalid_production_config){
 ScenarioSamplerConfig cfg{};cfg.blind_index=-1;bool threw=false;try{ScenarioSampler a(1,cfg);}catch(const std::invalid_argument&){threw=true;}REQUIRE(threw);
 cfg={};cfg.small_blind=30;cfg.big_blind=20;threw=false;try{ScenarioSampler a(1,cfg);}catch(const std::invalid_argument&){threw=true;}REQUIRE(threw);
}
