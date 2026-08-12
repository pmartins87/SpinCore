#include "spincore/scenario_sampler.hpp"
#include "spincore/ruleset_contract.hpp"
#include <algorithm>
namespace spincore {
ScenarioSampler::ScenarioSampler(std::uint64_t seed,ScenarioSamplerConfig cfg):state_(seed),cfg_(cfg){
 if(cfg_.total_chips<2) throw std::invalid_argument("scenario sampler total_chips too small");
 if(cfg_.small_blind<=0||cfg_.big_blind<cfg_.small_blind) throw std::invalid_argument("scenario sampler invalid blinds");
 if(cfg_.blind_index<0) throw std::invalid_argument("scenario sampler negative blind_index");
}
std::uint64_t ScenarioSampler::next(){std::uint64_t z=(state_+=0x9E3779B97F4A7C15ULL);z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL;z=(z^(z>>27))*0x94D049BB133111EBULL;return z^(z>>31);}
EpisodeScenario ScenarioSampler::sample(bool hu){EpisodeScenario s{};s.state.total_chips=cfg_.total_chips;s.state.small_blind=cfg_.small_blind;s.state.big_blind=cfg_.big_blind;s.state.blind_index=cfg_.blind_index;
 if(hu){int dead=(int)(next()%3);int a=(dead+1)%3,b=(dead+2)%3;int cut=1+(int)(next()%(std::uint64_t)(cfg_.total_chips-1));s.state.stacks={0,0,0};s.state.stacks[(std::size_t)a]=cut;s.state.stacks[(std::size_t)b]=cfg_.total_chips-cut;s.state.game_is_hu=true;s.state.dead_players={dead,-1,-1};s.state.dead_player_count=1;s.dealer_id=(next()&1)?a:b;}
 else {if(cfg_.total_chips<3) throw std::invalid_argument("three-handed scenario needs at least three chips");int x=1+(int)(next()%(std::uint64_t)(cfg_.total_chips-2));int rem=cfg_.total_chips-x;int y=1+(int)(next()%(std::uint64_t)(rem-1));int z=rem-y;s.state.stacks={x,y,z};s.state.game_is_hu=false;s.state.dead_players={-1,-1,-1};s.state.dead_player_count=0;s.dealer_id=(int)(next()%3);}validate_episode_scenario(s);return s;}
}
