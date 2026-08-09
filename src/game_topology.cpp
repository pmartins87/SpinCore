#include "spincore/game_topology.hpp"
#include <stdexcept>
namespace spincore {
int next_live_clockwise(const GameTopology& t,int seat){for(int k=1;k<=3;++k){int q=(seat+k)%3;for(int i=0;i<t.live_count;++i)if(t.live[(std::size_t)i]==q)return q;}return -1;}
GameTopology make_game_topology(const EpisodeScenario& s){validate_episode_scenario(s);GameTopology t{};t.domain=s.state.game_is_hu?StrategyDomain::TrueHeadsUp:StrategyDomain::ThreeHanded;t.dealer=s.dealer_id;t.live_count=0;for(int i=0;i<3;++i)if(s.state.stacks[(std::size_t)i]>0)t.live[(std::size_t)t.live_count++]=i;
 if(t.domain==StrategyDomain::TrueHeadsUp){t.small_blind_seat=t.dealer;t.big_blind_seat=next_live_clockwise(t,t.dealer);t.first_preflop=t.dealer;t.first_postflop=t.big_blind_seat;}
 else {t.small_blind_seat=next_live_clockwise(t,t.dealer);t.big_blind_seat=next_live_clockwise(t,t.small_blind_seat);t.first_preflop=t.dealer;t.first_postflop=t.small_blind_seat;} return t;}
StrategyDomain strategy_domain(const GameTopology& t) noexcept{return t.domain;}
}
