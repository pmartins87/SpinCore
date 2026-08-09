#include "spincore/ruleset_contract.hpp"
#include <numeric>
#include <set>
#include <stdexcept>
namespace spincore {
void validate_episode_scenario(const EpisodeScenario& s) {
    const auto& x=s.state;
    if(x.total_chips<=0||x.small_blind<=0||x.big_blind<=0||x.small_blind>=x.big_blind) throw std::invalid_argument("invalid blind/total contract");
    if(x.blind_index<0||s.dealer_id<0||s.dealer_id>2) throw std::invalid_argument("invalid scenario index/dealer");
    int sum=0,live=0; for(auto st:x.stacks){if(st<0) throw std::invalid_argument("negative stack");sum+=st;if(st>0)++live;}
    if(sum!=x.total_chips) throw std::invalid_argument("stacks do not conserve total chips");
    if(x.dead_player_count<0||x.dead_player_count>2) throw std::invalid_argument("dead_player_count");
    std::set<int> dead;
    for(int i=0;i<x.dead_player_count;++i){int p=x.dead_players[(std::size_t)i];if(p<0||p>2||!dead.insert(p).second||x.stacks[(std::size_t)p]!=0)throw std::invalid_argument("invalid dead-player history");}
    if(x.game_is_hu){if(live!=2||x.dead_player_count!=1) throw std::invalid_argument("true HU requires two live and one locked elimination");}
    else {if(live!=3||x.dead_player_count!=0) throw std::invalid_argument("3H hand requires three live players");}
    if(x.stacks[(std::size_t)s.dealer_id]<=0) throw std::invalid_argument("dealer must be live");
}
}
