#pragma once
#include "spincore/ruleset_contract.hpp"
inline spincore::EpisodeScenario sc3(int dealer=0){spincore::EpisodeScenario s{};s.state.total_chips=1500;s.state.game_is_hu=false;s.state.small_blind=10;s.state.big_blind=20;s.state.stacks={500,500,500};s.dealer_id=dealer;return s;}
inline spincore::EpisodeScenario schu(int dealer=1){spincore::EpisodeScenario s{};s.state.total_chips=1500;s.state.game_is_hu=true;s.state.small_blind=10;s.state.big_blind=20;s.state.stacks={0,750,750};s.state.dead_players={0,-1,-1};s.state.dead_player_count=1;s.dealer_id=dealer;return s;}
