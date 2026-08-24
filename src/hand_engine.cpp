#include "spincore/hand_engine.hpp"
#include "spincore/hand_evaluator.hpp"
#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
namespace spincore {
HandEngine::HandEngine(const EpisodeScenario&s,std::uint64_t seed):scenario_(s),betting_(s,make_game_topology(s)){std::array<int,52>d{};std::iota(d.begin(),d.end(),0);std::mt19937_64 rng(seed);std::shuffle(d.begin(),d.end(),rng);int k=0;auto topo=betting_.topology();for(int r=0;r<2;++r)for(int j=0;j<topo.live_count;++j){int seat=topo.live[(std::size_t)j];hole_[(std::size_t)seat][(std::size_t)r]=card_from_id((std::uint8_t)d[(std::size_t)k++]);has_hole_[(std::size_t)seat]=true;}for(int i=0;i<5;++i)board_[(std::size_t)i]=card_from_id((std::uint8_t)d[(std::size_t)k++]);maybe_advance();}
HandEngine::HandEngine(const EpisodeScenario&s,const std::array<std::array<Card,2>,3>&holes,const std::array<Card,5>&board):scenario_(s),betting_(s,make_game_topology(s)),hole_(holes),board_(board){
 std::array<bool,52>seen{};auto topo=betting_.topology();
 auto add=[&](const Card&c){if(!c.valid())throw std::invalid_argument("explicit deal contains invalid card");auto id=(std::size_t)c.id();if(id>=seen.size()||seen[id])throw std::invalid_argument("explicit deal contains duplicate card");seen[id]=true;};
 for(int seat=0;seat<3;++seat){bool live=false;for(int j=0;j<topo.live_count;++j)if(topo.live[(std::size_t)j]==seat){live=true;break;}if(live){add(hole_[(std::size_t)seat][0]);add(hole_[(std::size_t)seat][1]);has_hole_[(std::size_t)seat]=true;}else{if(hole_[(std::size_t)seat][0].valid()||hole_[(std::size_t)seat][1].valid())throw std::invalid_argument("dead seat must not receive explicit hole cards");has_hole_[(std::size_t)seat]=false;}}
 for(const auto&c:board_)add(c);maybe_advance();}
ActionEvent HandEngine::apply(int seat,ExactAction a){if(terminal_)throw std::logic_error("terminal hand");auto ev=betting_.apply(seat,a);maybe_advance();return ev;}
void HandEngine::maybe_advance(){while(!terminal_&&betting_.street_complete()){if(betting_.hand_over_by_fold()){terminal_=true;return;}if(betting_.street()==Street::River){visible_board_count_=5;terminal_=true;return;}if(betting_.actionable_count()<=1){visible_board_count_=5;terminal_=true;return;}betting_.advance_street();switch(betting_.street()){case Street::Flop:visible_board_count_=3;break;case Street::Turn:visible_board_count_=4;break;case Street::River:visible_board_count_=5;break;default:break;}}
}
HandSettlement HandEngine::settle() const{if(!terminal_)throw std::logic_error("settle before terminal");HandSettlement out{};auto ps=betting_.players();for(int i=0;i<3;++i)out.final_stacks[(std::size_t)i]=ps[(std::size_t)i].stack;
 std::vector<int> levels;for(auto&p:ps)if(p.total_commitment>0)levels.push_back(p.total_commitment);std::sort(levels.begin(),levels.end());levels.erase(std::unique(levels.begin(),levels.end()),levels.end());int prev=0;
 for(int level:levels){std::vector<int> contributors,eligible;for(int i=0;i<3;++i)if(ps[(std::size_t)i].total_commitment>=level){contributors.push_back(i);if(!ps[(std::size_t)i].folded)eligible.push_back(i);}int pot=(level-prev)*(int)contributors.size();prev=level;if(pot<=0)continue;if(eligible.empty())throw std::logic_error("side pot without eligible player");std::vector<int>winners;
  if(eligible.size()==1)winners=eligible;else{HandRank best{};bool first=true;for(int seat:eligible){std::array<Card,7> c{};c[0]=hole_[(std::size_t)seat][0];c[1]=hole_[(std::size_t)seat][1];for(int j=0;j<5;++j)c[(std::size_t)(j+2)]=board_[(std::size_t)j];auto r=evaluate_best(c);if(first||r>best){best=r;winners={seat};first=false;}else if(r==best)winners.push_back(seat);}}
  int share=pot/(int)winners.size(),rem=pot%(int)winners.size();std::sort(winners.begin(),winners.end());for(int seat:winners){out.final_stacks[(std::size_t)seat]+=share+(rem-->0?1:0);} }
 int sum=std::accumulate(out.final_stacks.begin(),out.final_stacks.end(),0);if(sum!=scenario_.state.total_chips)throw std::logic_error("settlement chip conservation failure");return out;}
}
