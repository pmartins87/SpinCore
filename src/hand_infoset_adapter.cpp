#include "spincore/hand_infoset_adapter.hpp"
#include <stdexcept>
namespace spincore {
CanonicalInfoset build_current_actor_infoset(const HandEngine& h,std::int32_t blind_index){if(h.terminal())throw std::logic_error("terminal infoset");CanonicalInfoset o{};auto&b=h.betting();int actor=b.actor();if(actor<0)throw std::logic_error("no actor");o.hole=h.hole_cards()[(std::size_t)actor];o.board=h.board();o.visible_board=(std::uint8_t)h.visible_board_count();o.domain=strategy_domain(b.topology());o.street=b.street();o.live_count=(std::uint8_t)b.topology().live_count;o.pot=b.pot();auto legal=b.legal_actions(actor);o.to_call=legal.to_call;o.current_bet=b.current_bet();o.small_blind=h.scenario().state.small_blind;o.big_blind=h.scenario().state.big_blind;o.blind_index=blind_index;
 int rel=0;for(int k=0;k<3;++k){int seat=(actor+k)%3;auto&p=b.players()[(std::size_t)seat];o.stacks[(std::size_t)rel]=p.stack;o.street_commitments[(std::size_t)rel]=p.street_commitment;o.total_commitments[(std::size_t)rel]=p.total_commitment;o.statuses[(std::size_t)rel]=p.folded?1:(p.all_in?2:0);if(seat==b.topology().dealer)o.dealer_rel=(std::uint8_t)rel;++rel;}
 auto add=[&](int idx,bool yes){if(yes)o.legal_action_mask[(std::size_t)idx]=1;};add(0,legal.fold);add(1,legal.check||legal.call);add(2,b.street()==Street::Preflop&&(legal.bet||legal.raise));add(3,b.street()!=Street::Preflop&&(legal.bet||legal.raise));add(4,b.street()!=Street::Preflop&&(legal.bet||legal.raise));add(5,legal.all_in);
 for(auto&e:b.history()){
  // Frozen V1 token stream: intentionally unchanged.
  std::uint8_t tok=(std::uint8_t)(1+(int)e.action.type);tok=(std::uint8_t)(tok+8*(int)e.street);o.public_history.push_back(tok);
  int event_rel=(e.actor-actor+3)%3;
  o.public_events.push_back({(std::uint8_t)event_rel,e.street,e.action.type,e.paid,e.resulting_commitment,e.pot_before,e.pot_after,e.forced});
 }
 return o;}
}
