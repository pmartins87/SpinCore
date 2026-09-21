#include "spincore/lt2_openholdem_runtime.hpp"

#include "spincore/card.hpp"
#include "spincore/game_topology.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace spincore::lt2oh {
namespace {

constexpr std::array<std::array<std::int32_t,2>,9> kBlindLevels{{
    {{10,20}},{{15,30}},{{20,40}},{{30,60}},{{40,80}},
    {{50,100}},{{60,120}},{{80,160}},{{100,200}}
}};

std::set<int> bits(std::uint32_t value) {
    if ((value & ~0x3ffU) != 0U) {
        throw std::runtime_error("invalid 10-chair bitmask");
    }
    std::set<int> out;
    for (int i=0;i<10;++i) if ((value & (1U<<static_cast<unsigned>(i)))!=0U) out.insert(i);
    return out;
}

int validate_chair(int chair, const char* name) {
    if (chair<0 || chair>9) throw std::runtime_error(std::string(name)+" outside 0..9");
    return chair;
}

int blind_index(int sb,int bb) {
    for (std::size_t i=0;i<kBlindLevels.size();++i) {
        if (kBlindLevels[i][0]==sb && kBlindLevels[i][1]==bb) return static_cast<int>(i);
    }
    throw std::runtime_error("unsupported blind level");
}

std::vector<int> clockwise_from(int dealer,const std::set<int>& seats) {
    std::vector<int> out;
    for (int step=0;step<10;++step) {
        int chair=(dealer+step)%10;
        if (seats.count(chair)!=0) out.push_back(chair);
    }
    return out;
}

void validate_cards(
    const std::array<std::int32_t,2>& hero,
    const std::array<std::int32_t,5>& board,
    int visible) {
    std::array<bool,52> seen{};
    for (int c:hero) {
        if (c<0 || c>=52) throw std::runtime_error("invalid Hero card id");
        if (seen[static_cast<std::size_t>(c)]) throw std::runtime_error("duplicate Hero card");
        seen[static_cast<std::size_t>(c)]=true;
    }
    for (int i=0;i<5;++i) {
        int c=board[static_cast<std::size_t>(i)];
        if (i<visible) {
            if (c<0 || c>=52) throw std::runtime_error("visible board card missing/invalid");
            if (seen[static_cast<std::size_t>(c)]) throw std::runtime_error("duplicate known card");
            seen[static_cast<std::size_t>(c)]=true;
        } else if (c!=-1) {
            throw std::runtime_error("unrevealed board slot must be -1");
        }
    }
}

std::pair<std::array<std::array<Card,2>,3>,std::array<Card,5>> runtime_deal(
    const HandAnchor& anchor,
    const ObservedSnapshot& observed) {
    std::array<std::array<Card,2>,3> holes{};
    std::array<Card,5> board{};
    std::array<bool,52> used{};

    const int hero=anchor.hero_logical_seat;
    for (int r=0;r<2;++r) {
        const int id=anchor.hero_cards[static_cast<std::size_t>(r)];
        if (id<0 || id>=52 || used[static_cast<std::size_t>(id)]) {
            throw std::runtime_error("invalid Hero card while rebuilding");
        }
        used[static_cast<std::size_t>(id)]=true;
        holes[static_cast<std::size_t>(hero)][static_cast<std::size_t>(r)] =
            card_from_id(static_cast<std::uint8_t>(id));
    }

    for (int i=0;i<observed.visible_board_count;++i) {
        const int id=observed.board_cards[static_cast<std::size_t>(i)];
        if (id<0 || id>=52 || used[static_cast<std::size_t>(id)]) {
            throw std::runtime_error("invalid visible board while rebuilding");
        }
        used[static_cast<std::size_t>(id)]=true;
        board[static_cast<std::size_t>(i)]=card_from_id(static_cast<std::uint8_t>(id));
    }

    std::vector<int> remaining;
    for (int c=0;c<52;++c) if (!used[static_cast<std::size_t>(c)]) remaining.push_back(c);
    std::size_t cursor=0;

    const auto topology=make_game_topology(anchor.scenario);
    std::array<bool,3> live{};
    for (int i=0;i<topology.live_count;++i) {
        live[static_cast<std::size_t>(topology.live[static_cast<std::size_t>(i)])]=true;
    }
    for (int seat=0;seat<3;++seat) {
        if (!live[static_cast<std::size_t>(seat)] || seat==hero) continue;
        for (int r=0;r<2;++r) {
            const int id=remaining.at(cursor++);
            holes[static_cast<std::size_t>(seat)][static_cast<std::size_t>(r)] =
                card_from_id(static_cast<std::uint8_t>(id));
        }
    }
    for (int i=observed.visible_board_count;i<5;++i) {
        const int id=remaining.at(cursor++);
        board[static_cast<std::size_t>(i)]=card_from_id(static_cast<std::uint8_t>(id));
    }
    return {holes,board};
}

bool exact_equal(const ExactAction& a,const ExactAction& b) {
    return a.type==b.type && a.amount_to==b.amount_to;
}

} // namespace

std::int32_t openholdem_betround_from_visible_count(std::int32_t visible) {
    switch (visible) {
        case 0:return 1;
        case 3:return 2;
        case 4:return 3;
        case 5:return 4;
        default:throw std::runtime_error("unsupported visible community-card count");
    }
}

HandAnchor anchor_from_raw_frame(const RawFrame& frame) {
    if (frame.hand_id.empty()) throw std::runtime_error("empty hand id");
    const int user=validate_chair(frame.user_chair,"userchair");
    const int dealer=validate_chair(frame.dealer_chair,"dealerchair");
    if (frame.betround!=1 || frame.common_cards_known!=0) {
        throw std::runtime_error("hand anchor must be preflop with zero common cards");
    }
    if (frame.small_blind<=0 || frame.big_blind<frame.small_blind) {
        throw std::runtime_error("invalid blind amounts");
    }

    const auto dealt=bits(frame.players_dealt_bits);
    const auto playing=bits(frame.players_playing_bits);
    const auto allin=bits(frame.players_allin_bits);
    if (dealt.size()!=2 && dealt.size()!=3) throw std::runtime_error("runtime supports 2 or 3 dealt players");
    if (dealt.count(user)==0 || dealt.count(dealer)==0) throw std::runtime_error("user/dealer not dealt");
    if (!std::includes(dealt.begin(),dealt.end(),playing.begin(),playing.end())) {
        throw std::runtime_error("playing contains undealt chair");
    }
    if (!std::includes(dealt.begin(),dealt.end(),allin.begin(),allin.end())) {
        throw std::runtime_error("allin contains undealt chair");
    }

    const auto order=clockwise_from(dealer,dealt);
    if (order.size()!=dealt.size() || order.front()!=dealer) {
        throw std::runtime_error("failed clockwise dealt order");
    }

    HandAnchor anchor{};
    anchor.hand_id=frame.hand_id;
    anchor.chair_to_logical.fill(-1);
    const bool hu=order.size()==2;
    if (hu) {
        anchor.logical_to_chair={order[0],order[1],-1};
    } else {
        anchor.logical_to_chair={order[0],order[1],order[2]};
    }
    for (int logical=0;logical<3;++logical) {
        int chair=anchor.logical_to_chair[static_cast<std::size_t>(logical)];
        if (chair>=0) anchor.chair_to_logical[static_cast<std::size_t>(chair)]=logical;
    }
    anchor.hero_logical_seat=anchor.chair_to_logical[static_cast<std::size_t>(user)];
    if (anchor.hero_logical_seat<0) throw std::runtime_error("failed Hero logical mapping");
    anchor.hero_cards=frame.hero_cards;
    validate_cards(anchor.hero_cards,frame.board_cards,0);

    EpisodeScenario scenario{};
    scenario.state.game_is_hu=hu;
    scenario.state.small_blind=frame.small_blind;
    scenario.state.big_blind=frame.big_blind;
    scenario.state.blind_index=blind_index(frame.small_blind,frame.big_blind);
    scenario.dealer_id=0;
    scenario.state.dead_players={-1,-1,-1};
    scenario.state.dead_player_count=0;
    if (hu) {
        scenario.state.dead_players={2,-1,-1};
        scenario.state.dead_player_count=1;
    }

    for (int logical=0;logical<3;++logical) {
        int chair=anchor.logical_to_chair[static_cast<std::size_t>(logical)];
        if (chair<0) {
            scenario.state.stacks[static_cast<std::size_t>(logical)]=0;
            continue;
        }
        const int bal=frame.balances[static_cast<std::size_t>(chair)];
        const int cur=frame.current_bets[static_cast<std::size_t>(chair)];
        if (bal<0 || cur<0) throw std::runtime_error("negative balance/currentbet");
        scenario.state.stacks[static_cast<std::size_t>(logical)]=bal+cur;
    }
    scenario.state.total_chips=
        scenario.state.stacks[0]+scenario.state.stacks[1]+scenario.state.stacks[2];

    std::array<int,3> expected{};
    if (hu) {
        expected[0]=std::min(scenario.state.stacks[0],scenario.state.small_blind);
        expected[1]=std::min(scenario.state.stacks[1],scenario.state.big_blind);
    } else {
        expected[1]=std::min(scenario.state.stacks[1],scenario.state.small_blind);
        expected[2]=std::min(scenario.state.stacks[2],scenario.state.big_blind);
    }
    for (int logical=0;logical<3;++logical) {
        int chair=anchor.logical_to_chair[static_cast<std::size_t>(logical)];
        int actual=chair<0?0:frame.current_bets[static_cast<std::size_t>(chair)];
        if (actual!=expected[static_cast<std::size_t>(logical)]) {
            throw std::runtime_error("hand-start blind-post mismatch");
        }
    }
    validate_episode_scenario(scenario);
    anchor.scenario=scenario;
    return anchor;
}

ObservedSnapshot normalize_raw_frame(const RawFrame& frame,const HandAnchor& anchor) {
    if (frame.hand_id!=anchor.hand_id) throw std::runtime_error("hand id changed without reset");
    if (validate_chair(frame.user_chair,"userchair") !=
        anchor.logical_to_chair[static_cast<std::size_t>(anchor.hero_logical_seat)]) {
        throw std::runtime_error("userchair changed");
    }
    if (validate_chair(frame.dealer_chair,"dealerchair") != anchor.logical_to_chair[0]) {
        throw std::runtime_error("dealerchair changed");
    }
    if (frame.small_blind!=anchor.scenario.state.small_blind ||
        frame.big_blind!=anchor.scenario.state.big_blind) {
        throw std::runtime_error("blinds changed");
    }

    const auto dealt=bits(frame.players_dealt_bits);
    std::set<int> expected_dealt;
    for (int chair:anchor.logical_to_chair) if (chair>=0) expected_dealt.insert(chair);
    if (dealt!=expected_dealt) throw std::runtime_error("dealt set changed");

    const auto playing=bits(frame.players_playing_bits);
    const auto allin=bits(frame.players_allin_bits);
    if (!std::includes(dealt.begin(),dealt.end(),playing.begin(),playing.end()) ||
        !std::includes(dealt.begin(),dealt.end(),allin.begin(),allin.end()) ||
        !std::includes(playing.begin(),playing.end(),allin.begin(),allin.end())) {
        throw std::runtime_error("playing/allin bitset inconsistency");
    }

    if (frame.betround<1 || frame.betround>4) throw std::runtime_error("betround outside 1..4");
    const int expected_br=openholdem_betround_from_visible_count(frame.common_cards_known);
    if (frame.betround!=expected_br) throw std::runtime_error("board-count/betround mismatch");
    if (frame.hero_cards!=anchor.hero_cards) throw std::runtime_error("Hero cards changed");
    validate_cards(frame.hero_cards,frame.board_cards,frame.common_cards_known);

    ObservedSnapshot out{};
    out.street=frame.betround-1;
    out.pot=frame.pot;
    out.visible_board_count=frame.common_cards_known;
    out.board_cards=frame.board_cards;
    for (int logical=0;logical<3;++logical) {
        const int chair=anchor.logical_to_chair[static_cast<std::size_t>(logical)];
        if (chair<0) {
            out.stacks[static_cast<std::size_t>(logical)]=0;
            out.street_commitments[static_cast<std::size_t>(logical)]=0;
            out.folded[static_cast<std::size_t>(logical)]=0;
            out.all_in[static_cast<std::size_t>(logical)]=1;
            continue;
        }
        const int bal=frame.balances[static_cast<std::size_t>(chair)];
        const int cur=frame.current_bets[static_cast<std::size_t>(chair)];
        if (bal<0 || cur<0) throw std::runtime_error("negative balance/currentbet");
        out.stacks[static_cast<std::size_t>(logical)]=bal;
        out.street_commitments[static_cast<std::size_t>(logical)]=cur;
        const bool is_allin=allin.count(chair)!=0;
        out.all_in[static_cast<std::size_t>(logical)]=is_allin?1U:0U;
        out.folded[static_cast<std::size_t>(logical)]=
            (playing.count(chair)==0 && !is_allin)?1U:0U;
    }
    int expected_pot=0;
    for (int logical=0;logical<3;++logical) {
        if (anchor.logical_to_chair[static_cast<std::size_t>(logical)]>=0) {
            expected_pot += anchor.scenario.state.stacks[static_cast<std::size_t>(logical)]
                - out.stacks[static_cast<std::size_t>(logical)];
        }
    }
    if (out.pot!=expected_pot) throw std::runtime_error("pot/chip conservation mismatch");
    return out;
}

ObservedSnapshot observable_projection(
    const SpinTraversalState& state,
    const std::array<std::int32_t,5>& visible_board_cards) {
    const auto& hand=state.hand();
    const auto& betting=hand.betting();
    const auto& players=betting.players();
    ObservedSnapshot out{};
    out.visible_board_count=static_cast<int>(hand.visible_board_count());
    out.street=openholdem_betround_from_visible_count(out.visible_board_count)-1;
    out.pot=betting.pot();
    out.board_cards=visible_board_cards;
    for (int i=0;i<3;++i) {
        const auto& p=players[static_cast<std::size_t>(i)];
        out.stacks[static_cast<std::size_t>(i)]=p.stack;
        out.street_commitments[static_cast<std::size_t>(i)]=p.street_commitment;
        out.folded[static_cast<std::size_t>(i)]=p.folded?1U:0U;
        out.all_in[static_cast<std::size_t>(i)]=p.all_in?1U:0U;
    }
    return out;
}

void ObservableRuntimeTracker::reset() noexcept {
    have_anchor_=false;
    state_.reset();
    observed_.reset();
    transcript_.clear();
    failed_=false;
    failure_reason_.clear();
    generation_=0;
}

SyncResult ObservableRuntimeTracker::fail(std::string reason) {
    failed_=true;
    failure_reason_=std::move(reason);
    return {SyncKind::Failed,std::nullopt,failure_reason_};
}

SpinTraversalState ObservableRuntimeTracker::rebuild(
    const ObservedSnapshot& observed,
    const std::vector<ExactAction>& transcript) const {
    if (!have_anchor_) throw std::runtime_error("rebuild without anchor");
    const auto [holes,board]=runtime_deal(anchor_,observed);
    SpinTraversalState state(anchor_.scenario,holes,board);
    for (const auto& action:transcript) state.apply_exact(action);
    if (observable_projection(state,observed.board_cards)!=observed) {
        throw std::runtime_error("from-scratch transcript rebuild mismatch");
    }
    return state;
}

SyncResult ObservableRuntimeTracker::start_hand(
    const HandAnchor& anchor,
    const ObservedSnapshot& observed) {
    reset();
    anchor_=anchor;
    have_anchor_=true;
    try {
        state_=rebuild(observed,{});
    } catch (const std::exception& e) {
        return fail(std::string("observable hand initialization failed: ")+e.what());
    }
    observed_=observed;
    return {SyncKind::Start,std::nullopt,{}};
}

SyncResult ObservableRuntimeTracker::heartbeat(
    const HandAnchor& anchor,
    const ObservedSnapshot& observed) {
    return sync(anchor,observed,std::nullopt);
}

SyncResult ObservableRuntimeTracker::synchronize_to_actor(
    const HandAnchor& anchor,
    const ObservedSnapshot& observed,
    std::int32_t target_actor) {
    return sync(anchor,observed,target_actor);
}

SyncResult ObservableRuntimeTracker::sync(
    const HandAnchor& anchor,
    const ObservedSnapshot& observed,
    std::optional<std::int32_t> target_actor) {
    if (failed_) return {SyncKind::Failed,std::nullopt,failure_reason_};
    if (!have_anchor_ || !state_.has_value() || !observed_.has_value()) {
        return fail("sync without active hand");
    }
    if (!(anchor==anchor_)) return fail("hand anchor changed without reset");

    const auto& current=*state_;
    if (observed==*observed_ &&
        (!target_actor.has_value() ||
         (!current.terminal() && current.actor()==*target_actor))) {
        return {SyncKind::NoChange,std::nullopt,{}};
    }

    std::vector<ExactAction> inferred;
    int visible_actions=0;
    try {
        SpinTraversalState probe=current;
        bool matched=false;
        for (int step=0;step<=6;++step) {
            const auto projected=observable_projection(probe,observed.board_cards);
            const bool actor_ok=!target_actor.has_value() ||
                (!probe.terminal() && probe.actor()==*target_actor);
            if (projected==observed && actor_ok) {
                matched=true;
                break;
            }
            if (probe.terminal()) throw std::runtime_error("terminal state does not match observed frame");
            const auto& betting=probe.hand().betting();
            const int actor=probe.actor();
            if (actor<0 || actor>2) throw std::runtime_error("invalid actor");
            const auto legal=betting.legal_actions(actor);
            const auto& player=betting.players()[static_cast<std::size_t>(actor)];
            if (observed.stacks[static_cast<std::size_t>(actor)]>player.stack) {
                throw std::runtime_error("acting stack increased");
            }
            const int paid=player.stack-observed.stacks[static_cast<std::size_t>(actor)];
            const bool folded_now=observed.folded[static_cast<std::size_t>(actor)]!=0 &&
                !player.folded;

            ExactAction action{};
            if (folded_now) {
                if (paid!=0) throw std::runtime_error("fold transition paid chips");
                action={ExactActionType::Fold,0};
                ++visible_actions;
            } else if (paid==0) {
                action={ExactActionType::Check,0};
            } else {
                const int target=player.street_commitment+paid;
                if (legal.to_call>0 && target<=betting.current_bet()) {
                    action={ExactActionType::Call,0};
                } else if (observed.stacks[static_cast<std::size_t>(actor)]==0) {
                    action={ExactActionType::AllIn,0};
                } else if (betting.current_bet()==0) {
                    action={ExactActionType::BetTo,target};
                } else {
                    action={ExactActionType::RaiseTo,target};
                }
                ++visible_actions;
            }
            if (visible_actions>1) {
                throw std::runtime_error("more than one chip/fold-changing action between frames");
            }
            probe.apply_exact(action);
            inferred.push_back(action);
        }
        if (!matched) throw std::runtime_error("observable reconciliation exceeded action cap");
        if (inferred.empty()) return {SyncKind::NoChange,std::nullopt,{}};

        auto new_transcript=transcript_;
        new_transcript.insert(new_transcript.end(),inferred.begin(),inferred.end());
        auto rebuilt=rebuild(observed,new_transcript);
        if (target_actor.has_value() &&
            (rebuilt.terminal() || rebuilt.actor()!=*target_actor)) {
            throw std::runtime_error("rebuild did not reach target actor");
        }
        state_=std::move(rebuilt);
        observed_=observed;
        transcript_=std::move(new_transcript);
        ++generation_;
        return {SyncKind::Action,inferred.back(),{}};
    } catch (const std::exception& e) {
        return fail(std::string("observable reconciliation/rebuild failed: ")+e.what());
    }
}

} // namespace spincore::lt2oh
