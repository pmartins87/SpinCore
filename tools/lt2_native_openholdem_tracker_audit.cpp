#include "spincore/lt2_openholdem_runtime.hpp"

#include "spincore/lean_action_abstraction.hpp"
#include "spincore/neural_encoder.hpp"
#include "spincore/neural_encoder_v2.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace spincore;

namespace {

constexpr std::array<std::array<int,2>,9> kBlinds{{
    {{10,20}},{{15,30}},{{20,40}},{{30,60}},{{40,80}},
    {{50,100}},{{60,120}},{{80,160}},{{100,200}}
}};

UniversalActionMaskV2 active_mask() {
    UniversalActionMaskV2 m{};
    for (int slot : {0,1,3,5,7,8,9}) m[static_cast<std::size_t>(slot)]=1U;
    return m;
}

EpisodeScenario scenario_for(std::uint64_t index,std::mt19937_64& rng) {
    EpisodeScenario s{};
    const bool hu=(index%2U)==1U;
    const auto blind=kBlinds[index%kBlinds.size()];
    const int sb=blind[0],bb=blind[1];
    s.state.total_chips=1500;
    s.state.game_is_hu=hu;
    s.state.blind_index=static_cast<int>(index%kBlinds.size());
    s.state.small_blind=sb;
    s.state.big_blind=bb;
    s.dealer_id=0;
    s.state.dead_players={-1,-1,-1};
    s.state.dead_player_count=0;

    if (hu) {
        const int rem=1500-2*bb;
        const int add=rem>0?static_cast<int>(rng()%static_cast<std::uint64_t>(rem+1)):0;
        s.state.stacks={bb+add,1500-(bb+add),0};
        s.state.dead_players={2,-1,-1};
        s.state.dead_player_count=1;
    } else {
        const int rem=1500-3*bb;
        const int a=rem>0?static_cast<int>(rng()%static_cast<std::uint64_t>(rem+1)):0;
        const int rest=rem-a;
        const int b=rest>0?static_cast<int>(rng()%static_cast<std::uint64_t>(rest+1)):0;
        const int c=rest-b;
        s.state.stacks={bb+a,bb+b,bb+c};
    }
    validate_episode_scenario(s);
    return s;
}

std::array<int,3> physical_chairs(const EpisodeScenario& s,std::mt19937_64& rng) {
    const int dealer=static_cast<int>(rng()%10U);
    if (s.state.game_is_hu) {
        const int gap=1+static_cast<int>(rng()%9U);
        return {dealer,(dealer+gap)%10,-1};
    }
    int g1=1+static_cast<int>(rng()%8U);
    int g2=g1+1+static_cast<int>(rng()%static_cast<std::uint64_t>(9-g1));
    return {dealer,(dealer+g1)%10,(dealer+g2)%10};
}

lt2oh::RawFrame frame_from(
    const std::string& hand_id,
    const SpinTraversalState& state,
    const std::array<int,3>& chairs,
    int hero) {
    lt2oh::RawFrame f{};
    f.hand_id=hand_id;
    f.user_chair=chairs[static_cast<std::size_t>(hero)];
    f.dealer_chair=chairs[0];
    f.small_blind=state.scenario().state.small_blind;
    f.big_blind=state.scenario().state.big_blind;

    const auto& hand=state.hand();
    const auto& betting=hand.betting();
    const auto& ps=betting.players();
    const int visible=static_cast<int>(hand.visible_board_count());
    f.common_cards_known=visible;
    f.betround=lt2oh::openholdem_betround_from_visible_count(visible);
    f.pot=betting.pot();

    for (int logical=0;logical<3;++logical) {
        const int chair=chairs[static_cast<std::size_t>(logical)];
        if (chair<0) continue;
        f.players_dealt_bits |= 1U<<static_cast<unsigned>(chair);
        const auto& p=ps[static_cast<std::size_t>(logical)];
        f.balances[static_cast<std::size_t>(chair)]=p.stack;
        f.current_bets[static_cast<std::size_t>(chair)]=p.street_commitment;
        if (!p.folded) f.players_playing_bits |= 1U<<static_cast<unsigned>(chair);
        if (p.all_in) {
            f.players_allin_bits |= 1U<<static_cast<unsigned>(chair);
            f.players_playing_bits |= 1U<<static_cast<unsigned>(chair);
        }
    }

    const auto& holes=hand.hole_cards();
    f.hero_cards={
        static_cast<int>(holes[static_cast<std::size_t>(hero)][0].id()),
        static_cast<int>(holes[static_cast<std::size_t>(hero)][1].id())
    };
    f.board_cards.fill(-1);
    const auto& board=hand.board();
    for (int i=0;i<visible;++i) f.board_cards[static_cast<std::size_t>(i)]=static_cast<int>(board[static_cast<std::size_t>(i)].id());
    return f;
}

bool hero_context_equal(const SpinTraversalState& a,const SpinTraversalState& b) {
    if (a.terminal() || b.terminal()) return a.terminal()==b.terminal();
    if (a.actor()!=b.actor()) return false;
    if (strategy_domain(a.hand().betting().topology())!=strategy_domain(b.hand().betting().topology())) return false;

    const auto av1=serialize_neural_input_v1(a.neural_input());
    const auto bv1=serialize_neural_input_v1(b.neural_input());
    if (av1!=bv1) return false;

    const auto av2=serialize_neural_input_v2(encode_neural_input_v2(a.hand(),a.blind_index()));
    const auto bv2=serialize_neural_input_v2(encode_neural_input_v2(b.hand(),b.blind_index()));
    if (av2!=bv2) return false;

    const auto mask=active_mask();
    return resolve_lean_legacy_actions_v1(a.hand().betting(),mask)
        == resolve_lean_legacy_actions_v1(b.hand().betting(),mask);
}

ResolvedUniversalActionV2 choose_action(
    const SpinTraversalState& state,
    std::mt19937_64& rng) {
    auto legal=resolve_lean_legacy_actions_v1(state.hand().betting(),active_mask());
    if (legal.empty()) throw std::runtime_error("nonterminal state without lean action");
    std::vector<std::size_t> passive;
    for (std::size_t i=0;i<legal.size();++i) {
        if (legal[i].exact.type==ExactActionType::Check || legal[i].exact.type==ExactActionType::Call) {
            passive.push_back(i);
        }
    }
    if (!passive.empty() && (rng()%100U)<60U) {
        return legal[passive[static_cast<std::size_t>(rng()%passive.size())]];
    }
    return legal[static_cast<std::size_t>(rng()%legal.size())];
}

bool visible_impact(const ExactAction& a) {
    return a.type!=ExactActionType::Check;
}

void write_report(
    const std::filesystem::path& path,
    const std::string& verdict,
    std::uint64_t transitions,
    std::uint64_t hero_checks,
    std::uint64_t silent_checks,
    std::uint64_t myturn_delayed,
    std::uint64_t multi_sync,
    std::uint64_t street_reveals,
    std::uint64_t corrupt_attempts,
    std::uint64_t corrupt_rejects,
    std::uint64_t skipped_attempts,
    std::uint64_t skipped_rejects,
    std::uint64_t failures) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write report");
    out<<"{\n"
       <<"  \"schema\": \"SPINCORE_LT2_NATIVE_OPENHOLDEM_TRACKER_V1\",\n"
       <<"  \"verdict\": \""<<verdict<<"\",\n"
       <<"  \"transitions_checked\": "<<transitions<<",\n"
       <<"  \"hero_canonical_state_checks\": "<<hero_checks<<",\n"
       <<"  \"silent_check_deferrals\": "<<silent_checks<<",\n"
       <<"  \"delayed_actions_reconciled_at_myturn\": "<<myturn_delayed<<",\n"
       <<"  \"multi_action_sync_events\": "<<multi_sync<<",\n"
       <<"  \"street_reveals_checked\": "<<street_reveals<<",\n"
       <<"  \"faults\": {\n"
       <<"    \"corrupt_attempts\": "<<corrupt_attempts<<",\n"
       <<"    \"corrupt_rejections\": "<<corrupt_rejects<<",\n"
       <<"    \"skipped_attempts\": "<<skipped_attempts<<",\n"
       <<"    \"skipped_rejections\": "<<skipped_rejects<<"\n"
       <<"  },\n"
       <<"  \"failure_count\": "<<failures<<",\n"
       <<"  \"method\": {\n"
       <<"    \"training_roots\": 0,\n"
       <<"    \"optimizer_steps\": 0,\n"
       <<"    \"strategic_ev_evaluations\": 0,\n"
       <<"    \"deployment_model_inference\": 0,\n"
       <<"    \"holdout_reused\": false\n"
       <<"  }\n"
       <<"}\n";
}

} // namespace

int main(int argc,char** argv) {
    try {
        if (argc!=2) throw std::runtime_error("usage: spincore_lt2_native_openholdem_tracker_audit REPORT.json");
        const std::filesystem::path report=argv[1];
        std::mt19937_64 rng(20260921ULL);

        std::uint64_t transitions=0,hero_checks=0,silent_checks=0,myturn_delayed=0,multi_sync=0,street_reveals=0;
        std::uint64_t corrupt_attempts=0,corrupt_rejects=0,skipped_attempts=0,skipped_rejects=0,failures=0;
        constexpr std::uint64_t kTransitionTarget=12000;
        constexpr std::uint64_t kFaultTarget=500;

        for (std::uint64_t hand_index=0; transitions<kTransitionTarget && hand_index<10000; ++hand_index) {
            auto scenario=scenario_for(hand_index,rng);
            SpinTraversalState truth(scenario,rng());
            auto chairs=physical_chairs(scenario,rng);
            std::vector<int> live;
            for (int i=0;i<3;++i) if (scenario.state.stacks[static_cast<std::size_t>(i)]>0) live.push_back(i);
            const int hero=live[static_cast<std::size_t>(rng()%live.size())];
            const std::string hand_id="NATIVE-"+std::to_string(hand_index);

            auto raw0=frame_from(hand_id,truth,chairs,hero);
            auto anchor=lt2oh::anchor_from_raw_frame(raw0);
            auto obs0=lt2oh::normalize_raw_frame(raw0,anchor);
            lt2oh::ObservableRuntimeTracker tracker;
            auto start=tracker.start_hand(anchor,obs0);
            if (start.kind!=lt2oh::SyncKind::Start) { ++failures; break; }

            std::vector<ExactAction> expected;
            for (int step=0;step<80 && !truth.terminal() && transitions<kTransitionTarget;++step) {
                auto raw=frame_from(hand_id,truth,chairs,hero);
                auto obs=lt2oh::normalize_raw_frame(raw,anchor);

                auto dup=tracker.heartbeat(anchor,obs);
                if (dup.kind!=lt2oh::SyncKind::NoChange) { ++failures; break; }

                if (truth.actor()==hero) {
                    const auto before_len=tracker.transcript().size();
                    auto sync=tracker.synchronize_to_actor(anchor,obs,hero);
                    if (sync.kind==lt2oh::SyncKind::Failed) { ++failures; break; }
                    myturn_delayed += tracker.transcript().size()-before_len;
                    if (tracker.transcript().size()!=expected.size()) { ++failures; break; }
                    const auto* rebuilt=tracker.state();
                    if (!rebuilt || !hero_context_equal(*rebuilt,truth)) { ++failures; break; }
                    ++hero_checks;
                }

                const auto chosen=choose_action(truth,rng);
                const int old_visible=static_cast<int>(truth.hand().visible_board_count());
                truth.apply_exact(chosen.exact);
                expected.push_back(chosen.exact);
                const int new_visible=static_cast<int>(truth.hand().visible_board_count());
                if (new_visible!=old_visible) ++street_reveals;

                auto raw2=frame_from(hand_id,truth,chairs,hero);
                auto obs2=lt2oh::normalize_raw_frame(raw2,anchor);
                const bool changed=!(obs2==obs);
                const auto before_len=tracker.transcript().size();
                auto event=tracker.heartbeat(anchor,obs2);
                ++transitions;

                if (!changed) {
                    if (chosen.exact.type!=ExactActionType::Check ||
                        event.kind!=lt2oh::SyncKind::NoChange) {
                        ++failures; break;
                    }
                    ++silent_checks;
                } else {
                    if (event.kind!=lt2oh::SyncKind::Action) { ++failures; break; }
                    if (tracker.transcript().size()-before_len>1U) ++multi_sync;
                    if (!event.action.has_value() ||
                        event.action->type!=chosen.exact.type ||
                        event.action->amount_to!=chosen.exact.amount_to) {
                        ++failures; break;
                    }
                    if (tracker.transcript().size()!=expected.size()) { ++failures; break; }
                    const auto* rebuilt=tracker.state();
                    if (!rebuilt ||
                        lt2oh::observable_projection(*rebuilt,obs2.board_cards)!=
                        lt2oh::observable_projection(truth,obs2.board_cards)) {
                        ++failures; break;
                    }
                }
            }
            if (failures) break;

            if (corrupt_attempts<kFaultTarget) {
                SpinTraversalState base(scenario,rng());
                auto f0=frame_from(hand_id+"-C",base,chairs,hero);
                auto a0=lt2oh::anchor_from_raw_frame(f0);
                auto o0=lt2oh::normalize_raw_frame(f0,a0);
                lt2oh::ObservableRuntimeTracker t;
                if (t.start_hand(a0,o0).kind!=lt2oh::SyncKind::Start) { ++failures; break; }
                if (!base.terminal()) {
                    auto c=choose_action(base,rng);
                    base.apply_exact(c.exact);
                    auto f1=frame_from(hand_id+"-C",base,chairs,hero);
                    ++f1.pot;
                    ++corrupt_attempts;
                    try {
                        auto bad=lt2oh::normalize_raw_frame(f1,a0);
                        if (t.heartbeat(a0,bad).kind==lt2oh::SyncKind::Failed) ++corrupt_rejects;
                    } catch (...) {
                        ++corrupt_rejects;
                    }
                }
            }

            if (skipped_attempts<kFaultTarget) {
                SpinTraversalState base(scenario,rng());
                auto f0=frame_from(hand_id+"-S",base,chairs,hero);
                auto a0=lt2oh::anchor_from_raw_frame(f0);
                auto o0=lt2oh::normalize_raw_frame(f0,a0);
                lt2oh::ObservableRuntimeTracker t;
                if (t.start_hand(a0,o0).kind!=lt2oh::SyncKind::Start) { ++failures; break; }

                bool attempted=false;
                auto first=resolve_lean_legacy_actions_v1(base.hand().betting(),active_mask());
                for (const auto& a1:first) {
                    if (!visible_impact(a1.exact)) continue;
                    SpinTraversalState p=base;
                    p.apply_exact(a1.exact);
                    if (p.terminal()) continue;
                    auto second=resolve_lean_legacy_actions_v1(p.hand().betting(),active_mask());
                    auto it=std::find_if(second.begin(),second.end(),[](const auto& x){return visible_impact(x.exact);});
                    if (it==second.end()) continue;
                    p.apply_exact(it->exact);
                    auto fs=frame_from(hand_id+"-S",p,chairs,hero);
                    auto os=lt2oh::normalize_raw_frame(fs,a0);
                    ++skipped_attempts;
                    if (t.heartbeat(a0,os).kind==lt2oh::SyncKind::Failed) ++skipped_rejects;
                    attempted=true;
                    break;
                }
                (void)attempted;
            }
        }

        const bool pass=
            failures==0 &&
            transitions==kTransitionTarget &&
            hero_checks>0 &&
            silent_checks>0 &&
            (myturn_delayed>0 || multi_sync>0) &&
            street_reveals>0 &&
            corrupt_attempts==kFaultTarget &&
            corrupt_rejects==corrupt_attempts &&
            skipped_attempts==kFaultTarget &&
            skipped_rejects==skipped_attempts;

        write_report(
            report,pass?"PASS":"FAIL",transitions,hero_checks,silent_checks,
            myturn_delayed,multi_sync,street_reveals,corrupt_attempts,corrupt_rejects,
            skipped_attempts,skipped_rejects,failures);

        std::cout<<"=== LT2 NATIVE OPENHOLDEM TRACKER AUDIT ===\n"
                 <<"VERDICT="<<(pass?"PASS":"FAIL")<<"\n"
                 <<"transitions="<<transitions<<" hero_checks="<<hero_checks
                 <<" silent_checks="<<silent_checks<<" myturn_delayed="<<myturn_delayed
                 <<" multi_sync="<<multi_sync<<" street_reveals="<<street_reveals<<"\n"
                 <<"faults corrupt="<<corrupt_rejects<<"/"<<corrupt_attempts
                 <<" skipped="<<skipped_rejects<<"/"<<skipped_attempts<<"\n"
                 <<"LT2_NATIVE_OPENHOLDEM_TRACKER_COMPLETE\n"
                 <<"report="<<std::filesystem::absolute(report)<<"\n";
        return pass?0:2;
    } catch (const std::exception& e) {
        std::cerr<<"FATAL: "<<e.what()<<"\n";
        return 3;
    }
}
