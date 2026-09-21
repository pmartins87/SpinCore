#include "spincore/lt2_openholdem_shadow_engine.hpp"

#include "spincore/game_topology.hpp"
#include "spincore/lean_action_abstraction.hpp"
#include "spincore/neural_encoder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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

EpisodeScenario make_scenario(std::uint64_t index,std::mt19937_64& rng) {
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
        const int min_stack=bb;
        const int span=1500-2*min_stack;
        const int add=span>0?static_cast<int>(rng()%static_cast<std::uint64_t>(span+1)):0;
        s.state.stacks={min_stack+add,1500-min_stack-add,0};
        s.state.dead_players={2,-1,-1};
        s.state.dead_player_count=1;
    } else {
        const int min_stack=bb;
        const int span=1500-3*min_stack;
        const int a=span>0?static_cast<int>(rng()%static_cast<std::uint64_t>(span+1)):0;
        const int rest=span-a;
        const int b=rest>0?static_cast<int>(rng()%static_cast<std::uint64_t>(rest+1)):0;
        s.state.stacks={min_stack+a,min_stack+b,min_stack+(rest-b)};
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
    const int g1=1+static_cast<int>(rng()%8U);
    const int g2=g1+1+static_cast<int>(rng()%static_cast<std::uint64_t>(9-g1));
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
    for (int i=0;i<visible;++i) {
        f.board_cards[static_cast<std::size_t>(i)]=static_cast<int>(board[static_cast<std::size_t>(i)].id());
    }
    return f;
}

ResolvedUniversalActionV2 choose_truth_action(
    const SpinTraversalState& state,
    std::mt19937_64& rng) {
    auto legal=resolve_lean_legacy_actions_v1(state.hand().betting(),active_mask());
    if (legal.empty()) throw std::runtime_error("truth state without lean legal action");

    std::vector<std::size_t> passive;
    for (std::size_t i=0;i<legal.size();++i) {
        if (legal[i].exact.type==ExactActionType::Check ||
            legal[i].exact.type==ExactActionType::Call) passive.push_back(i);
    }
    if (!passive.empty() && (rng()%100U)<68U) {
        return legal[passive[static_cast<std::size_t>(rng()%passive.size())]];
    }

    std::vector<std::size_t> nonfold;
    for (std::size_t i=0;i<legal.size();++i) {
        if (legal[i].exact.type!=ExactActionType::Fold) nonfold.push_back(i);
    }
    const auto& pool=nonfold.empty()?std::vector<std::size_t>{}:nonfold;
    if (!pool.empty() && (rng()%100U)<90U) {
        return legal[pool[static_cast<std::size_t>(rng()%pool.size())]];
    }
    return legal[static_cast<std::size_t>(rng()%legal.size())];
}

bool decision_matches_truth(
    const lt2oh::ShadowDecision& d,
    const SpinTraversalState& truth) {
    const auto legal=resolve_lean_legacy_actions_v1(truth.hand().betting(),active_mask());
    std::array<std::uint8_t,10> expected_mask{};
    const ResolvedUniversalActionV2* selected=nullptr;
    for (const auto& item:legal) {
        const auto slot=static_cast<std::size_t>(item.slot);
        expected_mask[slot]=1U;
        if (static_cast<int>(item.slot)==d.action_slot) selected=&item;
    }
    if (expected_mask!=d.legal_mask || selected==nullptr) return false;
    if (!(selected->exact==d.exact)) return false;

    const auto domain=strategy_domain(truth.hand().betting().topology());
    const int expected_domain=domain==StrategyDomain::TrueHeadsUp?1:0;
    if (d.domain!=expected_domain) return false;

    double sum=0.0;
    for (std::size_t i=0;i<d.probabilities.size();++i) {
        const float p=d.probabilities[i];
        if (!std::isfinite(p) || p<0.0F) return false;
        if (d.legal_mask[i]==0U && p!=0.0F) return false;
        sum+=static_cast<double>(p);
    }
    return std::abs(sum-1.0)<=1e-5;
}

std::vector<std::uint8_t> neural_bytes(const SpinTraversalState& state) {
    return serialize_neural_input_v1(state.neural_input());
}

void write_report(
    const std::filesystem::path& path,
    const std::string& verdict,
    std::uint64_t transitions,
    std::uint64_t decisions,
    std::uint64_t inference_count,
    std::uint64_t cache_hits,
    std::uint64_t cache_invalidations,
    std::uint64_t exact_matches,
    std::uint64_t three_handed,
    std::uint64_t heads_up,
    const std::array<std::uint64_t,4>& streets,
    std::uint64_t postflop,
    std::uint64_t failures) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write report");
    out<<"{\n"
       <<"  \"schema\": \"SPINCORE_LT2_NATIVE_OPENHOLDEM_SHADOW_ENGINE_V1\",\n"
       <<"  \"verdict\": \""<<verdict<<"\",\n"
       <<"  \"transitions_checked\": "<<transitions<<",\n"
       <<"  \"hero_decisions_checked\": "<<decisions<<",\n"
       <<"  \"native_inference_calls\": "<<inference_count<<",\n"
       <<"  \"repeated_myturn_cache_hits\": "<<cache_hits<<",\n"
       <<"  \"cache_invalidations_checked\": "<<cache_invalidations<<",\n"
       <<"  \"exact_selected_action_matches\": "<<exact_matches<<",\n"
       <<"  \"decisions_by_domain\": {\n"
       <<"    \"THREE_HANDED\": "<<three_handed<<",\n"
       <<"    \"TRUE_HEADS_UP\": "<<heads_up<<"\n"
       <<"  },\n"
       <<"  \"decisions_by_street\": {\n"
       <<"    \"0\": "<<streets[0]<<",\n"
       <<"    \"1\": "<<streets[1]<<",\n"
       <<"    \"2\": "<<streets[2]<<",\n"
       <<"    \"3\": "<<streets[3]<<"\n"
       <<"  },\n"
       <<"  \"postflop_decisions\": "<<postflop<<",\n"
       <<"  \"failure_count\": "<<failures<<",\n"
       <<"  \"bundle_identity\": {\n"
       <<"    \"native_file_sha256_expected\": \""<<lt2oh::kFrozenNativeBundleFileSha256<<"\",\n"
       <<"    \"python_bundle_sha256\": \""<<lt2oh::kFrozenPythonBundleSha256<<"\",\n"
       <<"    \"checkpoint_sha256\": \""<<lt2oh::kFrozenCheckpointSha256<<"\",\n"
       <<"    \"ensemble_sha256\": \""<<lt2oh::kFrozenEnsembleSha256<<"\"\n"
       <<"  },\n"
       <<"  \"method\": {\n"
       <<"    \"training_roots\": 0,\n"
       <<"    \"optimizer_steps\": 0,\n"
       <<"    \"strategic_ev_evaluations\": 0,\n"
       <<"    \"holdout_reused\": false,\n"
       <<"    \"mode\": \"SHADOW_NO_TABLE_ACTION\"\n"
       <<"  }\n"
       <<"}\n";
}

} // namespace

int main(int argc,char** argv) {
    try {
        if (argc!=3) {
            throw std::runtime_error(
                "usage: spincore_lt2_native_openholdem_shadow_engine_audit BUNDLE.bin REPORT.json");
        }
        const std::filesystem::path bundle_path=argv[1];
        const std::filesystem::path report=argv[2];
        auto bundle=lt2oh::load_frozen_deployment_bundle(bundle_path);
        lt2oh::NativeShadowDecisionEngine engine(std::move(bundle),0x5A17C0DE12345678ULL);

        std::mt19937_64 rng(20260921ULL);
        constexpr std::uint64_t kDecisionTarget=1600;
        constexpr std::uint64_t kTransitionCap=16000;

        std::uint64_t transitions=0,decisions=0,cache_hits=0,cache_invalidations=0;
        std::uint64_t exact_matches=0,three_handed=0,heads_up=0,postflop=0,failures=0;
        std::array<std::uint64_t,4> streets{};

        for (std::uint64_t hand=0;
             decisions<kDecisionTarget && transitions<kTransitionCap && hand<10000;
             ++hand) {
            const auto scenario=make_scenario(hand,rng);
            SpinTraversalState truth(scenario,rng());
            const auto chairs=physical_chairs(scenario,rng);
            std::vector<int> live;
            for (int i=0;i<3;++i) {
                if (scenario.state.stacks[static_cast<std::size_t>(i)]>0) live.push_back(i);
            }
            const int hero=live[static_cast<std::size_t>(rng()%live.size())];
            const std::string hand_id="SHADOW-"+std::to_string(hand);

            const auto raw0=frame_from(hand_id,truth,chairs,hero);
            const auto anchor=lt2oh::anchor_from_raw_frame(raw0);
            const auto obs0=lt2oh::normalize_raw_frame(raw0,anchor);
            const auto start=engine.start_hand(anchor,obs0);
            if (start.kind!=lt2oh::SyncKind::Start) { ++failures; break; }

            for (int step=0;
                 step<100 && !truth.terminal() &&
                 decisions<kDecisionTarget && transitions<kTransitionCap;
                 ++step) {
                const auto raw=frame_from(hand_id,truth,chairs,hero);
                const auto obs=lt2oh::normalize_raw_frame(raw,anchor);

                const auto hb=engine.heartbeat(anchor,obs);
                if (hb.kind==lt2oh::SyncKind::Failed) { ++failures; break; }

                if (truth.actor()==hero) {
                    const auto before_inference=engine.inference_count();
                    auto first=engine.on_my_turn(anchor,obs);
                    if (!first.has_value()) { ++failures; break; }
                    const auto after_first=engine.inference_count();

                    auto second=engine.on_my_turn(anchor,obs);
                    if (!second.has_value() || !(*second==*first)) { ++failures; break; }
                    if (engine.inference_count()!=after_first ||
                        after_first!=before_inference+1U) {
                        ++failures; break;
                    }
                    ++cache_hits;

                    const auto cached=engine.cached_decision();
                    if (!cached.has_value() || !(*cached==*first)) { ++failures; break; }

                    const auto* tracker_state=engine.tracker().state();
                    if (!tracker_state || neural_bytes(*tracker_state)!=neural_bytes(truth)) {
                        ++failures; break;
                    }
                    if (!decision_matches_truth(*first,truth)) {
                        ++failures; break;
                    }

                    ++exact_matches;
                    ++decisions;
                    const auto street=static_cast<std::size_t>(truth.hand().betting().street());
                    if (street>=4U) { ++failures; break; }
                    ++streets[street];
                    if (street>0U) ++postflop;
                    if (first->domain==1) ++heads_up; else ++three_handed;
                }

                const auto chosen=choose_truth_action(truth,rng);
                truth.apply_exact(chosen.exact);
                const auto raw2=frame_from(hand_id,truth,chairs,hero);
                const auto obs2=lt2oh::normalize_raw_frame(raw2,anchor);
                const auto generation_before=engine.tracker().generation();
                const auto event=engine.heartbeat(anchor,obs2);
                ++transitions;
                if (event.kind==lt2oh::SyncKind::Failed) { ++failures; break; }
                if (engine.tracker().generation()!=generation_before) {
                    ++cache_invalidations;
                    if (engine.cached_decision().has_value()) { ++failures; break; }
                }
            }
            if (failures) break;
        }

        const auto inference_count=engine.inference_count();
        const bool pass=
            failures==0 &&
            decisions==kDecisionTarget &&
            inference_count==decisions &&
            exact_matches==decisions &&
            cache_hits==decisions &&
            cache_invalidations>0 &&
            three_handed>0 &&
            heads_up>0 &&
            streets[0]>0 &&
            streets[1]>0 &&
            streets[2]>0 &&
            streets[3]>0 &&
            postflop>0;

        write_report(
            report,pass?"PASS":"FAIL",transitions,decisions,inference_count,
            cache_hits,cache_invalidations,exact_matches,three_handed,heads_up,
            streets,postflop,failures);

        std::cout<<"=== LT2 NATIVE OPENHOLDEM SHADOW ENGINE AUDIT ===\n"
                 <<"VERDICT="<<(pass?"PASS":"FAIL")<<"\n"
                 <<"decisions="<<decisions<<" inference="<<inference_count
                 <<" cache_hits="<<cache_hits<<" invalidations="<<cache_invalidations
                 <<" exact="<<exact_matches<<"\n"
                 <<"domains 3H="<<three_handed<<" HU="<<heads_up
                 <<" streets=["<<streets[0]<<","<<streets[1]<<","<<streets[2]<<","<<streets[3]<<"]\n"
                 <<"LT2_NATIVE_OPENHOLDEM_SHADOW_ENGINE_COMPLETE\n"
                 <<"report="<<std::filesystem::absolute(report)<<"\n";
        return pass?0:2;
    } catch (const std::exception& e) {
        std::cerr<<"FATAL: "<<e.what()<<"\n";
        return 3;
    }
}
