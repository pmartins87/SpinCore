#include "spincore/action_abstraction_v2.hpp"
#include "spincore/spin_traversal_state.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace spincore;

namespace {

struct Candidate {
    std::string id;
    UniversalActionMaskV2 preflop{};
    UniversalActionMaskV2 postflop{};
    bool audit_postflop{true};
};

struct CandidateStats {
    std::uint64_t states{0};
    std::uint64_t nominal_aggressive{0};
    std::uint64_t effective_aggressive{0};
    std::uint64_t aliases_suppressed{0};
    std::uint64_t fractional_nominal{0};
    std::uint64_t fractional_clamped_min{0};
    std::uint64_t fractional_clamped_allin{0};
    std::uint64_t exact_application_failures{0};
    std::uint64_t duplicates_after_dedup{0};
    std::uint64_t monotonicity_violations{0};
    std::uint64_t allin_representation_failures{0};
    std::uint64_t min_representation_failures{0};
    std::uint64_t raise_ratio_count{0};
    double raise_ratio_sum{0.0};
    double raise_ratio_min{1e100};
    double raise_ratio_max{-1e100};
    std::map<std::string, std::uint64_t> alias_by_spr;
    std::map<std::string, std::uint64_t> alias_by_effective_stack;
};

struct DomainStats {
    std::uint64_t trajectories{0};
    std::uint64_t visited_states{0};
    std::uint64_t terminal_trajectories{0};
    std::uint64_t nonterminal_stalls{0};
    std::uint64_t terminal_settlement_failures{0};
    std::uint64_t control_equivalence_checks{0};
    std::uint64_t control_equivalence_differences{0};
    std::map<int, std::uint64_t> street_counts;
    std::map<std::string, CandidateStats> candidates;
};

UniversalActionMaskV2 make_mask(std::initializer_list<UniversalActionSlotV2> slots) {
    UniversalActionMaskV2 out{};
    out[static_cast<std::size_t>(UniversalActionSlotV2::Fold)] = 1;
    out[static_cast<std::size_t>(UniversalActionSlotV2::CheckCall)] = 1;
    for (auto slot : slots) {
        out[static_cast<std::size_t>(slot)] = 1;
    }
    return out;
}

UniversalActionMaskV2 singleton(UniversalActionSlotV2 slot) {
    UniversalActionMaskV2 out{};
    out[static_cast<std::size_t>(slot)] = 1;
    return out;
}

std::vector<Candidate> candidates() {
    const auto pre_control = make_mask({UniversalActionSlotV2::MinRaise, UniversalActionSlotV2::AllIn});
    return {
        {"PF0_CONTROL_33_75_AI", pre_control, make_mask({UniversalActionSlotV2::Pot33, UniversalActionSlotV2::Pot75, UniversalActionSlotV2::AllIn}), true},
        {"PF1_33_50_75_AI", pre_control, make_mask({UniversalActionSlotV2::Pot33, UniversalActionSlotV2::Pot50, UniversalActionSlotV2::Pot75, UniversalActionSlotV2::AllIn}), true},
        {"PF2_33_50_75_100_AI", pre_control, make_mask({UniversalActionSlotV2::Pot33, UniversalActionSlotV2::Pot50, UniversalActionSlotV2::Pot75, UniversalActionSlotV2::Pot100, UniversalActionSlotV2::AllIn}), true},
        {"PF3_COMPACT_33_66_100_AI", pre_control, make_mask({UniversalActionSlotV2::Pot33, UniversalActionSlotV2::Pot66, UniversalActionSlotV2::Pot100, UniversalActionSlotV2::AllIn}), true},
        {"PF4_CRUSHER_COMPACT_40_66_100_AI", pre_control, make_mask({UniversalActionSlotV2::Pot40, UniversalActionSlotV2::Pot66, UniversalActionSlotV2::Pot100, UniversalActionSlotV2::AllIn}), true},
        {"PF_DENSE_REFERENCE", pre_control, make_mask({UniversalActionSlotV2::MinRaise, UniversalActionSlotV2::Pot33, UniversalActionSlotV2::Pot40, UniversalActionSlotV2::Pot50, UniversalActionSlotV2::Pot66, UniversalActionSlotV2::Pot75, UniversalActionSlotV2::Pot100, UniversalActionSlotV2::AllIn}), true},
        {"PR0_CONTROL_MIN_AI", make_mask({UniversalActionSlotV2::MinRaise, UniversalActionSlotV2::AllIn}), {}, false},
        {"PR1_MIN_75_AI", make_mask({UniversalActionSlotV2::MinRaise, UniversalActionSlotV2::Pot75, UniversalActionSlotV2::AllIn}), {}, false},
        {"PR2_MIN_50_75_AI", make_mask({UniversalActionSlotV2::MinRaise, UniversalActionSlotV2::Pot50, UniversalActionSlotV2::Pot75, UniversalActionSlotV2::AllIn}), {}, false},
        {"PR3_MIN_75_100_AI", make_mask({UniversalActionSlotV2::MinRaise, UniversalActionSlotV2::Pot75, UniversalActionSlotV2::Pot100, UniversalActionSlotV2::AllIn}), {}, false},
        {"PR4_MIN_50_75_100_AI", make_mask({UniversalActionSlotV2::MinRaise, UniversalActionSlotV2::Pot50, UniversalActionSlotV2::Pot75, UniversalActionSlotV2::Pot100, UniversalActionSlotV2::AllIn}), {}, false},
    };
}

UniversalActionMaskV2 dense_path_mask() {
    return make_mask({
        UniversalActionSlotV2::MinRaise,
        UniversalActionSlotV2::Pot33,
        UniversalActionSlotV2::Pot40,
        UniversalActionSlotV2::Pot50,
        UniversalActionSlotV2::Pot66,
        UniversalActionSlotV2::Pot75,
        UniversalActionSlotV2::Pot100,
        UniversalActionSlotV2::AllIn,
    });
}

bool is_aggressive(UniversalActionSlotV2 slot) {
    return slot != UniversalActionSlotV2::Fold && slot != UniversalActionSlotV2::CheckCall;
}

std::string spr_bucket(const SpinTraversalState& state) {
    const auto& betting = state.hand().betting();
    const auto pot = std::max(1, betting.pot());
    const auto actor = state.actor();
    const auto hero_stack = betting.players()[static_cast<std::size_t>(actor)].stack;
    int effective = hero_stack;
    for (int seat = 0; seat < 3; ++seat) {
        if (seat == actor) continue;
        const auto& p = betting.players()[static_cast<std::size_t>(seat)];
        if (p.folded) continue;
        effective = std::min(effective, p.stack);
    }
    const double spr = static_cast<double>(std::max(0, effective)) / static_cast<double>(pot);
    if (spr < 1.0) return "lt1";
    if (spr < 2.0) return "1to2";
    if (spr < 4.0) return "2to4";
    if (spr < 8.0) return "4to8";
    return "ge8";
}

std::string effective_stack_bucket(const SpinTraversalState& state) {
    const auto& betting = state.hand().betting();
    const auto actor = state.actor();
    const auto hero_stack = betting.players()[static_cast<std::size_t>(actor)].stack;
    int effective = hero_stack;
    for (int seat = 0; seat < 3; ++seat) {
        if (seat == actor) continue;
        const auto& p = betting.players()[static_cast<std::size_t>(seat)];
        if (p.folded) continue;
        effective = std::min(effective, p.stack);
    }
    const double bb = static_cast<double>(std::max(1, state.scenario().state.big_blind));
    const double stack_bb = static_cast<double>(std::max(0, effective)) / bb;
    if (stack_bb < 5.0) return "lt5bb";
    if (stack_bb < 10.0) return "5to10bb";
    if (stack_bb < 20.0) return "10to20bb";
    if (stack_bb < 40.0) return "20to40bb";
    return "ge40bb";
}

std::tuple<int,int> exact_key(const ExactAction& action) {
    return {static_cast<int>(action.type), action.amount_to};
}

void audit_control_equivalence(const SpinTraversalState& state, DomainStats& stats) {
    const std::array<std::pair<AbstractActionSlot, UniversalActionSlotV2>, 6> mapping{{
        {AbstractActionSlot::Fold, UniversalActionSlotV2::Fold},
        {AbstractActionSlot::CheckCall, UniversalActionSlotV2::CheckCall},
        {AbstractActionSlot::ContextRaise, UniversalActionSlotV2::MinRaise},
        {AbstractActionSlot::SmallPot, UniversalActionSlotV2::Pot33},
        {AbstractActionSlot::LargePot, UniversalActionSlotV2::Pot75},
        {AbstractActionSlot::AllIn, UniversalActionSlotV2::AllIn},
    }};
    const auto legal_old = state.legal_abstract_actions();
    for (const auto& [old_slot, new_slot] : mapping) {
        if (std::find(legal_old.begin(), legal_old.end(), old_slot) == legal_old.end()) {
            continue;
        }
        const auto resolved = resolve_universal_actions_v2(state.hand().betting(), singleton(new_slot));
        ++stats.control_equivalence_checks;
        if (resolved.size() != 1 || !(resolved.front().exact == state.resolve(old_slot))) {
            ++stats.control_equivalence_differences;
        }
    }
}

void audit_candidate(
    const SpinTraversalState& state,
    const Candidate& candidate,
    CandidateStats& stats) {
    const auto street = state.hand().betting().street();
    if (street != Street::Preflop && !candidate.audit_postflop) {
        return;
    }
    const auto& active_mask = street == Street::Preflop ? candidate.preflop : candidate.postflop;
    ++stats.states;

    std::uint64_t nominal_aggressive = 0;
    std::uint64_t raw_fractional = 0;
    for (std::size_t i = 0; i < kUniversalActionCountV2; ++i) {
        if (!active_mask[i]) continue;
        const auto slot = static_cast<UniversalActionSlotV2>(i);
        if (!is_aggressive(slot)) continue;
        const auto one = resolve_universal_actions_v2(state.hand().betting(), singleton(slot));
        if (one.empty()) continue;
        ++nominal_aggressive;
        if (universal_is_fractional_v2(slot)) {
            ++raw_fractional;
            ++stats.fractional_nominal;
            if (one.front().clamped_to_min) ++stats.fractional_clamped_min;
            if (one.front().clamped_to_allin) ++stats.fractional_clamped_allin;
        }
    }

    const auto effective = resolve_universal_actions_v2(state.hand().betting(), active_mask);
    std::uint64_t effective_aggressive = 0;
    std::set<std::tuple<int,int>> exact_seen;
    int last_fractional_target = -1;
    for (const auto& action : effective) {
        if (!exact_seen.insert(exact_key(action.exact)).second) {
            ++stats.duplicates_after_dedup;
        }
        if (is_aggressive(action.slot)) {
            ++effective_aggressive;
        }
        if (universal_is_fractional_v2(action.slot) && action.exact.type != ExactActionType::AllIn) {
            if (last_fractional_target >= 0 && action.realized_target <= last_fractional_target) {
                ++stats.monotonicity_violations;
            }
            last_fractional_target = action.realized_target;
            const auto legal = state.hand().betting().legal_actions(state.actor());
            const int pot_after_call = std::max(1, state.hand().betting().pot() + legal.to_call);
            const double ratio = static_cast<double>(action.realized_target) / static_cast<double>(pot_after_call);
            ++stats.raise_ratio_count;
            stats.raise_ratio_sum += ratio;
            stats.raise_ratio_min = std::min(stats.raise_ratio_min, ratio);
            stats.raise_ratio_max = std::max(stats.raise_ratio_max, ratio);
        }
        try {
            auto clone = state;
            clone.apply_exact(action.exact);
            if (clone.terminal()) {
                (void)clone.terminal_chip_delta();
                (void)clone.terminal_icm_delta(PayoutProfile{{0.5,0.3,0.2}});
            }
        } catch (...) {
            ++stats.exact_application_failures;
        }
    }

    stats.nominal_aggressive += nominal_aggressive;
    stats.effective_aggressive += effective_aggressive;
    const auto suppressed = nominal_aggressive >= effective_aggressive
        ? nominal_aggressive - effective_aggressive
        : 0;
    stats.aliases_suppressed += suppressed;
    if (suppressed) {
        stats.alias_by_spr[spr_bucket(state)] += suppressed;
        stats.alias_by_effective_stack[effective_stack_bucket(state)] += suppressed;
    }

    const auto legal = state.hand().betting().legal_actions(state.actor());
    if (active_mask[static_cast<std::size_t>(UniversalActionSlotV2::AllIn)] && legal.all_in) {
        const auto found = std::find_if(effective.begin(), effective.end(), [](const auto& action) {
            return action.slot == UniversalActionSlotV2::AllIn && action.exact.type == ExactActionType::AllIn;
        });
        if (found == effective.end()) ++stats.allin_representation_failures;
    }
    if (active_mask[static_cast<std::size_t>(UniversalActionSlotV2::MinRaise)] &&
        (legal.bet || legal.raise) && legal.min_raise_to < legal.max_raise_to) {
        const auto found = std::find_if(effective.begin(), effective.end(), [&](const auto& action) {
            return action.slot == UniversalActionSlotV2::MinRaise &&
                   (action.exact.type == ExactActionType::BetTo || action.exact.type == ExactActionType::RaiseTo) &&
                   action.exact.amount_to == legal.min_raise_to;
        });
        if (found == effective.end()) ++stats.min_representation_failures;
    }
    (void)raw_fractional;
}

std::vector<EpisodeScenario> hu_scenarios() {
    std::vector<EpisodeScenario> out;
    const std::array<std::array<int,3>,3> profiles{{
        {{0,750,750}}, {{0,500,1000}}, {{0,1000,500}}
    }};
    for (const auto& stacks : profiles) {
        for (int dealer : {1,2}) {
            EpisodeScenario s{};
            s.state.total_chips = 1500;
            s.state.game_is_hu = true;
            s.state.blind_index = 0;
            s.state.small_blind = 10;
            s.state.big_blind = 20;
            s.state.stacks = stacks;
            s.state.dead_players = {0,-1,-1};
            s.state.dead_player_count = 1;
            s.dealer_id = dealer;
            out.push_back(s);
        }
    }
    return out;
}

std::vector<EpisodeScenario> three_handed_scenarios() {
    std::vector<EpisodeScenario> out;
    const std::array<std::array<int,3>,5> profiles{{
        {{500,500,500}}, {{250,500,750}}, {{250,750,500}}, {{500,250,750}}, {{750,250,500}}
    }};
    for (const auto& stacks : profiles) {
        for (int dealer : {0,1,2}) {
            EpisodeScenario s{};
            s.state.total_chips = 1500;
            s.state.game_is_hu = false;
            s.state.blind_index = 0;
            s.state.small_blind = 10;
            s.state.big_blind = 20;
            s.state.stacks = stacks;
            s.dealer_id = dealer;
            out.push_back(s);
        }
    }
    return out;
}

std::uint64_t deck_seed(std::uint64_t evaluation_seed, std::uint64_t scenario_index, std::uint64_t trajectory_index) {
    return evaluation_seed * 1000003ULL + scenario_index * 10007ULL + trajectory_index * 97ULL + 0x754AULL;
}

std::uint64_t path_seed(std::uint64_t evaluation_seed, std::uint64_t scenario_index, std::uint64_t trajectory_index) {
    return evaluation_seed ^ (scenario_index * 0x9E3779B1ULL) ^ (trajectory_index * 0x45D9F3BULL);
}

void audit_domain(
    const std::string& domain,
    const std::vector<EpisodeScenario>& scenarios,
    DomainStats& stats) {
    const std::array<std::uint64_t,2> evaluation_seeds{{1412935182ULL,816404962ULL}};
    const auto candidate_defs = candidates();
    for (const auto& candidate : candidate_defs) {
        stats.candidates.emplace(candidate.id, CandidateStats{});
    }
    const auto dense = dense_path_mask();

    for (auto evaluation_seed : evaluation_seeds) {
        for (std::size_t scenario_index = 0; scenario_index < scenarios.size(); ++scenario_index) {
            for (std::uint64_t trajectory_index = 0; trajectory_index < 64; ++trajectory_index) {
                ++stats.trajectories;
                SpinTraversalState state(
                    scenarios[scenario_index],
                    deck_seed(evaluation_seed, scenario_index, trajectory_index));
                std::mt19937_64 rng(path_seed(evaluation_seed, scenario_index, trajectory_index));
                int guard = 0;
                while (!state.terminal() && guard++ < 128) {
                    ++stats.visited_states;
                    ++stats.street_counts[static_cast<int>(state.hand().betting().street())];
                    audit_control_equivalence(state, stats);
                    for (const auto& candidate : candidate_defs) {
                        audit_candidate(state, candidate, stats.candidates.at(candidate.id));
                    }

                    const auto actions = resolve_universal_actions_v2(state.hand().betting(), dense);
                    if (actions.empty()) {
                        ++stats.nonterminal_stalls;
                        break;
                    }
                    const auto& chosen = actions[static_cast<std::size_t>(rng() % actions.size())];
                    try {
                        state.apply_exact(chosen.exact);
                    } catch (...) {
                        ++stats.nonterminal_stalls;
                        break;
                    }
                }
                if (state.terminal()) {
                    ++stats.terminal_trajectories;
                    try {
                        const auto chips = state.terminal_chip_delta();
                        if (chips[0] + chips[1] + chips[2] != 0) {
                            ++stats.terminal_settlement_failures;
                        }
                        (void)state.terminal_icm_delta(PayoutProfile{{0.5,0.3,0.2}});
                    } catch (...) {
                        ++stats.terminal_settlement_failures;
                    }
                } else if (guard >= 128) {
                    ++stats.nonterminal_stalls;
                }
            }
        }
    }
    (void)domain;
}

void write_map(std::ostream& out, const std::map<std::string,std::uint64_t>& values) {
    out << "{";
    bool first = true;
    for (const auto& [key,value] : values) {
        if (!first) out << ",";
        first = false;
        out << "\"" << key << "\":" << value;
    }
    out << "}";
}

void write_domain(std::ostream& out, const std::string& name, const DomainStats& stats) {
    out << "\"" << name << "\":{";
    out << "\"trajectories\":" << stats.trajectories << ",";
    out << "\"visited_states\":" << stats.visited_states << ",";
    out << "\"terminal_trajectories\":" << stats.terminal_trajectories << ",";
    out << "\"nonterminal_stalls\":" << stats.nonterminal_stalls << ",";
    out << "\"terminal_settlement_failures\":" << stats.terminal_settlement_failures << ",";
    out << "\"control_equivalence_checks\":" << stats.control_equivalence_checks << ",";
    out << "\"control_equivalence_differences\":" << stats.control_equivalence_differences << ",";
    out << "\"street_counts\":{";
    bool first_street = true;
    for (const auto& [street,count] : stats.street_counts) {
        if (!first_street) out << ",";
        first_street = false;
        out << "\"" << street << "\":" << count;
    }
    out << "},\"candidates\":{";
    bool first_candidate = true;
    for (const auto& [id,s] : stats.candidates) {
        if (!first_candidate) out << ",";
        first_candidate = false;
        const double alias_fraction = s.nominal_aggressive
            ? static_cast<double>(s.aliases_suppressed) / static_cast<double>(s.nominal_aggressive)
            : 0.0;
        const double clamp_min_fraction = s.fractional_nominal
            ? static_cast<double>(s.fractional_clamped_min) / static_cast<double>(s.fractional_nominal)
            : 0.0;
        const double clamp_allin_fraction = s.fractional_nominal
            ? static_cast<double>(s.fractional_clamped_allin) / static_cast<double>(s.fractional_nominal)
            : 0.0;
        out << "\"" << id << "\":{";
        out << "\"states\":" << s.states << ",";
        out << "\"nominal_aggressive\":" << s.nominal_aggressive << ",";
        out << "\"effective_aggressive\":" << s.effective_aggressive << ",";
        out << "\"aliases_suppressed\":" << s.aliases_suppressed << ",";
        out << "\"alias_suppression_fraction\":" << std::setprecision(17) << alias_fraction << ",";
        out << "\"fractional_nominal\":" << s.fractional_nominal << ",";
        out << "\"fractional_clamped_min\":" << s.fractional_clamped_min << ",";
        out << "\"fractional_clamped_min_fraction\":" << clamp_min_fraction << ",";
        out << "\"fractional_clamped_allin\":" << s.fractional_clamped_allin << ",";
        out << "\"fractional_clamped_allin_fraction\":" << clamp_allin_fraction << ",";
        out << "\"exact_application_failures\":" << s.exact_application_failures << ",";
        out << "\"duplicates_after_dedup\":" << s.duplicates_after_dedup << ",";
        out << "\"monotonicity_violations\":" << s.monotonicity_violations << ",";
        out << "\"allin_representation_failures\":" << s.allin_representation_failures << ",";
        out << "\"min_representation_failures\":" << s.min_representation_failures << ",";
        out << "\"raise_target_over_pot_after_call\":{";
        out << "\"count\":" << s.raise_ratio_count << ",";
        if (s.raise_ratio_count) {
            out << "\"mean\":" << (s.raise_ratio_sum / static_cast<double>(s.raise_ratio_count)) << ",";
            out << "\"min\":" << s.raise_ratio_min << ",";
            out << "\"max\":" << s.raise_ratio_max;
        } else {
            out << "\"mean\":null,\"min\":null,\"max\":null";
        }
        out << "},\"alias_by_spr\":";
        write_map(out, s.alias_by_spr);
        out << ",\"alias_by_effective_stack\":";
        write_map(out, s.alias_by_effective_stack);
        out << "}";
    }
    out << "}}";
}

bool pass_domain(const DomainStats& stats) {
    if (stats.nonterminal_stalls != 0 || stats.terminal_settlement_failures != 0 ||
        stats.control_equivalence_differences != 0) {
        return false;
    }
    for (const auto& [id,s] : stats.candidates) {
        if (s.states < 1000 || s.exact_application_failures != 0 || s.duplicates_after_dedup != 0 ||
            s.monotonicity_violations != 0 || s.allin_representation_failures != 0 ||
            s.min_representation_failures != 0) {
            return false;
        }
        (void)id;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::string out_path;
    if (argc == 3 && std::string(argv[1]) == "--out") {
        out_path = argv[2];
    } else if (argc != 1) {
        std::cerr << "usage: spincore_r7_5_action_structural_audit [--out path]\n";
        return 64;
    }

    DomainStats hu;
    DomainStats three;
    try {
        audit_domain("TRUE_HEADS_UP", hu_scenarios(), hu);
        audit_domain("THREE_HANDED", three_handed_scenarios(), three);
    } catch (const std::exception& e) {
        std::cerr << "structural audit exception: " << e.what() << "\n";
        return 3;
    }

    const bool passed = pass_domain(hu) && pass_domain(three);
    std::ostringstream json;
    json << "{";
    json << "\"schema\":\"SPINCORE_R7_5_4_ACTION_STRUCTURAL_AUDIT_V1\",";
    json << "\"freeze_schema\":\"SPINCORE_R7_5_4_STRUCTURAL_AUDIT_FREEZE_V1\",";
    json << "\"authoritative_action_precommit\":\"SPINCORE_R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V2\",";
    json << "\"path_policy_is_strategic_evidence\":false,";
    json << "\"domains\":{";
    write_domain(json, "TRUE_HEADS_UP", hu);
    json << ",";
    write_domain(json, "THREE_HANDED", three);
    json << "},";
    json << "\"structural_gate_pass\":" << (passed ? "true" : "false") << ",";
    json << "\"production_training_authorized\":false,";
    json << "\"ready_for_tables\":false";
    json << "}\n";

    if (!out_path.empty()) {
        std::ofstream file(out_path);
        if (!file) {
            std::cerr << "cannot open output path: " << out_path << "\n";
            return 65;
        }
        file << json.str();
    }
    std::cout << json.str();
    return passed ? 0 : 2;
}
