// Authoritative R7.5.4 structural-audit executor.
//
// The V1 implementation is included as a library-like translation unit so its
// already regression-compiled resolver/candidate/statistics code remains
// unchanged. We replace only the evaluation-seed stream and evidence schema,
// as required by PRECOMMIT_V3 + STRUCTURAL_AUDIT_FREEZE_V2.
#define main spincore_r7_5_action_structural_audit_v1_disabled_main
#include "r7_5_action_structural_audit.cpp"
#undef main

namespace {

void audit_domain_v2(
    const std::vector<EpisodeScenario>& scenarios,
    DomainStats& stats) {
    // Mechanically derived by
    // SHA256("SpinCore|R7.5.4|paired_evaluation|<index>|" +
    //        "ba2eab3c51cdf86057c18dd160cb6febe8cc60f7"),
    // first 8 bytes big-endian & 0x7fffffff.
    const std::array<std::uint64_t,2> evaluation_seeds{{1817694185ULL,1617273629ULL}};
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
        audit_domain_v2(hu_scenarios(), hu);
        audit_domain_v2(three_handed_scenarios(), three);
    } catch (const std::exception& e) {
        std::cerr << "structural audit exception: " << e.what() << "\n";
        return 3;
    }

    const bool passed = pass_domain(hu) && pass_domain(three);
    std::ostringstream json;
    json << "{";
    json << "\"schema\":\"SPINCORE_R7_5_4_ACTION_STRUCTURAL_AUDIT_V2\",";
    json << "\"freeze_schema\":\"SPINCORE_R7_5_4_STRUCTURAL_AUDIT_FREEZE_V2\",";
    json << "\"authoritative_action_precommit\":\"SPINCORE_R7_5_4_ACTION_ABSTRACTION_ABLATION_PRECOMMIT_V3\",";
    json << "\"evaluation_seeds\":[1817694185,1617273629],";
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
