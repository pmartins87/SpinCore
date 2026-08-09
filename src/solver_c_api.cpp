#define SPINCORE_SOLVER_C_EXPORTS
#include "spincore/solver_c_api.h"
#include "spincore/spin_traversal_state.hpp"
#include "spincore/game_topology.hpp"

#include <cstring>
#include <exception>
#include <memory>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>

using spincore::AbstractActionSlot;
using spincore::EpisodeScenario;
using spincore::PayoutProfile;
using spincore::SpinTraversalState;
using spincore::StrategyDomain;

struct spincore_solver_state {
    SpinTraversalState impl;
    explicit spincore_solver_state(SpinTraversalState s): impl(std::move(s)) {}
};

struct spincore_solver_frontier {
    std::vector<SpinTraversalState> states;
    std::vector<std::uint8_t> terminal;
    std::size_t nodes_visited{0};
    std::size_t max_depth_reached{0};
};

static void expand_until_actor_impl(
    const SpinTraversalState& state,
    std::int32_t target_actor,
    std::size_t depth,
    std::size_t max_nodes,
    std::size_t max_depth,
    spincore_solver_frontier& out
) {
    if (++out.nodes_visited > max_nodes) {
        throw std::runtime_error("native frontier node cap exceeded");
    }
    out.max_depth_reached = std::max(out.max_depth_reached, depth);
    if (depth > max_depth) {
        throw std::runtime_error("native frontier depth cap exceeded");
    }
    if (state.terminal()) {
        out.states.push_back(state);
        out.terminal.push_back(1);
        return;
    }
    if (state.actor() == target_actor) {
        out.states.push_back(state);
        out.terminal.push_back(0);
        return;
    }
    const auto legal = state.legal_abstract_actions();
    if (legal.empty()) throw std::runtime_error("nonterminal frontier node without legal action");
    for (const auto action : legal) {
        const auto child = state.child(action);
        expand_until_actor_impl(child, target_actor, depth + 1, max_nodes, max_depth, out);
    }
}

static thread_local std::string g_error;

static void set_error(const char* s) { g_error = s ? s : "unknown error"; }

template<class F, class R>
static R guard(F&& fn, R fail) noexcept {
    try {
        g_error.clear();
        return fn();
    } catch (const std::exception& e) {
        set_error(e.what());
        return fail;
    } catch (...) {
        set_error("non-standard exception");
        return fail;
    }
}

static EpisodeScenario decode_scenario(const spincore_solver_scenario_v2& in) {
    EpisodeScenario s{};
    s.state.total_chips = in.total_chips;
    s.state.game_is_hu = in.game_is_hu != 0;
    s.state.blind_index = in.blind_index;
    s.state.small_blind = in.small_blind;
    s.state.big_blind = in.big_blind;
    s.state.stacks = {in.stack_0, in.stack_1, in.stack_2};
    s.dealer_id = in.dealer_id;

    if (in.dead_player_count < 0 || in.dead_player_count > 2) {
        throw std::invalid_argument("dead_player_count outside ABI v2 bounds");
    }
    s.state.dead_player_count = in.dead_player_count;
    s.state.dead_players = {-1,-1,-1};
    if (in.dead_player_count >= 1) s.state.dead_players[0] = in.dead_player_0;
    if (in.dead_player_count >= 2) s.state.dead_players[1] = in.dead_player_1;
    return s;
}

extern "C" {

int32_t spincore_solver_c_abi_version(void) {
    return SPINCORE_SOLVER_C_ABI_VERSION;
}

const char* spincore_solver_last_error(void) {
    return g_error.c_str();
}

spincore_solver_state* spincore_solver_state_create_v2(
    const spincore_solver_scenario_v2* scenario,
    uint64_t deck_seed
) {
    return guard([&]() -> spincore_solver_state* {
        if (!scenario) throw std::invalid_argument("null scenario");
        return new spincore_solver_state(SpinTraversalState(decode_scenario(*scenario), deck_seed));
    }, static_cast<spincore_solver_state*>(nullptr));
}

spincore_solver_state* spincore_solver_state_clone(const spincore_solver_state* state) {
    return guard([&]() -> spincore_solver_state* {
        if (!state) throw std::invalid_argument("null state");
        return new spincore_solver_state(state->impl);
    }, static_cast<spincore_solver_state*>(nullptr));
}

void spincore_solver_state_destroy(spincore_solver_state* state) {
    delete state;
}

int32_t spincore_solver_state_terminal(const spincore_solver_state* state) {
    return state ? (state->impl.terminal() ? 1 : 0) : 1;
}

int32_t spincore_solver_state_actor(const spincore_solver_state* state) {
    if (!state || state->impl.terminal()) return -1;
    return state->impl.actor();
}

int32_t spincore_solver_state_domain(const spincore_solver_state* state) {
    if (!state) return -1;
    return spincore::strategy_domain(state->impl.hand().betting().topology()) == StrategyDomain::TrueHeadsUp ? 1 : 0;
}

uint32_t spincore_solver_state_legal_mask(const spincore_solver_state* state) {
    return guard([&]() -> uint32_t {
        if (!state || state->impl.terminal()) return 0;
        uint32_t mask = 0;
        for (const auto a : state->impl.legal_abstract_actions()) {
            mask |= (1u << static_cast<uint32_t>(a));
        }
        return mask;
    }, 0u);
}

int32_t spincore_solver_state_apply_abstract(spincore_solver_state* state, int32_t abstract_action) {
    return guard([&]() -> int32_t {
        if (!state) throw std::invalid_argument("null state");
        if (abstract_action < 0 || abstract_action > static_cast<int32_t>(AbstractActionSlot::AllIn)) {
            throw std::invalid_argument("bad abstract action");
        }
        (void)state->impl.apply(static_cast<AbstractActionSlot>(abstract_action));
        return 0;
    }, -1);
}

size_t spincore_solver_state_neural_input(
    const spincore_solver_state* state,
    uint8_t* out,
    size_t capacity
) {
    return guard([&]() -> size_t {
        if (!state) throw std::invalid_argument("null state");
        if (state->impl.terminal()) throw std::logic_error("terminal state has no neural input");
        const auto bytes = spincore::serialize_neural_input_v1(state->impl.neural_input());
        if (!out || capacity == 0) return bytes.size();
        if (capacity < bytes.size()) throw std::invalid_argument("neural buffer too small");
        std::memcpy(out, bytes.data(), bytes.size());
        return bytes.size();
    }, static_cast<size_t>(0));
}

int32_t spincore_solver_state_terminal_chip_delta(
    const spincore_solver_state* state,
    int32_t out_delta[3]
) {
    return guard([&]() -> int32_t {
        if (!state || !out_delta) throw std::invalid_argument("null terminal chip delta argument");
        const auto d = state->impl.terminal_chip_delta();
        out_delta[0]=d[0]; out_delta[1]=d[1]; out_delta[2]=d[2];
        return 0;
    }, -1);
}

int32_t spincore_solver_state_terminal_icm_delta(
    const spincore_solver_state* state,
    const double payout_by_place[3],
    double out_delta[3]
) {
    return guard([&]() -> int32_t {
        if (!state || !payout_by_place || !out_delta) {
            throw std::invalid_argument("null terminal ICM argument");
        }
        const PayoutProfile payout{{payout_by_place[0],payout_by_place[1],payout_by_place[2]}};
        const auto d = state->impl.terminal_icm_delta(payout);
        out_delta[0]=d[0]; out_delta[1]=d[1]; out_delta[2]=d[2];
        return 0;
    }, -1);
}

spincore_solver_frontier* spincore_solver_frontier_create_until_actor(
    const spincore_solver_state* root,
    int32_t target_actor,
    size_t max_nodes,
    size_t max_depth
) {
    return guard([&]() -> spincore_solver_frontier* {
        if (!root) throw std::invalid_argument("null frontier root");
        if (target_actor < 0 || target_actor > 2) throw std::invalid_argument("target_actor outside seat range");
        if (max_nodes == 0 || max_depth == 0) throw std::invalid_argument("frontier caps must be positive");
        auto out = std::make_unique<spincore_solver_frontier>();
        expand_until_actor_impl(root->impl, target_actor, 0, max_nodes, max_depth, *out);
        if (out->states.size() != out->terminal.size()) throw std::logic_error("frontier internal size mismatch");
        return out.release();
    }, static_cast<spincore_solver_frontier*>(nullptr));
}

void spincore_solver_frontier_destroy(spincore_solver_frontier* frontier) { delete frontier; }

size_t spincore_solver_frontier_size(const spincore_solver_frontier* frontier) {
    return frontier ? frontier->states.size() : 0;
}

size_t spincore_solver_frontier_nodes_visited(const spincore_solver_frontier* frontier) {
    return frontier ? frontier->nodes_visited : 0;
}

size_t spincore_solver_frontier_max_depth_reached(const spincore_solver_frontier* frontier) {
    return frontier ? frontier->max_depth_reached : 0;
}

int32_t spincore_solver_frontier_is_terminal(const spincore_solver_frontier* frontier, size_t index) {
    if (!frontier || index >= frontier->terminal.size()) return -1;
    return frontier->terminal[index] ? 1 : 0;
}

spincore_solver_state* spincore_solver_frontier_clone_state(const spincore_solver_frontier* frontier, size_t index) {
    return guard([&]() -> spincore_solver_state* {
        if (!frontier || index >= frontier->states.size()) throw std::out_of_range("frontier index");
        return new spincore_solver_state(frontier->states[index]);
    }, static_cast<spincore_solver_state*>(nullptr));
}

} // extern "C"
