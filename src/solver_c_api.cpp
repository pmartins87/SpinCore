#define SPINCORE_SOLVER_C_EXPORTS
#include "spincore/solver_c_api.h"
#include "spincore/spin_traversal_state.hpp"
#include "spincore/neural_encoder_v2.hpp"
#include "spincore/neural_encoder_v3.hpp"
#include "spincore/action_abstraction_v2.hpp"
#include "spincore/game_topology.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using spincore::AbstractActionSlot;
using spincore::Card;
using spincore::EpisodeScenario;
using spincore::PayoutProfile;
using spincore::SpinTraversalState;
using spincore::StrategyDomain;
using spincore::UniversalActionMaskV2;
using spincore::UniversalActionSlotV2;

struct spincore_solver_state {
    SpinTraversalState impl;
    explicit spincore_solver_state(SpinTraversalState s) : impl(std::move(s)) {}
};

struct spincore_solver_frontier {
    std::vector<SpinTraversalState> states;
    std::vector<std::uint8_t> terminal;
    std::size_t nodes_visited{0};
    std::size_t max_depth_reached{0};
};

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

static EpisodeScenario decode(const spincore_solver_scenario_v2& i) {
    EpisodeScenario s{};
    s.state.total_chips = i.total_chips;
    s.state.game_is_hu = i.game_is_hu != 0;
    s.state.blind_index = i.blind_index;
    s.state.small_blind = i.small_blind;
    s.state.big_blind = i.big_blind;
    s.state.stacks = {i.stack_0, i.stack_1, i.stack_2};
    s.state.dead_players = {i.dead_player_0, i.dead_player_1, -1};
    s.state.dead_player_count = i.dead_player_count;
    s.dealer_id = i.dealer_id;
    return s;
}

struct DecodedExplicitDeal {
    std::array<std::array<Card,2>,3> holes{};
    std::array<Card,5> board{};
};

static DecodedExplicitDeal decode_deal(const EpisodeScenario& scenario, const spincore_solver_deal_v1& d) {
    const int hole_ids[3][2] = {
        {d.hole_0_0, d.hole_0_1},
        {d.hole_1_0, d.hole_1_1},
        {d.hole_2_0, d.hole_2_1},
    };
    const int board_ids[5] = {d.board_0, d.board_1, d.board_2, d.board_3, d.board_4};
    DecodedExplicitDeal out{};
    const auto topo = spincore::make_game_topology(scenario);
    for (int seat = 0; seat < 3; ++seat) {
        bool live = false;
        for (int j = 0; j < topo.live_count; ++j) {
            if (topo.live[static_cast<std::size_t>(j)] == seat) { live = true; break; }
        }
        for (int r = 0; r < 2; ++r) {
            const int id = hole_ids[seat][r];
            if (!live) {
                if (id != -1) throw std::invalid_argument("dead-seat explicit hole id must be -1");
                out.holes[static_cast<std::size_t>(seat)][static_cast<std::size_t>(r)] = Card{};
            } else {
                if (id < 0 || id >= 52) throw std::invalid_argument("live-seat explicit hole id outside 0..51");
                out.holes[static_cast<std::size_t>(seat)][static_cast<std::size_t>(r)] = spincore::card_from_id(static_cast<std::uint8_t>(id));
            }
        }
    }
    for (int i = 0; i < 5; ++i) {
        const int id = board_ids[i];
        if (id < 0 || id >= 52) throw std::invalid_argument("explicit board id outside 0..51");
        out.board[static_cast<std::size_t>(i)] = spincore::card_from_id(static_cast<std::uint8_t>(id));
    }
    return out;
}

static UniversalActionMaskV2 decode_universal_mask(uint32_t mask) {
    UniversalActionMaskV2 out{};
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = (mask & (1u << static_cast<uint32_t>(i))) ? 1u : 0u;
    }
    return out;
}

static void expand(
    const SpinTraversalState& s,
    int target,
    std::size_t depth,
    std::size_t max_nodes,
    std::size_t max_depth,
    spincore_solver_frontier& o
) {
    if (++o.nodes_visited > max_nodes) throw std::runtime_error("native frontier node cap exceeded");
    o.max_depth_reached = std::max(o.max_depth_reached, depth);
    if (depth > max_depth) throw std::runtime_error("native frontier depth cap exceeded");
    if (s.terminal()) {
        o.states.push_back(s);
        o.terminal.push_back(1);
        return;
    }
    if (s.actor() == target) {
        o.states.push_back(s);
        o.terminal.push_back(0);
        return;
    }
    auto legal = s.legal_abstract_actions();
    if (legal.empty()) throw std::runtime_error("nonterminal frontier node without legal action");
    for (auto a : legal) {
        expand(s.child(a), target, depth + 1, max_nodes, max_depth, o);
    }
}

extern "C" {

int32_t spincore_solver_c_abi_version(void) { return 2; }
const char* spincore_solver_last_error(void) { return g_error.c_str(); }

spincore_solver_state* spincore_solver_state_create_v2(
    const spincore_solver_scenario_v2* s,
    uint64_t seed
) {
    return guard([&]() -> spincore_solver_state* {
        if (!s) throw std::invalid_argument("null scenario");
        return new spincore_solver_state(SpinTraversalState(decode(*s), seed));
    }, static_cast<spincore_solver_state*>(nullptr));
}

spincore_solver_state* spincore_solver_state_create_v2_deal(
    const spincore_solver_scenario_v2* s,
    const spincore_solver_deal_v1* d
) {
    return guard([&]() -> spincore_solver_state* {
        if (!s || !d) throw std::invalid_argument("null explicit-deal arguments");
        const auto scenario = decode(*s);
        const auto deal = decode_deal(scenario, *d);
        return new spincore_solver_state(SpinTraversalState(scenario, deal.holes, deal.board));
    }, static_cast<spincore_solver_state*>(nullptr));
}

int32_t spincore_solver_state_deal_snapshot_v1(
    const spincore_solver_state* s,
    spincore_solver_deal_v1* out,
    int32_t* visible_board_count
) {
    return guard([&]() {
        if (!s || !out || !visible_board_count) throw std::invalid_argument("null deal-snapshot arguments");
        const auto& holes = s->impl.hand().hole_cards();
        const auto& board = s->impl.hand().board();
        auto id_or_minus_one = [](const Card& c) -> int32_t { return c.valid() ? static_cast<int32_t>(c.id()) : -1; };
        out->hole_0_0 = id_or_minus_one(holes[0][0]); out->hole_0_1 = id_or_minus_one(holes[0][1]);
        out->hole_1_0 = id_or_minus_one(holes[1][0]); out->hole_1_1 = id_or_minus_one(holes[1][1]);
        out->hole_2_0 = id_or_minus_one(holes[2][0]); out->hole_2_1 = id_or_minus_one(holes[2][1]);
        out->board_0 = static_cast<int32_t>(board[0].id()); out->board_1 = static_cast<int32_t>(board[1].id());
        out->board_2 = static_cast<int32_t>(board[2].id()); out->board_3 = static_cast<int32_t>(board[3].id());
        out->board_4 = static_cast<int32_t>(board[4].id());
        *visible_board_count = static_cast<int32_t>(s->impl.hand().visible_board_count());
        return 0;
    }, -1);
}

spincore_solver_state* spincore_solver_state_clone(const spincore_solver_state* s) {
    return guard([&]() -> spincore_solver_state* {
        if (!s) throw std::invalid_argument("null state");
        return new spincore_solver_state(s->impl);
    }, static_cast<spincore_solver_state*>(nullptr));
}

void spincore_solver_state_destroy(spincore_solver_state* s) { delete s; }

int32_t spincore_solver_state_terminal(const spincore_solver_state* s) {
    return s ? (s->impl.terminal() ? 1 : 0) : 1;
}

int32_t spincore_solver_state_actor(const spincore_solver_state* s) {
    return (!s || s->impl.terminal()) ? -1 : s->impl.actor();
}

int32_t spincore_solver_state_domain(const spincore_solver_state* s) {
    if (!s) return -1;
    return spincore::strategy_domain(s->impl.hand().betting().topology()) == StrategyDomain::TrueHeadsUp ? 1 : 0;
}

uint32_t spincore_solver_state_legal_mask(const spincore_solver_state* s) {
    return guard([&]() {
        if (!s || s->impl.terminal()) return 0u;
        uint32_t m = 0;
        for (auto a : s->impl.legal_abstract_actions()) m |= 1u << static_cast<uint32_t>(a);
        return m;
    }, 0u);
}

int32_t spincore_solver_state_apply_abstract(spincore_solver_state* s, int32_t a) {
    return guard([&]() {
        if (!s || a < 0 || a > 5) throw std::invalid_argument("bad action");
        s->impl.apply(static_cast<AbstractActionSlot>(a));
        return 0;
    }, -1);
}

uint32_t spincore_solver_state_universal_legal_mask(
    const spincore_solver_state* s,
    uint32_t active_mask
) {
    return guard([&]() {
        if (!s || s->impl.terminal()) return 0u;
        const auto active = decode_universal_mask(active_mask);
        const auto actions = spincore::resolve_universal_actions_v2(s->impl.hand().betting(), active);
        uint32_t m = 0;
        for (const auto& a : actions) m |= 1u << static_cast<uint32_t>(a.slot);
        return m;
    }, 0u);
}

int32_t spincore_solver_state_apply_universal(
    spincore_solver_state* s,
    uint32_t active_mask,
    int32_t action_slot
) {
    return guard([&]() {
        if (!s || s->impl.terminal() || action_slot < 0 ||
            action_slot >= static_cast<int32_t>(spincore::kUniversalActionCountV2)) {
            throw std::invalid_argument("bad universal action");
        }
        const auto active = decode_universal_mask(active_mask);
        const auto actions = spincore::resolve_universal_actions_v2(s->impl.hand().betting(), active);
        const auto wanted = static_cast<UniversalActionSlotV2>(action_slot);
        const auto it = std::find_if(actions.begin(), actions.end(), [&](const auto& a) {
            return a.slot == wanted;
        });
        if (it == actions.end()) {
            throw std::invalid_argument("universal action is inactive, illegal, or state-local alias");
        }
        s->impl.apply_exact(it->exact);
        return 0;
    }, -1);
}

int32_t spincore_solver_state_resolve_universal_exact(
    const spincore_solver_state* s,
    uint32_t active_mask,
    int32_t action_slot,
    int32_t* out_type,
    int32_t* out_amount_to
) {
    return guard([&]() {
        if (!s || s->impl.terminal() || !out_type || !out_amount_to || action_slot < 0 ||
            action_slot >= static_cast<int32_t>(spincore::kUniversalActionCountV2)) {
            throw std::invalid_argument("bad universal exact-resolution arguments");
        }
        const auto active = decode_universal_mask(active_mask);
        const auto actions = spincore::resolve_universal_actions_v2(s->impl.hand().betting(), active);
        const auto wanted = static_cast<UniversalActionSlotV2>(action_slot);
        const auto it = std::find_if(actions.begin(), actions.end(), [&](const auto& a) {
            return a.slot == wanted;
        });
        if (it == actions.end()) {
            throw std::invalid_argument("universal action is inactive, illegal, or state-local alias");
        }
        *out_type = static_cast<int32_t>(it->exact.type);
        *out_amount_to = it->exact.amount_to;
        return 0;
    }, -1);
}

size_t spincore_solver_state_neural_input(
    const spincore_solver_state* s,
    uint8_t* out,
    size_t cap
) {
    return guard([&]() {
        if (!s || s->impl.terminal()) throw std::logic_error("no neural input");
        auto b = spincore::serialize_neural_input_v1(s->impl.neural_input());
        if (!out || !cap) return b.size();
        if (cap < b.size()) throw std::invalid_argument("neural buffer too small");
        std::memcpy(out, b.data(), b.size());
        return b.size();
    }, static_cast<size_t>(0));
}

size_t spincore_solver_state_neural_input_v2(
    const spincore_solver_state* s,
    uint8_t* out,
    size_t cap
) {
    return guard([&]() {
        if (!s || s->impl.terminal()) throw std::logic_error("no neural V2 input");
        auto b = spincore::serialize_neural_input_v2(
            spincore::encode_neural_input_v2(s->impl.hand(), s->impl.blind_index())
        );
        if (!out || !cap) return b.size();
        if (cap < b.size()) throw std::invalid_argument("neural V2 buffer too small");
        std::memcpy(out, b.data(), b.size());
        return b.size();
    }, static_cast<size_t>(0));
}

size_t spincore_solver_state_neural_input_v3(
    const spincore_solver_state* s,
    uint8_t* out,
    size_t cap
) {
    return guard([&]() {
        if (!s || s->impl.terminal()) throw std::logic_error("no neural V3 input");
        auto b = spincore::serialize_neural_input_v3(
            spincore::encode_neural_input_v3(s->impl.hand(), s->impl.blind_index())
        );
        if (!out || !cap) return b.size();
        if (cap < b.size()) throw std::invalid_argument("neural V3 buffer too small");
        std::memcpy(out, b.data(), b.size());
        return b.size();
    }, static_cast<size_t>(0));
}

int32_t spincore_solver_state_terminal_chip_delta(const spincore_solver_state* s, int32_t out[3]) {
    return guard([&]() {
        if (!s || !out) throw std::invalid_argument("null");
        auto d = s->impl.terminal_chip_delta();
        for (int i = 0; i < 3; ++i) out[i] = d[static_cast<std::size_t>(i)];
        return 0;
    }, -1);
}

int32_t spincore_solver_state_terminal_icm_delta(
    const spincore_solver_state* s,
    const double p[3],
    double out[3]
) {
    return guard([&]() {
        if (!s || !p || !out) throw std::invalid_argument("null");
        auto d = s->impl.terminal_icm_delta(PayoutProfile{{p[0], p[1], p[2]}});
        for (int i = 0; i < 3; ++i) out[i] = d[static_cast<std::size_t>(i)];
        return 0;
    }, -1);
}

spincore_solver_frontier* spincore_solver_frontier_create_until_actor(
    const spincore_solver_state* r,
    int32_t target,
    size_t mn,
    size_t md
) {
    return guard([&]() -> spincore_solver_frontier* {
        if (!r || target < 0 || target > 2 || !mn || !md) {
            throw std::invalid_argument("bad frontier args");
        }
        auto o = std::make_unique<spincore_solver_frontier>();
        expand(r->impl, target, 0, mn, md, *o);
        return o.release();
    }, static_cast<spincore_solver_frontier*>(nullptr));
}

void spincore_solver_frontier_destroy(spincore_solver_frontier* f) { delete f; }
size_t spincore_solver_frontier_size(const spincore_solver_frontier* f) { return f ? f->states.size() : 0; }
size_t spincore_solver_frontier_nodes_visited(const spincore_solver_frontier* f) { return f ? f->nodes_visited : 0; }
size_t spincore_solver_frontier_max_depth_reached(const spincore_solver_frontier* f) { return f ? f->max_depth_reached : 0; }
int32_t spincore_solver_frontier_is_terminal(const spincore_solver_frontier* f, size_t i) {
    return (!f || i >= f->terminal.size()) ? -1 : (f->terminal[i] ? 1 : 0);
}

spincore_solver_state* spincore_solver_frontier_clone_state(
    const spincore_solver_frontier* f,
    size_t i
) {
    return guard([&]() -> spincore_solver_state* {
        if (!f || i >= f->states.size()) throw std::out_of_range("frontier index");
        return new spincore_solver_state(f->states[i]);
    }, static_cast<spincore_solver_state*>(nullptr));
}

}  // extern "C"
