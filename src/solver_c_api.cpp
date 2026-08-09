#define SPINCORE_SOLVER_C_EXPORTS
#include "spincore/solver_c_api.h"
#include "spincore/spin_traversal_state.hpp"
#include "spincore/game_topology.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

using spincore::AbstractActionSlot;
using spincore::EpisodeScenario;
using spincore::SpinTraversalState;
using spincore::StrategyDomain;

struct spincore_state { SpinTraversalState impl; explicit spincore_state(SpinTraversalState s): impl(std::move(s)) {} };
static thread_local std::string g_error;

static void set_error(const char* s) { g_error = s ? s : "unknown error"; }
template<class F, class R>
static R guard(F&& fn, R fail) noexcept {
    try { g_error.clear(); return fn(); }
    catch (const std::exception& e) { set_error(e.what()); return fail; }
    catch (...) { set_error("non-standard exception"); return fail; }
}

extern "C" {
int spincore_solver_abi_version(void) { return SPINCORE_SOLVER_C_ABI_VERSION; }

spincore_state* spincore_state_create(const spincore_episode_v1* ep, uint64_t seed) {
    return guard([&]() -> spincore_state* {
        if (!ep) throw std::invalid_argument("null episode");
        EpisodeScenario s{};
        s.state.total_chips = ep->total_chips;
        s.state.game_is_hu = ep->game_is_hu != 0;
        s.state.blind_index = ep->blind_index;
        s.state.small_blind = ep->small_blind;
        s.state.big_blind = ep->big_blind;
        s.state.stacks = {ep->stacks[0], ep->stacks[1], ep->stacks[2]};
        s.dealer_id = ep->dealer_id;
        if (s.state.game_is_hu) {
            for (int i=0;i<3;++i) if (s.state.stacks[static_cast<size_t>(i)]<=0) {
                s.state.dead_players[static_cast<size_t>(s.state.dead_player_count++)]=i;
            }
        }
        return new spincore_state(SpinTraversalState(s, seed));
    }, static_cast<spincore_state*>(nullptr));
}

spincore_state* spincore_state_clone(const spincore_state* st) {
    return guard([&]() -> spincore_state* {
        if (!st) throw std::invalid_argument("null state");
        return new spincore_state(st->impl);
    }, static_cast<spincore_state*>(nullptr));
}
void spincore_state_destroy(spincore_state* st) { delete st; }
int spincore_state_terminal(const spincore_state* st) { return st ? (st->impl.terminal()?1:0) : 1; }
int32_t spincore_state_actor(const spincore_state* st) { return st ? st->impl.actor() : -1; }
int spincore_state_domain(const spincore_state* st) {
    if (!st) return -1;
    return spincore::strategy_domain(st->impl.hand().betting().topology()) == StrategyDomain::TrueHeadsUp ? 1 : 0;
}
uint8_t spincore_state_legal_mask(const spincore_state* st) {
    return guard([&]() -> uint8_t {
        if (!st || st->impl.terminal()) return 0;
        uint8_t m=0;
        for (auto a: st->impl.legal_abstract_actions()) m |= static_cast<uint8_t>(1u << static_cast<uint8_t>(a));
        return m;
    }, static_cast<uint8_t>(0));
}
int spincore_state_apply(spincore_state* st, uint8_t action) {
    return guard([&]() -> int {
        if (!st) throw std::invalid_argument("null state");
        if (action > static_cast<uint8_t>(AbstractActionSlot::AllIn)) throw std::invalid_argument("bad abstract action");
        (void)st->impl.apply(static_cast<AbstractActionSlot>(action));
        return 0;
    }, -1);
}
size_t spincore_state_neural_size(const spincore_state* st) {
    return guard([&]() -> size_t {
        if (!st || st->impl.terminal()) return 0;
        return spincore::serialize_neural_input_v1(st->impl.neural_input()).size();
    }, static_cast<size_t>(0));
}
int spincore_state_neural_copy(const spincore_state* st, uint8_t* out, size_t n) {
    return guard([&]() -> int {
        if (!st || !out) throw std::invalid_argument("null neural copy argument");
        auto bytes=spincore::serialize_neural_input_v1(st->impl.neural_input());
        if (n < bytes.size()) throw std::invalid_argument("neural buffer too small");
        std::memcpy(out,bytes.data(),bytes.size());
        return static_cast<int>(bytes.size());
    }, -1);
}
int spincore_state_terminal_chip_delta(const spincore_state* st, int32_t out[3]) {
    return guard([&]() -> int {
        if (!st || !out) throw std::invalid_argument("null terminal delta argument");
        const auto d=st->impl.terminal_chip_delta();
        out[0]=d[0]; out[1]=d[1]; out[2]=d[2];
        return 0;
    }, -1);
}
const char* spincore_last_error(void) { return g_error.c_str(); }
}
