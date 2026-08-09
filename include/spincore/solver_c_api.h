#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#  ifdef SPINCORE_SOLVER_C_EXPORTS
#    define SPINCORE_SOLVER_C_API __declspec(dllexport)
#  else
#    define SPINCORE_SOLVER_C_API __declspec(dllimport)
#  endif
#else
#  define SPINCORE_SOLVER_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SPINCORE_SOLVER_C_ABI_VERSION 2

typedef struct spincore_solver_state spincore_solver_state;
typedef struct spincore_solver_frontier spincore_solver_frontier;

typedef struct spincore_solver_scenario_v2 {
    int32_t total_chips;
    int32_t game_is_hu;
    int32_t blind_index;
    int32_t small_blind;
    int32_t big_blind;
    int32_t stack_0;
    int32_t stack_1;
    int32_t stack_2;
    int32_t dead_player_0;
    int32_t dead_player_1;
    int32_t dead_player_count;
    int32_t dealer_id;
} spincore_solver_scenario_v2;

SPINCORE_SOLVER_C_API int32_t spincore_solver_c_abi_version(void);
SPINCORE_SOLVER_C_API const char* spincore_solver_last_error(void);

SPINCORE_SOLVER_C_API spincore_solver_state* spincore_solver_state_create_v2(
    const spincore_solver_scenario_v2* scenario,
    uint64_t deck_seed);
SPINCORE_SOLVER_C_API spincore_solver_state* spincore_solver_state_clone(
    const spincore_solver_state* state);
SPINCORE_SOLVER_C_API void spincore_solver_state_destroy(spincore_solver_state* state);

SPINCORE_SOLVER_C_API int32_t spincore_solver_state_terminal(const spincore_solver_state* state);
SPINCORE_SOLVER_C_API int32_t spincore_solver_state_actor(const spincore_solver_state* state);
SPINCORE_SOLVER_C_API int32_t spincore_solver_state_domain(const spincore_solver_state* state);
SPINCORE_SOLVER_C_API uint32_t spincore_solver_state_legal_mask(const spincore_solver_state* state);
SPINCORE_SOLVER_C_API int32_t spincore_solver_state_apply_abstract(
    spincore_solver_state* state,
    int32_t abstract_action);

/* Returns required byte size if out == NULL or capacity == 0. */
SPINCORE_SOLVER_C_API size_t spincore_solver_state_neural_input(
    const spincore_solver_state* state,
    uint8_t* out,
    size_t capacity);

/* Diagnostic chip delta. Production Deep CFR must use the explicit payout API below. */
SPINCORE_SOLVER_C_API int32_t spincore_solver_state_terminal_chip_delta(
    const spincore_solver_state* state,
    int32_t out_delta[3]);

/*
 * Exact ICM continuation utility V(after)-V(before).
 * payout_by_place is mandatory [1st,2nd,3rd]; no hidden default exists.
 * Ambiguous simultaneous elimination under unequal payouts fails closed.
 */
SPINCORE_SOLVER_C_API int32_t spincore_solver_state_terminal_icm_delta(
    const spincore_solver_state* state,
    const double payout_by_place[3],
    double out_delta[3]);

/*
 * Native R7.1 frontier: enumerate every non-target action until either the
 * target player is to act or a terminal state is reached. The object owns its
 * exact hidden-state copies. max_nodes/max_depth are hard fail-closed guards.
 */
SPINCORE_SOLVER_C_API spincore_solver_frontier* spincore_solver_frontier_create_until_actor(
    const spincore_solver_state* root,
    int32_t target_actor,
    size_t max_nodes,
    size_t max_depth);
SPINCORE_SOLVER_C_API void spincore_solver_frontier_destroy(spincore_solver_frontier* frontier);
SPINCORE_SOLVER_C_API size_t spincore_solver_frontier_size(const spincore_solver_frontier* frontier);
SPINCORE_SOLVER_C_API size_t spincore_solver_frontier_nodes_visited(const spincore_solver_frontier* frontier);
SPINCORE_SOLVER_C_API size_t spincore_solver_frontier_max_depth_reached(const spincore_solver_frontier* frontier);
SPINCORE_SOLVER_C_API int32_t spincore_solver_frontier_is_terminal(
    const spincore_solver_frontier* frontier, size_t index);
SPINCORE_SOLVER_C_API spincore_solver_state* spincore_solver_frontier_clone_state(
    const spincore_solver_frontier* frontier, size_t index);

#ifdef __cplusplus
}
#endif
