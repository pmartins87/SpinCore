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

/*
 * Scenario fields are kept scalar to make the ABI stable for Python/ctypes.
 * Dead players use -1 in dead_player_0/dead_player_1 when absent.
 */
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

/* abstract_action is the numeric value of AbstractActionSlot [0,5]. */
SPINCORE_SOLVER_C_API int32_t spincore_solver_state_apply_abstract(
    spincore_solver_state* state,
    int32_t abstract_action);

/* Returns required byte size if out == NULL or capacity == 0. */
SPINCORE_SOLVER_C_API size_t spincore_solver_state_neural_input(
    const spincore_solver_state* state,
    uint8_t* out,
    size_t capacity);

/* Writes exactly three chip deltas for a terminal state. */
SPINCORE_SOLVER_C_API int32_t spincore_solver_state_terminal_chip_delta(
    const spincore_solver_state* state,
    int32_t out_delta[3]);

#ifdef __cplusplus
}
#endif
