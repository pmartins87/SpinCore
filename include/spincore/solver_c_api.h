#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
  #ifdef SPINCORE_SOLVER_C_EXPORTS
    #define SPINCORE_C_API __declspec(dllexport)
  #else
    #define SPINCORE_C_API __declspec(dllimport)
  #endif
#else
  #define SPINCORE_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SPINCORE_SOLVER_C_ABI_VERSION 2

typedef struct spincore_state spincore_state;

typedef struct spincore_episode_v1 {
    int32_t total_chips;
    int32_t game_is_hu;
    int32_t blind_index;
    int32_t small_blind;
    int32_t big_blind;
    int32_t stacks[3];
    int32_t dealer_id;
} spincore_episode_v1;

SPINCORE_C_API int spincore_solver_abi_version(void);
SPINCORE_C_API spincore_state* spincore_state_create(const spincore_episode_v1* episode, uint64_t deck_seed);
SPINCORE_C_API spincore_state* spincore_state_clone(const spincore_state* state);
SPINCORE_C_API void spincore_state_destroy(spincore_state* state);
SPINCORE_C_API int spincore_state_terminal(const spincore_state* state);
SPINCORE_C_API int32_t spincore_state_actor(const spincore_state* state);
SPINCORE_C_API int spincore_state_domain(const spincore_state* state); /* 0=3H, 1=true HU */
SPINCORE_C_API uint8_t spincore_state_legal_mask(const spincore_state* state); /* six abstract slots */
SPINCORE_C_API int spincore_state_apply(spincore_state* state, uint8_t abstract_action);
SPINCORE_C_API size_t spincore_state_neural_size(const spincore_state* state);
SPINCORE_C_API int spincore_state_neural_copy(const spincore_state* state, uint8_t* out, size_t out_size);
SPINCORE_C_API int spincore_state_terminal_chip_delta(const spincore_state* state, int32_t out_delta[3]);
SPINCORE_C_API const char* spincore_last_error(void);

#ifdef __cplusplus
}
#endif
