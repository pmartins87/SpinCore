#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef _WIN32
# ifdef SPINCORE_SOLVER_C_EXPORTS
#  define SPINCORE_SOLVER_C_API __declspec(dllexport)
# else
#  define SPINCORE_SOLVER_C_API __declspec(dllimport)
# endif
#else
# define SPINCORE_SOLVER_C_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif
#define SPINCORE_SOLVER_C_ABI_VERSION 2
typedef struct spincore_solver_state spincore_solver_state; typedef struct spincore_solver_frontier spincore_solver_frontier;
typedef struct spincore_solver_scenario_v2 {int32_t total_chips,game_is_hu,blind_index,small_blind,big_blind,stack_0,stack_1,stack_2,dead_player_0,dead_player_1,dead_player_count,dealer_id;} spincore_solver_scenario_v2;
SPINCORE_SOLVER_C_API int32_t spincore_solver_c_abi_version(void); SPINCORE_SOLVER_C_API const char* spincore_solver_last_error(void);
SPINCORE_SOLVER_C_API spincore_solver_state*spincore_solver_state_create_v2(const spincore_solver_scenario_v2*,uint64_t); SPINCORE_SOLVER_C_API spincore_solver_state*spincore_solver_state_clone(const spincore_solver_state*); SPINCORE_SOLVER_C_API void spincore_solver_state_destroy(spincore_solver_state*);
SPINCORE_SOLVER_C_API int32_t spincore_solver_state_terminal(const spincore_solver_state*); SPINCORE_SOLVER_C_API int32_t spincore_solver_state_actor(const spincore_solver_state*); SPINCORE_SOLVER_C_API int32_t spincore_solver_state_domain(const spincore_solver_state*); SPINCORE_SOLVER_C_API uint32_t spincore_solver_state_legal_mask(const spincore_solver_state*); SPINCORE_SOLVER_C_API int32_t spincore_solver_state_apply_abstract(spincore_solver_state*,int32_t);
/* Parallel R7.5.4 universal-action API. active_mask uses slots 0..9 from UniversalActionSlotV2. The returned legal mask contains only state-local deduplicated exact actions. Old six-slot calls above remain unchanged. */
SPINCORE_SOLVER_C_API uint32_t spincore_solver_state_universal_legal_mask(const spincore_solver_state*,uint32_t active_mask); SPINCORE_SOLVER_C_API int32_t spincore_solver_state_apply_universal(spincore_solver_state*,uint32_t active_mask,int32_t action_slot);
/* Read-only exact identity for one *effective* post-dedup universal slot. Returns 0 on success and writes ExactActionType (0..5) plus amount_to. Inactive, illegal, or suppressed alias slots fail with -1 and set last_error. This function never mutates state or RNG. */
SPINCORE_SOLVER_C_API int32_t spincore_solver_state_resolve_universal_exact(const spincore_solver_state*,uint32_t active_mask,int32_t action_slot,int32_t*out_type,int32_t*out_amount_to);
SPINCORE_SOLVER_C_API size_t spincore_solver_state_neural_input(const spincore_solver_state*,uint8_t*,size_t); SPINCORE_SOLVER_C_API size_t spincore_solver_state_neural_input_v2(const spincore_solver_state*,uint8_t*,size_t); SPINCORE_SOLVER_C_API int32_t spincore_solver_state_terminal_chip_delta(const spincore_solver_state*,int32_t[3]); SPINCORE_SOLVER_C_API int32_t spincore_solver_state_terminal_icm_delta(const spincore_solver_state*,const double[3],double[3]);
SPINCORE_SOLVER_C_API spincore_solver_frontier*spincore_solver_frontier_create_until_actor(const spincore_solver_state*,int32_t,size_t,size_t); SPINCORE_SOLVER_C_API void spincore_solver_frontier_destroy(spincore_solver_frontier*); SPINCORE_SOLVER_C_API size_t spincore_solver_frontier_size(const spincore_solver_frontier*); SPINCORE_SOLVER_C_API size_t spincore_solver_frontier_nodes_visited(const spincore_solver_frontier*); SPINCORE_SOLVER_C_API size_t spincore_solver_frontier_max_depth_reached(const spincore_solver_frontier*); SPINCORE_SOLVER_C_API int32_t spincore_solver_frontier_is_terminal(const spincore_solver_frontier*,size_t); SPINCORE_SOLVER_C_API spincore_solver_state*spincore_solver_frontier_clone_state(const spincore_solver_frontier*,size_t);
#ifdef __cplusplus
}
#endif
