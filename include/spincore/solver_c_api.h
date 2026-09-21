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
typedef struct spincore_solver_deal_v1 {int32_t hole_0_0,hole_0_1,hole_1_0,hole_1_1,hole_2_0,hole_2_1,board_0,board_1,board_2,board_3,board_4;} spincore_solver_deal_v1;
typedef struct spincore_solver_public_snapshot_v1 {
 int32_t terminal,street,actor,domain,current_bet,pot,visible_board_count;
 int32_t stack_0,stack_1,stack_2;
 int32_t street_commitment_0,street_commitment_1,street_commitment_2;
 int32_t total_commitment_0,total_commitment_1,total_commitment_2;
 int32_t folded_0,folded_1,folded_2;
 int32_t all_in_0,all_in_1,all_in_2;
 int32_t legal_fold,legal_check,legal_call,legal_bet,legal_raise,legal_all_in;
 int32_t to_call,min_raise_to,max_raise_to;
} spincore_solver_public_snapshot_v1;
SPINCORE_SOLVER_C_API int32_t spincore_solver_c_abi_version(void); SPINCORE_SOLVER_C_API const char* spincore_solver_last_error(void);
SPINCORE_SOLVER_C_API spincore_solver_state*spincore_solver_state_create_v2(const spincore_solver_scenario_v2*,uint64_t); SPINCORE_SOLVER_C_API spincore_solver_state*spincore_solver_state_clone(const spincore_solver_state*); SPINCORE_SOLVER_C_API void spincore_solver_state_destroy(spincore_solver_state*);
/* Phase2B10 additive diagnostic extension. Existing seed-based creation remains authoritative. The explicit-deal constructor accepts card ids 0..51 for live seats and board; dead-seat holes must be -1. The snapshot is read-only and returns the exact private/public deal plus current visible-board count. */
SPINCORE_SOLVER_C_API spincore_solver_state*spincore_solver_state_create_v2_deal(const spincore_solver_scenario_v2*,const spincore_solver_deal_v1*); SPINCORE_SOLVER_C_API int32_t spincore_solver_state_deal_snapshot_v1(const spincore_solver_state*,spincore_solver_deal_v1*,int32_t*visible_board_count);
SPINCORE_SOLVER_C_API int32_t spincore_solver_state_public_snapshot_v1(const spincore_solver_state*,spincore_solver_public_snapshot_v1*);
SPINCORE_SOLVER_C_API int32_t spincore_solver_state_terminal(const spincore_solver_state*); SPINCORE_SOLVER_C_API int32_t spincore_solver_state_actor(const spincore_solver_state*); SPINCORE_SOLVER_C_API int32_t spincore_solver_state_domain(const spincore_solver_state*); SPINCORE_SOLVER_C_API uint32_t spincore_solver_state_legal_mask(const spincore_solver_state*); SPINCORE_SOLVER_C_API int32_t spincore_solver_state_apply_abstract(spincore_solver_state*,int32_t);
/* Benchmark/runtime bridge: apply one already-resolved exact poker action. type uses ExactActionType: 0 Fold, 1 Check, 2 Call, 3 BetTo, 4 RaiseTo, 5 AllIn. amount_to is ignored for non BetTo/RaiseTo actions. BettingEngine/SpinTraversalState remains the legal-action authority and rejects illegal exact actions. This is required so external reference strategies such as DeepCrusher can retain their own exact sizings instead of being forced through SpinCore's abstraction. */
SPINCORE_SOLVER_C_API int32_t spincore_solver_state_apply_exact(spincore_solver_state*,int32_t action_type,int32_t amount_to);
/* Parallel R7.5.4 universal-action API. active_mask uses slots 0..9 from UniversalActionSlotV2. The returned legal mask contains only state-local deduplicated exact actions. Old six-slot calls above remain unchanged. */
SPINCORE_SOLVER_C_API uint32_t spincore_solver_state_universal_legal_mask(const spincore_solver_state*,uint32_t active_mask); SPINCORE_SOLVER_C_API int32_t spincore_solver_state_apply_universal(spincore_solver_state*,uint32_t active_mask,int32_t action_slot);
/* Read-only exact identity for one effective generic universal slot. */
SPINCORE_SOLVER_C_API int32_t spincore_solver_state_resolve_universal_exact(const spincore_solver_state*,uint32_t active_mask,int32_t action_slot,int32_t*out_type,int32_t*out_amount_to);
/* Legacy-first lean action semantics. Uses the same 10 slot ids/masks but resolves the active legacy seven-slot subset with DeepSpin's mature context-sensitive preflop sizes and postflop pruning/near-all-in collapse. Additive ABI-v2 extension; generic R7.5 calls above remain unchanged. */
SPINCORE_SOLVER_C_API uint32_t spincore_solver_state_lean_legal_mask(const spincore_solver_state*,uint32_t active_mask); SPINCORE_SOLVER_C_API int32_t spincore_solver_state_apply_lean(spincore_solver_state*,uint32_t active_mask,int32_t action_slot); SPINCORE_SOLVER_C_API int32_t spincore_solver_state_resolve_lean_exact(const spincore_solver_state*,uint32_t active_mask,int32_t action_slot,int32_t*out_type,int32_t*out_amount_to);
SPINCORE_SOLVER_C_API size_t spincore_solver_state_neural_input(const spincore_solver_state*,uint8_t*,size_t); SPINCORE_SOLVER_C_API size_t spincore_solver_state_neural_input_v2(const spincore_solver_state*,uint8_t*,size_t);
/* R7.5.3C lossless observation carrier. SPNNIV3 is variable-length because it contains the complete public history. As with V1/V2, pass out=NULL/cap=0 to query the exact required size; no legal history is silently truncated. This is an additive ABI-v2 extension and does not change existing entry points. */
SPINCORE_SOLVER_C_API size_t spincore_solver_state_neural_input_v3(const spincore_solver_state*,uint8_t*,size_t);
SPINCORE_SOLVER_C_API int32_t spincore_solver_state_terminal_chip_delta(const spincore_solver_state*,int32_t[3]); SPINCORE_SOLVER_C_API int32_t spincore_solver_state_terminal_icm_delta(const spincore_solver_state*,const double[3],double[3]);
SPINCORE_SOLVER_C_API spincore_solver_frontier*spincore_solver_frontier_create_until_actor(const spincore_solver_state*,int32_t,size_t,size_t); SPINCORE_SOLVER_C_API void spincore_solver_frontier_destroy(spincore_solver_frontier*); SPINCORE_SOLVER_C_API size_t spincore_solver_frontier_size(const spincore_solver_frontier*); SPINCORE_SOLVER_C_API size_t spincore_solver_frontier_nodes_visited(const spincore_solver_frontier*); SPINCORE_SOLVER_C_API size_t spincore_solver_frontier_max_depth_reached(const spincore_solver_frontier*); SPINCORE_SOLVER_C_API int32_t spincore_solver_frontier_is_terminal(const spincore_solver_frontier*,size_t); SPINCORE_SOLVER_C_API spincore_solver_state*spincore_solver_frontier_clone_state(const spincore_solver_frontier*,size_t);
#ifdef __cplusplus
}
#endif