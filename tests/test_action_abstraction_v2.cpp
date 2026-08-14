#include "test_framework.hpp"
#include "test_helpers.hpp"

#include "spincore/action_abstraction_v2.hpp"
#include "spincore/spin_traversal_state.hpp"

#include <set>
#include <tuple>

using namespace spincore;

namespace {

UniversalActionMaskV2 mask(std::initializer_list<UniversalActionSlotV2> slots) {
    UniversalActionMaskV2 out{};
    for (auto slot : slots) {
        out[static_cast<std::size_t>(slot)] = 1;
    }
    return out;
}

const ResolvedUniversalActionV2& find_slot(
    const std::vector<ResolvedUniversalActionV2>& actions,
    UniversalActionSlotV2 slot) {
    for (const auto& action : actions) {
        if (action.slot == slot) {
            return action;
        }
    }
    throw std::runtime_error("universal action slot not found");
}

void reach_flop_by_check_call(SpinTraversalState& state) {
    int guard = 0;
    while (!state.terminal() && state.hand().betting().street() == Street::Preflop && guard++ < 16) {
        const auto legal = state.legal_abstract_actions();
        bool applied = false;
        for (auto action : legal) {
            if (action == AbstractActionSlot::CheckCall) {
                state.apply(action);
                applied = true;
                break;
            }
        }
        REQUIRE(applied);
    }
    REQUIRE(!state.terminal());
    REQUIRE(state.hand().betting().street() == Street::Flop);
}

} // namespace

SPIN_TEST(universal_action_slots_are_stable_0_to_9) {
    REQUIRE(static_cast<int>(UniversalActionSlotV2::Fold) == 0);
    REQUIRE(static_cast<int>(UniversalActionSlotV2::MinRaise) == 2);
    REQUIRE(static_cast<int>(UniversalActionSlotV2::Pot33) == 3);
    REQUIRE(static_cast<int>(UniversalActionSlotV2::Pot100) == 8);
    REQUIRE(static_cast<int>(UniversalActionSlotV2::AllIn) == 9);
    for (int i = 0; i < 10; ++i) {
        REQUIRE(universal_action_name_v2(static_cast<UniversalActionSlotV2>(i))[0]);
    }
}

SPIN_TEST(universal_preflop_min_raise_is_exact_current_context_raise_control) {
    SpinTraversalState state(sc3(0), 12345);
    const auto old_exact = state.resolve(AbstractActionSlot::ContextRaise);
    const auto actions = resolve_universal_actions_v2(
        state.hand().betting(),
        mask({UniversalActionSlotV2::Fold,
              UniversalActionSlotV2::CheckCall,
              UniversalActionSlotV2::MinRaise,
              UniversalActionSlotV2::AllIn}));
    const auto& min_raise = find_slot(actions, UniversalActionSlotV2::MinRaise);
    REQUIRE(min_raise.exact == old_exact);
}

SPIN_TEST(universal_postflop_33_75_are_bit_exact_current_control) {
    SpinTraversalState state(sc3(0), 54321);
    reach_flop_by_check_call(state);

    const auto old_33 = state.resolve(AbstractActionSlot::SmallPot);
    const auto old_75 = state.resolve(AbstractActionSlot::LargePot);
    const auto actions = resolve_universal_actions_v2(
        state.hand().betting(),
        mask({UniversalActionSlotV2::Fold,
              UniversalActionSlotV2::CheckCall,
              UniversalActionSlotV2::Pot33,
              UniversalActionSlotV2::Pot75,
              UniversalActionSlotV2::AllIn}));

    REQUIRE(find_slot(actions, UniversalActionSlotV2::Pot33).exact == old_33);
    REQUIRE(find_slot(actions, UniversalActionSlotV2::Pot75).exact == old_75);
}

SPIN_TEST(universal_dense_mask_never_exposes_duplicate_exact_actions) {
    SpinTraversalState state(sc3(1), 919191);
    reach_flop_by_check_call(state);
    const auto actions = resolve_universal_actions_v2(
        state.hand().betting(),
        mask({UniversalActionSlotV2::Fold,
              UniversalActionSlotV2::CheckCall,
              UniversalActionSlotV2::MinRaise,
              UniversalActionSlotV2::Pot33,
              UniversalActionSlotV2::Pot40,
              UniversalActionSlotV2::Pot50,
              UniversalActionSlotV2::Pot66,
              UniversalActionSlotV2::Pot75,
              UniversalActionSlotV2::Pot100,
              UniversalActionSlotV2::AllIn}));

    std::set<std::tuple<int,int>> exact;
    for (const auto& action : actions) {
        const auto key = std::make_tuple(static_cast<int>(action.exact.type), action.exact.amount_to);
        REQUIRE(exact.insert(key).second);

        auto child = state;
        child.apply_exact(action.exact);
    }
}

SPIN_TEST(universal_min_raise_wins_fractional_alias_at_legal_minimum) {
    SpinTraversalState state(sc3(0), 314159);
    const auto actions = resolve_universal_actions_v2(
        state.hand().betting(),
        mask({UniversalActionSlotV2::CheckCall,
              UniversalActionSlotV2::MinRaise,
              UniversalActionSlotV2::Pot33,
              UniversalActionSlotV2::Pot40,
              UniversalActionSlotV2::Pot50,
              UniversalActionSlotV2::AllIn}));

    const auto legal = state.hand().betting().legal_actions(state.actor());
    const auto& minimum = find_slot(actions, UniversalActionSlotV2::MinRaise);
    REQUIRE(minimum.realized_target == legal.min_raise_to);

    for (const auto& action : actions) {
        if (action.slot == UniversalActionSlotV2::MinRaise) {
            continue;
        }
        if (action.exact.type == ExactActionType::RaiseTo || action.exact.type == ExactActionType::BetTo) {
            REQUIRE(action.exact.amount_to != legal.min_raise_to);
        }
    }
}

SPIN_TEST(universal_allin_wins_every_clamped_max_alias) {
    auto shallow = sc3(0);
    shallow.state.total_chips = 90;
    shallow.state.stacks = {30, 30, 30};
    SpinTraversalState state(shallow, 271828);

    const auto actions = resolve_universal_actions_v2(
        state.hand().betting(),
        mask({UniversalActionSlotV2::Fold,
              UniversalActionSlotV2::CheckCall,
              UniversalActionSlotV2::MinRaise,
              UniversalActionSlotV2::Pot33,
              UniversalActionSlotV2::Pot40,
              UniversalActionSlotV2::Pot50,
              UniversalActionSlotV2::Pot66,
              UniversalActionSlotV2::Pot75,
              UniversalActionSlotV2::Pot100,
              UniversalActionSlotV2::AllIn}));

    int allin_exact_count = 0;
    for (const auto& action : actions) {
        if (action.exact.type == ExactActionType::AllIn) {
            ++allin_exact_count;
            REQUIRE(action.slot == UniversalActionSlotV2::AllIn);
        }
    }
    REQUIRE(allin_exact_count == 1);
}

SPIN_TEST(universal_fractional_targets_are_monotone_after_dedup) {
    SpinTraversalState state(sc3(2), 161803);
    reach_flop_by_check_call(state);
    const auto actions = resolve_universal_actions_v2(
        state.hand().betting(),
        mask({UniversalActionSlotV2::Pot33,
              UniversalActionSlotV2::Pot40,
              UniversalActionSlotV2::Pot50,
              UniversalActionSlotV2::Pot66,
              UniversalActionSlotV2::Pot75,
              UniversalActionSlotV2::Pot100,
              UniversalActionSlotV2::AllIn}));

    int previous = -1;
    for (const auto& action : actions) {
        if (!universal_is_fractional_v2(action.slot) || action.exact.type == ExactActionType::AllIn) {
            continue;
        }
        REQUIRE(action.realized_target > previous);
        previous = action.realized_target;
    }
}
