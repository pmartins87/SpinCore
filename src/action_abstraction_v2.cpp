#include "spincore/action_abstraction_v2.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <tuple>

namespace spincore {
namespace {

int clamp_target(int target, const LegalActions& legal) {
    return std::min(std::max(target, legal.min_raise_to), legal.max_raise_to);
}

bool active(const UniversalActionMaskV2& mask, UniversalActionSlotV2 slot) {
    return mask[static_cast<std::size_t>(slot)] != 0;
}

int slot_id(UniversalActionSlotV2 slot) {
    return static_cast<int>(slot);
}

bool same_exact(const ExactAction& a, const ExactAction& b) {
    return a.type == b.type && a.amount_to == b.amount_to;
}

bool is_raise_target(const ExactAction& action) {
    return action.type == ExactActionType::BetTo || action.type == ExactActionType::RaiseTo;
}

ResolvedUniversalActionV2 resolve_min_raise(
    const BettingEngine& betting,
    const LegalActions& legal) {
    const int target = clamp_target(legal.min_raise_to, legal);
    if (target >= legal.max_raise_to) {
        return {
            UniversalActionSlotV2::MinRaise,
            {ExactActionType::AllIn, 0},
            legal.min_raise_to,
            legal.max_raise_to,
            target == legal.min_raise_to,
            true,
        };
    }
    return {
        UniversalActionSlotV2::MinRaise,
        {betting.current_bet() == 0 ? ExactActionType::BetTo : ExactActionType::RaiseTo, target},
        legal.min_raise_to,
        target,
        target == legal.min_raise_to,
        false,
    };
}

ResolvedUniversalActionV2 resolve_fractional(
    const BettingEngine& betting,
    const LegalActions& legal,
    UniversalActionSlotV2 slot,
    double fraction) {
    const int actor = betting.actor();
    if (actor < 0) {
        throw std::logic_error("universal action resolution without actor");
    }
    const auto& player = betting.players()[static_cast<std::size_t>(actor)];

    // Exact legacy-control semantics from SpinTraversalState::resolve:
    // llround(fraction * pot_after_call), minimum one chip of increment, then
    // clamp to the legal exact raise range.
    const int pot_after_call = betting.pot() + legal.to_call;
    const auto rounded_increment = static_cast<int>(std::llround(fraction * pot_after_call));
    const int call_target = player.street_commitment + legal.to_call;
    const int raw_target = call_target + std::max(1, rounded_increment);
    const int target = clamp_target(raw_target, legal);

    if (target >= legal.max_raise_to) {
        return {
            slot,
            {ExactActionType::AllIn, 0},
            raw_target,
            legal.max_raise_to,
            target == legal.min_raise_to,
            true,
        };
    }
    return {
        slot,
        {betting.current_bet() == 0 ? ExactActionType::BetTo : ExactActionType::RaiseTo, target},
        raw_target,
        target,
        target == legal.min_raise_to,
        false,
    };
}

bool candidate_better_for_alias(
    const ResolvedUniversalActionV2& candidate,
    const ResolvedUniversalActionV2& incumbent,
    const LegalActions& legal) {
    if (candidate.exact.type == ExactActionType::AllIn) {
        if (candidate.slot == UniversalActionSlotV2::AllIn && incumbent.slot != UniversalActionSlotV2::AllIn) {
            return true;
        }
        if (incumbent.slot == UniversalActionSlotV2::AllIn && candidate.slot != UniversalActionSlotV2::AllIn) {
            return false;
        }
    }

    if (is_raise_target(candidate.exact) && candidate.realized_target == legal.min_raise_to) {
        if (candidate.slot == UniversalActionSlotV2::MinRaise && incumbent.slot != UniversalActionSlotV2::MinRaise) {
            return true;
        }
        if (incumbent.slot == UniversalActionSlotV2::MinRaise && candidate.slot != UniversalActionSlotV2::MinRaise) {
            return false;
        }
    }

    const bool candidate_fractional = universal_is_fractional_v2(candidate.slot);
    const bool incumbent_fractional = universal_is_fractional_v2(incumbent.slot);
    if (candidate_fractional && incumbent_fractional) {
        const long candidate_distance = std::labs(
            static_cast<long>(candidate.unclamped_target) - static_cast<long>(candidate.realized_target));
        const long incumbent_distance = std::labs(
            static_cast<long>(incumbent.unclamped_target) - static_cast<long>(incumbent.realized_target));
        if (candidate_distance != incumbent_distance) {
            return candidate_distance < incumbent_distance;
        }
        const double candidate_fraction = universal_pot_fraction_v2(candidate.slot);
        const double incumbent_fraction = universal_pot_fraction_v2(incumbent.slot);
        if (candidate_fraction != incumbent_fraction) {
            return candidate_fraction < incumbent_fraction;
        }
    }

    return slot_id(candidate.slot) < slot_id(incumbent.slot);
}

void insert_deduplicated(
    std::vector<ResolvedUniversalActionV2>& out,
    const ResolvedUniversalActionV2& candidate,
    const LegalActions& legal) {
    for (auto& incumbent : out) {
        if (!same_exact(incumbent.exact, candidate.exact)) {
            continue;
        }
        if (candidate_better_for_alias(candidate, incumbent, legal)) {
            incumbent = candidate;
        }
        return;
    }
    out.push_back(candidate);
}

} // namespace

const char* universal_action_name_v2(UniversalActionSlotV2 slot) noexcept {
    switch (slot) {
        case UniversalActionSlotV2::Fold: return "fold";
        case UniversalActionSlotV2::CheckCall: return "check_call";
        case UniversalActionSlotV2::MinRaise: return "min_raise";
        case UniversalActionSlotV2::Pot33: return "pot_33";
        case UniversalActionSlotV2::Pot40: return "pot_40";
        case UniversalActionSlotV2::Pot50: return "pot_50";
        case UniversalActionSlotV2::Pot66: return "pot_66";
        case UniversalActionSlotV2::Pot75: return "pot_75";
        case UniversalActionSlotV2::Pot100: return "pot_100";
        case UniversalActionSlotV2::AllIn: return "all_in";
    }
    return "?";
}

double universal_pot_fraction_v2(UniversalActionSlotV2 slot) noexcept {
    switch (slot) {
        case UniversalActionSlotV2::Pot33: return 0.33;
        case UniversalActionSlotV2::Pot40: return 0.40;
        case UniversalActionSlotV2::Pot50: return 0.50;
        case UniversalActionSlotV2::Pot66: return 0.66;
        case UniversalActionSlotV2::Pot75: return 0.75;
        case UniversalActionSlotV2::Pot100: return 1.00;
        default: return 0.0;
    }
}

bool universal_is_fractional_v2(UniversalActionSlotV2 slot) noexcept {
    return slot == UniversalActionSlotV2::Pot33 ||
           slot == UniversalActionSlotV2::Pot40 ||
           slot == UniversalActionSlotV2::Pot50 ||
           slot == UniversalActionSlotV2::Pot66 ||
           slot == UniversalActionSlotV2::Pot75 ||
           slot == UniversalActionSlotV2::Pot100;
}

std::vector<ResolvedUniversalActionV2> resolve_universal_actions_v2(
    const BettingEngine& betting,
    const UniversalActionMaskV2& active_mask) {
    std::vector<ResolvedUniversalActionV2> out;
    const int actor = betting.actor();
    if (actor < 0) {
        return out;
    }
    const auto legal = betting.legal_actions(actor);

    if (active(active_mask, UniversalActionSlotV2::Fold) && legal.fold) {
        insert_deduplicated(
            out,
            {UniversalActionSlotV2::Fold, {ExactActionType::Fold, 0}, 0, 0, false, false},
            legal);
    }
    if (active(active_mask, UniversalActionSlotV2::CheckCall)) {
        if (legal.check) {
            insert_deduplicated(
                out,
                {UniversalActionSlotV2::CheckCall, {ExactActionType::Check, 0}, 0, 0, false, false},
                legal);
        } else if (legal.call) {
            insert_deduplicated(
                out,
                {UniversalActionSlotV2::CheckCall, {ExactActionType::Call, 0}, 0, 0, false, false},
                legal);
        }
    }

    const bool can_raise = legal.bet || legal.raise;
    if (can_raise && active(active_mask, UniversalActionSlotV2::MinRaise)) {
        insert_deduplicated(out, resolve_min_raise(betting, legal), legal);
    }
    if (can_raise) {
        for (const auto slot : {
                 UniversalActionSlotV2::Pot33,
                 UniversalActionSlotV2::Pot40,
                 UniversalActionSlotV2::Pot50,
                 UniversalActionSlotV2::Pot66,
                 UniversalActionSlotV2::Pot75,
                 UniversalActionSlotV2::Pot100}) {
            if (!active(active_mask, slot)) {
                continue;
            }
            insert_deduplicated(
                out,
                resolve_fractional(betting, legal, slot, universal_pot_fraction_v2(slot)),
                legal);
        }
    }
    if (active(active_mask, UniversalActionSlotV2::AllIn) && legal.all_in) {
        insert_deduplicated(
            out,
            {UniversalActionSlotV2::AllIn, {ExactActionType::AllIn, 0}, 0, legal.max_raise_to, false, true},
            legal);
    }

    std::sort(
        out.begin(),
        out.end(),
        [](const ResolvedUniversalActionV2& a, const ResolvedUniversalActionV2& b) {
            return slot_id(a.slot) < slot_id(b.slot);
        });
    return out;
}

} // namespace spincore
