#include "spincore/lean_action_abstraction.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace spincore {
namespace {

bool active(const UniversalActionMaskV2& mask, UniversalActionSlotV2 slot) {
    return mask[static_cast<std::size_t>(slot)] != 0;
}

int slot_id(UniversalActionSlotV2 slot) {
    return static_cast<int>(slot);
}

bool same_exact(const ExactAction& a, const ExactAction& b) {
    return a.type == b.type && a.amount_to == b.amount_to;
}

void insert_deduplicated(
    std::vector<ResolvedUniversalActionV2>& out,
    const ResolvedUniversalActionV2& candidate) {
    for (auto& incumbent : out) {
        if (!same_exact(incumbent.exact, candidate.exact)) continue;
        // Prefer the explicit ALL_IN neural slot for an exact all-in alias.
        if (candidate.exact.type == ExactActionType::AllIn) {
            if (candidate.slot == UniversalActionSlotV2::AllIn &&
                incumbent.slot != UniversalActionSlotV2::AllIn) {
                incumbent = candidate;
            }
            return;
        }
        // Otherwise keep the lower stable slot id.
        if (slot_id(candidate.slot) < slot_id(incumbent.slot)) incumbent = candidate;
        return;
    }
    out.push_back(candidate);
}

struct PreflopDerived {
    int num_calls{0};
    int num_raises{0};
    int num_limpers{0};
    int num_callers_after_raise{0};
    int first_limper{-1};
    int iso_raiser{-1};
    bool limp_before_raise{false};
    bool got_isolated{false};
};

PreflopDerived derive_preflop(const BettingEngine& betting, int hero) {
    PreflopDerived d{};
    bool seen_raise = false;
    int max_commitment = 0;

    for (const auto& ev : betting.history()) {
        if (ev.street != Street::Preflop) continue;

        const int max_before = max_commitment;
        max_commitment = std::max(max_commitment, static_cast<int>(ev.resulting_commitment));
        if (ev.forced) continue;

        bool is_call = ev.action.type == ExactActionType::Call;
        const bool is_check = ev.action.type == ExactActionType::Check;
        bool is_raise = ev.action.type == ExactActionType::BetTo ||
                        ev.action.type == ExactActionType::RaiseTo;
        if (ev.action.type == ExactActionType::AllIn) {
            // Legacy DeepSpin recorded an all-in that merely called as CHECK_CALL.
            is_raise = ev.resulting_commitment > max_before;
            is_call = !is_raise;
        }

        if (is_check || ev.action.type == ExactActionType::Fold) continue;

        if (is_call) {
            ++d.num_calls;
            if (!seen_raise) {
                ++d.num_limpers;
                if (d.first_limper < 0) d.first_limper = ev.actor;
            } else if (d.num_raises == 1) {
                ++d.num_callers_after_raise;
            }
            continue;
        }

        if (is_raise) {
            ++d.num_raises;
            if (!seen_raise && d.num_limpers > 0) {
                d.limp_before_raise = true;
                d.iso_raiser = ev.actor;
            }
            seen_raise = true;
        }
    }

    d.got_isolated = d.limp_before_raise && d.first_limper == hero &&
                     d.num_raises >= 1 && d.iso_raiser != hero;
    return d;
}

ResolvedUniversalActionV2 passive_action(
    UniversalActionSlotV2 slot,
    ExactActionType type) {
    return {slot, {type, 0}, 0, 0, false, false};
}

ResolvedUniversalActionV2 allin_action() {
    return {UniversalActionSlotV2::AllIn, {ExactActionType::AllIn, 0}, 0, 0, false, true};
}

ResolvedUniversalActionV2 target_action(
    const BettingEngine& betting,
    UniversalActionSlotV2 slot,
    int target) {
    const auto legal = betting.legal_actions(betting.actor());
    if (target >= legal.max_raise_to) {
        return {slot, {ExactActionType::AllIn, 0}, target, legal.max_raise_to, false, true};
    }
    return {
        slot,
        {betting.current_bet() == 0 ? ExactActionType::BetTo : ExactActionType::RaiseTo, target},
        target,
        target,
        false,
        false,
    };
}

void add_passive(
    std::vector<ResolvedUniversalActionV2>& out,
    const BettingEngine& betting,
    const UniversalActionMaskV2& mask,
    bool allow_check_call) {
    const auto legal = betting.legal_actions(betting.actor());
    if (active(mask, UniversalActionSlotV2::Fold) && legal.fold) {
        insert_deduplicated(out, passive_action(UniversalActionSlotV2::Fold, ExactActionType::Fold));
    }
    if (allow_check_call && active(mask, UniversalActionSlotV2::CheckCall)) {
        if (legal.check) {
            insert_deduplicated(out, passive_action(UniversalActionSlotV2::CheckCall, ExactActionType::Check));
        } else if (legal.call) {
            insert_deduplicated(out, passive_action(UniversalActionSlotV2::CheckCall, ExactActionType::Call));
        }
    }
}

std::vector<ResolvedUniversalActionV2> resolve_preflop(
    const BettingEngine& betting,
    const UniversalActionMaskV2& mask) {
    std::vector<ResolvedUniversalActionV2> out;
    const int actor = betting.actor();
    if (actor < 0) return out;
    const auto legal = betting.legal_actions(actor);
    const auto& player = betting.players()[static_cast<std::size_t>(actor)];
    const auto d = derive_preflop(betting, actor);

    // Legacy: after 3+ raises, if Hero still has chips beyond the call, no flat
    // call is offered: the branch is fold or shove.
    const bool allow_check_call = !(d.num_raises >= 3 && legal.to_call < player.stack);
    add_passive(out, betting, mask, allow_check_call);

    // Legacy removes ALL_IN as a second label when the shove is merely an
    // all-in call; CHECK_CALL already represents that exact economic action.
    if (active(mask, UniversalActionSlotV2::AllIn) && legal.all_in &&
        legal.to_call < player.stack) {
        insert_deduplicated(out, allin_action());
    }

    if (!(legal.bet || legal.raise)) return out;
    if (d.got_isolated && d.num_raises == 1) return out;
    if (d.num_raises >= 2) return out;

    UniversalActionSlotV2 slot = UniversalActionSlotV2::Pot33;
    double target_bb = 0.0;
    if (d.num_raises == 0 && d.num_limpers == 0) {
        slot = UniversalActionSlotV2::Pot33;
        target_bb = 2.0;
    } else if (d.num_raises == 0 && d.num_limpers >= 1) {
        slot = UniversalActionSlotV2::Pot75;
        const int extra_limpers = std::max(0, d.num_limpers - 1);
        target_bb = 2.5 + static_cast<double>(extra_limpers);
    } else if (d.num_raises == 1) {
        slot = UniversalActionSlotV2::Pot50;
        target_bb = 5.0 + 2.0 * static_cast<double>(d.num_callers_after_raise);
    }

    if (target_bb <= 0.0 || !active(mask, slot)) return out;
    const int bb = std::max(1, betting.big_blind());
    const int target = static_cast<int>(std::lround(target_bb * static_cast<double>(bb)));
    if (target <= betting.current_bet()) return out;
    if (target < legal.min_raise_to) return out; // legacy prunes, never clamps upward

    insert_deduplicated(out, target_action(betting, slot, target));
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        return slot_id(a.slot) < slot_id(b.slot);
    });
    return out;
}

bool near_allin_legacy(const BettingEngine& betting, int put_now) {
    const int actor = betting.actor();
    const auto& players = betting.players();
    const auto& hero = players[static_cast<std::size_t>(actor)];
    const int hero_total = hero.stack + hero.street_commitment;
    int max_villain_total = 0;
    for (int i = 0; i < 3; ++i) {
        if (i == actor) continue;
        const auto& villain = players[static_cast<std::size_t>(i)];
        if (!villain.folded && (villain.stack > 0 || villain.street_commitment > 0)) {
            max_villain_total = std::max(max_villain_total, villain.stack + villain.street_commitment);
        }
    }
    const int effective_total = std::min(hero_total, max_villain_total);
    if (effective_total <= 0) return false;
    const int capped_put = std::min(put_now, hero.stack);
    const int total_committed = hero.street_commitment + capped_put;
    const int threshold = static_cast<int>(std::ceil(0.60 * static_cast<double>(effective_total)));
    const int leftover = static_cast<int>(std::floor(0.40 * static_cast<double>(effective_total)));
    return total_committed >= threshold || (effective_total - total_committed) <= leftover;
}

int legacy_fraction_increment(UniversalActionSlotV2 slot, int base) {
    switch (slot) {
        case UniversalActionSlotV2::Pot33: return (base * 33) / 100;
        case UniversalActionSlotV2::Pot50: return base / 2;
        case UniversalActionSlotV2::Pot75: return (base * 75) / 100;
        case UniversalActionSlotV2::Pot100: return base;
        default: return 0;
    }
}

std::vector<ResolvedUniversalActionV2> resolve_postflop(
    const BettingEngine& betting,
    const UniversalActionMaskV2& mask) {
    std::vector<ResolvedUniversalActionV2> out;
    const int actor = betting.actor();
    if (actor < 0) return out;
    const auto legal = betting.legal_actions(actor);
    const auto& player = betting.players()[static_cast<std::size_t>(actor)];

    add_passive(out, betting, mask, true);

    // Legacy does not expose ALL_IN as a separate action when the stack can only call.
    if (active(mask, UniversalActionSlotV2::AllIn) && legal.all_in &&
        legal.to_call < player.stack) {
        insert_deduplicated(out, allin_action());
    }

    if (!(legal.bet || legal.raise)) return out;
    const int base = betting.pot() + legal.to_call; // pot after paying the call
    const int call_target = player.street_commitment + legal.to_call;

    for (const auto slot : {
             UniversalActionSlotV2::Pot33,
             UniversalActionSlotV2::Pot50,
             UniversalActionSlotV2::Pot75,
             UniversalActionSlotV2::Pot100}) {
        if (!active(mask, slot)) continue;
        const int increment = legacy_fraction_increment(slot, base);
        if (increment <= 0) continue;
        const int put_now = legal.to_call + increment;
        if (put_now > player.stack) continue; // legacy prunes unaffordable fractional size
        const int target = call_target + increment;
        if (target < legal.min_raise_to) continue; // legacy prunes too-small raise
        if (near_allin_legacy(betting, put_now)) continue; // 60% collapse
        insert_deduplicated(out, target_action(betting, slot, target));
    }

    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        return slot_id(a.slot) < slot_id(b.slot);
    });
    return out;
}

} // namespace

std::vector<ResolvedUniversalActionV2> resolve_lean_legacy_actions_v1(
    const BettingEngine& betting,
    const UniversalActionMaskV2& active_mask) {
    if (betting.actor() < 0) return {};
    return betting.street() == Street::Preflop
        ? resolve_preflop(betting, active_mask)
        : resolve_postflop(betting, active_mask);
}

} // namespace spincore
