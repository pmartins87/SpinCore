#include "spincore/preflop_ontology.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace spincore {
namespace {

[[nodiscard]] std::uint8_t sat_inc(std::uint8_t value) noexcept {
    return value == 255U ? value : static_cast<std::uint8_t>(value + 1U);
}

}  // namespace

PreflopOntologyContext derive_preflop_ontology(const BettingEngine& betting) {
    PreflopOntologyContext out{};
    out.actor = betting.actor();
    out.pot = betting.pot();
    if (out.actor >= 0 && betting.street() == Street::Preflop) {
        out.to_call = betting.legal_actions(out.actor).to_call;
    }

    std::array<bool, 3> limped{};
    std::int32_t prior_max = 0;

    for (const auto& event : betting.history()) {
        if (event.street != Street::Preflop) {
            continue;
        }

        if (event.forced) {
            prior_max = std::max(prior_max, event.resulting_commitment);
            continue;
        }

        out.voluntary_action_count = sat_inc(out.voluntary_action_count);
        out.last_voluntary_actor = event.actor;

        bool aggressive = false;
        switch (event.action.type) {
            case ExactActionType::BetTo:
            case ExactActionType::RaiseTo:
                aggressive = true;
                break;
            case ExactActionType::AllIn:
                // All-in calls exist, so only an increase above the prior
                // street maximum is aggression.
                aggressive = event.resulting_commitment > prior_max;
                break;
            case ExactActionType::Fold:
            case ExactActionType::Check:
            case ExactActionType::Call:
                break;
        }

        if (aggressive) {
            const std::int32_t increment = std::max(0, event.resulting_commitment - prior_max);
            if (out.aggression_count == 0U) {
                out.first_aggressor = event.actor;
                out.first_raise_to = event.resulting_commitment;
                out.first_raise_increment = increment;
                out.had_limp_before_first_aggression = out.limp_count > 0U;
            } else if (event.actor >= 0 && event.actor < static_cast<std::int32_t>(limped.size()) && limped[static_cast<std::size_t>(event.actor)]) {
                out.limper_became_reraiser = true;
            }
            out.aggression_count = sat_inc(out.aggression_count);
            out.last_aggressor = event.actor;
            out.last_raise_to = event.resulting_commitment;
            out.last_raise_increment = increment;
        } else if (event.action.type == ExactActionType::Call || event.action.type == ExactActionType::AllIn) {
            out.voluntary_call_count = sat_inc(out.voluntary_call_count);
            if (out.aggression_count == 0U) {
                out.limp_count = sat_inc(out.limp_count);
                if (event.actor >= 0 && event.actor < static_cast<std::int32_t>(limped.size())) {
                    limped[static_cast<std::size_t>(event.actor)] = true;
                }
            } else {
                out.calls_after_aggression = sat_inc(out.calls_after_aggression);
            }
        }

        prior_max = std::max(prior_max, event.resulting_commitment);
    }

    if (out.aggression_count == 0U) {
        out.lineage = out.limp_count > 0U ? PreflopLineageType::Limped : PreflopLineageType::Unopened;
    } else if (out.aggression_count == 1U) {
        out.lineage = out.had_limp_before_first_aggression
            ? PreflopLineageType::RaiseOverLimp
            : PreflopLineageType::OpenRaised;
    } else if (out.limper_became_reraiser) {
        out.lineage = PreflopLineageType::LimpReraised;
    } else {
        out.lineage = PreflopLineageType::Reraised;
    }

    return out;
}

const char* preflop_lineage_name(PreflopLineageType lineage) noexcept {
    switch (lineage) {
        case PreflopLineageType::Unopened: return "unopened";
        case PreflopLineageType::Limped: return "limped";
        case PreflopLineageType::OpenRaised: return "open_raised";
        case PreflopLineageType::RaiseOverLimp: return "raise_over_limp";
        case PreflopLineageType::Reraised: return "reraised";
        case PreflopLineageType::LimpReraised: return "limp_reraised";
    }
    return "unknown";
}

}  // namespace spincore
