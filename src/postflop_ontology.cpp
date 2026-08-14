#include "spincore/postflop_ontology.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <vector>

namespace spincore {
namespace {

struct AggressionRef {
    std::size_t index{0};
    std::int32_t actor{-1};
    Street street{Street::Preflop};
};

[[nodiscard]] int street_index(Street street) noexcept {
    return static_cast<int>(street);
}

[[nodiscard]] std::int32_t prior_max_commitment_on_street(
    const std::vector<ActionEvent>& history,
    std::size_t index,
    Street street
) noexcept {
    std::int32_t out = 0;
    for (std::size_t i = 0; i < index && i < history.size(); ++i) {
        const auto& event = history[i];
        if (event.street != street) {
            continue;
        }
        out = std::max(out, event.resulting_commitment);
    }
    return out;
}

[[nodiscard]] bool is_aggressive_event(const std::vector<ActionEvent>& history, std::size_t index) noexcept {
    if (index >= history.size() || history[index].forced) {
        return false;
    }
    const auto& event = history[index];
    switch (event.action.type) {
        case ExactActionType::BetTo:
        case ExactActionType::RaiseTo:
            return true;
        case ExactActionType::AllIn:
            // AllIn is also used for an all-in call. It is aggression only when
            // it increases the street's previous maximum commitment.
            return event.resulting_commitment > prior_max_commitment_on_street(history, index, event.street);
        case ExactActionType::Fold:
        case ExactActionType::Check:
        case ExactActionType::Call:
            return false;
    }
    return false;
}

[[nodiscard]] std::optional<AggressionRef> last_aggression_before_street(
    const std::vector<ActionEvent>& history,
    Street current
) noexcept {
    std::optional<AggressionRef> out;
    for (std::size_t i = 0; i < history.size(); ++i) {
        const auto& event = history[i];
        if (street_index(event.street) >= street_index(current)) {
            continue;
        }
        if (is_aggressive_event(history, i)) {
            out = AggressionRef{i, event.actor, event.street};
        }
    }
    return out;
}

[[nodiscard]] std::vector<std::size_t> aggressive_events_on_street(
    const std::vector<ActionEvent>& history,
    Street street
) {
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < history.size(); ++i) {
        if (history[i].street == street && is_aggressive_event(history, i)) {
            out.push_back(i);
        }
    }
    return out;
}

[[nodiscard]] bool actor_checked_before(
    const std::vector<ActionEvent>& history,
    Street street,
    std::int32_t actor,
    std::size_t boundary
) noexcept {
    const std::size_t limit = std::min(boundary, history.size());
    for (std::size_t i = 0; i < limit; ++i) {
        const auto& event = history[i];
        if (event.street == street && event.actor == actor && event.action.type == ExactActionType::Check) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool actor_called_lineage_aggression(
    const std::vector<ActionEvent>& history,
    std::int32_t actor,
    const std::optional<AggressionRef>& lineage
) noexcept {
    if (!lineage.has_value()) {
        return false;
    }
    for (std::size_t i = lineage->index + 1U; i < history.size(); ++i) {
        const auto& event = history[i];
        if (event.street != lineage->street) {
            if (street_index(event.street) > street_index(lineage->street)) {
                break;
            }
            continue;
        }
        if (event.actor != actor) {
            continue;
        }
        if (event.action.type == ExactActionType::Call) {
            return true;
        }
        if (event.action.type == ExactActionType::AllIn && !is_aggressive_event(history, i)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::uint8_t skipped_streets(Street current, const AggressionRef& lineage) noexcept {
    const int gap = street_index(current) - street_index(lineage.street) - 1;
    return static_cast<std::uint8_t>(std::max(0, gap));
}

[[nodiscard]] PostflopLineType classify_first_bet(
    const std::vector<ActionEvent>& history,
    Street current,
    std::int32_t bettor,
    std::size_t boundary,
    const std::optional<AggressionRef>& lineage
) noexcept {
    if (!lineage.has_value()) {
        return PostflopLineType::GenericBet;
    }

    const std::uint8_t skipped = skipped_streets(current, *lineage);
    if (bettor == lineage->actor) {
        if (skipped == 0U) {
            return PostflopLineType::CBet;
        }
        if (skipped == 1U) {
            return PostflopLineType::DelayedCBet;
        }
        return PostflopLineType::DoubleDelayedCBet;
    }

    const bool lineage_checked = actor_checked_before(history, current, lineage->actor, boundary);
    const bool bettor_called_lineage = actor_called_lineage_aggression(history, bettor, lineage);

    if (lineage_checked && bettor_called_lineage) {
        return skipped == 0U ? PostflopLineType::FloatBet : PostflopLineType::DelayedFloatBet;
    }
    if (skipped > 0U) {
        return PostflopLineType::ProbeBet;
    }
    if (!lineage_checked) {
        return PostflopLineType::DonkBet;
    }
    return PostflopLineType::GenericBet;
}

}  // namespace

PostflopOntologyContext derive_postflop_ontology(const BettingEngine& betting) {
    PostflopOntologyContext out{};
    out.street = betting.street();
    out.actor = betting.actor();
    out.pot = betting.pot();

    if (out.actor >= 0) {
        out.to_call = betting.legal_actions(out.actor).to_call;
    }
    if (out.street == Street::Preflop || out.actor < 0) {
        return out;
    }

    const auto& history = betting.history();
    const auto lineage = last_aggression_before_street(history, out.street);
    if (lineage.has_value()) {
        out.has_lineage_aggressor = true;
        out.lineage_aggressor = lineage->actor;
        out.lineage_aggression_street = static_cast<std::int8_t>(street_index(lineage->street));
        out.skipped_streets_since_lineage = skipped_streets(out.street, *lineage);
        out.lineage_checked_current_street = actor_checked_before(
            history,
            out.street,
            lineage->actor,
            history.size()
        );
        out.actor_called_lineage_aggression = actor_called_lineage_aggression(history, out.actor, lineage);
    }

    const auto current_aggressions = aggressive_events_on_street(history, out.street);
    out.current_street_aggression_count = static_cast<std::uint8_t>(
        std::min<std::size_t>(current_aggressions.size(), 255U)
    );

    if (!current_aggressions.empty()) {
        const std::size_t first_index = current_aggressions.front();
        const auto& first = history[first_index];
        out.opening_line = classify_first_bet(history, out.street, first.actor, first_index, lineage);
        out.raise_depth = static_cast<std::uint8_t>(
            std::min<std::size_t>(current_aggressions.size() - 1U, 255U)
        );
        out.facing_line = out.raise_depth > 0U ? PostflopLineType::Raise : out.opening_line;
        return out;
    }

    // No bet exists yet on the street. Classify what semantic line the current
    // actor would create by making the opening bet. This is an opportunity
    // feature only; it does not tell the policy to bet.
    out.attack_opportunity = classify_first_bet(
        history,
        out.street,
        out.actor,
        history.size(),
        lineage
    );
    return out;
}

const char* postflop_line_name(PostflopLineType line) noexcept {
    switch (line) {
        case PostflopLineType::None: return "none";
        case PostflopLineType::CBet: return "cbet";
        case PostflopLineType::DonkBet: return "donk_bet";
        case PostflopLineType::ProbeBet: return "probe_bet";
        case PostflopLineType::FloatBet: return "float_bet";
        case PostflopLineType::DelayedFloatBet: return "delayed_float_bet";
        case PostflopLineType::DelayedCBet: return "delayed_cbet";
        case PostflopLineType::DoubleDelayedCBet: return "double_delayed_cbet";
        case PostflopLineType::GenericBet: return "generic_bet";
        case PostflopLineType::Raise: return "raise";
    }
    return "unknown";
}

}  // namespace spincore
