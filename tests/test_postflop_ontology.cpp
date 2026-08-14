#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "spincore/postflop_ontology.hpp"

using namespace spincore;

namespace {

void hu_single_raise_call(BettingEngine& betting) {
    auto open = betting.legal_actions(1);
    REQUIRE(open.raise);
    betting.apply(1, {ExactActionType::RaiseTo, open.min_raise_to});
    REQUIRE(betting.actor() == 2);
    betting.apply(2, {ExactActionType::Call, 0});
    REQUIRE(betting.street_complete());
    betting.advance_street();
    REQUIRE(betting.street() == Street::Flop);
    REQUIRE(betting.actor() == 2);
}

void hu_bb_threebet_sb_call(BettingEngine& betting) {
    auto open = betting.legal_actions(1);
    REQUIRE(open.raise);
    betting.apply(1, {ExactActionType::RaiseTo, open.min_raise_to});
    auto threebet = betting.legal_actions(2);
    REQUIRE(threebet.raise);
    betting.apply(2, {ExactActionType::RaiseTo, threebet.min_raise_to});
    REQUIRE(betting.actor() == 1);
    betting.apply(1, {ExactActionType::Call, 0});
    REQUIRE(betting.street_complete());
    betting.advance_street();
    REQUIRE(betting.street() == Street::Flop);
    REQUIRE(betting.actor() == 2);
}

void bet_min(BettingEngine& betting) {
    const int actor = betting.actor();
    auto legal = betting.legal_actions(actor);
    REQUIRE(legal.bet);
    betting.apply(actor, {ExactActionType::BetTo, legal.min_raise_to});
}

void raise_min(BettingEngine& betting) {
    const int actor = betting.actor();
    auto legal = betting.legal_actions(actor);
    REQUIRE(legal.raise);
    betting.apply(actor, {ExactActionType::RaiseTo, legal.min_raise_to});
}

}  // namespace

SPIN_TEST(postflop_ontology_distinguishes_cbet_from_donk) {
    {
        auto scenario = schu(1);
        BettingEngine betting(scenario, make_game_topology(scenario));
        hu_single_raise_call(betting);
        betting.apply(2, {ExactActionType::Check, 0});
        auto opportunity = derive_postflop_ontology(betting);
        REQUIRE(opportunity.attack_opportunity == PostflopLineType::CBet);
        bet_min(betting);
        auto facing = derive_postflop_ontology(betting);
        REQUIRE(facing.opening_line == PostflopLineType::CBet);
        REQUIRE(facing.facing_line == PostflopLineType::CBet);
        REQUIRE(facing.raise_depth == 0);
    }
    {
        auto scenario = schu(1);
        BettingEngine betting(scenario, make_game_topology(scenario));
        hu_single_raise_call(betting);
        bet_min(betting);
        auto facing = derive_postflop_ontology(betting);
        REQUIRE(facing.opening_line == PostflopLineType::DonkBet);
        REQUIRE(facing.facing_line == PostflopLineType::DonkBet);
    }
}

SPIN_TEST(postflop_ontology_detects_probe_after_missed_flop_cbet) {
    auto scenario = schu(1);
    BettingEngine betting(scenario, make_game_topology(scenario));
    hu_single_raise_call(betting);
    betting.apply(2, {ExactActionType::Check, 0});
    betting.apply(1, {ExactActionType::Check, 0});
    REQUIRE(betting.street_complete());
    betting.advance_street();
    auto context = derive_postflop_ontology(betting);
    REQUIRE(context.street == Street::Turn);
    REQUIRE(context.skipped_streets_since_lineage == 1);
    REQUIRE(context.attack_opportunity == PostflopLineType::ProbeBet);
}

SPIN_TEST(postflop_ontology_detects_delayed_and_double_delayed_cbet) {
    auto scenario = schu(1);
    BettingEngine betting(scenario, make_game_topology(scenario));
    hu_single_raise_call(betting);

    betting.apply(2, {ExactActionType::Check, 0});
    betting.apply(1, {ExactActionType::Check, 0});
    betting.advance_street();
    betting.apply(2, {ExactActionType::Check, 0});
    auto delayed = derive_postflop_ontology(betting);
    REQUIRE(delayed.attack_opportunity == PostflopLineType::DelayedCBet);
    betting.apply(1, {ExactActionType::Check, 0});

    betting.advance_street();
    betting.apply(2, {ExactActionType::Check, 0});
    auto double_delayed = derive_postflop_ontology(betting);
    REQUIRE(double_delayed.street == Street::River);
    REQUIRE(double_delayed.skipped_streets_since_lineage == 2);
    REQUIRE(double_delayed.attack_opportunity == PostflopLineType::DoubleDelayedCBet);
}

SPIN_TEST(postflop_ontology_detects_float_and_delayed_float) {
    {
        auto scenario = schu(1);
        BettingEngine betting(scenario, make_game_topology(scenario));
        hu_bb_threebet_sb_call(betting);
        betting.apply(2, {ExactActionType::Check, 0});
        auto context = derive_postflop_ontology(betting);
        REQUIRE(context.lineage_aggressor == 2);
        REQUIRE(context.actor == 1);
        REQUIRE(context.actor_called_lineage_aggression);
        REQUIRE(context.attack_opportunity == PostflopLineType::FloatBet);
    }
    {
        auto scenario = schu(1);
        BettingEngine betting(scenario, make_game_topology(scenario));
        hu_bb_threebet_sb_call(betting);
        betting.apply(2, {ExactActionType::Check, 0});
        betting.apply(1, {ExactActionType::Check, 0});
        betting.advance_street();
        betting.apply(2, {ExactActionType::Check, 0});
        auto context = derive_postflop_ontology(betting);
        REQUIRE(context.skipped_streets_since_lineage == 1);
        REQUIRE(context.actor_called_lineage_aggression);
        REQUIRE(context.attack_opportunity == PostflopLineType::DelayedFloatBet);
    }
}

SPIN_TEST(postflop_ontology_preserves_opening_line_when_facing_raise) {
    auto scenario = schu(1);
    BettingEngine betting(scenario, make_game_topology(scenario));
    hu_single_raise_call(betting);
    betting.apply(2, {ExactActionType::Check, 0});
    bet_min(betting);
    raise_min(betting);

    auto context = derive_postflop_ontology(betting);
    REQUIRE(context.actor == 1);
    REQUIRE(context.current_street_aggression_count == 2);
    REQUIRE(context.opening_line == PostflopLineType::CBet);
    REQUIRE(context.facing_line == PostflopLineType::Raise);
    REQUIRE(context.raise_depth == 1);
    REQUIRE(context.to_call > 0);
}
