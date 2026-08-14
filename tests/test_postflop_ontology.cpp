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

void three_way_btn_raise_two_calls(BettingEngine& betting) {
    REQUIRE(betting.actor() == 0);
    auto open = betting.legal_actions(0);
    REQUIRE(open.raise);
    betting.apply(0, {ExactActionType::RaiseTo, open.min_raise_to});
    REQUIRE(betting.actor() == 1);
    betting.apply(1, {ExactActionType::Call, 0});
    REQUIRE(betting.actor() == 2);
    betting.apply(2, {ExactActionType::Call, 0});
    REQUIRE(betting.street_complete());
    betting.advance_street();
    REQUIRE(betting.street() == Street::Flop);
    REQUIRE(betting.actor() == 1); // SB first postflop with dealer/BTN at seat 0.
}

void three_way_bb_threebet_both_call(BettingEngine& betting) {
    REQUIRE(betting.actor() == 0);
    auto open = betting.legal_actions(0);
    REQUIRE(open.raise);
    betting.apply(0, {ExactActionType::RaiseTo, open.min_raise_to});
    REQUIRE(betting.actor() == 1);
    betting.apply(1, {ExactActionType::Call, 0});
    REQUIRE(betting.actor() == 2);
    auto threebet = betting.legal_actions(2);
    REQUIRE(threebet.raise);
    betting.apply(2, {ExactActionType::RaiseTo, threebet.min_raise_to});
    REQUIRE(betting.actor() == 0);
    betting.apply(0, {ExactActionType::Call, 0});
    REQUIRE(betting.actor() == 1);
    betting.apply(1, {ExactActionType::Call, 0});
    REQUIRE(betting.street_complete());
    betting.advance_street();
    REQUIRE(betting.street() == Street::Flop);
    REQUIRE(betting.actor() == 1);
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

SPIN_TEST(postflop_ontology_three_way_cbet_survives_two_preflop_callers_and_partial_fold) {
    auto scenario = sc3(0); // BTN 0, SB 1, BB 2.
    BettingEngine betting(scenario, make_game_topology(scenario));
    three_way_btn_raise_two_calls(betting);

    betting.apply(1, {ExactActionType::Check, 0});
    betting.apply(2, {ExactActionType::Check, 0});
    auto opportunity = derive_postflop_ontology(betting);
    REQUIRE(opportunity.actor == 0);
    REQUIRE(opportunity.lineage_aggressor == 0);
    REQUIRE(opportunity.attack_opportunity == PostflopLineType::CBet);

    bet_min(betting); // BTN c-bets.
    auto sb_facing = derive_postflop_ontology(betting);
    REQUIRE(sb_facing.actor == 1);
    REQUIRE(sb_facing.opening_line == PostflopLineType::CBet);
    REQUIRE(sb_facing.facing_line == PostflopLineType::CBet);
    betting.apply(1, {ExactActionType::Fold, 0});

    auto bb_facing = derive_postflop_ontology(betting);
    REQUIRE(bb_facing.actor == 2);
    REQUIRE(bb_facing.opening_line == PostflopLineType::CBet);
    REQUIRE(bb_facing.facing_line == PostflopLineType::CBet);
    REQUIRE(bb_facing.current_street_aggression_count == 1);
}

SPIN_TEST(postflop_ontology_three_way_caller_lead_before_aggressor_is_donk) {
    auto scenario = sc3(0);
    BettingEngine betting(scenario, make_game_topology(scenario));
    three_way_btn_raise_two_calls(betting);

    REQUIRE(betting.actor() == 1);
    bet_min(betting); // SB was a caller and acts before BTN aggressor.
    auto context = derive_postflop_ontology(betting);
    REQUIRE(context.actor == 2);
    REQUIRE(context.lineage_aggressor == 0);
    REQUIRE(context.opening_line == PostflopLineType::DonkBet);
    REQUIRE(context.facing_line == PostflopLineType::DonkBet);
}

SPIN_TEST(postflop_ontology_three_way_probe_after_missed_cbet) {
    auto scenario = sc3(0);
    BettingEngine betting(scenario, make_game_topology(scenario));
    three_way_btn_raise_two_calls(betting);

    betting.apply(1, {ExactActionType::Check, 0});
    betting.apply(2, {ExactActionType::Check, 0});
    betting.apply(0, {ExactActionType::Check, 0});
    REQUIRE(betting.street_complete());
    betting.advance_street();
    REQUIRE(betting.actor() == 1);

    auto context = derive_postflop_ontology(betting);
    REQUIRE(context.street == Street::Turn);
    REQUIRE(context.lineage_aggressor == 0);
    REQUIRE(context.skipped_streets_since_lineage == 1);
    REQUIRE(context.attack_opportunity == PostflopLineType::ProbeBet);
}

SPIN_TEST(postflop_ontology_three_way_float_after_bb_threebet_and_check) {
    auto scenario = sc3(0);
    BettingEngine betting(scenario, make_game_topology(scenario));
    three_way_bb_threebet_both_call(betting);

    // SB caller acts first and checks; BB is the preflop aggressor and checks;
    // BTN also called the 3-bet and now has an attack opportunity with SB still
    // active behind. This is deliberately multiway, not a HU reconstruction.
    betting.apply(1, {ExactActionType::Check, 0});
    REQUIRE(betting.actor() == 2);
    betting.apply(2, {ExactActionType::Check, 0});
    REQUIRE(betting.actor() == 0);

    auto context = derive_postflop_ontology(betting);
    REQUIRE(context.lineage_aggressor == 2);
    REQUIRE(context.actor == 0);
    REQUIRE(context.actor_called_lineage_aggression);
    REQUIRE(context.attack_opportunity == PostflopLineType::FloatBet);
}
