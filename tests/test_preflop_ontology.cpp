#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "spincore/preflop_ontology.hpp"

using namespace spincore;

SPIN_TEST(preflop_ontology_ignores_forced_blinds) {
    auto s = sc3(0);
    BettingEngine b(s, make_game_topology(s));
    const auto o = derive_preflop_ontology(b);
    REQUIRE(o.lineage == PreflopLineageType::Unopened);
    REQUIRE(o.voluntary_action_count == 0);
    REQUIRE(o.aggression_count == 0);
    REQUIRE(o.limp_count == 0);
}

SPIN_TEST(preflop_ontology_detects_limp_and_open_raise) {
    auto s = sc3(0);
    BettingEngine limp(s, make_game_topology(s));
    limp.apply(0, {ExactActionType::Call, 0});
    auto lo = derive_preflop_ontology(limp);
    REQUIRE(lo.lineage == PreflopLineageType::Limped);
    REQUIRE(lo.limp_count == 1);

    BettingEngine raised(s, make_game_topology(s));
    auto legal = raised.legal_actions(0);
    raised.apply(0, {ExactActionType::RaiseTo, legal.min_raise_to});
    auto ro = derive_preflop_ontology(raised);
    REQUIRE(ro.lineage == PreflopLineageType::OpenRaised);
    REQUIRE(ro.aggression_count == 1);
    REQUIRE(ro.first_aggressor == 0);
    REQUIRE(ro.first_raise_to == legal.min_raise_to);
}

SPIN_TEST(preflop_ontology_detects_raise_over_limp) {
    auto s = sc3(0);
    BettingEngine b(s, make_game_topology(s));
    b.apply(0, {ExactActionType::Call, 0});
    auto legal = b.legal_actions(1);
    b.apply(1, {ExactActionType::RaiseTo, legal.min_raise_to});
    const auto o = derive_preflop_ontology(b);
    REQUIRE(o.lineage == PreflopLineageType::RaiseOverLimp);
    REQUIRE(o.had_limp_before_first_aggression);
    REQUIRE(o.aggression_count == 1);
}

SPIN_TEST(preflop_ontology_detects_reraise_and_true_limp_reraise) {
    auto s = sc3(0);

    BettingEngine reraised(s, make_game_topology(s));
    auto open = reraised.legal_actions(0);
    reraised.apply(0, {ExactActionType::RaiseTo, open.min_raise_to});
    auto threebet = reraised.legal_actions(1);
    reraised.apply(1, {ExactActionType::RaiseTo, threebet.min_raise_to});
    const auto rr = derive_preflop_ontology(reraised);
    REQUIRE(rr.lineage == PreflopLineageType::Reraised);
    REQUIRE(rr.aggression_count == 2);

    BettingEngine limp_rr(s, make_game_topology(s));
    limp_rr.apply(0, {ExactActionType::Call, 0});
    auto iso = limp_rr.legal_actions(1);
    limp_rr.apply(1, {ExactActionType::RaiseTo, iso.min_raise_to});
    limp_rr.apply(2, {ExactActionType::Call, 0});
    auto backraise = limp_rr.legal_actions(0);
    limp_rr.apply(0, {ExactActionType::RaiseTo, backraise.min_raise_to});
    const auto lr = derive_preflop_ontology(limp_rr);
    REQUIRE(lr.lineage == PreflopLineageType::LimpReraised);
    REQUIRE(lr.limper_became_reraiser);
    REQUIRE(lr.aggression_count == 2);
}
