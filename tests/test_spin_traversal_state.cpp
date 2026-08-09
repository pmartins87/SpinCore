#include "test_framework.hpp"
#include "test_helpers.hpp"
#include "spincore/spin_traversal_state.hpp"

using namespace spincore;

SPIN_TEST(traversal_clone_preserves_hidden_deal_and_parent) {
    EpisodeScenario sc{}; sc.state.total_chips=1500; sc.state.game_is_hu=false; sc.state.blind_index=0; sc.state.small_blind=10; sc.state.big_blind=20; sc.state.stacks={500,500,500}; sc.dealer_id=0;
    SpinTraversalState root(sc, 1234567);
    const auto before_hole = root.hand().hole_cards();
    const auto before_board = root.hand().board();
    const auto before_hist = root.hand().betting().history();

    auto legal = root.legal_abstract_actions();
    REQUIRE(!legal.empty());
    auto child = root.child(legal.front());

    REQUIRE(root.hand().hole_cards() == before_hole);
    REQUIRE(root.hand().board() == before_board);
    REQUIRE(root.hand().betting().history() == before_hist);
    REQUIRE(child.hand().hole_cards() == before_hole);
    REQUIRE(child.hand().board() == before_board);
    REQUIRE(child.hand().betting().history().size() >= before_hist.size());
}

SPIN_TEST(traversal_infoset_never_exposes_opponent_private_cards) {
    EpisodeScenario sc{}; sc.state.total_chips=1500; sc.state.game_is_hu=false; sc.state.blind_index=0; sc.state.small_blind=10; sc.state.big_blind=20; sc.state.stacks={500,500,500}; sc.dealer_id=1;
    SpinTraversalState s(sc, 99);
    const auto input = s.neural_input();

    int nonzero=0;
    for (auto x: input.card_tokens) if (x) ++nonzero;
    REQUIRE(nonzero == 2);
}

SPIN_TEST(traversal_can_reach_terminal_and_is_zero_sum_in_chips) {
    EpisodeScenario sc{}; sc.state.total_chips=1500; sc.state.game_is_hu=true; sc.state.blind_index=0; sc.state.small_blind=10; sc.state.big_blind=20; sc.state.stacks={0,750,750}; sc.state.dead_players={0,-1,-1}; sc.state.dead_player_count=1; sc.dealer_id=1;
    SpinTraversalState s(sc, 777);
    int guard=0;
    while (!s.terminal() && guard++ < 64) {
        const auto acts=s.legal_abstract_actions();
        REQUIRE(!acts.empty());
        auto a=acts.front();
        for (auto x: acts) if (x==AbstractActionSlot::CheckCall) { a=x; break; }
        s.apply(a);
    }
    REQUIRE(s.terminal());
    const auto d=s.terminal_chip_delta();
    REQUIRE(d[0]+d[1]+d[2] == 0);
}
