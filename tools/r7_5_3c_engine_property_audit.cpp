#include "spincore/hand_engine.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace spincore;

namespace {

EpisodeScenario make_three_handed(std::mt19937_64& rng, int dealer) {
    EpisodeScenario s{};
    s.state.total_chips = 1500;
    s.state.game_is_hu = false;
    s.state.small_blind = 10;
    s.state.big_blind = 20;
    // Positive integer partition, deliberately including shallow/asymmetric states.
    std::uniform_int_distribution<int> first(1, 1498);
    const int a = first(rng);
    std::uniform_int_distribution<int> second(1, 1499 - a);
    const int b = second(rng);
    const int c = 1500 - a - b;
    s.state.stacks = {a, b, c};
    s.dealer_id = dealer;
    return s;
}

EpisodeScenario make_true_hu(std::mt19937_64& rng, int dead, int dealer) {
    EpisodeScenario s{};
    s.state.total_chips = 1500;
    s.state.game_is_hu = true;
    s.state.small_blind = 10;
    s.state.big_blind = 20;
    std::uniform_int_distribution<int> split(1, 1499);
    const int first_stack = split(rng);
    const int second_stack = 1500 - first_stack;
    s.state.stacks = {0, 0, 0};
    std::array<int, 2> live{};
    int cursor = 0;
    for (int seat = 0; seat < 3; ++seat) {
        if (seat != dead) {
            live[static_cast<std::size_t>(cursor++)] = seat;
        }
    }
    s.state.stacks[static_cast<std::size_t>(live[0])] = first_stack;
    s.state.stacks[static_cast<std::size_t>(live[1])] = second_stack;
    s.state.dead_players = {dead, -1, -1};
    s.state.dead_player_count = 1;
    s.dealer_id = dealer;
    return s;
}

void assert_public_chip_invariants(const HandEngine& hand) {
    const auto& betting = hand.betting();
    std::int64_t stack_sum = 0;
    std::int64_t commitment_sum = 0;
    for (const auto& player : betting.players()) {
        if (player.stack < 0 || player.street_commitment < 0 || player.total_commitment < 0) {
            throw std::runtime_error("negative player chip field");
        }
        if (player.street_commitment > player.total_commitment) {
            throw std::runtime_error("street commitment exceeds total commitment");
        }
        stack_sum += player.stack;
        commitment_sum += player.total_commitment;
    }
    if (betting.pot() < 0) {
        throw std::runtime_error("negative pot");
    }
    if (commitment_sum != betting.pot()) {
        throw std::runtime_error("pot != sum(total_commitments)");
    }
    if (stack_sum + static_cast<std::int64_t>(betting.pot()) != hand.scenario().state.total_chips) {
        throw std::runtime_error("chip conservation violated during hand");
    }
}

void assert_deal_unique(const HandEngine& hand) {
    std::array<bool, 52> seen{};
    const auto& scenario = hand.scenario();
    for (int seat = 0; seat < 3; ++seat) {
        const bool live = scenario.state.stacks[static_cast<std::size_t>(seat)] > 0;
        for (const auto& card : hand.hole_cards()[static_cast<std::size_t>(seat)]) {
            if (!live) {
                if (card.valid()) {
                    throw std::runtime_error("dead seat received a valid hole card");
                }
                continue;
            }
            if (!card.valid()) {
                throw std::runtime_error("live seat missing hole card");
            }
            const auto id = static_cast<std::size_t>(card.id());
            if (seen[id]) {
                throw std::runtime_error("duplicate hole card");
            }
            seen[id] = true;
        }
    }
    for (const auto& card : hand.board()) {
        if (!card.valid()) {
            throw std::runtime_error("predealt board contains invalid card");
        }
        const auto id = static_cast<std::size_t>(card.id());
        if (seen[id]) {
            throw std::runtime_error("duplicate board/hole card");
        }
        seen[id] = true;
    }
}

std::vector<ExactAction> legal_exact_choices(const BettingEngine& betting, int actor) {
    const auto legal = betting.legal_actions(actor);
    std::vector<ExactAction> choices;
    if (legal.fold) choices.push_back({ExactActionType::Fold, 0});
    if (legal.check) choices.push_back({ExactActionType::Check, 0});
    if (legal.call) choices.push_back({ExactActionType::Call, 0});
    if (legal.bet) {
        choices.push_back({ExactActionType::BetTo, legal.min_raise_to});
        if (legal.max_raise_to > legal.min_raise_to) {
            const int midpoint = legal.min_raise_to + (legal.max_raise_to - legal.min_raise_to) / 2;
            if (midpoint > legal.min_raise_to && midpoint < legal.max_raise_to) {
                choices.push_back({ExactActionType::BetTo, midpoint});
            }
        }
    }
    if (legal.raise) {
        choices.push_back({ExactActionType::RaiseTo, legal.min_raise_to});
        if (legal.max_raise_to > legal.min_raise_to) {
            const int midpoint = legal.min_raise_to + (legal.max_raise_to - legal.min_raise_to) / 2;
            if (midpoint > legal.min_raise_to && midpoint < legal.max_raise_to) {
                choices.push_back({ExactActionType::RaiseTo, midpoint});
            }
        }
    }
    if (legal.all_in) choices.push_back({ExactActionType::AllIn, 0});
    return choices;
}

void play_one(const EpisodeScenario& scenario, std::uint64_t deck_seed, std::mt19937_64& action_rng) {
    HandEngine hand(scenario, deck_seed);
    assert_deal_unique(hand);
    assert_public_chip_invariants(hand);

    std::size_t actions = 0;
    while (!hand.terminal()) {
        const int actor = hand.betting().actor();
        if (actor < 0 || actor >= 3) {
            throw std::runtime_error("nonterminal state has invalid actor");
        }
        const auto choices = legal_exact_choices(hand.betting(), actor);
        if (choices.empty()) {
            throw std::runtime_error("nonterminal actor has no legal exact action");
        }
        std::uniform_int_distribution<std::size_t> pick(0, choices.size() - 1);
        hand.apply(actor, choices[pick(action_rng)]);
        ++actions;
        if (actions > 256) {
            throw std::runtime_error("legal hand exceeded 256 actions / possible loop");
        }
        assert_public_chip_invariants(hand);
    }

    const auto settlement = hand.settle();
    std::int64_t final_sum = 0;
    for (int stack : settlement.final_stacks) {
        if (stack < 0) {
            throw std::runtime_error("negative final settlement stack");
        }
        final_sum += stack;
    }
    if (final_sum != scenario.state.total_chips) {
        throw std::runtime_error("terminal settlement violates chip conservation");
    }
}

} // namespace

int main() {
    constexpr int kThreeHandedHands = 3000;
    constexpr int kHeadsUpHands = 3000;
    std::mt19937_64 scenario_rng(0x53A1D17ULL);
    std::mt19937_64 action_rng(0xA0F5EEDULL);

    for (int i = 0; i < kThreeHandedHands; ++i) {
        const int dealer = i % 3;
        auto scenario = make_three_handed(scenario_rng, dealer);
        play_one(scenario, scenario_rng(), action_rng);
    }

    for (int i = 0; i < kHeadsUpHands; ++i) {
        const int dead = i % 3;
        const int live_a = (dead + 1) % 3;
        const int live_b = (dead + 2) % 3;
        const int dealer = (i & 1) ? live_a : live_b;
        auto scenario = make_true_hu(scenario_rng, dead, dealer);
        play_one(scenario, scenario_rng(), action_rng);
    }

    std::cout << "R7.5.3C engine property audit PASS: "
              << kThreeHandedHands << " 3H + " << kHeadsUpHands << " HU hands\n";
    return 0;
}
