#pragma once

#include "spincore/ruleset_contract.hpp"
#include "spincore/spin_traversal_state.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace spincore::lt2oh {

struct RawFrame {
    std::string hand_id;
    std::int32_t user_chair{-1};
    std::int32_t dealer_chair{-1};
    std::int32_t betround{1};
    std::int32_t small_blind{0};
    std::int32_t big_blind{0};
    std::uint32_t players_dealt_bits{0};
    std::uint32_t players_playing_bits{0};
    std::uint32_t players_allin_bits{0};
    std::array<std::int32_t,10> balances{};
    std::array<std::int32_t,10> current_bets{};
    std::int32_t pot{0};
    std::int32_t common_cards_known{0};
    std::array<std::int32_t,2> hero_cards{-1,-1};
    std::array<std::int32_t,5> board_cards{-1,-1,-1,-1,-1};
};

struct HandAnchor {
    std::string hand_id;
    EpisodeScenario scenario{};
    std::array<std::int32_t,3> logical_to_chair{-1,-1,-1};
    std::array<std::int32_t,10> chair_to_logical{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
    std::int32_t hero_logical_seat{-1};
    std::array<std::int32_t,2> hero_cards{-1,-1};

    friend bool operator==(const HandAnchor&, const HandAnchor&) = default;
};

struct ObservedSnapshot {
    std::int32_t street{0}; // OpenHoldem card-derived street, 0..3
    std::int32_t pot{0};
    std::array<std::int32_t,3> stacks{};
    std::array<std::int32_t,3> street_commitments{};
    std::array<std::uint8_t,3> folded{};
    std::array<std::uint8_t,3> all_in{};
    std::int32_t visible_board_count{0};
    std::array<std::int32_t,5> board_cards{-1,-1,-1,-1,-1};

    friend bool operator==(const ObservedSnapshot&, const ObservedSnapshot&) = default;
};

enum class SyncKind : std::uint8_t { Start=0, NoChange=1, Action=2, Failed=3 };

struct SyncResult {
    SyncKind kind{SyncKind::NoChange};
    std::optional<ExactAction> action{};
    std::string reason{};
};

[[nodiscard]] std::int32_t openholdem_betround_from_visible_count(std::int32_t visible);
[[nodiscard]] std::int32_t openholdem_card_id_from_rank_suit(
    std::int32_t rank,
    std::int32_t openholdem_suit);
[[nodiscard]] HandAnchor anchor_from_raw_frame(const RawFrame& frame);
[[nodiscard]] ObservedSnapshot normalize_raw_frame(const RawFrame& frame, const HandAnchor& anchor);
[[nodiscard]] ObservedSnapshot observable_projection(
    const SpinTraversalState& state,
    const std::array<std::int32_t,5>& visible_board_cards);

class ObservableRuntimeTracker {
public:
    void reset() noexcept;
    [[nodiscard]] SyncResult start_hand(const HandAnchor& anchor, const ObservedSnapshot& observed);
    [[nodiscard]] SyncResult heartbeat(const HandAnchor& anchor, const ObservedSnapshot& observed);
    [[nodiscard]] SyncResult synchronize_to_actor(
        const HandAnchor& anchor,
        const ObservedSnapshot& observed,
        std::int32_t target_actor);

    [[nodiscard]] bool failed() const noexcept { return failed_; }
    [[nodiscard]] const std::string& failure_reason() const noexcept { return failure_reason_; }
    [[nodiscard]] const std::vector<ExactAction>& transcript() const noexcept { return transcript_; }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
    [[nodiscard]] const SpinTraversalState* state() const noexcept {
        return state_.has_value() && !failed_ ? &*state_ : nullptr;
    }

private:
    HandAnchor anchor_{};
    bool have_anchor_{false};
    std::optional<SpinTraversalState> state_{};
    std::optional<ObservedSnapshot> observed_{};
    std::vector<ExactAction> transcript_{};
    bool failed_{false};
    std::string failure_reason_{};
    std::uint64_t generation_{0};

    [[nodiscard]] SyncResult fail(std::string reason);
    [[nodiscard]] SpinTraversalState rebuild(
        const ObservedSnapshot& observed,
        const std::vector<ExactAction>& transcript) const;
    [[nodiscard]] SyncResult sync(
        const HandAnchor& anchor,
        const ObservedSnapshot& observed,
        std::optional<std::int32_t> target_actor);
};

} // namespace spincore::lt2oh
