#pragma once

#include "spincore/lt2_deployment_runtime.hpp"
#include "spincore/lt2_openholdem_runtime.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>

namespace spincore::lt2oh {

inline constexpr const char* kFrozenNativeBundleFileSha256 =
    "2b79ab7ff746a9c1c3dd73dbc0a1d6884a471813cf34b9cb126790c4c4cbb123";
inline constexpr const char* kFrozenPythonBundleSha256 =
    "87e46b40cb43bb89cb46bf3b760bbac5c8282491fd3d6da73d1bbe28329b278c";
inline constexpr const char* kFrozenCheckpointSha256 =
    "a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf";
inline constexpr const char* kFrozenEnsembleSha256 =
    "c44b817f75304db352eedc33febbd180e6d038e0580c91be0b9befdbdb08f181";

struct ShadowDecision {
    std::uint64_t tracker_generation{0};
    std::uint64_t decision_index{0};
    std::int32_t domain{0}; // 0 THREE_HANDED, 1 TRUE_HEADS_UP
    std::int32_t action_slot{-1};
    ExactAction exact{};
    std::array<float,lt2::kActionCount> probabilities{};
    std::array<std::uint8_t,lt2::kActionCount> legal_mask{};
    double sample_u{0.0};

    friend bool operator==(const ShadowDecision&, const ShadowDecision&) = default;
};

[[nodiscard]] lt2::DeploymentBundle load_frozen_deployment_bundle(
    const std::filesystem::path& path);

class NativeShadowDecisionEngine {
public:
    NativeShadowDecisionEngine(lt2::DeploymentBundle bundle,std::uint64_t sampling_seed);

    void reset() noexcept;
    [[nodiscard]] SyncResult start_hand(
        const HandAnchor& anchor,
        const ObservedSnapshot& observed);
    [[nodiscard]] SyncResult heartbeat(
        const HandAnchor& anchor,
        const ObservedSnapshot& observed);

    [[nodiscard]] std::optional<ShadowDecision> on_my_turn(
        const HandAnchor& anchor,
        const ObservedSnapshot& observed);

    [[nodiscard]] std::optional<ShadowDecision> cached_decision() const noexcept;
    [[nodiscard]] const ObservableRuntimeTracker& tracker() const noexcept { return tracker_; }
    [[nodiscard]] std::uint64_t inference_count() const noexcept { return inference_count_; }
    [[nodiscard]] std::uint64_t decision_count() const noexcept { return decision_count_; }

private:
    lt2::DeploymentBundle bundle_;
    ObservableRuntimeTracker tracker_{};
    std::uint64_t rng_state_{0};
    std::uint64_t inference_count_{0};
    std::uint64_t decision_count_{0};
    std::optional<ShadowDecision> cached_{};

    void clear_cache() noexcept { cached_.reset(); }
    [[nodiscard]] std::uint64_t next_u64() noexcept;
    [[nodiscard]] double next_unit() noexcept;
    [[nodiscard]] ShadowDecision compute_decision();
};

} // namespace spincore::lt2oh
