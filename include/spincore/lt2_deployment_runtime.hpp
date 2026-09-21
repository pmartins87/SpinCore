#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace spincore::lt2 {

inline constexpr std::size_t kActionCount = 10;
inline constexpr std::size_t kObservationSizeV1 = 126;
inline constexpr std::size_t kHuEnsembleSize = 8;

enum class DeploymentModelKind : std::uint32_t {
    ThreeHandedAveragePolicy = 1,
    HeadsUpAdvantage = 2,
};

struct TensorF32 {
    std::vector<std::uint32_t> dims;
    std::vector<float> values;
};

struct ActionModelWeights {
    DeploymentModelKind kind{DeploymentModelKind::ThreeHandedAveragePolicy};
    std::uint32_t member_index{0};
    std::unordered_map<std::string, TensorF32> tensors;
};

struct DeploymentBundle {
    std::string deployment_bundle_sha256;
    std::string source_checkpoint_sha256;
    std::string source_ensemble_sha256;
    ActionModelWeights three_handed_policy;
    std::array<ActionModelWeights, kHuEnsembleSize> hu_members;
};

struct DecodedObservationV1 {
    std::array<std::uint8_t, 7> cards{};
    std::array<float, 16> numeric{};
    std::array<std::uint8_t, 8> categorical{};
    std::array<std::uint8_t, 32> history{};
};

[[nodiscard]] DeploymentBundle load_deployment_bundle(
    const std::filesystem::path& path);

[[nodiscard]] DecodedObservationV1 decode_observation_v1(
    const std::array<std::uint8_t, kObservationSizeV1>& bytes);

[[nodiscard]] std::array<float, kActionCount> infer_raw_action_model(
    const ActionModelWeights& model,
    const DecodedObservationV1& observation);

[[nodiscard]] std::array<float, kActionCount> infer_three_handed_policy(
    const DeploymentBundle& bundle,
    const DecodedObservationV1& observation,
    const std::array<std::uint8_t, kActionCount>& legal_mask);

[[nodiscard]] std::array<float, kActionCount> infer_heads_up_ensemble_policy(
    const DeploymentBundle& bundle,
    const DecodedObservationV1& observation,
    const std::array<std::uint8_t, kActionCount>& legal_mask);

}  // namespace spincore::lt2
