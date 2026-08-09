#pragma once
#include "spincore/ruleset_contract.hpp"
#include <cstdint>
namespace spincore {
struct ScenarioSamplerConfig { std::int32_t total_chips{1500}; std::int32_t small_blind{10}; std::int32_t big_blind{20}; };
class ScenarioSampler { public: explicit ScenarioSampler(std::uint64_t seed, ScenarioSamplerConfig cfg={}); [[nodiscard]] EpisodeScenario sample(bool true_hu); private: std::uint64_t state_; ScenarioSamplerConfig cfg_; std::uint64_t next(); };
}
