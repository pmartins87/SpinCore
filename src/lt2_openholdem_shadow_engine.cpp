#include "spincore/lt2_openholdem_shadow_engine.hpp"

#include "spincore/game_topology.hpp"
#include "spincore/lean_action_abstraction.hpp"
#include "spincore/neural_encoder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

namespace spincore::lt2oh {
namespace {

UniversalActionMaskV2 frozen_active_mask() {
    UniversalActionMaskV2 mask{};
    for (const int slot : {0,1,3,5,7,8,9}) {
        mask[static_cast<std::size_t>(slot)]=1U;
    }
    return mask;
}

void validate_identity(const lt2::DeploymentBundle& bundle) {
    if (bundle.deployment_bundle_sha256!=kFrozenPythonBundleSha256) {
        throw std::runtime_error("LT2 deployment source-bundle identity mismatch");
    }
    if (bundle.source_checkpoint_sha256!=kFrozenCheckpointSha256) {
        throw std::runtime_error("LT2 checkpoint identity mismatch");
    }
    if (bundle.source_ensemble_sha256!=kFrozenEnsembleSha256) {
        throw std::runtime_error("LT2 ensemble identity mismatch");
    }
}

std::array<std::uint8_t,lt2::kObservationSizeV1> observation_bytes(
    const SpinTraversalState& state) {
    const auto bytes=serialize_neural_input_v1(state.neural_input());
    if (bytes.size()!=lt2::kObservationSizeV1) {
        throw std::runtime_error("SPNNIV1 deployment length drift");
    }
    std::array<std::uint8_t,lt2::kObservationSizeV1> out{};
    std::copy(bytes.begin(),bytes.end(),out.begin());
    return out;
}

std::array<std::uint8_t,lt2::kActionCount> legal_mask(
    const std::vector<ResolvedUniversalActionV2>& legal) {
    std::array<std::uint8_t,lt2::kActionCount> out{};
    for (const auto& action:legal) {
        const auto slot=static_cast<std::size_t>(action.slot);
        if (slot>=out.size()) throw std::runtime_error("lean action slot outside deployment carrier");
        out[slot]=1U;
    }
    return out;
}

const ResolvedUniversalActionV2& find_slot(
    const std::vector<ResolvedUniversalActionV2>& legal,
    std::int32_t slot) {
    const auto it=std::find_if(legal.begin(),legal.end(),[&](const auto& item) {
        return static_cast<std::int32_t>(item.slot)==slot;
    });
    if (it==legal.end()) throw std::runtime_error("sampled deployment slot not legal");
    return *it;
}

} // namespace

lt2::DeploymentBundle load_frozen_deployment_bundle(const std::filesystem::path& path) {
    auto bundle=lt2::load_deployment_bundle(path);
    validate_identity(bundle);
    return bundle;
}

NativeShadowDecisionEngine::NativeShadowDecisionEngine(
    lt2::DeploymentBundle bundle,
    std::uint64_t sampling_seed)
    :bundle_(std::move(bundle)),rng_state_(sampling_seed) {
    validate_identity(bundle_);
}

void NativeShadowDecisionEngine::reset() noexcept {
    tracker_.reset();
    inference_count_=0;
    decision_count_=0;
    cached_.reset();
}

SyncResult NativeShadowDecisionEngine::start_hand(
    const HandAnchor& anchor,
    const ObservedSnapshot& observed) {
    clear_cache();
    return tracker_.start_hand(anchor,observed);
}

SyncResult NativeShadowDecisionEngine::heartbeat(
    const HandAnchor& anchor,
    const ObservedSnapshot& observed) {
    const auto before=tracker_.generation();
    auto result=tracker_.heartbeat(anchor,observed);
    if (result.kind==SyncKind::Failed || tracker_.generation()!=before) clear_cache();
    return result;
}

std::optional<ShadowDecision> NativeShadowDecisionEngine::on_my_turn(
    const HandAnchor& anchor,
    const ObservedSnapshot& observed) {
    const auto before=tracker_.generation();
    const auto sync=tracker_.synchronize_to_actor(anchor,observed,anchor.hero_logical_seat);
    if (sync.kind==SyncKind::Failed) {
        clear_cache();
        return std::nullopt;
    }
    if (tracker_.generation()!=before) clear_cache();

    if (cached_.has_value() && cached_->tracker_generation==tracker_.generation()) {
        return cached_;
    }
    try {
        cached_=compute_decision();
        return cached_;
    } catch (...) {
        clear_cache();
        return std::nullopt;
    }
}

std::optional<ShadowDecision> NativeShadowDecisionEngine::cached_decision() const noexcept {
    if (!cached_.has_value()) return std::nullopt;
    if (cached_->tracker_generation!=tracker_.generation()) return std::nullopt;
    return cached_;
}

std::uint64_t NativeShadowDecisionEngine::next_u64() noexcept {
    std::uint64_t z=(rng_state_+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30U))*0xBF58476D1CE4E5B9ULL;
    z=(z^(z>>27U))*0x94D049BB133111EBULL;
    return z^(z>>31U);
}

double NativeShadowDecisionEngine::next_unit() noexcept {
    return static_cast<double>(next_u64()>>11U) * (1.0/9007199254740992.0);
}

ShadowDecision NativeShadowDecisionEngine::compute_decision() {
    const auto* state=tracker_.state();
    if (!state || state->terminal()) throw std::runtime_error("decision requested without live canonical state");
    if (state->actor()<0) throw std::runtime_error("decision requested without actor");

    const auto resolved=resolve_lean_legacy_actions_v1(state->hand().betting(),frozen_active_mask());
    if (resolved.empty()) throw std::runtime_error("empty lean legal action set at decision");
    const auto legal=legal_mask(resolved);

    const auto bytes=observation_bytes(*state);
    const auto observation=lt2::decode_observation_v1(bytes);

    const auto domain=strategy_domain(state->hand().betting().topology());
    std::array<float,lt2::kActionCount> probs{};
    std::int32_t domain_code=0;
    if (domain==StrategyDomain::ThreeHanded) {
        probs=lt2::infer_three_handed_policy(bundle_,observation,legal);
        domain_code=0;
    } else {
        probs=lt2::infer_heads_up_ensemble_policy(bundle_,observation,legal);
        domain_code=1;
    }
    ++inference_count_;

    double total=0.0;
    for (std::size_t i=0;i<probs.size();++i) {
        if (!std::isfinite(probs[i]) || probs[i]<0.0F) {
            throw std::runtime_error("invalid deployment probability");
        }
        if (legal[i]==0U && probs[i]!=0.0F) {
            throw std::runtime_error("deployment placed probability mass on illegal action");
        }
        total+=static_cast<double>(probs[i]);
    }
    if (std::abs(total-1.0)>1e-5) throw std::runtime_error("deployment probability mass drift");

    const double u=next_unit();
    double cumulative=0.0;
    std::int32_t selected=-1;
    std::int32_t fallback=-1;
    for (std::size_t i=0;i<probs.size();++i) {
        if (legal[i]==0U) continue;
        fallback=static_cast<std::int32_t>(i);
        cumulative+=static_cast<double>(probs[i]);
        if (selected<0 && u<cumulative) selected=static_cast<std::int32_t>(i);
    }
    if (selected<0) selected=fallback;
    if (selected<0) throw std::runtime_error("failed to sample legal deployment action");

    const auto& exact=find_slot(resolved,selected);
    ShadowDecision out{};
    out.tracker_generation=tracker_.generation();
    out.decision_index=decision_count_;
    out.domain=domain_code;
    out.action_slot=selected;
    out.exact=exact.exact;
    out.probabilities=probs;
    out.legal_mask=legal;
    out.sample_u=u;
    ++decision_count_;
    return out;
}

} // namespace spincore::lt2oh
