#include "spincore/lt2_deployment_runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace spincore::lt2 {
namespace {

constexpr std::array<char, 8> kBundleMagic{'S','C','L','T','2','D','1','\0'};
constexpr std::uint32_t kBundleVersion = 1;
constexpr std::size_t kCardVocab = 53;
constexpr std::size_t kCardEmb = 16;
constexpr std::size_t kCatVocab = 32;
constexpr std::size_t kCatEmb = 8;
constexpr std::size_t kHistVocab = 64;
constexpr std::size_t kHistEmb = 8;
constexpr std::size_t kHistorySteps = 32;
constexpr std::size_t kGruHidden = 80;
constexpr std::size_t kInputDim = 272;
constexpr std::size_t kBodyHidden = 320;
constexpr std::size_t kHeadHidden = 128;

template <typename T>
T read_pod(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    if (!in) {
        throw std::runtime_error("unexpected EOF in LT2 deployment bundle");
    }
    return value;
}

std::string read_fixed_ascii(std::ifstream& in, std::size_t n) {
    std::string value(n, '\0');
    in.read(value.data(), static_cast<std::streamsize>(n));
    if (!in) {
        throw std::runtime_error("unexpected EOF in fixed deployment metadata");
    }
    const auto zero = value.find('\0');
    if (zero != std::string::npos) {
        value.resize(zero);
    }
    return value;
}

std::string read_string(std::ifstream& in) {
    const auto n = read_pod<std::uint16_t>(in);
    if (n == 0 || n > 4096) {
        throw std::runtime_error("invalid tensor name length");
    }
    std::string value(static_cast<std::size_t>(n), '\0');
    in.read(value.data(), static_cast<std::streamsize>(n));
    if (!in) {
        throw std::runtime_error("unexpected EOF in tensor name");
    }
    return value;
}

const TensorF32& tensor(
    const ActionModelWeights& model,
    std::string_view name,
    std::initializer_list<std::uint32_t> dims) {
    const auto it = model.tensors.find(std::string(name));
    if (it == model.tensors.end()) {
        throw std::runtime_error("missing deployment tensor: " + std::string(name));
    }
    const std::vector<std::uint32_t> expected(dims);
    if (it->second.dims != expected) {
        throw std::runtime_error("deployment tensor shape drift: " + std::string(name));
    }
    std::size_t count = 1;
    for (const auto d : expected) {
        count *= static_cast<std::size_t>(d);
    }
    if (it->second.values.size() != count) {
        throw std::runtime_error("deployment tensor value-count drift: " + std::string(name));
    }
    return it->second;
}

void validate_model(const ActionModelWeights& model) {
    (void)tensor(model, "card_emb.weight", {53, 16});
    (void)tensor(model, "cat_emb.weight", {32, 8});
    (void)tensor(model, "hist_emb.weight", {64, 8});
    (void)tensor(model, "gru.weight_ih_l0", {240, 8});
    (void)tensor(model, "gru.weight_hh_l0", {240, 80});
    (void)tensor(model, "gru.bias_ih_l0", {240});
    (void)tensor(model, "gru.bias_hh_l0", {240});
    (void)tensor(model, "body.0.weight", {320, 272});
    (void)tensor(model, "body.0.bias", {320});
    (void)tensor(model, "body.2.weight", {128, 320});
    (void)tensor(model, "body.2.bias", {128});
    (void)tensor(model, "head.weight", {10, 128});
    (void)tensor(model, "head.bias", {10});
}

float sigmoid(float x) {
    if (x >= 0.0F) {
        const float z = std::exp(-x);
        return 1.0F / (1.0F + z);
    }
    const float z = std::exp(x);
    return z / (1.0F + z);
}

std::vector<float> linear(
    const TensorF32& weight,
    const TensorF32& bias,
    const std::vector<float>& input,
    std::size_t out_dim,
    std::size_t in_dim) {
    if (input.size() != in_dim) {
        throw std::runtime_error("linear input dimension drift");
    }
    std::vector<float> out(out_dim, 0.0F);
    for (std::size_t row = 0; row < out_dim; ++row) {
        float value = bias.values[row];
        const std::size_t base = row * in_dim;
        for (std::size_t col = 0; col < in_dim; ++col) {
            value += weight.values[base + col] * input[col];
        }
        out[row] = value;
    }
    return out;
}

std::array<float, kGruHidden> gru_hidden(
    const ActionModelWeights& model,
    const DecodedObservationV1& observation) {
    const auto& embedding = tensor(model, "hist_emb.weight", {64, 8});
    const auto& w_ih = tensor(model, "gru.weight_ih_l0", {240, 8});
    const auto& w_hh = tensor(model, "gru.weight_hh_l0", {240, 80});
    const auto& b_ih = tensor(model, "gru.bias_ih_l0", {240});
    const auto& b_hh = tensor(model, "gru.bias_hh_l0", {240});

    std::array<float, kGruHidden> hidden{};
    for (std::size_t step = 0; step < kHistorySteps; ++step) {
        const auto token = static_cast<std::size_t>(observation.history[step]);
        if (token >= kHistVocab) {
            throw std::runtime_error("history token out of range");
        }
        std::array<float, kHistEmb> x{};
        for (std::size_t j = 0; j < kHistEmb; ++j) {
            x[j] = embedding.values[token * kHistEmb + j];
        }

        std::array<float, kGruHidden> reset{};
        std::array<float, kGruHidden> update{};
        std::array<float, kGruHidden> candidate{};

        for (std::size_t h = 0; h < kGruHidden; ++h) {
            float ir = b_ih.values[h];
            float iz = b_ih.values[kGruHidden + h];
            float in = b_ih.values[2 * kGruHidden + h];
            float hr = b_hh.values[h];
            float hz = b_hh.values[kGruHidden + h];
            float hn = b_hh.values[2 * kGruHidden + h];

            for (std::size_t j = 0; j < kHistEmb; ++j) {
                ir += w_ih.values[h * kHistEmb + j] * x[j];
                iz += w_ih.values[(kGruHidden + h) * kHistEmb + j] * x[j];
                in += w_ih.values[(2 * kGruHidden + h) * kHistEmb + j] * x[j];
            }
            for (std::size_t j = 0; j < kGruHidden; ++j) {
                hr += w_hh.values[h * kGruHidden + j] * hidden[j];
                hz += w_hh.values[(kGruHidden + h) * kGruHidden + j] * hidden[j];
                hn += w_hh.values[(2 * kGruHidden + h) * kGruHidden + j] * hidden[j];
            }
            reset[h] = sigmoid(ir + hr);
            update[h] = sigmoid(iz + hz);
            candidate[h] = std::tanh(in + reset[h] * hn);
        }

        std::array<float, kGruHidden> next{};
        for (std::size_t h = 0; h < kGruHidden; ++h) {
            next[h] = (1.0F - update[h]) * candidate[h] + update[h] * hidden[h];
        }
        hidden = next;
    }
    return hidden;
}

std::array<float, kActionCount> softmax_legal(
    const std::array<float, kActionCount>& logits,
    const std::array<std::uint8_t, kActionCount>& legal) {
    bool any = false;
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t a = 0; a < kActionCount; ++a) {
        if (legal[a] != 0U) {
            any = true;
            maximum = std::max(maximum, logits[a]);
        }
    }
    if (!any) {
        throw std::runtime_error("empty legal action mask");
    }
    std::array<float, kActionCount> out{};
    float total = 0.0F;
    for (std::size_t a = 0; a < kActionCount; ++a) {
        if (legal[a] != 0U) {
            out[a] = std::exp(logits[a] - maximum);
            total += out[a];
        }
    }
    if (!(total > 0.0F) || !std::isfinite(total)) {
        throw std::runtime_error("invalid softmax total");
    }
    for (auto& v : out) {
        v /= total;
    }
    return out;
}

std::array<float, kActionCount> regret_match(
    const std::array<float, kActionCount>& raw,
    const std::array<std::uint8_t, kActionCount>& legal) {
    std::array<float, kActionCount> out{};
    float positive_total = 0.0F;
    std::size_t legal_count = 0;
    for (std::size_t a = 0; a < kActionCount; ++a) {
        if (legal[a] != 0U) {
            ++legal_count;
            if (!std::isfinite(raw[a])) {
                throw std::runtime_error("nonfinite HU advantage");
            }
            positive_total += std::max(0.0F, raw[a]);
        }
    }
    if (legal_count == 0) {
        throw std::runtime_error("empty legal action mask");
    }
    if (positive_total > 0.0F) {
        for (std::size_t a = 0; a < kActionCount; ++a) {
            if (legal[a] != 0U) {
                out[a] = std::max(0.0F, raw[a]) / positive_total;
            }
        }
        return out;
    }

    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t a = 0; a < kActionCount; ++a) {
        if (legal[a] != 0U) {
            maximum = std::max(maximum, raw[a]);
        }
    }
    float total = 0.0F;
    for (std::size_t a = 0; a < kActionCount; ++a) {
        if (legal[a] != 0U) {
            const float shifted = std::clamp(raw[a] - maximum, -60.0F, 60.0F);
            out[a] = std::exp(shifted);
            total += out[a];
        }
    }
    if (!(total > 0.0F) || !std::isfinite(total)) {
        const float uniform = 1.0F / static_cast<float>(legal_count);
        for (std::size_t a = 0; a < kActionCount; ++a) {
            if (legal[a] != 0U) {
                out[a] = uniform;
            }
        }
        return out;
    }
    for (auto& v : out) {
        v /= total;
    }
    return out;
}

}  // namespace

DeploymentBundle load_deployment_bundle(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open LT2 deployment bundle");
    }
    std::array<char, 8> magic{};
    in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!in || magic != kBundleMagic) {
        throw std::runtime_error("bad LT2 deployment bundle magic");
    }
    const auto version = read_pod<std::uint32_t>(in);
    const auto action_count = read_pod<std::uint32_t>(in);
    const auto ensemble_size = read_pod<std::uint32_t>(in);
    const auto model_count = read_pod<std::uint32_t>(in);
    if (version != kBundleVersion || action_count != kActionCount ||
        ensemble_size != kHuEnsembleSize || model_count != 1U + kHuEnsembleSize) {
        throw std::runtime_error("LT2 deployment bundle contract drift");
    }

    DeploymentBundle bundle{};
    bundle.deployment_bundle_sha256 = read_fixed_ascii(in, 64);
    bundle.source_checkpoint_sha256 = read_fixed_ascii(in, 64);
    bundle.source_ensemble_sha256 = read_fixed_ascii(in, 64);

    std::array<bool, kHuEnsembleSize> seen_hu{};
    bool seen_three = false;
    for (std::uint32_t m = 0; m < model_count; ++m) {
        ActionModelWeights model{};
        model.kind = static_cast<DeploymentModelKind>(read_pod<std::uint32_t>(in));
        model.member_index = read_pod<std::uint32_t>(in);
        const auto tensor_count = read_pod<std::uint32_t>(in);
        if (tensor_count == 0 || tensor_count > 64) {
            throw std::runtime_error("invalid deployment tensor count");
        }

        for (std::uint32_t t = 0; t < tensor_count; ++t) {
            const std::string name = read_string(in);
            const auto ndim = read_pod<std::uint32_t>(in);
            if (ndim == 0 || ndim > 4) {
                throw std::runtime_error("invalid tensor ndim");
            }
            TensorF32 tensor_value{};
            std::size_t expected_count = 1;
            for (std::uint32_t d = 0; d < ndim; ++d) {
                const auto dim = read_pod<std::uint32_t>(in);
                tensor_value.dims.push_back(dim);
                expected_count *= static_cast<std::size_t>(dim);
            }
            const auto numel = read_pod<std::uint64_t>(in);
            if (numel != expected_count || numel > 100000000ULL) {
                throw std::runtime_error("invalid tensor element count");
            }
            tensor_value.values.resize(static_cast<std::size_t>(numel));
            in.read(
                reinterpret_cast<char*>(tensor_value.values.data()),
                static_cast<std::streamsize>(numel * sizeof(float)));
            if (!in) {
                throw std::runtime_error("unexpected EOF in tensor data");
            }
            if (!model.tensors.emplace(name, std::move(tensor_value)).second) {
                throw std::runtime_error("duplicate tensor name");
            }
        }
        validate_model(model);

        if (model.kind == DeploymentModelKind::ThreeHandedAveragePolicy) {
            if (seen_three || model.member_index != 0U) {
                throw std::runtime_error("invalid THREE_HANDED deployment model identity");
            }
            seen_three = true;
            bundle.three_handed_policy = std::move(model);
        } else if (model.kind == DeploymentModelKind::HeadsUpAdvantage) {
            if (model.member_index >= kHuEnsembleSize || seen_hu[model.member_index]) {
                throw std::runtime_error("invalid HU member identity");
            }
            seen_hu[model.member_index] = true;
            bundle.hu_members[model.member_index] = std::move(model);
        } else {
            throw std::runtime_error("unknown deployment model kind");
        }
    }
    if (!seen_three || !std::all_of(seen_hu.begin(), seen_hu.end(), [](bool v) { return v; })) {
        throw std::runtime_error("incomplete LT2 deployment bundle");
    }
    return bundle;
}

DecodedObservationV1 decode_observation_v1(
    const std::array<std::uint8_t, kObservationSizeV1>& bytes) {
    constexpr std::array<std::uint8_t, 8> magic{'S','P','N','N','I','V','1','\0'};
    if (!std::equal(magic.begin(), magic.end(), bytes.begin())) {
        throw std::runtime_error("bad SPNNIV1 observation magic");
    }
    DecodedObservationV1 out{};
    std::size_t p = 8;
    for (auto& v : out.cards) {
        v = bytes[p++];
        if (v >= kCardVocab) {
            throw std::runtime_error("card token out of range");
        }
    }
    for (auto& v : out.numeric) {
        std::uint32_t raw = 0;
        std::memcpy(&raw, bytes.data() + p, sizeof(raw));
        std::memcpy(&v, &raw, sizeof(v));
        p += sizeof(float);
        if (!std::isfinite(v)) {
            throw std::runtime_error("nonfinite observation numeric");
        }
    }
    for (auto& v : out.categorical) {
        v = bytes[p++];
        if (v >= kCatVocab) {
            throw std::runtime_error("categorical token out of range");
        }
    }
    p += 6;  // legacy six-action mask in SPNNIV1; live ten-action legal mask is external.
    const auto history_len = bytes[p++];
    if (history_len > kHistorySteps) {
        throw std::runtime_error("history length out of range");
    }
    for (auto& v : out.history) {
        v = bytes[p++];
        if (v >= kHistVocab) {
            throw std::runtime_error("history token out of range");
        }
    }
    if (p != kObservationSizeV1) {
        throw std::runtime_error("SPNNIV1 decode length drift");
    }
    return out;
}

std::array<float, kActionCount> infer_raw_action_model(
    const ActionModelWeights& model,
    const DecodedObservationV1& observation) {
    const auto& card_emb = tensor(model, "card_emb.weight", {53, 16});
    const auto& cat_emb = tensor(model, "cat_emb.weight", {32, 8});

    std::vector<float> input;
    input.reserve(kInputDim);
    for (const auto token_u8 : observation.cards) {
        const auto token = static_cast<std::size_t>(token_u8);
        for (std::size_t j = 0; j < kCardEmb; ++j) {
            input.push_back(card_emb.values[token * kCardEmb + j]);
        }
    }
    for (const auto token_u8 : observation.categorical) {
        const auto token = static_cast<std::size_t>(token_u8);
        for (std::size_t j = 0; j < kCatEmb; ++j) {
            input.push_back(cat_emb.values[token * kCatEmb + j]);
        }
    }
    input.insert(input.end(), observation.numeric.begin(), observation.numeric.end());

    const auto hidden = gru_hidden(model, observation);
    input.insert(input.end(), hidden.begin(), hidden.end());
    if (input.size() != kInputDim) {
        throw std::runtime_error("action model concatenated input drift");
    }

    auto h1 = linear(
        tensor(model, "body.0.weight", {320, 272}),
        tensor(model, "body.0.bias", {320}),
        input, kBodyHidden, kInputDim);
    for (auto& v : h1) {
        v = std::max(0.0F, v);
    }
    auto h2 = linear(
        tensor(model, "body.2.weight", {128, 320}),
        tensor(model, "body.2.bias", {128}),
        h1, kHeadHidden, kBodyHidden);
    for (auto& v : h2) {
        v = std::max(0.0F, v);
    }
    const auto logits = linear(
        tensor(model, "head.weight", {10, 128}),
        tensor(model, "head.bias", {10}),
        h2, kActionCount, kHeadHidden);

    std::array<float, kActionCount> out{};
    std::copy(logits.begin(), logits.end(), out.begin());
    return out;
}

std::array<float, kActionCount> infer_three_handed_policy(
    const DeploymentBundle& bundle,
    const DecodedObservationV1& observation,
    const std::array<std::uint8_t, kActionCount>& legal_mask) {
    return softmax_legal(
        infer_raw_action_model(bundle.three_handed_policy, observation),
        legal_mask);
}

std::array<float, kActionCount> infer_heads_up_ensemble_policy(
    const DeploymentBundle& bundle,
    const DecodedObservationV1& observation,
    const std::array<std::uint8_t, kActionCount>& legal_mask) {
    std::array<float, kActionCount> mean{};
    for (const auto& member : bundle.hu_members) {
        const auto raw = infer_raw_action_model(member, observation);
        for (std::size_t a = 0; a < kActionCount; ++a) {
            mean[a] += raw[a];
        }
    }
    const float denom = static_cast<float>(kHuEnsembleSize);
    for (auto& v : mean) {
        v /= denom;
    }
    return regret_match(mean, legal_mask);
}

}  // namespace spincore::lt2
