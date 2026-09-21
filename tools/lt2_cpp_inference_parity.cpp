#include "spincore/lt2_deployment_runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr std::array<char, 8> kFixtureMagic{'S','C','L','T','2','F','1','\0'};
constexpr std::uint32_t kFixtureVersion = 1;

template <typename T>
T read_pod(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
    if (!in) {
        throw std::runtime_error("unexpected EOF in fixture file");
    }
    return value;
}

struct Fixture {
    std::uint8_t domain{0};
    std::uint8_t street{0};
    std::array<std::uint8_t, spincore::lt2::kObservationSizeV1> observation{};
    std::array<std::uint8_t, spincore::lt2::kActionCount> legal{};
    std::array<float, spincore::lt2::kActionCount> expected{};
};

Fixture read_fixture(std::ifstream& in) {
    Fixture f{};
    f.domain = read_pod<std::uint8_t>(in);
    f.street = read_pod<std::uint8_t>(in);
    (void)read_pod<std::uint16_t>(in);
    in.read(
        reinterpret_cast<char*>(f.observation.data()),
        static_cast<std::streamsize>(f.observation.size()));
    in.read(
        reinterpret_cast<char*>(f.legal.data()),
        static_cast<std::streamsize>(f.legal.size()));
    in.read(
        reinterpret_cast<char*>(f.expected.data()),
        static_cast<std::streamsize>(f.expected.size() * sizeof(float)));
    if (!in) {
        throw std::runtime_error("unexpected EOF in fixture payload");
    }
    return f;
}

std::size_t argmax_legal(
    const std::array<float, spincore::lt2::kActionCount>& probs,
    const std::array<std::uint8_t, spincore::lt2::kActionCount>& legal) {
    bool found = false;
    std::size_t best = 0;
    float best_value = -std::numeric_limits<float>::infinity();
    for (std::size_t a = 0; a < spincore::lt2::kActionCount; ++a) {
        if (legal[a] == 0U) {
            continue;
        }
        if (!found || probs[a] > best_value) {
            found = true;
            best = a;
            best_value = probs[a];
        }
    }
    if (!found) {
        throw std::runtime_error("fixture has empty legal mask");
    }
    return best;
}

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (const char c : value) {
        switch (c) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += c; break;
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr << "usage: spincore_lt2_cpp_inference_parity "
                      << "<deployment.bin> <fixtures.bin> <report.json> <tolerance>\n";
            return 2;
        }
        const std::filesystem::path deployment_path(argv[1]);
        const std::filesystem::path fixture_path(argv[2]);
        const std::filesystem::path report_path(argv[3]);
        const float tolerance = std::stof(argv[4]);
        if (!(tolerance > 0.0F)) {
            throw std::runtime_error("tolerance must be positive");
        }

        const auto bundle = spincore::lt2::load_deployment_bundle(deployment_path);

        std::ifstream in(fixture_path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("cannot open parity fixture file");
        }
        std::array<char, 8> magic{};
        in.read(magic.data(), static_cast<std::streamsize>(magic.size()));
        if (!in || magic != kFixtureMagic) {
            throw std::runtime_error("bad parity fixture magic");
        }
        const auto version = read_pod<std::uint32_t>(in);
        const auto count = read_pod<std::uint32_t>(in);
        if (version != kFixtureVersion || count == 0U) {
            throw std::runtime_error("bad parity fixture header");
        }

        std::uint64_t three_handed = 0;
        std::uint64_t heads_up = 0;
        std::array<std::uint64_t, 4> streets{};
        std::uint64_t argmax_mismatches = 0;
        std::uint64_t illegal_mass_failures = 0;
        std::uint64_t nonfinite_failures = 0;
        std::uint64_t compared_probabilities = 0;
        double sum_abs = 0.0;
        float max_abs = 0.0F;
        float max_mass_error = 0.0F;

        for (std::uint32_t i = 0; i < count; ++i) {
            const auto fixture = read_fixture(in);
            if (fixture.domain > 1U || fixture.street > 3U) {
                throw std::runtime_error("fixture domain/street out of range");
            }
            const auto obs = spincore::lt2::decode_observation_v1(fixture.observation);
            std::array<float, spincore::lt2::kActionCount> actual{};
            if (fixture.domain == 0U) {
                ++three_handed;
                actual = spincore::lt2::infer_three_handed_policy(
                    bundle, obs, fixture.legal);
            } else {
                ++heads_up;
                actual = spincore::lt2::infer_heads_up_ensemble_policy(
                    bundle, obs, fixture.legal);
            }
            ++streets[fixture.street];

            float mass = 0.0F;
            for (std::size_t a = 0; a < spincore::lt2::kActionCount; ++a) {
                if (!std::isfinite(actual[a])) {
                    ++nonfinite_failures;
                }
                if (fixture.legal[a] == 0U && actual[a] != 0.0F) {
                    ++illegal_mass_failures;
                }
                if (fixture.legal[a] != 0U) {
                    mass += actual[a];
                }
                const float diff = std::fabs(actual[a] - fixture.expected[a]);
                max_abs = std::max(max_abs, diff);
                sum_abs += static_cast<double>(diff);
                ++compared_probabilities;
            }
            max_mass_error = std::max(max_mass_error, std::fabs(mass - 1.0F));
            if (argmax_legal(actual, fixture.legal) !=
                argmax_legal(fixture.expected, fixture.legal)) {
                ++argmax_mismatches;
            }
        }

        const bool coverage =
            three_handed > 0U && heads_up > 0U &&
            streets[0] > 0U && (streets[1] + streets[2] + streets[3]) > 0U;
        const bool pass =
            coverage &&
            argmax_mismatches == 0U &&
            illegal_mass_failures == 0U &&
            nonfinite_failures == 0U &&
            max_abs <= tolerance &&
            max_mass_error <= tolerance;

        report_path.parent_path().empty()
            ? void()
            : std::filesystem::create_directories(report_path.parent_path());
        std::ofstream report(report_path);
        if (!report) {
            throw std::runtime_error("cannot create parity report");
        }
        report << std::setprecision(12);
        report << "{\n";
        report << "  \"schema\": \"SPINCORE_LT2_NATIVE_CPP_INFERENCE_PARITY_V1\",\n";
        report << "  \"verdict\": \"" << (pass ? "PASS" : "FAIL") << "\",\n";
        report << "  \"deployment_path\": \"" << json_escape(deployment_path.string()) << "\",\n";
        report << "  \"fixture_path\": \"" << json_escape(fixture_path.string()) << "\",\n";
        report << "  \"source_identity\": {\n";
        report << "    \"deployment_bundle_sha256\": \""
               << json_escape(bundle.deployment_bundle_sha256) << "\",\n";
        report << "    \"source_checkpoint_sha256\": \""
               << json_escape(bundle.source_checkpoint_sha256) << "\",\n";
        report << "    \"source_ensemble_sha256\": \""
               << json_escape(bundle.source_ensemble_sha256) << "\"\n";
        report << "  },\n";
        report << "  \"records\": " << count << ",\n";
        report << "  \"three_handed_records\": " << three_handed << ",\n";
        report << "  \"heads_up_records\": " << heads_up << ",\n";
        report << "  \"streets\": {"
               << "\"preflop\":" << streets[0] << ","
               << "\"flop\":" << streets[1] << ","
               << "\"turn\":" << streets[2] << ","
               << "\"river\":" << streets[3] << "},\n";
        report << "  \"compared_probabilities\": " << compared_probabilities << ",\n";
        report << "  \"max_abs_probability_diff\": " << max_abs << ",\n";
        report << "  \"mean_abs_probability_diff\": "
               << (sum_abs / static_cast<double>(std::max<std::uint64_t>(1U, compared_probabilities)))
               << ",\n";
        report << "  \"max_probability_mass_error\": " << max_mass_error << ",\n";
        report << "  \"argmax_mismatches\": " << argmax_mismatches << ",\n";
        report << "  \"illegal_mass_failures\": " << illegal_mass_failures << ",\n";
        report << "  \"nonfinite_failures\": " << nonfinite_failures << ",\n";
        report << "  \"tolerance\": " << tolerance << ",\n";
        report << "  \"coverage_pass\": " << (coverage ? "true" : "false") << "\n";
        report << "}\n";
        report.close();

        std::cout << "=== LT2 NATIVE C++ INFERENCE PARITY ===\n";
        std::cout << "VERDICT=" << (pass ? "PASS" : "FAIL") << "\n";
        std::cout << "records=" << count
                  << " 3H=" << three_handed
                  << " HU=" << heads_up << "\n";
        std::cout << "max_abs_probability_diff=" << std::setprecision(12)
                  << max_abs << "\n";
        std::cout << "argmax_mismatches=" << argmax_mismatches << "\n";
        std::cout << "LT2_NATIVE_CPP_INFERENCE_PARITY_COMPLETE\n";
        std::cout << "report=" << report_path << "\n";
        return pass ? 0 : 3;
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << "\n";
        return 1;
    }
}
