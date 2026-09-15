#pragma once

#include "spincore/action_abstraction_v2.hpp"

namespace spincore {

// Legacy-first seven-action resolver used only by the lean functional path.
// Slots stay in the universal 10-slot vocabulary, but their legal/exact
// semantics reproduce the mature DeepSpin action abstraction:
//
// preflop:
//   POT_33  -> 2.0 BB open
//   POT_75  -> 2.5 BB + 1 BB per extra limper isolation raise
//   POT_50  -> 5.0 BB + 2 BB per caller after the first raise
//   special limp-vs-isolation / multi-raise restrictions preserved
// postflop:
//   POT_33/POT_50/POT_75/POT_100 use pot-after-call sizing,
//   invalid too-small/unaffordable sizes are pruned (not clamped), and
//   legacy 60% near-all-in collapse removes redundant fractional branches.
//
// The generic R7.5 universal resolver remains untouched and available for old
// experiments.  This resolver is the action contract for first functional
// SpinCore training/runtime parity.
[[nodiscard]] std::vector<ResolvedUniversalActionV2> resolve_lean_legacy_actions_v1(
    const BettingEngine& betting,
    const UniversalActionMaskV2& active_mask);

} // namespace spincore
