# Legacy DeepSpin vs SpinCore — Architecture Map

Status: **IN PROGRESS**
Started: 2026-09-09
Purpose: preserve multi-year legacy knowledge and identify only genuine SpinCore improvements before more heavy compute.

## 1. Scenario distribution — LEGACY STRONGER / MUST RESTORE

Legacy `deepspin/scenario.py` already models the tournament-state distribution instead of a single fixed blind:

- 3-handed and true-HU sampled separately;
- blind ladder: 10/20, 15/30, 20/40, 30/60, 40/80, 50/100, 60/120, 80/160, 100/200;
- separate empirical blind-frequency weights for 3H and HU;
- blind-conditioned empirical stack distributions;
- 1500 total chips;
- random dead seat/live-seat permutation for HU;
- random dealer among live seats;
- sparse late levels approximated from nearest useful empirical level instead of silently removed.

SpinCore R7.5.4's 10/20-only matrix is therefore a regression in scenario realism and may be used only as localized evidence.

Decision: restore/adapt the legacy scenario-distribution semantics before global action-abstraction selection or final training.

## 2. Action abstraction — LEGACY ALREADY HAD A RICH PRACTICAL SET

Legacy DeepSpin exposes seven neural actions:

- fold;
- check/call;
- bet/raise 33% pot;
- 50% pot;
- 75% pot;
- 100% pot;
- all-in.

The C++ environment also contains legality/near-all-in pruning. SpinCore may improve this set, but must compare against this baseline under the full tournament distribution rather than inventing candidates only at 10/20.

## 3. Observation/state — LEGACY RICH BUT HISTORICALLY BUG-PRONE

Legacy neural observation dimension is 292. It contains raw card one-hot features plus numeric/game-state, hand-strength, draw, board-texture and other strategic features. The current archived C++ source explicitly distinguishes board-only made hands from Hero-contributed made hands in its 40-way hand-strength classification (for example, separate two-pair-on-board vs one-hole vs two-hole categories).

Historical lesson: earlier DeepSpin training/debugging exposed semantic mistakes in this area, including board-created made-hand/two-pair structures being treated as Hero strength. The archived source appears to contain later corrective logic, so migration must preserve the repaired semantics rather than revive the faulty earlier implementation.

Decision: audit feature-by-feature against SpinCore exact state/encoder. Preserve semantic knowledge; prefer exact state underneath and a neural boundary that cannot confuse board texture with Hero contribution.

## 4. Learning algorithm — LEGACY DEEP CFR FOUNDATION IS REAL

Legacy stack includes:

- external-sampling traversal;
- per-player advantage networks;
- average-policy networks;
- reservoir buffers;
- clone-based C++ traversal fast path;
- deterministic episode reconstruction fallback;
- checkpointing of Python/NumPy/Torch/scenario RNG state;
- multiprocessing rollout workers.

This is not a toy baseline. SpinCore changes must be evaluated as refinements of this foundation.

Historical code also records a repaired regret-matching fallback: the older uniform fallback, when all legal regrets were non-positive, injected grossly wrong actions. The archived traversal uses masked softmax over raw advantage values instead. This is a concrete example of knowledge that must not be lost.

## 5. Utility/objective — CHIP EV REMAINS THE DEFAULT; NO CHANGE WITHOUT A REAL PAYOUT REASON

Legacy `ExternalSamplingTraverser._terminal_value()` reads `g.get_payoffs()` and normalizes the traverser's single-hand chip payoff by the current BB. The C++ `get_payoffs()` computes chips won/lost in that hand.

Previous audit wording prematurely called SpinCore's explicit-payout/ICM utility a likely improvement. That conclusion is withdrawn.

For a winner-take-all payout vector, ICM first-place equity is linear in chips: `equity_i = prize * stack_i / total_chips`. Therefore ranking actions by expected ICM delta is exactly the same as ranking them by expected chip delta, up to a positive constant factor. Replacing chip EV with payout/ICM in such games adds complexity without strategic gain.

Current decision:

- **chip EV is the primary training/evaluation objective for ordinary winner-take-all SpinGo states**;
- raw monetary tournament results are not the primary quality metric because multiplier and card variance add noise unrelated to decision quality;
- payout-aware utility is justified only for variants where more than one finishing position is actually paid and the payout vector can change optimal decisions;
- if those multi-place variants are included, add payout awareness only for those states (condition the policy on payout vector or use a dedicated variant), rather than replacing the core chip-EV architecture wholesale;
- no utility change is allowed merely because SpinCore already implemented one.

This preserves the user's legacy design unless a concrete, strategically material counterexample shows that another objective improves the intended game.

## 6. Runtime/integration — LEGACY ASSET TO PRESERVE

Legacy archive includes `user_deepspin.cpp`, a substantial OpenHoldem inference/runtime implementation with neural-brain loading, dimension/action checks, state construction, decision routing and logging.

Decision: do not rebuild runtime from zero. Audit training/runtime observation identity and action mapping, preserve mature integration pieces, and replace only where SpinCore has a demonstrated correctness advantage.

## 7. Why the historical trained agent failed

Known from the user's history and subsequent debugging:

- roughly three months of continuous Ryzen training still produced gross strategic errors, ruling out 'simply train longer' as the default diagnosis;
- excessive complexity made defects harder to isolate;
- basic hand/state semantic errors existed during the historical line, including made-hand/two-pair interpretation problems;
- an earlier uniform regret fallback injected strategically absurd legal actions when positive regrets were absent;
- version/local-patch drift made it difficult to guarantee that trainer, checkpoint, exporter and runtime represented the same semantics.

Chip EV itself is **not** currently classified as a historical failure cause. It remains the baseline objective unless an actual payout structure creates a demonstrated reason to depart from it.

## 8. Current synthesis

Target direction, subject to the rest of this audit:

**legacy realistic SpinGo scenario distribution + repaired legacy poker semantics + legacy chip-EV objective by default + only demonstrated SpinCore state/traversal improvements + payout awareness only where the real payout vector requires it + lean evidence-driven action-abstraction selection + preserved/reconciled OpenHoldem runtime.**

No heavy training resumes until the remaining legacy components (buffers, networks, runtime, Crusher hardcoded material, solver-v2/184 assets) are fully mapped and the consolidated architecture is explicit.
