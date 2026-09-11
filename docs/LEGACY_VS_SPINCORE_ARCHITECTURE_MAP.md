# Legacy DeepSpin vs SpinCore — Architecture Map

Status: **IN PROGRESS**
Started: 2026-09-09
Updated: 2026-09-10
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

Decision: restore/adapt the legacy scenario-distribution semantics before global action-abstraction selection or final training. The exact legacy tables are now preserved in `python/spincore/legacy_scenario_data.json`, and `python/spincore/legacy_scenario.py` adapts them to current `spincore.solver.Episode` while preserving the historical sampling semantics.

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

Historical code also records a repaired regret-matching fallback: the older uniform fallback, when all legal advantages were non-positive, injected grossly wrong actions. Current SpinCore had regressed to that same uniform fallback. On 2026-09-10 it was corrected in `python/spincore/deep_cfr.py`: ordinary positive regret matching remains unchanged, but fitted-network states with no positive legal regret now use stable masked softmax over raw legal advantages, preserving the model's ranking. The true untrained zero-regret state remains uniform through `NeuralAdvantagePolicy.ready=False`.

## 5. Utility/objective — ONE WTA CHIP-EV POLICY FIRST; MULTIPAY SPECIALIZATION DEFERRED

Legacy `ExternalSamplingTraverser._terminal_value()` reads `g.get_payoffs()` and normalizes the traverser's single-hand chip payoff by the current BB. The C++ `get_payoffs()` computes chips won/lost in that hand.

Previous audit wording prematurely called SpinCore's explicit-payout/ICM utility a likely improvement. That conclusion is withdrawn.

For a winner-take-all payout vector, ICM first-place equity is linear in chips. Therefore expected ICM delta and expected chip delta rank actions identically, up to a positive constant factor. Replacing chip EV with payout/ICM inside the WTA training problem adds no strategic information.

Actual GGPoker Spin & Gold does sometimes pay more than one place at high multipliers. In those 3-handed multi-place states, payout-aware utility can change optimal decisions and is theoretically more accurate. However, that does **not** imply that first-release SpinCore should multiply its already expensive training scope by payout structures.

Current product decision:

- train the first strong functional SpinCore policy family on **winner-take-all chip EV**;
- use that same WTA-trained policy initially for all payout variants, including rare multi-place games;
- do not train separate 70/30, 50/30/20, or other payout-specific policies now;
- keep the solver's ICM capability available but dormant for first-release training;
- only revisit multi-place specialization after the WTA agent is strong and functional, and only if the estimated real-world gain justifies the added compute;
- if specialization is later justified, first investigate cheaper transfer/fine-tuning or payout-conditioned approaches before full independent training from scratch.

This decision is encoded in `python/spincore/lean_training_scope.py` as `WTA_SHARED_ACROSS_PAYOUTS_V1` with multi-pay specialization disabled.

One separate legacy question remains open: the old target is `chip_payoff / current_bb`, not raw chip delta. Dividing by BB does not change action ordering within a state, but it changes target magnitude across blind levels for a shared neural approximator. Treat this as an audit hypothesis only; do not change it until its learning effect is understood.

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

Chip EV itself is **not** classified as a historical failure cause. It remains the first-release objective. Multi-place payout modeling is a possible later refinement, not a prerequisite for making DeepSpin/SpinCore work.

## 8. Current synthesis

Target direction, subject to the rest of this audit:

**legacy realistic SpinGo scenario distribution + repaired legacy poker semantics + one WTA chip-EV policy family + only demonstrated SpinCore state/traversal improvements + corrected legacy regret fallback + lean evidence-driven action-abstraction selection + preserved/reconciled OpenHoldem runtime.**

No heavy training resumes until the remaining legacy components (292 features, buffers/networks, runtime, Crusher hardcoded material, solver-v2/184 assets) are mapped far enough to eliminate known structural failure modes and define the first functional training path. The audit is not an excuse for exhaustive certification: once the architecture-affecting questions are resolved, training should start.
