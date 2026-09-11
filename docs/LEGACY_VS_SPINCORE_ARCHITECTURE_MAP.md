# Legacy DeepSpin vs SpinCore — Architecture Map

Status: **IN PROGRESS**
Started: 2026-09-09
Updated: 2026-09-11
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

## 3. Observation/state — KEEP LEGACY KNOWLEDGE, NOT THE 292-FLOAT INPUT

The archived DeepSpin v60 observation is exactly 292 floats: 104 card one-hots, 20 numeric values, 4 street flags, 11 position flags, 40 hand-strength categories, 12 draw flags, 29 board-texture flags, 26 action-context values, 39 history values and 7 legal-action flags.

That representation contains real poker knowledge, and the later archived source repaired important semantics by distinguishing board-only made hands from hands that actually use Hero's hole cards. Those definitions remain valuable as diagnostics/regression knowledge.

However, the 292-dimensional flat input duplicates information heavily. Cards already determine hand strength, draws and board texture; action-context/history blocks summarize information also represented by the exact betting state/history; legal actions are separately masked. Derived features can improve sample efficiency, but each hand-coded semantic feature is also another place where a bug can poison learning — exactly what happened historically with board-only/two-pair interpretation.

The number 292 alone was not the main performance problem. The legacy AdvantageNet used `[1024,1024,512,512]` hidden layers (~2.14M parameters) and the PolicyNet `[1024,512,512]` (~1.09M), about 3.23M parameters combined. Merely shrinking the flat input from 292 to 128 while preserving those hidden layers would save only about 10% of model parameters.

Current SpinCore V1 is materially leaner. Its neural boundary uses 7 card tokens, 16 numeric values, 8 categorical values, a legal mask and up to 32 history tokens, while the exact `CanonicalInfoset` retains cards, stacks, commitments, domain, street, blinds, blind index, statuses, pot, to-call, current bet and public history. The recovered V1 network is about 152,438 parameters per model and embeds/encodes structured inputs rather than feeding a huge flat one-hot vector.

Decision for the first functional SpinCore:

- do **not** restore the full 292-float vector as the neural input;
- use compact V1 as the default neural representation;
- preserve the legacy 292-feature definitions as a semantic checklist/diagnostic library;
- add back a specific derived feature only if a concrete strategic weakness shows that compact V1 cannot distinguish/learn the relevant situation efficiently;
- do not open another broad representation tournament merely to seek novelty;
- every future added feature must be computable from the canonical exact state and identical between training and runtime.

Detailed rationale is frozen in `docs/LEGACY_292_FEATURE_AUDIT.md`.

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

Target direction, subject to the remaining finite audit:

**legacy realistic SpinGo scenario distribution + repaired legacy poker semantics + compact V1 exact-state neural boundary + one WTA chip-EV policy family + only demonstrated SpinCore traversal/state improvements + corrected legacy regret fallback + lean action-abstraction selection + preserved/reconciled OpenHoldem runtime.**

The 292-feature question is now closed for first release: preserve its poker semantics as reference knowledge, but do not reintroduce the full flat vector. Remaining architecture-affecting questions are `/BB` utility normalization, action-set reconciliation and training/runtime parity. Crusher/hardcoded and solver-v2/184 assets are consulted only where they can improve one of those concrete decisions. Once those questions are resolved, training should start rather than opening another certification cycle.
