# R7.5 architecture reset — V1+ Advantage-trajectory forensic precommit

Date: 2026-08-22
Status: FROZEN_BEFORE_ADVANTAGE_FORENSIC_OUTPUT
READY FOR TABLES: NO
Production training authorized: NO

## Governance scope

This is post-R7.5.3 architecture-reset diagnosis. It does **not** reopen R7.5.3, does not create another H2/H3 readmission remediation, and does not weaken the finite-closure decision. The final x16 H2/H3 admission attempt remains closed `STABILITY_BLOCKED_FINAL`.

The only purpose of this step is to use already-existing Phase 2A artifacts to determine which upstream mechanism should be isolated next before any additional heavy training.

## Evidence entering this step

The completed V1+ Phase 2A Strategy-memory capacity ablation established:

- H2 / THREE_HANDED / x4, two frozen training seeds;
- all local Advantage gates passed;
- all COMMON learner local policy-fit gates passed;
- increasing Strategy reservoir capacity from 100k to 800k retained most of the generated Strategy stream but reduced pooled COMMON cross-seed mean TV by only about 0.0052 (about 2.4%);
- the paired bootstrap confidence interval for that COMMON improvement crossed zero;
- no capacity arm passed the frozen cross-seed stability gates;
- therefore Strategy-memory truncation is at most a secondary amplifier at this scale and blind multi-million Strategy-memory escalation is rejected.

Earlier x16 and Phase 1/1B diagnostics also showed that independent 3H seeds visit materially different geometry/history support and that the dominant variance family is upstream sampling/deck/chance rather than final-policy learner randomness.

## Frozen questions

The read-only Advantage forensic must distinguish as far as the preserved artifacts permit among four mechanisms.

### HYP-A — Advantage-memory pressure

The fixed 100k Advantage reservoir may itself be too small relative to the number/diversity of Advantage samples generated in 3H. Evidence favoring this hypothesis would include strong saturation/replacement pressure and materially better coarse support agreement than retained exact-support agreement, without large conditional target disagreement on exact infosets shared by both seeds.

This forensic can support or weaken HYP-A but cannot prove the causal effect of a larger Advantage reservoir because no passive larger-Advantage shadow reservoir was recorded in Phase 2A.

### HYP-B — chance/return target noise

Even when the same exact observable infoset is retained in both seeds, sampled counterfactual Advantage targets may disagree materially because future chance realization and sampled opponent continuation create high-variance returns. Evidence favoring HYP-B includes:

- substantial within-seed target variance among repeated exact infosets;
- frequent positive/negative target sign instability for the same legal slot;
- high cross-seed target disagreement on shared exact infosets; and/or
- high regret-matching behavior-policy TV derived from those shared target means.

### HYP-C — trajectory-support divergence

Independent training seeds may visit materially different regions before target estimation is even compared. Evidence favoring HYP-C includes:

- low cross-seed overlap for exact/current-state Advantage support;
- large cross-seed distribution TV by street, legal mask, history length or current geometry;
- substantial differences in Advantage samples generated per root/iteration; and
- final Advantage networks that induce different regret-matching behavior policies on the same frozen heldout states despite each seed passing its own local fit gate.

### HYP-D — representation fragmentation

SPNNIV3 may split strategically nearby states so finely that overlap disappears even when coarser information content aligns. Evidence favoring HYP-D requires a large increase in overlap when exact observations are projected to deliberately coarser, predeclared views such as current geometry without cards, current state without history, structured categorical history, or a V1-like history projection.

Coarse target comparisons are diagnostic only because those projections intentionally alias strategically distinct states. They cannot by themselves justify a lossy production representation.

## Frozen read-only measurements

Use exactly the two completed Phase 2A seed checkpoints under the original execution SHA:

`4bfa55d69029cd69536fa6dbfcadd162719cb887`

Expected seeds:

- `1342191342`
- `1801739323`

Expected representation/domain:

- `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`
- `THREE_HANDED`

Expected completed shape:

- 3 iterations;
- 4 x 64-root chunks per iteration;
- 768 roots per seed;
- final resume stage index 12;
- Advantage reservoir capacity 100,000.

The forensic must perform no solver traversal, no reservoir mutation, no optimizer step and no policy refit.

For each seed, record:

- Advantage samples seen/retained/retention fraction;
- retained samples by iteration;
- unique exact observations and duplicate fraction;
- support-set hashes for exact observation, cards only, current state without history, geometry/current state without cards, exact history, structured categorical history, V1-like last-32 `(street, action_type)` history, current+V1-like history, and geometry+V1-like history;
- street, legal-mask and history-length distributions;
- within-seed repeated-exact-infoset target variance and legal-slot sign instability;
- frozen iteration reports including Advantage samples generated per iteration/root and local Advantage fit gate values.

Across seeds, record:

- Jaccard overlap for every frozen support projection;
- total-variation distance between street/legal-mask/history-length distributions;
- shared exact-infoset coverage on each retained reservoir;
- for shared exact observation + legal-mask groups, compare weighted mean Advantage targets, target MAE, legal-slot sign flips and regret-matching behavior-policy TV;
- coarse-group target comparisons only as explicitly labeled diagnostics.

On the two frozen THREE_HANDED heldout corpora (`2029384436`, `1150634112`), evaluate the two final Advantage networks on the same first 1024 states and record:

- legal-slot raw Advantage-output disagreement;
- legal-slot positive-sign disagreement;
- regret-matching behavior-policy mean/p50/p95/max TV;
- dominant legal action mismatch rate;
- the same summaries by street.

The heldout Advantage comparison is a diagnostic of upstream behavior-policy divergence, not the R7.5 admission metric and not an architecture selection score.

## Predeclared interpretation rules

1. **Target-noise priority:** if shared exact infosets have materially high target-derived regret-matching TV/sign disagreement and/or repeated exact infosets show large within-seed target variance, prioritize a chance/return variance-reduction ablation. Do not respond by merely enlarging Strategy memory.
2. **Support-divergence priority:** if conditional shared-exact targets are comparatively consistent but exact/current support overlap is low and final Advantage behavior is divergent on common heldout states, prioritize traversal/chance support stabilization or stratified/canonical sampling.
3. **Advantage-capacity candidate:** if the Advantage reservoir is strongly saturated while coarse/current support is substantially more aligned than retained exact support and shared-exact targets are comparatively consistent, a controlled passive Advantage-capacity ablation is eligible. No blind capacity escalation is allowed.
4. **Representation candidate:** representation compression becomes a first-order candidate only if deliberately coarser projections recover substantial cross-seed support alignment that exact/current representations lose. Strategic non-inferiority would still be mandatory later.
5. Several mechanisms may coexist. The next training experiment must isolate at most one primary upstream mechanism plus a frozen control; no multi-change rescue bundle is allowed.

No numeric threshold is invented after seeing the forensic output. The output is diagnostic; any causal Phase 2B training gate must be separately precommitted before training.

## Strategic-strength safeguard

Stability remains an eligibility condition, never a strength metric. Any eventual V1+ candidate must separately demonstrate strategic non-inferiority or superiority to the certified stable V1 control on paired common deals before production architecture selection.

## Prohibitions

- no new training in this forensic;
- no solver traversal;
- no checkpoint mutation;
- no seed shopping;
- no threshold relaxation;
- no H2/H3 winner declaration;
- no R7.5.3 reopening;
- no production authorization;
- no blind Strategy-memory or Advantage-memory escalation from this readout alone.
