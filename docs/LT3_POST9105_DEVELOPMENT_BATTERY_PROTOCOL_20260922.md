# LT3 post-9105 development battery protocol — frozen 2026-09-22

## Purpose

Compare the three predeclared LT3 research checkpoints after the uninterrupted
8200 -> 9105 continuation:

- finalized LT2/LT3 baseline @8100;
- raw internal milestone @8600, finalized only on a **derived copy**;
- finalized endpoint @9105.

This is a development battery. It is not the LT3 sealed holdout and cannot
authorize production promotion by itself.

## Frozen source artifacts

### Raw 8600 milestone

- checkpoint SHA256:
  `11cee67aaa626224f59a89ee2b18a1a521e2679f0152144728729186ee580cd6`
- HU ENS8 sidecar SHA256:
  `4717fcb5ea3c132a19687cf0848c95694c920e7ac4e42300d740c1bdf3c73816`
- source checkpoint must have `completed_iteration=8600` and `finalized=false`;
- it must remain immutable;
- AveragePolicy finalization is performed on an in-memory derived copy and saved
  to a new evaluation checkpoint.

### 9105 endpoint

- checkpoint SHA256:
  `21945e27c43c7e6c1cdb77018cd66dc90b3fab72c29a9034bb4a9f97cc0e6c68`
- HU ENS8 sidecar SHA256:
  `b9c3ffffc7139eeb77c4b4182136e10ada5cad2f023aa6e560e164e2e0ac9256`
- checkpoint must have `completed_iteration=9105` and `finalized=true`.

The frozen 8100 baseline remains the same finalized checkpoint + HU ENS8
sidecar that passed the LT2 sealed holdout. Its runtime SHA256 is recorded by
the development run.

## Development data contract

- seed: `20260922`;
- AveragePolicy pairwise cross-play scenarios: 3000;
- HU current ENS8 pairwise scenarios: 3000 full-sampler draws, with non-HU
  draws skipped by the HU evaluator;
- policy-drift scenarios: 3000;
- weak-baseline quality scenarios per checkpoint: 2000;
- worker target: 31;
- exact same seed and scenario count for each pair within a battery component.

These are development seeds. The LT2 final sealed-holdout seeds and any future
LT3 sealed-holdout seeds must not be touched.

## Battery components

### A. Finalized AveragePolicy cross-play

Run all three ordered comparisons on identical development conditions:

1. 8100 -> 8600;
2. 8600 -> 9105;
3. 8100 -> 9105.

Primary metric is the existing paired `MIXTURE_HERO ALL` after-minus-before
chip EV with a 95% CI clustered by scenario. Three-handed and true-HU domain
breakdowns, direct HU and 3H invasion are retained as anti-collapse diagnostics.

Classification for a paired statistic:

- **POSITIVE**: CI95 lower bound > 0;
- **NEGATIVE**: CI95 upper bound < 0;
- **INCONCLUSIVE**: otherwise.

Development screen for 9105 versus 8100:

- `DEV_FAVORS_9105_OVER_8100` only if the ALL comparison is POSITIVE and
  neither major domain is NEGATIVE;
- `DEV_REJECTS_9105_VS_8100` if the ALL comparison is NEGATIVE;
- otherwise `DEV_INCONCLUSIVE_8100_VS_9105`.

This screen is not a production verdict.

### B. HU current ENS8 behavior

Compare the actual eight-member HU current-behavior sidecars for the same three
pairs. The primary metric is exact seat-balanced after-versus-before chip EV on
shared deals. Paired deltas against UNIFORM_LEGAL, PASSIVE_CALLER and JAMMER are
retained as transparent context.

The same POSITIVE / NEGATIVE / INCONCLUSIVE CI classification is used.

### C. AveragePolicy movement

Measure decision-level total-variation drift for all three pairs using the
checkpoint-independent uniform-legal probe. This is descriptive only and is not
a strength metric.

### D. Transparent weak baselines

Evaluate finalized AveragePolicy @8100, @8600 and @9105 against the same
UNIFORM_LEGAL, PASSIVE_CALLER and JAMMER development conditions. This is a
sanity/context diagnostic, not a GTO or exploitability proof.

## Immutability and stopping rules

- no CFR roots;
- no source training-memory writes;
- no source optimizer steps;
- no source checkpoint mutation;
- no sealed-holdout access;
- no further training decision until this battery is interpreted together with
  the DeepCrusher external-strength lane;
- do not promote 8600 or 9105 from this development battery alone.

## Canonical runner

`bash tools/run_lt3_post9105_dev_battery.sh`

Expected terminal sentinel:

`LT3_POST9105_DEV_BATTERY_COMPLETE`


## Execution incident and memory-safe correction — 2026-09-22

The first execution reached the frozen 8600 derived-finalization PASS and then
was terminated by the host at the first AveragePolicy 8100 -> 8600 cross-play,
before any development result was emitted.

Root cause: the orchestration script passed full multi-GB training checkpoints
directly to a 31-process evaluator. Each spawned worker independently called
`torch.load` on both full checkpoints, multiplying reservoir/optimizer memory
use. This violated the already established evaluation pattern used by
`run_lt2_checkpoint_crossplay.sh`, which exports compact inference-only
AveragePolicy artifacts before multiprocessing.

Correction: export compact policy-only checkpoints for 8100, derived 8600 and
9105 once in the parent process, validate that each contains only finalized
AveragePolicy weights, and use those compact artifacts for AveragePolicy
cross-play, policy drift and weak-baseline quality evaluation.

This is an execution/memory correction only. It changes no development seed,
scenario count, pairing, policy weights, metric, confidence interval,
classification rule or stopping rule. The first terminated attempt produced no
cross-play outcome, so no statistical criterion was adapted after observing a
result.


## Frozen outcome — completed 2026-09-22

The corrected battery completed under the preregistered development protocol.
No sealed holdout was touched and no promotion was authorized.

Primary AveragePolicy 8100 -> 9105 paired result:
- ALL: -0.055 chips/hand, 95% CI [-2.636,+2.526] — INCONCLUSIVE;
- THREE_HANDED: -2.303, 95% CI [-4.841,+0.235] — INCONCLUSIVE;
- TRUE_HEADS_UP: +2.630, 95% CI [-2.151,+7.411] — INCONCLUSIVE.

Current HU ENS8 8100 -> 9105 direct:
- -0.804 chips/hand, 95% CI [-7.391,+5.784] — INCONCLUSIVE.

The 8100 -> 9105 AveragePolicy drift is measurable rather than a no-op:
- overall mean TV 0.04182, p95 0.11972, argmax disagreement 11.76%;
- THREE_HANDED mean TV 0.03208;
- TRUE_HEADS_UP mean TV 0.06139, p95 0.14563, argmax disagreement 15.68%.

Thus the development evidence says that training materially changed the policy
but did not establish that the change improved strength. No further long
training block is justified from this battery alone. The next evidence gate is
the frozen DeepCrusher external-strength lane (DC0 -> DC1 -> DC2), comparing
8100, derived-finalized 8600 and 9105 under the same external protocol.
