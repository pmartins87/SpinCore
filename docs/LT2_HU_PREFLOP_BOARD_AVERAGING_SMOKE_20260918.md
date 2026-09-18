# SpinCore — LT2 HU preflop board-averaging mechanics smoke

Date: 2026-09-18
Status: **ACTIVE — READ-ONLY IMPLEMENTATION GATE**

## Trigger

The board-only estimator audit admitted K4 as the first candidate training semantic:
- K4 is the policy-space compute elbow;
- K8 doubles K4 nodes without resolved extra TV/regret/argmax improvement;
- board-only K4 captures most of the descriptive model-to-full-hidden-deal TV gap while avoiding posterior opponent-hand resampling.

No training roots are authorized until the implementation contract is verified.

## Implementation contract

The opt-in trainer parameter is:

`hu_preflop_board_average_k`

Default `1` preserves the canonical trainer.

For TRUE_HEADS_UP and K>1:
1. create the canonical root from the normal deterministic deck seed;
2. snapshot the exact hole cards and canonical future board;
3. retain the canonical board as board 0;
4. draw K-1 additional future boards conditional on exactly the same hole cards;
5. replay exactly the same external-sampling RNG state for every board variant;
6. collect K Advantage traversals for each traverser;
7. assert that all preflop information-state observations/legal masks/order agree;
8. average only the preflop target vectors;
9. retain all postflop samples and their targets from canonical board 0 only;
10. restore the RNG progression produced by canonical board 0.

Therefore the intended intervention changes preflop labels while preserving:
- root/scenario sampling;
- hole cards;
- sample count;
- sample insertion order;
- sample identity/legal mask/weight/iteration;
- postflop labels;
- canonical RNG progression for later traversers/roots.

The extra boards increase only traversal-node compute.

## Smoke

The smoke compares the same deterministic prospective Stage-B HU roots under K1 and K4 without writing training memory or running optimizer steps.

It must prove:
- K1/K4 root counts equal;
- K1/K4 sample counts equal;
- all sample identities and ordering equal;
- every postflop target is unchanged;
- at least one preflop target changes;
- K4 nodes > K1 nodes;
- preserved Stage-B checkpoint SHA is unchanged.

Launcher:

```bash
bash tools/run_lt2_hu_preflop_board_averaging_smoke.sh
```

Expected marker:

`LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_PASS`

After the smoke, use the measured K4/K1 node multiplier to size the bounded causal training pilot. Do not choose the pilot root budget before this measurement.
