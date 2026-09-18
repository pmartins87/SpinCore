# SpinCore — LT2 HU preflop board-averaging mechanics smoke

Date: 2026-09-18
Status: **ACTIVE — FIRST RUN FAILED SAFELY; RNG COUPLING FIXED; RERUN REQUIRED**

## First-run outcome

The first smoke stopped with:

`RuntimeError: HU board averaging changed preflop sample count across boards`.

This was not a strategy result. It was an implementation invariant failure, and the gate worked as intended.

No training memory was written, no optimizer step ran, and the preserved Stage-B checkpoint remained untouched.

Canonical diagnosis:

- `docs/LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_FAILURE_20260918.md`.

## Root cause

The original implementation reset one global external-sampling RNG state before each board variant.

That did not preserve later preflop opponent samples because the traversal is depth-first. A preflop traverser branch can enter postflop and consume a board-dependent number of RNG draws before recursion returns to a later preflop branch.

Thus different future boards could shift the RNG position seen by a later preflop opponent node.

## Corrected implementation contract

The opt-in trainer parameter is:

`hu_preflop_board_average_k`.

Default `1` remains canonical.

For TRUE_HEADS_UP and K>1:

1. create the canonical root from the deterministic deck seed;
2. snapshot fixed hole cards and canonical future board;
3. run board 0 canonically;
4. record every sampled **preflop opponent action**, plus its observation and legal set;
5. draw K-1 alternate future boards conditional on the same hole cards;
6. replay the exact canonical preflop opponent-action trace on every alternate board;
7. require identical preflop observation/legal set at every replayed node;
8. consume one dummy RNG draw per replayed preflop sample to preserve the local sampling-call count;
9. leave postflop external sampling ordinary;
10. average only preflop Advantage targets;
11. retain postflop samples/targets from canonical board 0 only;
12. restore the RNG state reached by canonical board 0.

The generic collector gained only an overridable opponent-sampling hook. With the default path, it still executes the same direct `sample_action` call as before.

## Rerun smoke contract

The smoke compares the same deterministic prospective Stage-B HU roots under K1 and K4 without writing training memory or running optimizer steps.

It must prove:

- K1/K4 root counts equal;
- K1/K4 sample counts equal;
- sample identities/order equal;
- every postflop target unchanged;
- at least one preflop target changed;
- K4 nodes > K1 nodes;
- preserved Stage-B checkpoint SHA unchanged.

Launcher:

```bash
bash tools/run_lt2_hu_preflop_board_averaging_smoke.sh
```

Expected marker:

`LT2_HU_PREFLOP_BOARD_AVERAGING_SMOKE_PASS`.

## What happens after PASS

Do **not** start the K4 training pilot automatically.

The next gate is Stage-A -> Stage-B causal attribution. We must show that the target-noise/sign errors corrected by K4 actually explain the observed HU-Jammer regression rather than merely improving a proxy chosen after seeing the benchmark.

Only after that causal link is demonstrated may K4 be trained.
