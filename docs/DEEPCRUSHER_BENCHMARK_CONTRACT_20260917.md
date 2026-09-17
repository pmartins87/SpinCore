# SpinCore — DeepCrusher external benchmark contract

Date: 2026-09-17
Status: **NEXT PRODUCT-STRENGTH GATE AFTER LT2 SELF-PLAY REVIEW**

## Why this gate exists

LT2 Stage B reached 4.5M roots with healthy resources and all four 2M reservoirs in replacement regime. Stage A -> Stage B policy drift is material, but neither weak fixed opponents nor two independent contemporary checkpoint cross-play runs established a reproducible Stage-B strength advantage.

The first cross-play run (3000 scenarios, seed 20260918) pointed mildly toward Stage A. The independent higher-power run (9000 scenarios, seed 20260919) reversed the primary point estimates toward Stage B, while all primary confidence intervals still included zero. Additional HU-direct and 3H-invasion diagnostics were also inconclusive and changed direction across runs.

Therefore more same-regime training and more repeated A-vs-B self-play are both paused. The next useful strength reference is an external deterministic strategy: faithful DeepCrusher R8 v22.

## Admission prerequisite for DeepCrusher

Do not use a simplified imitation of DeepCrusher for canonical benchmark claims.

The DeepCrusher C++ oracle must first pass a source-faithfulness gate:

- OpenPPL strategy translated structurally, preserving rule ordering/priority and action semantics;
- every relevant OpenPPL/library symbol used by the strategy implemented faithfully, including symbols such as `AmountToCall` and their actual definitions rather than name-based approximations;
- action sizing and all-in conversion semantics preserved;
- deterministic parity probes against the frozen OpenPPL source on representative states;
- any accepted intentional divergence explicitly documented before benchmark use.

Until that gate passes, DeepCrusher is **not admitted** as a strength oracle.

## Preserved SpinCore candidates

Benchmark both preserved policies rather than assuming the latest checkpoint is stronger.

Stage A:

- iteration 3000 / 1.8M roots;
- SHA256 `e7dd9c460fe103933ee1b025b1ac7936555aa2802e3520e029b8793f616f3b5c`.

Stage B:

- iteration 7500 / 4.5M roots;
- SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

Do not overwrite either checkpoint.

## Primary benchmark design

When the faithful DeepCrusher oracle is admitted, evaluate Stage A and Stage B independently against the **same DeepCrusher policy** using identical scenario/deal/seat seeds.

Required properties:

- empirical SpinGo scenario sampler consistent with the current SpinCore evaluation line;
- same sampled scenario for Stage A and Stage B comparisons;
- same deal seed;
- same hero seat rotation;
- same DeepCrusher opponent policy and deterministic/random streams where applicable;
- paired chip-EV delta with scenario-clustered uncertainty;
- 3H and HU reported separately whenever DeepCrusher supports the corresponding domain faithfully;
- retain row-level evidence for later blind/stack/position decomposition.

Primary questions:

1. Does Stage B outperform Stage A against the same external DeepCrusher reference?
2. Does either checkpoint outperform DeepCrusher, and by how much under the sampled game distribution?
3. Are 3H and HU conclusions directionally consistent?

## Secondary breakdowns

After the primary paired result is stable, break down by:

- domain (3H / HU);
- blind level;
- effective stack bucket;
- hero seat / position;
- later, street or action-family diagnostics when the row evidence supports it.

Do not multiply subgroup claims before the primary benchmark is stable.

## Decision logic

- **Stage B clearly stronger than Stage A vs DeepCrusher:** resume the current training line only through another bounded block, preserving Stage B as source and re-running the same external benchmark afterward.
- **Stage A clearly stronger than Stage B vs DeepCrusher:** treat Stage B as a likely training-quality regression/cycling signal; investigate AveragePolicy/training dynamics before more roots.
- **Stage A and Stage B externally indistinguishable:** do not spend another long block solely on root count; inspect training/evaluation sensitivity and consider architecture/optimization changes only through bounded experiments.
- **DeepCrusher oracle not yet faithful:** keep SpinCore training paused and finish oracle parity first.

This benchmark is still empirical sampled chip-EV, not a proof of GTO exploitability.

## Current stop condition

Do not continue SpinCore training beyond iteration 7500 before the faithful DeepCrusher benchmark gate is available and reviewed.
