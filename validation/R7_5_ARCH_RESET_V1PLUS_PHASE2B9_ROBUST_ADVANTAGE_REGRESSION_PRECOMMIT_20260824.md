# R7.5 Architecture Reset — Phase2B9 Robust Advantage Regression Screen

Status: **FROZEN BEFORE PHASE2B9 OUTPUTS**  
Date: 2026-08-24

## 1. Why this screen exists

Phase2B8 is a clean negative result: the iteration-1/2 equivalence gate passed, but replacing the Phase2B6 uniform continuation component with a 25% lagged learned behavior worsened COMMON pooled cross-seed mean TV from `0.18934816676149685` to `0.20333039705340733`; the paired 95% CI for Phase2B6-minus-Phase2B8 is entirely negative `[-0.02168171060108362, -0.006820936161345012]`. Root and continuation both worsened. Therefore no post-hoc lag-weight tuning is allowed.

The earlier causal chain remains the strongest one:

`chance/return noise -> small Advantage sign changes -> regret-matching zero-boundary amplification -> divergent preflop behavior -> divergent trajectory/Strategy streams -> unstable AveragePolicy`.

Phase2B1 showed that chance-only K1 target-policy TV was about `0.51537`, while traversal-only K1 TV was about `0.05194`; naive K4 target averaging reduced numeric target MAD but did **not** reduce sign/policy instability. Phase2B6 then showed that adding entropy only to preflop continuation behavior had a real but incomplete stabilizing effect. Phase2B8 showed that temporal learned anchoring is not an adequate substitute for that entropy.

Before any further trajectory-training intervention, Phase2B9 asks a narrower question: **is the squared-error Advantage regression itself giving excessive leverage to noisy chance-conditioned targets, and does a robust regression loss produce materially more cross-seed-stable Advantage behavior from the exact same frozen memories?**

## 2. Frozen source

Use the exact completed Phase2B6 trajectories as the source, because Phase2B6 is the best causally supported successor so far and Phase2B8 is inferior.

- Representation: `H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL`.
- Domain: `THREE_HANDED`.
- Training seeds: `1342191342`, `1801739323`.
- Evaluation seeds: `2029384436`, `1150634112`.
- Exact Phase2B6 result SHA-256: `33ec6ba89823dae632b7af935def17444379c96a28e59478c0b7c91f1ec3659a`.
- Exact Phase2B8 result SHA-256: `1fd9144a488cea6de0a7500320d552abf994908b5200146d4baa4bd6f81c4d98`.
- Phase2B6 final resume checkpoints must identify H2/3H, the Phase2B6 execution SHA `4fa96434321c32efc734a55ae75982018ff2d091`, `global_root=768`, iteration 3, stage 12, and the Phase2B6 checkpoint-extra schema.
- Advantage and Strategy memories are read-only.
- No solver traversal, reservoir insertion, AveragePolicy fit, heldout regeneration, or checkpoint mutation is permitted.

## 3. Paired fit design

For each frozen training-seed Advantage memory, fit two four-member Advantage ensembles from scratch:

1. `MSE_PAIRED_CONTROL`: the canonical weighted legal-action MSE used by the current training stack.
2. `HUBER_BETA_002`: weighted legal-action Smooth-L1/Huber regression with `beta=0.02`.

The Huber beta is frozen **before Phase2B9 outputs**. `0.02` is chosen from the independent Phase2B1 chance-noise scale: pooled chance-only K1 target mean-absolute difference was `0.02337436098850536`. This is not a parameter sweep and no alternate beta is authorized after seeing the result.

For each training seed and member index, MSE and Huber must use:

- exactly the same model initialization seed;
- exactly the same batch RNG seed;
- exactly the same memory items;
- exactly the same 4096 optimizer steps;
- batch size 256;
- Adam, LR 0.001;
- the same per-sample iteration/reach weights and legal masks;
- the same gradient norm cap 10.0.

Member 0 uses the canonical iteration-3 primary reset seed for initialization and a new deterministic Phase2B9 batch-seed namespace. Members 1-3 use the canonical iteration-3 side-member initialization and batch seeds. The Phase2B9 member-0 batch seed is frozen by code and identical between the paired losses; it is not selected from results.

## 4. Why this is a screen, not a solver change

The desired CFR target is an expected counterfactual regret. MSE estimates a conditional mean; Huber can trade some asymptotic mean fidelity for robustness under heavy/noisy tails. Therefore a Huber improvement is not automatically strategically acceptable. This screen only asks whether robust regression materially reduces the **observed cross-seed behavior instability from the same data**.

If the screen passes, it authorizes at most one later small causal trajectory-training pilot. Strategic strength remains a separate mandatory gate.

## 5. Readouts

On the first 1024 states of each frozen H2/3H heldout artifact, evaluate the four-member behavior algebra exactly as the canonical uncertainty-damped wrapper does.

Report for paired MSE and Huber:

- cross-seed behavior TV mean, p50, p95, max;
- dominant-action mismatch rate;
- results by `PREFLOP_ROOT`, `PREFLOP_CONTINUATION_1`, `PREFLOP_CONTINUATION_2PLUS`, and postflop street;
- pooled equal-heldout mean TV;
- paired bootstrap 95% CI for `MSE - HUBER` state TV;
- each member and ensemble weighted NRMSE on a frozen deterministic audit slice of its own source memory;
- exact source checkpoint hashes and memory counts.

The screen also reports the paired fit loss summaries but loss magnitude alone never authorizes training.

## 6. Precommitted eligibility rule

`HUBER_ROBUSTNESS_SCREEN_PASS` requires **all** of:

1. every Huber ensemble weighted NRMSE is `<= 0.75` on its frozen source-memory audit;
2. Huber cross-seed mean TV improves versus paired MSE on **both** heldout seeds;
3. pooled equal-heldout mean TV improves by at least `0.03` absolute **or** `10%` relative;
4. the paired equal-heldout bootstrap 95% CI for `MSE - HUBER` is strictly positive;
5. neither heldout Huber p95 worsens by more than `0.02`;
6. pooled dominant-action mismatch does not increase;
7. `PREFLOP_ROOT` mean TV does not worsen by more than `0.01`;
8. combined preflop-continuation mean TV does not worsen by more than `0.01`.

If all pass: status `HUBER_ROBUSTNESS_SCREEN_PASS_ELIGIBLE_FOR_SMALL_CAUSAL_PILOT`, and the only next route is to precommit one Phase2B10 small causal trajectory pilot using the Phase2B6 continuation-floor contract plus the frozen Huber loss.

If any fail: status `HUBER_ROBUSTNESS_SCREEN_FAIL_DO_NOT_TRAIN`, and no Huber trajectory pilot is allowed. The next route becomes `DESIGN_STRATIFIED_CHANCE_SUPPORT_OR_SOLVER_LEVEL_VARIANCE_REDUCTION`, not another behavior anchor/floor tweak.

## 7. Prohibitions

- no Phase2B8 lag-weight tuning;
- no higher uniform floor;
- no Huber beta sweep;
- no seed shopping;
- no threshold relaxation;
- no dropped domain/heldout;
- no solver traversal in Phase2B9;
- no reservoir mutation;
- no AveragePolicy refit;
- no production training;
- no architecture winner selection from this screen;
- `READY FOR TABLES = NO`.

## 8. Strategic firewall

Even a Phase2B9 PASS only establishes a robust-regression mechanism worth testing causally. A later trajectory pilot must independently satisfy the historical stability gates, and only then may the candidate enter a precommitted strategic-strength comparison against the certified stable V1 control.
