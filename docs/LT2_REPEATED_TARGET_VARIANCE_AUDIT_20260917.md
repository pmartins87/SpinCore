# SpinCore — LT2 repeated-state target variance audit

Date: 2026-09-17
Status: **NEXT GATE — READ-ONLY CAUSAL DIAGNOSTIC**

## Why this exists

The 100k value-sensitivity audit showed that the Stage-B Advantage model disagrees with stored sampled targets primarily in high-span states. However those stored targets come from external-sampling MCCFR and are not noise-free labels.

The next question is therefore causal: **how much of the observed target error is Monte-Carlo sampling variance, and how much remains as model approximation error after repeatedly evaluating the exact same state/deal?**

This must be answered before changing model capacity, representation, loss, fallback, or root count.

## Source and safety

Source checkpoint: Stage B, iteration 7500 / 4.5M roots, SHA256 `3463aa1dccac2c9f26cb45753b69490cfa52616bdeb21e075b320b1b0d40f7d0`.

The launcher SHA-checks the source before and after. The diagnostic performs no optimizer steps and writes no samples into the checkpoint's training memories. It does run fresh diagnostic solver traversals in memory.

## State sampling

Default diagnostic budget:

- 64 independent selected states per domain per street;
- domains: THREE_HANDED and TRUE_HEADS_UP;
- streets: preflop, flop, turn, river;
- state is the first decision reached on that street during a deterministic trajectory sampled from the stored Stage-B Advantage behavior policy;
- 8 repeated Advantage-target traversals from the **same selected solver state and same hidden deal/future board**.

The 64x8 counts are a bounded diagnostic budget, not PASS thresholds. Per-state metrics are aggregated with standard errors / 95% intervals.

## Exact-opponent comparison

Every selected state is evaluated twice:

- `exact_opponent_levels=0`: production Stage-B external-sampling setting;
- `exact_opponent_levels=1`: exactly expand the first opponent decision layer, then resume external sampling.

This measures whether one exact opponent layer materially reduces target variance and what node-cost multiplier it requires.

## MSE decomposition

For each exact selected state, let repeated sampled target vectors be `T_r`, their repeat mean be `T_bar`, and the Stage-B network output be `P`.

Over legal actions:

`mean_r ||P - T_r||^2 = mean_r ||T_r - T_bar||^2 + ||P - T_bar||^2`.

The audit reports:

- within-repeat target MSE = directly observed external-sampling noise for the fixed state/deal;
- model MSE to sampled targets;
- model MSE to repeat-mean target;
- exact decomposition closure;
- chip-equivalent RMSE terms;
- production-policy TV between model and repeat-mean target;
- target-value gap / regret using the repeat mean as a lower-noise diagnostic reference;
- repeat-induced policy variability;
- node cost.

The repeat mean is **not** a GTO oracle. Because the hidden deal and future board are fixed within repeats, this audit isolates opponent-action external-sampling noise but does not include across-deal chance/hidden-card variance. The measured noise is therefore a lower bound on total conditional target variance.

## Interpretation branches

If within-repeat noise explains a large fraction of sample-target MSE and exact level 1 reduces it substantially at acceptable node cost, target variance becomes the leading mechanism and an isolated exact-level candidate can be tested before any long training.

If within-repeat noise is small while model MSE to the repeat mean remains large, the leading suspects become representation/capacity and target-function complexity rather than optimizer budget.

If noise is high but exact level 1 barely reduces it, opponent sampling is not the dominant source; chance/hidden-state variance and representation generalization need separate diagnosis.

If the result differs sharply by street or domain, subsequent interventions must stay localized rather than globally changing the trainer.

No architecture or training change is authorized directly by this diagnostic. Any intervention must first produce a bounded candidate and beat preserved Stage B on the statistically powered weak-baseline evaluation.

## Launcher

```bash
bash tools/run_lt2_repeated_target_variance_audit.sh
```

Expected completion marker: `LT2_REPEATED_TARGET_VARIANCE_AUDIT_PASS`.

Root training remains paused.
