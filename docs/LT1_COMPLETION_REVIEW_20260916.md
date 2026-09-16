# LT1 completion review — 2026-09-16

Status: LT1 execution complete; preserve checkpoint. LT2 extension awaits targeted neural-fit optimization and one bounded resume/throughput check, not a restart or strategic-strength gate.

## Evidence
User-supplied report.json SHA256: 3cd21150451f70be5d3504df5b881e243b97b394adba71114d4405dafa390ee6.
Run: /home/rz9/spincore_lean_functional/runs/long_training_lt1/20260915_181249.
Terminal reports LT1_SCALE_VALIDATION_PASS and exit status 0.
GitHub main inspected for interpretation; exact local training commit has not yet been independently established.

2,000 consecutive iterations; 600 roots/iteration; seed 20260922.
3H: 654,000 roots, 115,262,111 nodes, 20,598,662 advantage samples seen, 1,485,285 strategy samples seen.
HU: 546,000 roots, 79,329,378 nodes, 16,312,588 advantage samples seen, 544,490 strategy samples seen.
All recorded advantage losses finite. Final policy losses: 1.11331534 (3H), 1.10055304 (HU). These are fit diagnostics, not poker-strength evidence.

## Complete timing attribution
report.wall_seconds = 24,961.87672 (6h56m01.9s).
- Advantage fit: 17,742.20762 s (71.08%).
- Trees: 5,042.20995 s (20.20%).
- Sampled policy collection: 876.64775 s (3.51%).
- 20 periodic checkpoints: 868.22447 s (3.48%).
- Final policy fit: 346.06717 s (1.39%).
- Remaining measured wall time: 86.51976 s (0.35%).

The current trainer measures wall_seconds before its final finalized=True checkpoint. That final save metric is printed but not appended to report.checkpoint_metrics. Hence report.json excludes final-save cost; 53.30093 s / 1.97414 GiB is the iteration-2000 pre-finalization checkpoint, not independently measured final-save cost.

Terminal /usr/bin/time reports 6:39:00, shorter than the trainer's wall interval; this remains an unresolved clock/measurement inconsistency. Do not average the two or promise LT2 runtime from them.
Reported CPU 904% is about 28.25% of 32 logical processors, conditional on that timing source.
Maximum RSS 13,358,660 KiB is not an aggregate worker-pool peak. Prior free -h: ~30 GiB WSL total, ~19 GiB available, swap zero. No current evidence that increasing WSL allocation fixes the neural-fit bottleneck.

## Sampling and reservoir interpretation
3H blind counts: 10/20 339088; 15/30 195071; 20/40 91134; 30/60 24514; 40/80 3604; 50/100 511; 60/120 78.
HU: 10/20 49224; 15/30 135335; 20/40 162153; 30/60 123438; 40/80 57339; 50/100 15394; 60/120 2667; 80/160 350; 100/200 100.
Current legacy_scenario_data.json assigns ZERO empirical 3H weight to 80/160 and 100/200. Their absence in 3H is consistent with that model, not a discovered sampling fault. Any rare-state oversampling is a separate explicit strategic change.

Reservoir capacity is 2M per memory per domain. Under current reservoir semantics, both advantage memories have reached capacity; policy memories have seen only 1,485,285 / 544,490 samples and have not reached capacity. LT1 therefore does NOT establish the fully occupied 8M-slot memory/checkpoint peak. Preserve capacity; monitor growth during LT2.

## Decision and finite next step
Preserve LT1 unchanged as the continuation source. Do not start another fresh campaign or launch tens of millions of roots with an unexamined neural-fit bottleneck.
1. Inspect legacy and current fit/batch construction before modifying execution.
2. Optimize the demonstrated neural-fit bottleneck without reducing steps, batch size, memory capacity or changing reset/RNG/learning semantics. Parallel domain fits are a candidate only after dependency inspection, not an approved assumption.
3. Use one bounded comparison on an isolated copy of the LT1 checkpoint: confirm successful load/resume and compare useful iteration throughput under identical work. Stop when a clear winner is established; retain current path if candidate is not faster or changes intended semantics.
4. Preserve the original LT1; write extension into a separate run directory. Extend from iteration 2000 only after that check, with resource monitoring as policy memories fill.
5. Do not require a DeepCrusher strength verdict to authorize LT2; that remains later learning-curve work.

Checkpoint load support is confirmed by source inspection (models, optimizers, reservoirs, batch RNG, sampler RNG and counters); this specific local final checkpoint has not been independently loaded in this review.
