# SpinCore Roadmap — active state 2026-09-19

## Active status

- LT0 — **DONE**.
- LT1 — **DONE**.
- LT2 Stage A — **PASS**.
- LT2 Stage B — **PASS** at 4.5M roots / iteration 7500.
- Jammer FAI loss — **CONFIRMED**.
- Jammer FAI infoset overfold — **CONFIRMED**.
- Raw center-vs-gap decomposition — **ACTION-GAP DRIFT DOMINANT; OFFSET/FALLBACK SECONDARY**.
- Fallback-only patch — **REJECTED AS INSUFFICIENT**.
- Controlled reservoir refit — **PASS; 100 STEPS FAIL, 400/1600 RECOVER**.
- Stage-B reservoir poisoning hypothesis — **NOT SUPPORTED**.
- Insufficient Advantage refit budget — **LEADING CAUSE**.
- B400 broad generalization — **NEXT**.
- K4 training — **NOT AUTHORIZED**.
- long root training — **PAUSED**.
- DeepCrusher — **DEFERRED**.

## Controlled refit evidence

B_MORE_FOLD, Stage-B-reservoir minus Stage-A-reservoir:

- 100 steps: `-11.64417`, CI95 `[-18.26748,-5.02086]`;
- 400 steps: `+13.91919`, CI95 `[+2.96579,+24.87259]`;
- 1600 steps: `+7.29093`, CI95 `[+3.38533,+11.19653]`.

All three 100-step reps are negative.
All three 400-step reps are positive.
All three 1600-step reps are positive.

Therefore more Stage-B data did not destroy the useful signal. The fitted network fails to extract it at the canonical 100-step budget.

## Candidate intervention

Use `advantage_steps=400` as the minimum tested sufficient budget.

Do not adopt 1600 yet: it is substantially more expensive and 400 already clears the structural gate in every replicate.

## Next gate

Broad natural-HU forensic evaluation of all three B400 candidates.

Policies:

- PROD_A;
- PROD_B;
- B400_R0;
- B400_R1;
- B400_R2.

Baselines:

- UNIFORM_LEGAL;
- PASSIVE_CALLER;
- JAMMER.

Primary:
- B400 vs PROD_B on JAMMER.

Secondary:
- B400 vs PROD_B on other baselines;
- B400 vs PROD_A on JAMMER.

If B400 passes broadly:
- freeze 400 as the intervention;
- run a bounded continuation pilot from Stage B before any long continuation.

If B400 fails only on Jammer:
- escalate once to 1600 broad gate.

If B400 causes a resolved tradeoff:
- diagnose before training.

Holdout `20261001..20261006` remains sealed.

## Immediate action

Run `bash tools/run_lt2_hu_b400_broad_generalization.sh`.

No root training yet.
