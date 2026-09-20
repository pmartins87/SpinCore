# SpinCore Current Work

Date: 2026-09-20
Status: **ITERATION 8000 SOURCE FROZEN — REPLICATED ENS8 BROAD EV PASS — ISOLATED ONLINE ENS8 PILOT 8000→8100 NEXT**

## Replicated ENS8 broad-EV result

Both independently composed size-8 ensembles are strongly positive against every forensic baseline.

ENS8_A:
- UNIFORM_LEGAL `+28.097`;
- PASSIVE_CALLER `+10.961`;
- JAMMER `+15.052`.

ENS8_B:
- UNIFORM_LEGAL `+30.034`;
- PASSIVE_CALLER `+13.462`;
- JAMMER `+17.212`.

Both are resolved improvements versus Stage B 7500 on all three baselines.

Versus the 7600 pilot:
- neither group has a resolved Uniform regression;
- both significantly improve Passive;
- both significantly improve Jammer.

## Composition robustness

ENS8_B minus ENS8_A:
- Uniform `+1.937`, unresolved;
- Passive `+2.501`, barely resolved;
- Jammer `+2.160`, unresolved.

The maximum ENS8 composition gap is about half the ENS4 maximum gap.

Size 8 is therefore the leading stabilization candidate.

## Selection discipline

Do not pick ENS8_B because it happened to score higher.

The online pilot uses ENS8_A (replicas 0..7), the first predeclared group.

## Active gate

Run an isolated +100-iteration online-feedback pilot:

- source iteration 8000 exact SHA;
- target 8100;
- THREE_HANDED unchanged single fresh100;
- TRUE_HEADS_UP = 8 independent fresh400 fits;
- exact ENS8_A fit seeds reused every iteration;
- raw Advantage average before unchanged lean regret matching;
- K4 off;
- holdout sealed.

The source 8000 checkpoint remains immutable.

## Immediate action

```bash
bash tools/run_lt2_hu_ens8_online_pilot.sh
```

Wait for `LT2_HU_ENS8_ONLINE_PILOT_PASS`, then send
`SpinCore_LT2_hu_ens8_online_pilot.json`.

Do not run the generic trainer beyond 8000.
