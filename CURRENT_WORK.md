# SpinCore Current Work

Date: 2026-09-21
Status: **NATIVE C++ TRACKER PASS — TRACKER + FROZEN NATIVE INFERENCE SHADOW GATE NEXT**

## Frozen strategy/runtime

All strategic/model identities remain frozen.

No training, EV tuning or holdout reuse is permitted.

## Native tracker result

PASS:
- 12,000 observable transitions;
- 4,785 Hero canonical-state checks;
- 2,742 invisible CHECK deferrals;
- 1,124 delayed actions reconciled at MyTurn;
- 962 multi-action synchronization events;
- 3,660 street reveals;
- 500/500 corrupt-frame rejections;
- 500/500 skipped-transition rejections;
- 0 failures.

The Python/reference architecture and the native C++ implementation now agree at the level needed for productionization.

## Active gate

The next component joins:

1. native OpenHoldem tracker/rebuild;
2. frozen native LT2 neural bundle;
3. exact lean legal mask;
4. domain routing:
   - 3H -> AveragePolicy;
   - HU -> ENS8 raw-mean + regret matching;
5. deterministic auditable sampling;
6. canonical exact action resolution;
7. one-decision-per-generation cache.

This is still **SHADOW_NO_TABLE_ACTION**.

The runner also verifies the native deployment binary SHA256 before execution.

Expected native bundle SHA256:

`2b79ab7ff746a9c1c3dd73dbc0a1d6884a471813cf34b9cb126790c4c4cbb123`

## Immediate action

```bash
bash tools/run_lt2_native_openholdem_shadow_engine_audit.sh
```

Wait for `LT2_NATIVE_OPENHOLDEM_SHADOW_ENGINE_PASS`, then send
`SpinCore_LT2_native_openholdem_shadow_engine.json`.
