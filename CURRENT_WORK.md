# SpinCore Current Work

Date: 2026-09-21
Status: **NATIVE SHADOW ENGINE PASS — WINDOWS OPENHOLDEM USER-DLL MOCK-HOST GATE NEXT**

## Frozen strategy/runtime

All strategic/model identities remain frozen.

No training, EV tuning or holdout reuse is permitted.

## Native shadow engine result

PASS:
- 1,600 Hero decisions;
- 1,600 native inference calls;
- 1,600 repeated MyTurn cache hits;
- 1,600 exact selected-action matches;
- 2,886 correct cache invalidations;
- 789 3H decisions;
- 811 HU decisions;
- all four streets;
- 992 postflop decisions;
- 0 failures.

## Windows binding now implemented

The real OpenHoldem callback/export surface is implemented in:

`dll/openholdem/user_spincore_lt2_shadow.cpp`

Exported callback ABI:
- `ProcessQuery`;
- `DLLUpdateOnNewFormula`;
- `DLLUpdateOnConnection`;
- `DLLUpdateOnHandreset`;
- `DLLUpdateOnNewRound`;
- `DLLUpdateOnMyTurn`;
- `DLLUpdateOnHeartbeat`.

The DLL resolves OpenHoldem's exported `GetSymbol` and `GetHandnumber`
functions from the host process at runtime.

## Safety barrier

This DLL is **shadow-only**.

The following action-control queries are hard-coded to zero:

- `dll$fold`;
- `dll$check`;
- `dll$call`;
- `dll$rais`;
- `dll$alli`;
- `dll$betsize`;
- `dll$deep_action`.

The DLL may compute/log what SpinCore would do, but it cannot authorize a table
action through those interfaces.

## Additional correctness work

OpenHoldem rank/suit card symbols now have an explicit conversion to SpinCore's
rank-major card id format.

This is exhaustively unit-tested across all 52 cards.

The DLL itself SHA256-verifies the frozen native bundle before loading it.

## Active gate

A Windows mock OpenHoldem host exports `GetSymbol`, `GetHandnumber` and
`WriteLog`, loads the actual DLL with `LoadLibrary`, invokes the real
callbacks, and checks:

- bundle load;
- hand-anchor creation;
- HU MyTurn inference;
- probability/legal-mask consistency;
- repeated-MyTurn cache stability;
- duplicate-heartbeat stability;
- hard-zero action query barrier.

No OpenHoldem client and no real poker table are involved.

## Immediate action

```bash
bash tools/run_lt2_openholdem_shadow_dll_windows_gate.sh
```

Wait for `LT2_OPENHOLDEM_SHADOW_DLL_WINDOWS_GATE_PASS`.

Then send:
- terminal output;
- `SpinCore_LT2_openholdem_shadow_dll_mock_gate.json`.

Do not load the DLL into OpenHoldem yet.
