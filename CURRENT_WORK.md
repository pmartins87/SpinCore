# SpinCore Current Work

Date: 2026-09-21
Status: **WINDOWS SHADOW DLL MOCK-HOST PASS — REAL OPENHOLDEM HOST ARCHITECTURE INSPECTION NEXT**

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

## Windows mock-host gate result

PASS:
- MSVC x64 build completed;
- 2/2 Windows core tests passed;
- DLL loaded through Windows LoadLibrary;
- mock OpenHoldem host exports resolved;
- frozen bundle identity verified;
- strict HU hand anchor started;
- MyTurn shadow inference produced a valid decision;
- probability sum = 1 over 4 legal actions;
- repeated MyTurn cache remained stable;
- action-control query barrier remained hard-zero;
- failure latch remained clear.

Tested DLL SHA256:

`7566be1b3c73207d437171c2b4e94f6a94477786a2a48599994a647808030062`

## Active gate

Do **not** copy the DLL into OpenHoldem yet.

The tested DLL is x64. The actual OpenHoldem executable must first be inspected
for PE architecture. If it is x86, we must build and validate an x86 DLL before
real loading.

## Immediate action

```bash
bash tools/run_lt2_openholdem_host_inspection.sh
```

Expected sentinel when the real host and tested DLL are architecture-compatible:

`LT2_OPENHOLDEM_HOST_INSPECTION_PASS`

If multiple OpenHoldem executables are found, the script will list them and stop.
No files are installed or overwritten by this inspection.


## Parallel research lane — LT3 Heavy H1

LT3 has been opened as a research-only lane while LT2 deployment work continues.

H1 is preregistered as a 500-iteration continuation from the exact frozen
LT2 ENS8@8100 checkpoint+sidecar pair:

- target 8600;
- +300,000 roots;
- 3H fresh100 unchanged;
- HU ENS8 = 8 x fresh400 unchanged;
- K4 off;
- 31 workers / Torch threads 8;
- LT2 artifacts read-only;
- LT2 final holdout retired;
- LT3 sealed holdout not touched.

Run:

```bash
bash tools/run_lt3_heavy_ens8_h1.sh
```

Expected completion sentinel:

`LT3_HEAVY_ENS8_H1_TRAINING_PASS`

The Windows OpenHoldem shadow-DLL mock-host gate remains the deployment-lane
next step and may be run separately.
