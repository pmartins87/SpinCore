# SpinCore — LT2 native OpenHoldem shadow decision engine result

Date: 2026-09-21  
Status: **PASS — NATIVE TRACKER + FROZEN INFERENCE COMPOSITION VALIDATED**

## Scope

The native shadow decision engine combined:

- native OpenHoldem observable tracker;
- from-scratch canonical state rebuild;
- frozen native LT2 deployment bundle;
- 3H AveragePolicy routing;
- HU ENS8 raw-mean + regret matching;
- exact lean legal mask;
- deterministic auditable sampling;
- canonical lean-slot -> ExactAction resolution;
- per-generation decision cache.

Mode remained:

`SHADOW_NO_TABLE_ACTION`

## Coverage

- transitions: `3,954`;
- Hero decisions: `1,600`;
- native inference calls: `1,600`;
- repeated MyTurn cache hits: `1,600`;
- cache invalidations: `2,886`;
- selected exact-action matches: `1,600`.

Domains:
- THREE_HANDED: `789`;
- TRUE_HEADS_UP: `811`.

Streets:
- preflop: `608`;
- flop: `435`;
- turn: `320`;
- river: `237`.

Postflop decisions: `992`.

Failures: **0**.

## Frozen identities

Native deployment file SHA256 expected:

`2b79ab7ff746a9c1c3dd73dbc0a1d6884a471813cf34b9cb126790c4c4cbb123`

Embedded source identities:
- Python bundle: `87e46b40cb43bb89cb46bf3b760bbac5c8282491fd3d6da73d1bbe28329b278c`;
- checkpoint: `a51dbbed71090e45f2c4f5db6297f72eab848990b38650702b436e2aa60ca4bf`;
- ensemble sidecar: `c44b817f75304db352eedc33febbd180e6d038e0580c91be0b9befdbdb08f181`.

## Decision

The native decision core is accepted for Windows OpenHoldem binding.

The next gate compiles a real Windows user-DLL export surface and loads it into
a mock OpenHoldem host that exports `GetSymbol` and `GetHandnumber`.

The mock gate must prove:
- Windows DLL loads;
- OpenHoldem host exports resolve;
- frozen bundle SHA verifies inside the DLL;
- raw OH symbols create a canonical hand anchor;
- MyTurn computes a shadow decision;
- repeated MyTurn remains cache-stable;
- debug probability/legal symbols are coherent;
- every action-control query is hard-zero;
- table actions executed = 0.
