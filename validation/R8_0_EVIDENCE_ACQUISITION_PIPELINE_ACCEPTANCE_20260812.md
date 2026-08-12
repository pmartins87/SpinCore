# R8.0 selected-state evidence acquisition pipeline — acceptance

Date: 2026-08-12

```text
R8.0 EVIDENCE ACQUISITION / VALIDATION PIPELINE = PASS
R8.0 PRODUCTION PROFILE GATE                  = NOT PASS / DATA BLOCKED
R8 OFFICIAL TRAINING                           = BLOCKED
READY FOR TABLES                               = NO
```

This record accepts the machinery used to ingest exact first-party GGPoker selected-state rules. It does **not** claim that the missing buy-in × multiplier production data have been obtained.

## Accepted components

### `SPINCORE_R8_SELECTED_STATE_EVIDENCE_PACKET_V1`

Implemented in `python/spincore/production_evidence_packet.py`.

One packet represents exactly one selected 3-Max state and binds:

- currency;
- buy-in in integer minor units;
- Spin multiplier;
- starting chips per player;
- complete captured blind levels;
- raw 1st/2nd/3rd payout amounts in integer minor units;
- normalized payout shares derived from those captured amounts;
- official client/rule-document source identity;
- capture UTC time;
- SHA-256 and byte size of the captured source.

The packet does not infer prize-pool value from `buy-in × multiplier`. The captured payout amounts are authoritative for payout shape.

Conflicting packets for the same `(table_size, buy_in, multiplier)` are rejected rather than silently choosing one.

### Production profile builder

Implemented in `python/spincore/production_profile_builder.py`.

The builder combines:

```text
SELECTED_PROFILE_STATE evidence
+ separately scoped GLOBAL_GAME tournament-fee evidence
+ frozen strategy identities
-> SPINCORE_R8_PRODUCTION_PROFILE_V3
```

State-dependent evidence cannot substitute for global fee evidence, and global evidence cannot substitute for state-dependent stack/blind/payout evidence.

This preserves the V3 fail-closed rule that a generic dynamic GGPoker page is provenance but is not proof that crawler-rendered values belong to a particular buy-in/multiplier selection.

## Regression evidence

Authoritative regression after the payout-normalization fix:

```text
workflow run: 31651412158
head: 22d58d816e8d2b0257498a1de43128a6791803d6
C++ regression: PASS
Python syntax: PASS
Python regression: PASS
```

An earlier run exposed a floating representation bug for legitimate `80% / 20% / 0%` payouts: computing the last share as `1-a-b` could produce a tiny negative float. The implementation was corrected to derive **all three shares directly from the captured integer payout amounts**, while validating total probability with an absolute tolerance. No strategic/economic assumption changed.

## Remaining R8.0 blocker

The current official public GGPoker Spin & Gold page proves that starting chips, blind structure and time bank vary with the Spin multiplier, but the crawler-visible representation does not reliably bind the dynamic table values to the selected 3-Max buy-in state.

Therefore R8.0 remains fail-closed until we have first-party selected-state evidence sufficient to populate, for every production state we intend to support:

- exact starting chips;
- exact complete blind structure;
- exact 1st/2nd/3rd payout amounts or an official payout vector;
- exact buy-in and multiplier binding.

No R7.4 pilot constants, approximate strategy-article values, community tables, or unbound dynamic-page rows may fill this gap.

## Consequence

Once exact selected-state captures are available, no additional architecture is required to turn them into immutable production-profile IDs: the capture packet, evidence binding and profile builder are now regression-tested.

This acceptance does not authorize R8.2 calibration, R8.3/R8.4 training, R8.5 policy freeze, OpenHoldem use, or table use.
