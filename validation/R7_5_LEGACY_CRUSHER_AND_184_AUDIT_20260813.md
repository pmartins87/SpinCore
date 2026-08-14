# R7.5 legacy Crusher ontology + Solver-V2 184-flop audit

Date: 2026-08-13
Status: **PHYSICAL LEGACY EVIDENCE AUDITED; PRODUCTION SELECTION NOT YET AUTHORIZED**

This audit is a child artifact of `R7_5_REPRESENTATION_AND_ACTION_ABSTRACTION_PRECOMMIT_20260813.md`.
It records newly recovered artifacts from the user-supplied `Tentativas anteriores de SpinGo.zip` and converts them into requirements without copying legacy strategy decisions.

`READY FOR TABLES = NO`.

## 1. Newly recovered source evidence

```text
solver v2/184Flops.json
sha256 7fb7aebee3b24b5bf6904915194f21835cb06773d7474b3ff8649153337afff8

hardcoded/Crusher Framework 5.txt
sha256 7ec68e2efc9790bfc02f47da690faa1856ab2be47a8d10bd178a2963fa78ce08
```

The legacy files are evidence/reference inputs only. Their strategic hand lists, thresholds and actions are not accepted as SpinCore strategy merely because they existed historically.

## 2. Physical audit of `184Flops.json`

The recovered JSON is the missing Solver-V2 physical-flop-to-representative map.

Direct audit:

```text
JSON rows                              22,100
unordered physical flops mapped       22,100
missing physical flops                     0
extra physical flops                       0
unique representative texts              184
unique representatives modulo suits       184
invalid rows                                0
duplicate/conflicting physical rows         0
```

Therefore the historical mapping has complete physical coverage.

### 2.1 Bucket-size distribution

```text
bucket count           184
min                       8
median                   96
mean             120.108696
max                     600
```

Largest observed bucket:

```text
representative 4s7sTd
physical flops 600
exact suit-isomorphic input classes 50
```

The complete mapping is substantially richer than the 53-class control, but bucket size alone does not establish strategic homogeneity.

### 2.2 Exact suit-isomorphism reference

SpinCore's lossless flop reference removes only absolute suit naming and flop-card order. It produces:

```text
physical flops                         22,100
exact classes modulo suit permutation   1,755
```

The historical 184 map was compared against all 1,755 exact classes.

Result:

```text
exact classes examined                      1,755
exact classes split across >1 legacy bucket    40
maximum legacy buckets for one exact class      2
historical suit-permutation invariance PASS    false
```

This is a material structural defect for direct production reuse. Absolute names of suits are strategically irrelevant; a pure global renaming of suits must not change a flop abstraction.

Concrete example from the recovered mapping:

```text
exact suit-isomorphic family: 2s5s7h

some physical suit spellings -> legacy representative 2s3d7s
some physical suit spellings -> legacy representative 2s4s7d
```

Those inputs differ only by suit permutation, yet the historical map can route them to distinct strategic buckets.

No pair of the 184 representative texts themselves collapses to the same exact suit-isomorphic representative, so this is not a duplicate-centroid problem. It is an inconsistent assignment of input suit variants.

### 2.3 Exact classes carried by each legacy bucket

Counting the 1,755 lossless suit-isomorphic flop classes represented inside each legacy bucket:

```text
min       1
median    7.5
mean      9.755435
max      50
```

A production audit still has to measure strategically meaningful within-bucket collisions: pairedness, suit texture, straight/flush transition structure, nut changes and policy/value dispersion. Physical coverage is necessary but not sufficient.

### 2.4 Eligibility decision

The recovered historical map is now classified as:

```text
LEGACY_184_PHYSICAL_COVERAGE              PASS
LEGACY_184_REPRESENTATIVE_COUNT           PASS
LEGACY_184_SUIT_PERMUTATION_INVARIANCE    FAIL
LEGACY_184_ELIGIBLE_UNCHANGED             NO
LEGACY_184_REFERENCE_VALUE                HIGH
```

SpinCore will not silently repair the 40 split exact classes by arbitrary tie-breaking. A corrected suit-invariant descendant may reuse the historical clustering only after a deterministic repair/selection rule is precommitted and audited.

## 3. `Crusher Framework 5` ontology evidence

The legacy framework explicitly separates postflop **DEFEND** and **ATTACK** contexts. The user's recollection is correct: the source enumerates 32 DEFEND situations and 13 ATTACK situations.

### 3.1 Exact 32 DEFEND headings recovered from source

```text
01 RAISE VS FLOP NORMAL CBET
02 CALL  VS FLOP NORMAL CBET
03 RAISE VS TURN NORMAL CBET
04 CALL  VS TURN NORMAL CBET
05 RAISE VS RIVER NORMAL CBET
06 CALL  VS RIVER NORMAL CBET
07 RAISE VS HIGH CBET
08 CALL  VS HIGH CBET
09 RAISE VS OVER CBET
10 CALL  VS OVER CBET
11 RAISE VS NORMAL DONK
12 RAISE VS HIGH DONK
13 RAISE VS OVER DONK
14 CALL  VS NORMAL DONK
15 CALL  VS HIGH DONK
16 CALL  VS OVER DONK
17 RAISE VS NORMAL FLOAT
18 RAISE VS HIGH FLOAT
19 RAISE VS OVER FLOAT
20 CALL  VS NORMAL FLOAT
21 CALL  VS HIGH FLOAT
22 CALL  VS OVER FLOAT
23 RAISE VS NORMAL BET
24 RAISE VS HIGH BET
25 RAISE VS OVER BET
26 CALL  VS NORMAL BET
27 CALL  VS HIGH BET
28 CALL  VS OVER BET
29 RAISE VS NORMAL RAISE
30 RAISE VS OVER RAISE
31 CALL  VS NORMAL RAISE
32 CALL  VS OVER RAISE
```

The source comment block has copy/paste label mistakes for entries 24/25/27/28 (it says FLOAT in places), while the actual function names are `...High_Bet` / `...Over_Bet`. SpinCore must use the executable semantic structure, not reproduce comment typos.

### 3.2 Exact 13 ATTACK headings recovered from source

```text
01 FLOP CBET
02 TURN CBET
03 RIVER CBET
04 FLOP FLOAT BET
05 TURN FLOAT BET
06 RIVER FLOAT BET
07 FLOP DONK BET
08 TURN DONK BET
09 RIVER DONK BET
10 TURN PROBE BET
11 RIVER PROBE BET
12 TURN DELAYED BETS
   - DELAYED CB
   - DELAYED FLOAT BET
13 RIVER DELAYED BET
```

The framework then routes flop/turn/river decisions according to initiative, relative position, prior-street action and whether the current amount is a first bet or a re-raise. This is valuable ontology evidence even where individual strategic rules are obsolete.

## 4. Critical decomposition: state is not action

The 32 DEFEND labels must **not** become 32 one-hot observation states exactly as written.

For example:

```text
CALL VS HIGH DONK
RAISE VS HIGH DONK
```

are the same environment state. `CALL` and `RAISE` are candidate responses and belong in the policy output/action set, not in the input observation.

SpinCore V2 therefore decomposes the legacy DEFEND catalogue into orthogonal facts:

```text
street
facing opening-line semantic:
  CBET / DONK / FLOAT / PROBE / GENERIC_BET / other derived line
raise depth / whether the faced action is a raise or re-raise
actor identities / relative position
last aggressor and initiative lineage
exact faced amount / exact pot geometry
normalized sizing
legal action mask
```

The network then chooses Fold / Call / Raise-size / All-in from the legal action set.

This preserves the strategic distinction in the Crusher catalogue while removing duplicated state-action labels and allowing generalization across streets and sizes.

## 5. ATTACK ontology becomes an opportunity/context feature

The 13 ATTACK headings are treated as descriptions of **why a bet opportunity exists**, not instructions that a bet must be made.

Required derived attack-opportunity semantics now include at minimum:

```text
CBET
DONK_BET
FLOAT_BET
PROBE_BET
DELAYED_CBET
DELAYED_FLOAT_BET
DOUBLE_DELAYED_CBET
GENERIC_STAB / UNCLASSIFIED_BET_OPPORTUNITY
```

`DOUBLE_DELAYED_CBET` is retained even though the Crusher heading list does not name it explicitly: it is deterministically reconstructible from exact history and was already required by the R7.5 precommit.

The source's use of `FLOAT` is treated as legacy vocabulary whose exact transition semantics must be preserved during comparison; SpinCore will not assume every historical label exactly matches modern textbook terminology.

## 6. Initiative lineage is a first-class state variable

The reusable idea in the Crusher routing is not a specific hand list. It is the chain:

```text
who was the last aggressor?
on which street?
did that aggressor decline the next betting opportunity?
did another player call the prior aggression?
who acts first/last relative to that player now?
is the current action the opening bet, raise, or re-raise?
```

A compact production representation can reconstruct c-bet/donk/probe/float/delayed semantics from those facts with far fewer dimensions than raw action-case one-hots.

This directly addresses a weakness of current SpinCore V1: its public-history token stream preserves action type and street but drops actor and exact sizing in the neural history representation.

## 7. Legacy sizing thresholds: reference only

Crusher contains legacy faced-size predicates based mainly on `AmountToCall` relative to `potcommon`, including boundaries around 35%, 52%, 60%, 76%, 100% and 150%, plus separate re-raise tests.

These thresholds are **not frozen as SpinCore production thresholds**. The production observation must retain exact pot/to-call/bet geometry. Coarse bins may be added as redundant auxiliary features only if their thresholds are precommitted and pass R7.5 ablation.

## 8. Relationship to the Solver-V2 context model

Solver V2 independently reinforces several low-cost semantic variables:

```text
preflop pot lineage: LIMP / SRP / ISO / 3BET
relative position / matchup
flop-start effective stack bucket SHORT / MID / DEEP
flop structural representative
```

SpinCore keeps those concepts but does not inherit Solver-V2's information-losing synthetic pot sizes or its heuristic remapping of future physical cards.

The resulting target is a hybrid:

```text
exact authoritative game state
+ compact structural board representation
+ exact continuous stack/pot/sizing geometry
+ explicit preflop lineage
+ explicit aggressor/initiative/action-line ontology
+ engineered objective hand/draw/board semantics
```

## 9. R7.5 consequences

Effective immediately:

```text
R7.5.0 design precommit                         PASS (design only)
R7.5.1 historical 184 map recovered             YES
R7.5.1 22,100 physical coverage                 PASS
R7.5.1 184 unique representatives               PASS
R7.5.1 historical suit invariance               FAIL
R7.5.1 historical map eligible unchanged        NO
R7.5.1 corrected/new candidate selection        PENDING
R7.5.2 Crusher-derived ontology requirements     FROZEN AS REQUIREMENTS
R7.5.2 production encoder implementation         PENDING
R7.5.3 representation ablation                   PENDING
R7.5.4 action-set ablation                       PENDING
R7.5.5 production freeze                         PENDING
R8 heavy production training                     BLOCKED BY R7.5
READY FOR TABLES                                 NO
```

No strategic gate, seed, tolerance or previously accepted R7.4 evidence is relaxed by this audit.
