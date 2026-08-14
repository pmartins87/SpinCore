# R7.5 — Full semantic audit of `Crusher Framework 5.txt`

Date: 2026-08-13
Status: **SOURCE AUDIT COMPLETE; DESIGN CONSEQUENCES PRECOMMITTED; NO STRATEGIC PASS**

`READY FOR TABLES = NO`.

## 1. Source identity and scope

Recovered source:

```text
Tentativas anteriores de SpinGo/hardcoded/Crusher Framework 5.txt
sha256 = 7ec68e2efc9790bfc02f47da690faa1856ab2be47a8d10bd178a2963fa78ce08
bytes  = 363654
lines  = 10831
```

The file is not treated as a solved strategy or as training labels. It is treated as a historical **poker-state vocabulary, implementation catalogue, heuristic catalogue and exploit catalogue**. Any source-derived hard-coded action remains historical evidence only.

Physical named-block inventory:

```text
#00 NOTES                         1
#01 CONFIG                        6
#02 CORE                         15
#03 POSTFLOP DEFEND              32
#04 POSTFLOP ATTACK              14
#05 PREFLOP ACTION               15
#06 FLOP ACTION                  10
#07 TURN ACTION                   8
#08 RIVER ACTION                  8
#09 SCENARIOS                   297
#10 HANDLISTS                   305
#10 HEADS-UP CHARTS             220
#11 EXPLOIT                     256
#12 BETSIZES                      8
-----------------------------------
TOTAL                           1195
```

The often-cited `32 DEFEND / 13 ATTACK` description captures only one subset. The physical file contains 32 named defend move formulas and 14 named attack/lead formulas; the fourteenth is `f$river_lead_noaction`.

## 2. Classification rule used by SpinCore

Every reusable concept is placed into one of four classes before implementation.

### A — Objective state semantics

Deterministically reconstructible from cards, chips, seats and exact public action history. These are eligible for `NeuralInputV2` because they tell the network **where it is**, not what to do.

### B — Derived diagnostic semantics

Deterministic but lossy human labels such as broad stack bands or dry/wet labels. These may be retained as diagnostics or ablation features but must not replace exact facts.

### C — Historical heuristics

Hand-written thresholds, outs approximations, implied-odds approximations, commitment rules and sizing bands. These are candidate hypotheses only and are not production truth without independent evidence.

### D — Historical strategic decisions

Handlists, chart actions, exploit substitutions and hard-coded bet/call/raise choices. These must **not** be copied into the base learned policy as targets.

## 3. Semantic catalogue and destination

### 3.1 Preflop pot lineage — Class A

Source concepts include:

```text
f$FirstAction
f$UnplayedPot
f$pot_Limped
f$pot_Got_Isolated
f$pot_SingleRaised
f$pot_Reraised
limp-raised contexts
last raiser / last caller by position
normal/over faced raise and isolation variants
```

SpinCore consequence:

The production candidate must reconstruct an objective preflop lineage from exact voluntary actions, including at minimum:

```text
UNOPENED
LIMPED
OPEN_RAISED
RAISE_OVER_LIMP
RERAISED
LIMP_RERAISED
```

and preserve counts, actor/position, last aggressor and exact realized sizes. Legacy `normal/over` thresholds are not frozen as truth.

### 3.2 Initiative and postflop line identity — Class A

Source concepts include:

```text
hero initiative
villain initiative
nobody initiative
got raised / isolated
hero/villain limped initiative
c-bet
donk
float
probe
delayed c-bet
delayed float
no-action street continuations
raise / re-raise contexts
```

The existing R7.5 postflop ontology already implements a compositional subset:

```text
CBET
DONK_BET
PROBE_BET
FLOAT_BET
DELAYED_FLOAT_BET
DELAYED_CBET
DOUBLE_DELAYED_CBET
GENERIC_BET
RAISE
```

with opening-line preservation and raise depth. This is an **observation ontology**, never a prescribed action.

### 3.3 Faced sizing and raise geometry — A + C

Historical scenario thresholds distinguish very-low, low, 60%, normal, high, overbet, 1–150%, 150%+ and normal/over re-raises.

The useful invariant is not the historical boundary itself. `NeuralInputV2` should preserve continuous facts sufficient to derive any later bin:

```text
pot before action
amount to call
bet size / pot
raise increment
raise-to / pot
stack behind
raise depth
```

Legacy percentage thresholds remain Class C until an ablation justifies them.

### 3.4 Position and lineup — Class A

The source repeatedly distinguishes HU SB/BB and 3-handed BTN/SB/BB matchups, plus first/last/middle postflop position.

SpinCore must preserve exact seat-relative facts:

```text
hero position
active opponent positions
IP/OOP relation versus each live opponent
number of live players
dealer relation
```

Human matchup labels may be redundantly derived but cannot be the sole source of truth.

### 3.5 Stack, effective stack, SPR and stack-behind — A + B

The framework contains exact/back-up effective-stack logic and many overlapping stack bands.

SpinCore consequence:

- preserve exact stacks in BB;
- preserve effective stack against each live opponent;
- preserve min/max live-opponent stack where useful;
- preserve exact pot and SPR;
- stack bands are optional Class B features only.

No SHORT/MID/DEEP label may replace exact geometry.

### 3.6 Private hand semantics — A + B/C

The framework contains a rich vocabulary for:

```text
made-hand tiers
overpair / top pair / lower pair relations
kicker quality and better-kicker counts
straight / flush contribution by hole cards
nutness relations
gutshot
OESD
flush draw
overcards
backdoors
combination draws
```

These are valuable because the network should not need to rediscover elementary poker structure from sparse samples.

SpinCore will implement the **objective primitives**. Aggregate historical labels such as `good / medium / weak draw`, hand-written outs estimates and stack-off rules remain B/C and require evidence before use.

### 3.7 Board and street-transition semantics — A + B

The source distinguishes:

```text
rainbow / flush-rich structures
monotone
paired boards
broadway counts
low/mid/high composition
connectedness
flop completed / super-completed
turn/river straight or flush completion
missed river flush draw
turn/river overcard or undercard
turn/river board pairing
street-to-street texture transitions
```

SpinCore will prefer objective primitives:

```text
suit multiplicities
rank multiplicities
broadway count
rank span / gaps
straight-window overlap counts
paired/trips structure
turn/river rank relation to prior board
straight/flush completion flags
prior-draw brick/completion flags
```

Opaque labels such as `dry/wet` may be diagnostics, not the sole representation.

### 3.8 Pot odds, price and commitment — A + C

The source contains pot-odds, implied-odds, outs and committed-opponent logic.

Eligible production observations:

```text
exact price to continue
pot odds
stack behind
SPR
```

Legacy implied-odds/outs/commitment formulas remain Class C. They are not imported as strategic truth.

### 3.9 Runtime integrity and backup logic — R10/R12 destination

The framework includes checks and backup logic around:

```text
chips in game
wrong pot
blinds / stack consistency
effective-stack recovery
OpenHoldem state/history fallbacks
tournament/table-state sanity
```

These concepts are not neural strategy features by default. They are recorded as requirements candidates for **R10 OpenHoldem runtime integration** and **R12 operational homologation**, where state corruption must fail safe rather than silently change strategy.

### 3.10 Exploitation catalogue — R11 destination

`#11 EXPLOIT` contains 256 named blocks. Source concepts include opponent statistics and exploit triggers around:

```text
fold to minraise
3bet total
3bet non-all-in / all-in
isolation total
isolation non-all-in / all-in
fold to c-bet
flop c-bet
flop check-raise
turn c-bet
VPIP
limp-raise
stack-band-specific reactions
```

The exploit action substitutions themselves are Class D and are not production base-policy targets.

The **opponent-feature vocabulary** is valuable for R11. SpinCore should prefer exact observed rates plus sample counts / uncertainty over rigid historical `low/std/high` labels whenever possible.

### 3.11 Betsize catalogue — R7.5.4 destination

The source exposes Min, 33, 40, 50, 66, 75, 100 and Max concepts, with street/HU/multiway differences.

This is evidence that a `33/75/AI` action tree is not automatically sufficient, but it is not evidence that every old size belongs in production.

R7.5.4 remains responsible for selecting a small action abstraction empirically. Existing precommitted candidates remain valid; `40`, `66` and `Min` may be diagnostic/additional candidates only if frozen before outputs and if their strategic gain can justify branching cost.

## 4. What the 525 historical handlist/chart blocks are useful for

They are **not** a source of target policy.

They can still provide value in three controlled roles:

1. **semantic coverage audit** — reveal contexts the new state representation must be capable of distinguishing;
2. **strategic sentinels** — identify representative state families where a future learned strategy deserves inspection, without requiring it to match the old action;
3. **future exploit research** — reveal opponent dimensions and response classes worth testing after the base strategy is frozen.

No R7.5 base-policy PASS may be obtained by measuring agreement with these charts.

## 5. Design consequences frozen before V2 learning outputs

The full-source audit strengthens the R7.5 architecture in the following way:

```text
exact authoritative state
    + exact voluntary public-history events
    + objective preflop lineage
    + objective initiative / attack-line ontology
    + exact sizing / price / raise geometry
    + exact stack / effective-stack / SPR geometry
    + objective hand/draw/nutness primitives
    + objective board/transition primitives
    + audited flop structural candidate
    -> NeuralInputV2 candidate
```

The following are explicitly **not** automatically imported:

```text
historical handlists
historical call/raise/bet decisions
historical exploit substitutions
historical implied-odds formula
historical outs approximations
historical commitment thresholds
historical low/normal/high sizing cutoffs
historical dry/wet labels as sole truth
```

This preserves the value of the framework without turning a hard-coded bot into hidden supervision.

## 6. Interaction with flop abstraction

The framework's detailed board vocabulary is additional evidence that centroid quality should be judged by strategic structural homogeneity, not by preserving old centroid names.

Therefore H3 is authorized to change representatives, recategorize physical flops and discard legacy centroid identities. It must still be deterministic, suit-invariant, fully cover all 1,755 exact flop classes / 22,100 physical flops, and be frozen before comparative learning outputs.

H1/H2 remain useful historical descendants; they are not protected from losing to H3 or to the exact 1,755-class reference.

## 7. Current gate meaning

This audit does **not** emit `R7_5_PASS`.

It closes the question of whether the Crusher source should be mined beyond the 32/14 postflop move formulas: **yes, but by semantic class and downstream destination, not by blind copying**.

Next authorized engineering:

1. make forced blind posts explicit in action history rather than inferred by event index;
2. preserve rich exact public-history events alongside the unchanged V1 token stream;
3. implement objective preflop lineage and board/private-card semantic primitives;
4. expose them through a separate `NeuralInputV2` candidate API;
5. generate deterministic H3 on the 1,755 exact suit-isomorphic flop classes;
6. freeze R7.5.3 comparative metrics before observing learning-quality outputs.
