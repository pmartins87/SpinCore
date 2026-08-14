# R7.5 — Strategic representation and action-abstraction precommit

Date: 2026-08-13
Status: **IN PROGRESS — DESIGN PRECOMMITTED, NO STRATEGIC PASS YET**

This gate is inserted after R7.4 and before official production training. Its purpose is to prevent SpinCore from repeating representation/action-space failures observed in earlier Spin & Go bot attempts while preserving exact game rules and keeping training feasible on the target Ryzen 9.

`READY FOR TABLES = NO` remains unchanged.

## 1. Why R7.5 exists

R7.4 proves statistical stability for the current recovered Deep CFR stack under its frozen pilot contract. It does **not** prove that the current neural representation or action abstraction is strategically sufficient for production Spin & Go play.

The legacy attempts supplied for review show two opposite failure modes that R7.5 must avoid:

1. overly raw/high-dimensional card representation that asks a small network to rediscover elementary poker structure from sparse samples;
2. overly lossy human abstraction that collapses strategically different states and silently destroys information.

The production representation therefore must be selected empirically between those extremes rather than inherited from either the current recovery or any legacy bot.

## 2. Legacy evidence audited

User-supplied legacy package: `Tentativas anteriores SpinGo`.

Audited source hashes:

```text
deepSpin/env/poker_env.cpp
  sha256 0defbc35ac2174f3a9dd4d896009ceb86febb906ee4758a68ba7f71a71595a20

deepSpin/user_deepspin.cpp
  sha256 0cc92148516d5dbf280ee26c92d78ab56b70ade8b3441b4397603fb8c98ce434

hardcoded/user_hardcoded.cpp
  sha256 df74cfeef10d7f0a538efe622e32d196ee68732890718ba5f0f982fbc70d853c

hardcoded/user_hardcoded_helper.cpp
  sha256 65cd5f0a6babf4a065ad3350f14af8c071b36015e576b14b55ab0c374197113e

solver v2/user_solver_v2.cpp
  sha256 d8df5d3776a0de4a81500ffe872128f15697903c25b36aceeb8f722dd2c6aca7

solver v2/classes_184_resumo.csv
  sha256 3d3d01a30cf5748ce26348563b8d3ba16928fec1b4f33b60f48af0ad494e3ecc
```

### 2.1 DeepSpin evidence

Recovered DeepSpin v60 observation layout is 292 direct dimensions:

```text
cards              104
numeric              20
street                4
positions             11
hand strength         40
draws                 12
board texture         29
action context        26
history               39
legal mask             7
TOTAL                 292
```

Therefore the recovered version did **not** contain `52 x 3 = 156` flop-card dimensions. It used 104 card dimensions total: 52 for the hero hole-card multi-hot block and 52 for all visible board cards. This still means raw absolute card identity consumed 35.6% of the direct observation vector.

DeepSpin already contained useful engineered poker semantics. It explicitly encoded hand strength, draws, board texture, preflop pot state, postflop faced-action classes and prior-street aggression. Its 7 actions were Fold, Check/Call, 33%, 50%, 75%, 100% and All-in. On preflop the four percentage labels collapsed into a context-dependent real raise group; postflop they represented distinct sizes.

**Conclusion:** DeepSpin's lack of success cannot responsibly be attributed to raw cards alone. However, its raw-card block was large and sample-inefficient, and the legacy evidence does not establish that absolute-suit card IDs were a good production representation.

### 2.2 Current SpinCore V1 evidence

Current `NeuralInputV1` uses seven exact card tokens plus 16 numerics, eight categorical tokens and a GRU history. In the present neural model, seven 16-dimensional card embeddings contribute 112 of the 272 concatenated features entering the first MLP layer (about 41.2%).

Dimension share is **not** the same as learned importance: a neural net can down-weight or transform those channels. Nevertheless, the current representation still spends a large fraction of its learnable input capacity on exact physical card IDs and absolute suit identity, so the same sample-efficiency concern remains plausible and must be tested rather than dismissed.

The current V1 action abstraction has six slots and postflop exposes approximately 33% pot, 75% pot and all-in as distinct aggressive sizes. Current public-history tokens preserve action type and street but not full actor/sizing semantics.

**Conclusion:** V1 remains the required control/baseline for R7.5. It is not automatically the production representation.

### 2.3 Solver V2 — 184-flop abstraction

`classes_184_resumo.csv` contains exactly 184 representatives. Their `quantidade_flops` values sum to all 22,100 physical three-card Hold'em flops.

Observed summary:

```text
representatives                 184
physical flops represented    22100
unique textual labels            99
bucket size min                   8
bucket size median               96
bucket size mean             120.11
bucket size max                 600
```

The fact that 184 centroids map to only 99 textual labels is useful: the old system sometimes split states more finely than its human-readable texture label alone.

Solver V2 also explicitly represented:

```text
relative matchup / position
pot lineage: LIMP / SRP / ISO / 3BET
flop-start effective-stack bucket: SHORT / MID / DEEP
IP/OOP on the board
```

Its historical thresholds were:

```text
DEEP  >= 15 BB
MID   >= 10 BB and < 15 BB
SHORT < 10 BB
```

This corrects the ambiguous verbal form `DEEP > 15`: in the recovered code, exactly 15 BB was DEEP.

The useful part is the **184-flop structural compression**. The unsafe part is to copy the entire old canonicalization pipeline. Solver V2 remapped hero hands and future turn/river cards to canonical analogues using a heuristic three-axis hand-strength signature and hand-written similarity scores. It also reconstructed the starting flop pot with fixed synthetic values (LIMP=2, SRP=4, ISO=6, 3BET=10 BB). Those are information-losing approximations and are not accepted as SpinCore production semantics.

The full `184Flops.json` mapping referenced by `user_solver_v2.cpp` is not present in the supplied legacy package and was not found in the accessible file library during this audit. Therefore the 184 scheme is a **primary candidate, not yet an accepted bucketization**. Full within-bucket homogeneity cannot be certified until the mapping is recovered or regenerated deterministically.

### 2.4 Hard-coded bot evidence

The hard-coded source contains a much richer action-history vocabulary than either current SpinCore V1 or a simple pot-type label. It explicitly tracks state concepts including c-bet, donk, probe, float, delayed lines, check-raise, small/big c-bets, called/raised continuations and street-to-street initiative transitions.

This source is useful as a **semantic requirements catalogue**, not as strategic truth. Its action choices, hand lists and thresholds must not be copied into the learned policy merely because they existed in the old bot.

## 3. Architecture decision frozen by this precommit

### 3.1 Exact game state stays exact

SpinCore will continue to preserve exact cards, chips, actions and rules inside the authoritative game/traversal state for:

- legal-action generation;
- chance dealing;
- showdown evaluation;
- terminal utility;
- reproducibility and replay;
- strategic audits.

Any lossy abstraction is allowed **only at the neural observation boundary**. The game engine itself will not be bucketized.

This prevents a neural compression decision from corrupting poker rules or terminal EV.

### 3.2 The 53-flop proposal is rejected as the primary production abstraction

The 53-type proposal remains useful as an interpretable diagnostic taxonomy, but it is too coarse to be the sole flop identity supplied to the network. It can merge materially different monotone/two-tone structures, paired-rank structures and connectivity/nut-transition properties.

It may be retained as auxiliary diagnostic features, but R7.5 will not select `53-only` as the production representation.

### 3.3 The 184-flop scheme is the leading candidate, not a foregone conclusion

R7.5 will benchmark a recovered/regenerated 184 mapping against the current V1 baseline and at least one more precise relational/canonical candidate. The winner is chosen by frozen evidence, not by historical preference.

No production training may assume that `184` is correct before its audit and comparative pilot pass.

### 3.4 Do not make absolute physical card identity the dominant production channel by default

The production candidate will avoid the DeepSpin-style 52-ID board multi-hot block and will not automatically preserve the current seven absolute-card embeddings as the dominant input channel.

If exact residual card information is retained, it must be represented compactly through suit-isomorphic / relational encodings or a deliberately small residual channel, and it must prove incremental value in the R7.5 ablation.

## 4. Candidate NeuralInputV2 semantic contract

The implementation may refine field packing, but it must preserve the following semantic information unless an ablation proves a field unnecessary.

### 4.1 Private hand

Preflop:

- exact 169-class identity (`AA`, `AKs`, `AKo`, ...), represented categorically/through an embedding;
- pair/suited/connectivity metadata only if useful as auxiliary features.

Postflop:

- objective made-hand class relative to the board;
- pair rank/relative pair tier where applicable;
- kicker/nutness information;
- flush/straight made-hand state;
- objective draw state: OESD, double-gutshot, gutshot, flush draw, nut-flush-draw relation, backdoors, overcards and combination draws;
- blocker/nut-transition information where it is deterministically meaningful.

The network should not need millions of samples to rediscover what a gutshot or top pair is.

### 4.2 Board

Primary candidate:

- audited flop structural bucket (184 candidate);
- independent suit texture (rainbow / two-tone / monotone);
- pairedness/trips and rank of paired structure;
- connectivity/dynamicity features;
- broadway/high-card structure;
- turn/river delta features: paired board, overcard/undercard relations, flush completion, straight completion, draw bricking/completion and other deterministic texture transitions.

A small exact/suit-isomorphic residual may be benchmarked. It is not mandatory.

### 4.3 Tournament / stack / pot geometry

Preserve **both** continuous facts and coarse buckets:

- each live stack in BB;
- effective stack(s);
- flop-start effective stack;
- stack bucket SHORT/MID/DEEP as auxiliary feature;
- current pot in BB;
- current amount to call;
- current bet/raise size;
- SPR;
- street commitments and total commitments where they add information.

Buckets must never replace exact pot/stack values when the exact value is already available cheaply.

### 4.4 Position and lineup

Explicitly preserve facts rather than relying only on ad-hoc labels:

- domain: HU-origin / 3-handed-origin;
- hero position BTN/SB/BB;
- active opponents and their positions;
- relative IP/OOP against each live opponent;
- number of live players;
- all-in/folded status.

Derived matchup labels such as `BTNvBB` may be supplied redundantly if useful, but they are not the sole source of truth.

### 4.5 Preflop lineage

Explicit pot lineage is required:

```text
UNOPENED
LIMPED
SINGLE_RAISED
ISOLATED
3BET_PLUS
LIMP_RAISED when genuinely distinguishable
```

Also preserve exact relevant sizes and counts:

- number of limpers;
- number of raises;
- callers after raise;
- last preflop aggressor;
- opening/iso/3bet sizing as BB and/or normalized ratio.

Thus `SRP` is a semantic feature, not a replacement for exact pot geometry.

### 4.6 Action history and initiative

Every strategically relevant history event must retain enough information to reconstruct:

```text
street
actor / relative position
action type
exact or normalized sizing
pot before / amount to call where required
aggressor / initiative transition
```

Deterministic derived semantic tags must include at least the concepts necessary to distinguish:

```text
c-bet
donk bet
probe bet
float bet
delayed c-bet
double-delayed c-bet
check-raise
bet-raise / raise-reraise contexts
```

These labels describe **what happened**. They do not hard-code what the bot should do.

A 30% c-bet and a 50% c-bet must be distinguishable. A 50% c-bet and a 50% donk must also be distinguishable.

## 5. Action abstraction candidates

Input semantics are cheap compared with action-tree branching. R7.5 therefore prioritizes rich observation semantics while keeping the aggressive action set small enough for Ryzen training.

The current action set remains control A:

```text
postflop: 33%, 75%, all-in
plus fold/check-call as legal
```

At minimum, R7.5 will benchmark:

```text
A: 33 / 75 / AI               current control
B: 33 / 50 / 75 / AI
C: 33 / 50 / 75 / 100 / AI   legacy-DeepSpin-sized candidate
```

Preflop will not naïvely expose all postflop percentage labels. It will use a small context-dependent raise abstraction plus all-in, while preserving the actual realized size in history.

No larger action set is authorized merely because it is more precise. Every extra branch must justify its strategic gain against nodes/root and seconds/root.

## 6. R7.5 finite gate sequence

### R7.5.0 — legacy evidence and architecture precommit

This document.

PASS condition: document persisted before comparative training results are observed.

Status: **PASS AS DESIGN PRECOMMIT ONLY**. This is not an R7.5 strategic PASS.

### R7.5.1 — recover/regenerate and audit flop mappings

Required artifacts:

- the historical `184Flops.json`, if recoverable, or a deterministic regenerated equivalent;
- mapping hash and generation provenance;
- complete coverage audit for all 22,100 physical flops;
- suit-permutation invariance audit;
- bucket-size distribution;
- within-bucket deterministic texture collision report;
- explicit comparison with 53 taxonomy and a more precise suit-isomorphic reference representation.

No bucket mapping with missing/duplicate physical flops is eligible.

### R7.5.2 — NeuralInputV2 implementation + semantic regression suite

Must prove, before learning-quality claims:

- deterministic encoding;
- replay/fresh-process equivalence;
- suit-isomorphic states map as intended;
- strategically distinct lineage states remain distinguishable;
- c-bet/donk/probe/float/delayed semantics reconstruct correctly;
- 30% vs 50% faced sizing is distinguishable;
- exact game/traversal utility is unchanged by encoder selection.

### R7.5.3 — frozen representation ablation

Before candidate results are run, persist:

- exact candidate list;
- seeds;
- root/sample budgets;
- training runtime;
- held-out state set;
- acceptance metrics and tie-break rules.

Required controls include current V1 and the 184-based V2 candidate. A 53-only representation may be tested only as a negative/control candidate; it is not eligible for production selection under this precommit.

Metrics must include at least:

- held-out Advantage fit/error;
- AveragePolicy held-out discrepancy/stability;
- cross-seed stability;
- strategically tagged sentinel errors;
- rare-state coverage;
- seconds/root;
- nodes/root;
- peak RAM;
- model parameter count.

The exact numerical thresholds/tie-breaks must be frozen in R7.5.3 **before candidate outputs are inspected**. They may not be chosen after seeing the winner.

### R7.5.4 — frozen action-abstraction ablation

Using the selected representation, compare the precommitted action sets under the same semantic RNG/deal contract. Record strategic benefit against tree-size/runtime cost.

Changing the action set changes the game abstraction, so old R7.4 numerical evidence cannot be relabelled as evidence for the new action set.

### R7.5.5 — production representation/action freeze

Only this subgate may emit:

```text
R7_5_PASS = true
R7_5_PRODUCTION_REPRESENTATION_FROZEN = true
R7_5_PRODUCTION_ACTION_ABSTRACTION_FROZEN = true
```

It must persist source hashes, mapping hashes, model schema, encoder schema, action schema, benchmark evidence and selected candidate.

## 7. Interaction with R8

R7.4 has passed and authorizes continued R8 engineering. R8.0 production-profile acquisition and non-strategic infrastructure work may proceed in parallel.

However:

```text
R8.3 official HU production training  BLOCKED until R7.5.5 PASS
R8.4 official 3H production training  BLOCKED until R7.5.5 PASS
R8.5 production policy freeze          BLOCKED until R7.5.5 PASS
```

This prevents spending the Ryzen production budget on an encoder/action abstraction that has not survived the legacy-informed design audit.

## 8. Explicitly rejected shortcuts

R7.5 must not:

- use the 53 flop classes merely because they were proposed by the user;
- use 184 merely because an older project used 184;
- copy hard-coded strategy decisions as neural targets and call them learned strategy;
- discard exact chips/pot/SPR in favor of SHORT/MID/DEEP alone;
- infer c-bet/donk/probe from a single generic aggressor bit when exact history can determine them;
- preserve absolute suit identities when a suit-isomorphic representation can preserve strategy-equivalent information more efficiently;
- increase the action tree without a measured strategic return;
- change R7.3/R7.4 gates retroactively;
- claim `READY FOR TABLES` from any R7.5 result.

## 9. Current decision

The leading production direction is now:

```text
exact authoritative poker state
    -> compact poker-semantic NeuralInputV2
       + audited flop abstraction (184 is leading candidate)
       + objective made-hand/draw/texture features
       + exact stack/pot/SPR facts with auxiliary buckets
       + explicit preflop lineage
       + actor-aware, sizing-aware action history
       + derived cbet/donk/probe/float/delayed semantics
    -> small empirically selected action abstraction
    -> Deep CFR
```

This is deliberately neither “give the network every physical card and hope” nor “compress the game into a few human buckets and erase information.”

R7.5 remains **IN PROGRESS** until the physical mapping audit and frozen ablations are completed.