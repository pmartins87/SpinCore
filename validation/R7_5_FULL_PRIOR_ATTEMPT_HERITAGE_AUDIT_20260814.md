# R7.5 — Full prior-attempt heritage audit

Date: 2026-08-14  
Status: **FULL READABLE-SOURCE HERITAGE REVIEW COMPLETE; MISSING HISTORICAL VARIANTS RECORDED AS DEBT; NO STRATEGIC PASS CREATED**  

`READY FOR TABLES = NO`.

## 1. Purpose

This audit formalizes a project rule that is stronger than “look at the old code for ideas.” Before further strategic action-abstraction training, SpinCore must preserve the strongest *correct* parts of earlier SpinGo/DeepSpin/AoF attempts while refusing to inherit their known strategic mistakes, hidden heuristics, impossible-state assumptions, train/runtime mismatches, or stale economic outputs.

The central invariant is **real-game reachability**:

> A represented state is valid only if it can be reached by a legal sequence of poker events from a valid hand start. The model must not be fed a synthetic combination of feature values that could not coexist in a real hand.

Consequently, any derived observation must remain traceable to the authoritative state and public action trajectory: dealt/live/folded/all-in seats, current actor, dealer/blind/position order, street, private/public cards, exact stacks/contributions/pot, amount to call, legal action set, aggressors, and ordered prior actions. A derived label may summarize these facts, but may not replace or contradict them.

## 2. Physical source coverage

The two recovered prior-attempt archives were compared after normalizing their top-level directory names:

- `Tentativas anteriores SpinGo.zip`: 27 entries.
- `Tentativas anteriores de SpinGo.zip`: 29 entries.
- All 27 shared entries are byte-identical.
- The larger archive additionally contains `hardcoded/Crusher Framework 5.txt` and `solver v2/184Flops.json`.

Therefore the larger archive is the authoritative review package and covers the smaller package without losing a byte of shared content.

All 17 human-readable code/data sources in the larger archive were read in full. Python sources were additionally parsed through the Python AST; JSON/CSV data were fully parsed. Compiled `.pyd/.obj/.lib/.exp/.pyc` products are inventoried and hashed, but are not treated as independent semantic sources when their corresponding source files are present.

The machine-readable source identity, hashes, sizes and full-read status are stored in `validation/R7_5_LEGACY_HERITAGE_SOURCE_MANIFEST_20260814.json`.

## 3. What “take the best of every attempt” means

The inheritance policy has four classes:

1. **Authoritative/objective mechanics** — legal game transitions, exact chip/accounting semantics, cards, positions, action history, RNG/checkpoint semantics. Preserve or reproduce exactly unless the current engine independently proves a correction.
2. **Useful semantic abstractions** — hand/draw/board/line vocabulary, canonicalization, diagnostic representations. Keep as candidate features/diagnostics only when derivable from exact state.
3. **Engineering mechanisms** — checkpointing, manifests, source hashes, paired comparisons, audit logs, safe fallbacks, runtime observability. Reuse aggressively when correct.
4. **Historical strategy/heuristics** — hard-coded decisions, old chart outputs, old trained policies, arbitrary thresholds, obsolete economic ranges. Never import as production truth merely because they existed before.

This prevents two opposite errors: throwing away years of useful engineering, or silently training the new system to imitate an old failed strategy.

## 4. DeepSpin neural attempt — what must be inherited

### 4.1 Exact environment and state progression

`deepspin/env/poker_env.cpp` is the most important old source for real-game state semantics. It contains:

- explicit player status and betting-round state;
- exact legal-action generation;
- affordability and minimum-raise filtering;
- near-all-in suppression of duplicate pot-size labels;
- exact application of fold/check-call/raise/all-in transitions;
- per-street ordered histories;
- HU dealer/SB correction when one seat is dead;
- exact pot/current-bet/stack accounting;
- separate live/folded/all-in counts;
- explicit previous aggressor/action context.

This is valuable architectural heritage. SpinCore should continue to derive learning observations by traversing an exact game state, never by independently inventing a feature vector.

### 4.2 OBS292 — useful vocabulary, not an object to copy blindly

The old training environment assembled 292 dimensions in one function `fill_obs_features`, explicitly documenting it as a single source of truth to prevent silent train/runtime mismatch. Its structure was:

```text
000..103  private + public card one-hots
104..123  20 numeric geometry/state facts
124..127  street
128..138  positions
139..178  hand-strength primitives
179..190  draw primitives
191..219  board texture
220..245  action context
246..284  history
285..291  legal-action mask
```

The 20 numeric facts included BB, effective stack, positional stacks/bets, max/min stacks, pot, pot-common estimate, amount-to-call, call/pot, SPR, last aggressive size, live/all-in/folded counts, and raises this street.

The strongest idea is **semantic parity from a common authoritative state**. SpinCore R7.5 V2 should preserve these useful categories where they remain objective, but not inherit the 292-dimensional layout as a frozen truth.

### 4.3 Critical old weakness exposed by full reading

The runtime bridge `deepspin/user_deepspin.cpp` contains comments acknowledging that exact environment recovery from a table snapshot is impossible without full action history. Some action-context/history features were therefore supplied through explicit OpenPPL placeholders rather than reconstructed solely by the bridge.

That is a key lesson: **do not reconstruct a strategic state from a late snapshot when the state depends on the path that led there**. SpinCore should keep the ordered trajectory as an authoritative object and derive lineage/initiative/context from it.

### 4.4 Training/RNG engineering worth keeping

The old Python stack contains good engineering independent of its eventual strategic quality:

- `buffers.py`: deterministic reservoir storage with state/save/load.
- `scenario.py`: isolated scenario RNG and explicit stack/blind/dealer episode sampling.
- `traversal.py`: exact `EpisodeSpec`, replay-to-node, legal-mask regret matching and external sampling.
- `trainer.py`: atomic checkpoints and restoration of Python/NumPy/Torch/scenario RNG state.
- `rollout_workers.py`: explicit episode specifications and worker-side legal masks.
- `networks.py`: masked advantage/policy losses.

These mechanisms support the current project policy of exact checkpoint/resume, fixed seeds and legal-mask discipline. They are heritage to preserve, not evidence that the old trained policy was good.

## 5. Old DeepSpin runtime bridge — what to preserve and what to reject

`deepspin/user_deepspin.cpp` attempted train/runtime parity by rebuilding the same feature families and had explicit statuses `ALIVE/FOLDED/ALLIN/EMPTY`. It also used the seven-action vocabulary:

```text
FOLD, CALL, 33, 50, 75, 100, ALLIN
```

Useful inheritance:

- explicit player-state status rather than inferring participation from stack alone;
- hole/board card validation;
- legal-action masking before inference;
- detailed validation logging for observation slices;
- separate translation from model action to OpenHoldem output.

Rejected as authoritative behavior:

- preflop collapsing of several nominal percentage actions into one real action;
- any runtime placeholder that cannot be traced to a complete ordered action history;
- any assumption that a snapshot alone can recover path-dependent state.

R7.5.4’s state-local exact-action deduplication is a stronger replacement for the old preflop action aliasing.

## 6. Hardcoded/Crusher attempt — richest semantic vocabulary, not target policy

### 6.1 Crusher Framework 5

`hardcoded/Crusher Framework 5.txt` was read in full: 10,831 lines and 1,195 named blocks. The already persisted Crusher semantic audit correctly classifies its reusable content.

The main inherited semantic families are:

- preflop lineage: unopened, limped, isolated, single-raised, reraised, limp-reraised, actor/position lineage;
- initiative and postflop line identity: c-bet, donk, probe, float, delayed lines, raise/reraise;
- continuous faced-sizing and raise geometry;
- exact lineup/IP-OOP/live-player relations;
- exact/effective stacks, pot and SPR;
- hand-made relation, kicker/nutness, draw and backdoor primitives;
- board rank/suit/connectivity and street-transition primitives;
- exact continuation price/pot odds;
- runtime sanity/fallback concepts;
- opponent-stat/exploitation vocabulary for the later R11 stage;
- size vocabulary Min/33/40/50/66/75/100/Max for empirical action-abstraction research.

Historical handlists, chart actions, exploit substitutions, outs estimates, implied-odds formulas and old threshold decisions remain non-authoritative.

### 6.2 `user_hardcoded.cpp`

The 19,803-line implementation adds implementation detail behind that vocabulary: overpair/top-pair/lower-pair relations, kicker quality, nutness, flush/straight contribution, OESD/gutshot/double-gutshot/backdoors, street-completion logic, board pairing/overcard/undercard transitions and line-dependent context.

Best-of inheritance: use these functions as a checklist for **objective semantic coverage tests**. Do not copy their final betting decisions as labels.

### 6.3 `user_hardcoded_helper.cpp`

This file contains one of the strongest runtime ideas from all prior attempts: a human-readable state snapshot and a PT4-like interpreted snapshot that explicitly separates two fault classes:

```text
if state snapshot differs from the reference -> fix state/symbol reading
if state snapshot matches but action is bad   -> fix strategy/branch
```

It logs hero/dealer/BB seats, relative position, board, each chair’s state, bet, stack, hole cards, pot, amount-to-call, effective stack, legal actions and history summary.

This should be carried to R10/R12 as **state-vs-strategy fault isolation**, adapted to SpinCore’s exact state rather than copied symbol-for-symbol.

## 7. Solver v2 attempt — canonicalization and solver integration heritage

### 7.1 184-flop data

`solver v2/184Flops.json` was fully parsed:

- 22,100 physical flop entries;
- each maps to one historical representative;
- `classes_184_resumo.csv` contains 184 representative classes;
- class quantities sum exactly to 22,100.

This is useful historical evidence and regression material. It is not protected as the final flop abstraction: the current R7.5 design may replace it with a stronger deterministic H3 or exact suit-isomorphic reference if validated.

### 7.2 `user_solver_v2.cpp`

The old solver bridge implemented:

- real hand and board extraction;
- representative-flop lookup;
- canonical hand/street-card mapping;
- solver pot/start-of-street pot/stack reconstruction;
- last-action/villain-action inference;
- range/GTO binary loading;
- legal/path action lookup and bet-size mapping;
- hand strength and detailed draw detection;
- solver process lifecycle and fallback paths.

Best-of inheritance:

- canonicalization test cases;
- physical-flop coverage and mapping audits;
- exact hand/board/action-path validation;
- solver-process diagnostics and safe fallback patterns.

Not inherited automatically: old strategic fallback actions, old solver ranges, or old 184-class identity as production truth.

## 8. AoF foundations — engineering patterns that transfer to SpinCore

The available AoF sources were also read in full because they contain mechanisms that proved useful outside the narrow AoF game.

### 8.1 Generators and standalone solvers

`aof_deep_generator_unificado_150k.py`, `aof_solver_standalone.py`, `stand_alone_kk.py`, `stand_alone_kk_all.py` and the ante variant contain strong reproducibility practices:

- self-contained resolved configuration;
- script/source hashes;
- argv/Python/platform manifests;
- explicit economy/rules;
- multiple deterministic seeds;
- CFR+/linear-average controls;
- coverage/min-visit accounting;
- pairwise/cross-seed stability summaries;
- preservation of best evidence per information set.

SpinCore should keep this evidence discipline while using its own exact NLHE rules and current training gates.

### 8.2 Paired decision comparison

`aof_best_response_vs_pool.py` is especially useful methodologically: compare alternative Hero actions on the same conditioned state/deal so variance cancels, report the EV gap, standard error and confidence interval. The transferable lesson is **paired causal comparison**, not the AoF action space.

### 8.3 State reconstruction, aliases and runtime safety

The AoF tracker/alias/schema/MES sources and the operational manual reinforce:

- preserve raw events instead of only aggregates;
- distinguish player identity from seat and from aliases;
- reconstruct action-level context before assigning statistics;
- fail safe on mode/position/context mismatch;
- never turn an uncertain lookup into an aggressive exploit action;
- preserve command, config, source hashes, logs, progress and installation evidence for every run.

These concepts belong later in SpinCore R10/R11/R12, not in the base-policy neural target.

### 8.4 Branch relevance from multiway AoF work

The AoF multiway plan contains a useful general simulation principle: once Hero folds, future actions may be irrelevant to Hero EV; if Hero continues/all-ins, later branches that change the showdown composition matter. This is a good guide for efficient exact traversal **only when the terminal utility proves that the omitted branch cannot affect Hero utility**. It must never be used to skip strategically relevant state evolution.

## 9. Exact “attention to what happens in the game” contract

Before a new representation/action candidate can be trusted, all of the following must hold for every observation sampled from the solver:

```text
hand start is legal
seat/dealer/blind assignment is legal
cards are unique and compatible with the street
folded players cannot act again
all-in players cannot take voluntary later actions
current actor is the next legal actor under the betting rules
ordered action history reaches the present chip contributions exactly
pot and per-street contributions reconcile with the history
amount-to-call and minimum/max raise come from the exact current state
legal actions are derived from the exact engine, not guessed by the network bridge
live-player/position/IP-OOP facts agree with statuses and dealer relation
preflop lineage and initiative are recomputed from ordered public events
postflop line labels are derived from those events, not independently set
board/hand/draw labels are deterministic functions of actual cards
no derived feature combination contradicts any authoritative fact above
```

This requirement is stronger than feature coverage. A network can have every desirable feature and still be wrong if the feature combination describes an impossible hand.

## 10. Heritage matrix — destination in the current roadmap

| Prior source | Best element to preserve | Current/next destination | Never copy as truth |
|---|---|---|---|
| DeepSpin `poker_env` | exact state, legal transitions, history, OBS source-of-truth concept | R7.5/R7.5.4 exact traversal | old learned policy |
| DeepSpin runtime | status/position/cards/legal-mask parity diagnostics | R10 runtime parity | snapshot-guessed history |
| DeepSpin trainer/traversal/buffers | RNG/checkpoint/replay/masked training | R7.5+ infrastructure | old training result |
| Crusher Framework | semantic vocabulary and sizing catalogue | R7.5 features, R7.5.4 actions, R11 exploit vocabulary | charts/actions/threshold decisions |
| hardcoded helper | state-vs-strategy fault isolation | R10/R12 | OH-specific assumptions without validation |
| solver v2 | canonicalization/flop mapping/solver diagnostics | R7.5 flop candidate tests, R10 | old GTO/range output, forced 184 production classes |
| AoF generators | manifests, source hashes, coverage, multi-seed evidence | all scientific stages | AoF economics/ranges |
| AoF best response | paired action EV/CI method | strategic referee/ablation methodology | AoF-specific actions |
| AoF tracker/MES | raw events, aliases, fail-safe lookup, audit | R10/R11/R12 | old opponent policy tables |

## 11. Already incorporated versus newly strengthened

Already present in current SpinCore work:

- exact authoritative game state/traversal;
- frozen candidate semantics and immutable compute SHA;
- full Crusher semantic audit;
- NeuralInputV2 work based on exact state/history semantics;
- exact-action resolution/dedup for the 10-action R7.5.4 vocabulary;
- exact checkpoint/RNG preservation;
- structural audits and fail-closed preflights;
- empirical action-abstraction selection rather than importing old charts.

This audit newly strengthens the project by making the **entire prior-attempt corpus** an explicit heritage requirement rather than relying mainly on the Crusher audit and remembered lessons. In particular, the old DeepSpin runtime-parity attempt, solver-v2 canonicalization, hardcoded helper observability and AoF evidence discipline are now explicitly assigned downstream destinations.

## 12. Unrecovered historical-source debt

Three generator names were explicitly referenced in earlier project work but their physical source files were not recovered in the current local uploads/File Library search:

```text
aof_deep_generator_unificado_150k_v2.py
aof_deep_generator_balanceado.py
aof_deep_generator_balanceado_v2.py
```

They are therefore **not falsely marked as read**. Their absence is recorded as source-recovery debt. Current R7.5.3 immutable computation does not depend on them. Before a final production/release freeze, any still-relevant unique behavior from those variants must either be recovered and reviewed or explicitly proven superseded by a source we do possess.

## 13. Gate consequence

This audit itself creates no strategic PASS.

The future R7.5.4 strategic preflight must require:

1. this heritage audit;
2. the machine-readable heritage manifest;
3. the real-game reachability contract;
4. the existing R7.5.3 representation winner;
5. R7.5.4 structural correctness;
6. uncertainty-equivalence certification;
7. the frozen action/training contracts.

Thus a manual workflow launch cannot bypass the requirement to use the strongest correct lessons from the old attempts.

`READY FOR TABLES = NO`.
