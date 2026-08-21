# R7.5.3D — V1+ forensic findings

Date: 2026-08-21  
Status: PHASE1_FINDINGS_PERSISTED / NO_ARCHITECTURE_SELECTED  
READY FOR TABLES: NO  
Production training authorized: NO

## Provenance

The findings below are derived from the completed read-only Ryzen forensic packet:

- x16 training execution SHA: `f44e05513721b59f63ed5c61f37de2c115c67315`
- diagnostic execution SHA: `2bfdcdffbed24d4c14d49d7cc1127974b8761f96`
- RAW SHA-256: `4cc6dbf5ee405a18eeedcd85f85348b8c265d29a3e5bb551f56dfeef37e12083`
- ENRICHED SHA-256: `481276add848787eb1b839f0c80b3682d011babb6d980e0e5701828fec763b10`
- LOCAL MANIFEST SHA-256: `69bcc1adb1a112a581ca4596e78656dce72b18c73c2ff453ae45023e32200d1b`
- 8,192 heldout state rows, 16 final reservoirs, 8 cross-seed reservoir-overlap rows.

No training, threshold change, seed change, weight mutation, or representation selection occurred during this readout.

## Core stability readout

Pooled over the two frozen evaluation seeds:

| representation | domain | mean TV | p95 TV |
|---|---:|---:|---:|
| H2 | TRUE_HEADS_UP | 0.157042 | 0.386854 |
| H3 | TRUE_HEADS_UP | 0.150440 | 0.347395 |
| H2 | THREE_HANDED | 0.200632 | 0.524370 |
| H3 | THREE_HANDED | 0.203994 | 0.554351 |

H3 is slightly more stable than H2 in HU and slightly less stable in 3H, but the H3-minus-H2 paired mean is small relative to the total 3H failure. This rejects H3 semantics as the primary explanation for the instability.

## Critical finding: instability already exists before voluntary history

On heldout states with `action_path_len == 0` and therefore only forced preflop history:

| representation | domain | mean TV | p95 TV |
|---|---:|---:|---:|
| H2 | TRUE_HEADS_UP | 0.143660 | 0.318071 |
| H3 | TRUE_HEADS_UP | 0.145567 | 0.283678 |
| H2 | THREE_HANDED | 0.216823 | 0.521308 |
| H3 | THREE_HANDED | 0.237185 | 0.559741 |

Therefore the full exact quantitative public history is not a necessary condition for the 3H instability. A representation-only diagnosis of “unbounded history is the root cause” is rejected.

History richness still contributes secondarily: when multiple exact histories collapse into one V1-like history projection, TV rises materially. For H2 3H, mean TV rises from 0.188395 (`variants=1`) to 0.241607 (`variants>1`); H3 3H rises from 0.198110 to 0.223696. The same direction appears in HU. Thus exact history refinement can amplify instability, but it does not explain the unstable 3H root states.

## Strongest evidence: 3H Strategy-memory pressure

All reservoirs have capacity 100,000.

### Strategy reservoir

HU:
- H2 seen: 147,381–147,885; saturation 1.474–1.479x; retained/seen 67.62–67.85%.
- H3 seen: 154,903–158,477; saturation 1.549–1.585x; retained/seen 63.10–64.56%.

3H:
- H2 seen: 3,424,960–3,852,182; saturation 34.25–38.52x; retained/seen 2.60–2.92%.
- H3 seen: 3,223,249–4,016,925; saturation 32.23–40.17x; retained/seen 2.49–3.10%.

The same fixed Strategy-memory budget is therefore operating in radically different regimes. 3H generates roughly 20–27 times as many Strategy samples per x16 run as HU, while capacity remains unchanged.

Cross-seed history-projection Jaccard for Strategy memory also collapses:

- HU H2: 0.7233
- HU H3: 0.7307
- 3H H2: 0.1574 (V1-like), 0.1555 (structured categorical)
- 3H H3: 0.1606 (V1-like), 0.1588 (structured categorical)

This domain split tracks the stability split closely.

### Advantage reservoir

Advantage saturation is high in both domains and does not show the same 3H-vs-HU pressure asymmetry:

- HU: about 14.2–14.7x
- 3H: about 11.5–13.2x

Therefore “all memory is too small” is too coarse. The strongest Phase-1 evidence points specifically to the Strategy-memory / 3H sampling regime, not generic reservoir capacity alone.

## H3 semantic features

Paired H3-minus-H2 state-level TV:
- HU: small improvement on average (approximately -0.0066 pooled).
- 3H: small deterioration on average (approximately +0.0034 pooled).
- Around half of states favor H3 and half favor H2.

Conclusion: H3 semantics are not the primary instability driver. They remain eligible for later V1+ ablation rather than being discarded.

## Action-output concentration

Universal action slots are:
`FOLD=0, CHECK_CALL=1, MIN_RAISE=2, POT_33=3, POT_40=4, POT_50=5, POT_66=6, POT_75=7, POT_100=8, ALL_IN=9`.

Across all rows, most L1 disagreement lies in FOLD and ALL_IN. The top two slots account for roughly 64–70% of total L1 mass. Conditioned on legality, FOLD disagreement is especially larger in 3H (about 0.181–0.189 mean absolute delta) than HU (about 0.131–0.133), and ALL_IN is also larger in 3H (about 0.117–0.123 versus 0.089–0.097).

This identifies the decision boundary where instability manifests, but does not establish the action vocabulary as the cause. These slots are also among the most frequently legal actions. Action-output changes remain downstream until sampling/memory causality is separated.

## Hypothesis ranking after Phase 1

1. **HYP-B Strategy-memory / sampling-distribution pressure: strongly supported and highest priority.**
2. **HYP-A exact-history fragmentation: partially supported as an amplifier, rejected as sole/root cause.**
3. **HYP-D H3 semantic amplification: not supported as primary cause.**
4. **HYP-E action-output instability: supported as a symptom/concentration, causality unresolved.**
5. **HYP-C model capacity/regularization: unresolved; do not spend a training run on it before the memory/chance decomposition is completed.**

## Important identification gap

The current reservoir post-mortem compares:
- exact full observation;
- V1-like history projection; and
- structured categorical history projection.

Exact full-observation overlap is near zero in both HU and 3H because it mixes cards, current chip geometry, legality and history. It therefore cannot tell whether the remaining seed divergence is driven primarily by chance/card support, current geometry, or exact quantitative history.

Before any new training, Phase 1B must decompose reservoir overlap into orthogonal projections:
- cards only;
- current geometry/legality only;
- fixed current state without history;
- exact history only;
- structured categorical history only;
- V1-like history only;
- state-without-cards + exact history;
- state-without-cards + structured history;
- state-without-cards + V1-like history.

## Decision

Do **not** start a V1+ training ablation yet.

The next step is a zero-training Phase 1B projection decomposition on the existing x16 checkpoints. If it confirms that 3H Strategy-memory support is the dominant failure mechanism independently of exact history, the first causal ablation should change the 3H Strategy-memory/sampling treatment before compressing the representation. If quantitative history remains a material independent source after removing card/chance identity, then a compressed structured-history V1+ arm becomes justified.

Stability remains an eligibility gate. Strategic strength remains a separate selection gate.
