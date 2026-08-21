from __future__ import annotations

"""Deterministic read-only enrichment for the R7.5.3D V1+ forensic readout.

Consumes the raw JSON produced by r7_5_3d_v1plus_forensic_postmortem.py and the
same frozen heldout V3 artifacts. It adds the diagnostic slices frozen in
validation/R7_5_3D_V1PLUS_POSTMORTEM_IMPLEMENTATION_AUDIT_20260821.md.

No model is trained or mutated by this tool.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

RAW_SCHEMA = "SPINCORE_R7_5_3D_V1PLUS_FORENSIC_POSTMORTEM_V1"
SCHEMA = "SPINCORE_R7_5_3D_V1PLUS_FORENSIC_ENRICHED_V1"
ZERO_TV_TOLERANCE = 1e-12
EXPECTED_POLICY_COUNT = 1024
H2 = "H2_RELATIONAL_EXACT_STRUCTURED_HISTORY_FINAL"
H3 = "H3_HYBRID_EXACT_SEMANTIC_FINAL"


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _linear_quantile(values: Sequence[float], q: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        raise ValueError("quantile requires non-empty values")
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("quantile must be in [0,1]")
    if len(rows) == 1:
        return rows[0]
    position = (len(rows) - 1) * float(q)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return rows[lo]
    weight = position - lo
    return rows[lo] * (1.0 - weight) + rows[hi] * weight


def _summary(values: Iterable[float]) -> dict:
    rows = [float(value) for value in values if _finite(value)]
    if not rows:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(rows),
        "mean": float(sum(rows) / len(rows)),
        "p50": float(_linear_quantile(rows, 0.50)),
        "p95": float(_linear_quantile(rows, 0.95)),
        "max": float(max(rows)),
    }


def _history_count_bin(value: object) -> str:
    x = int(value)
    if x == 0:
        return "0"
    if x <= 2:
        return "1-2"
    if x <= 4:
        return "3-4"
    if x <= 8:
        return "5-8"
    if x <= 16:
        return "9-16"
    return "17+"


def _pot_bin(value: object) -> str:
    x = float(value)
    if x <= 2.0:
        return "<=2"
    if x <= 5.0:
        return "(2,5]"
    if x <= 10.0:
        return "(5,10]"
    if x <= 20.0:
        return "(10,20]"
    return ">20"


def _call_bet_bin(value: object) -> str:
    x = float(value)
    if abs(x) <= 1e-12:
        return "0"
    if x <= 1.0:
        return "(0,1]"
    if x <= 2.0:
        return "(1,2]"
    if x <= 5.0:
        return "(2,5]"
    return ">5"


def _stack_bin(value: object) -> str:
    x = float(value)
    if abs(x) <= 1e-12:
        return "0"
    if x <= 5.0:
        return "(0,5]"
    if x <= 10.0:
        return "(5,10]"
    if x <= 20.0:
        return "(10,20]"
    if x <= 40.0:
        return "(20,40]"
    return ">40"


def _stack_spread_bin(value: object) -> str:
    x = float(value)
    if abs(x) <= 1e-12:
        return "0"
    if x <= 5.0:
        return "(0,5]"
    if x <= 10.0:
        return "(5,10]"
    if x <= 20.0:
        return "(10,20]"
    return ">20"


def _spr_bin(value: object) -> str:
    if value is None or not _finite(value):
        return "NA"
    x = float(value)
    if x <= 1.0:
        return "<=1"
    if x <= 2.0:
        return "(1,2]"
    if x <= 5.0:
        return "(2,5]"
    if x <= 10.0:
        return "(5,10]"
    return ">10"


def _ratio_bin(value: object) -> str:
    x = float(value)
    if abs(x) <= 1e-12:
        return "0"
    if x <= 0.25:
        return "(0,.25]"
    if x <= 0.50:
        return "(.25,.5]"
    if x <= 1.0:
        return "(.5,1]"
    if x <= 2.0:
        return "(1,2]"
    return ">2"


def _dominant_legal_delta_slot(
    legal_slots: Sequence[int],
    abs_delta_by_slot: Sequence[float],
    tv: float,
) -> int | None:
    if float(tv) <= ZERO_TV_TOLERANCE:
        return None
    legal = sorted({int(slot) for slot in legal_slots})
    if not legal:
        return None
    if any(slot < 0 or slot >= len(abs_delta_by_slot) for slot in legal):
        raise ValueError("legal slot outside delta vector")
    best = max(legal, key=lambda slot: (float(abs_delta_by_slot[slot]), -slot))
    if float(abs_delta_by_slot[best]) <= ZERO_TV_TOLERANCE:
        return None
    return int(best)


def _action_composition_key(row: dict) -> str:
    counts = tuple(int(row.get(f"history_action_type_{action_type}_count", 0)) for action_type in range(6))
    return "(" + ",".join(str(value) for value in counts) + ")"


def _grouped_tv(rows: Sequence[dict], field: str, transform=None) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        raw = row.get(field)
        key = transform(raw) if transform is not None else raw
        groups[str(key)].append(float(row["tv"]))
    return {key: _summary(values) for key, values in sorted(groups.items())}


def _grouped_delta(rows: Sequence[dict], field: str, transform=None) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        raw = row.get(field)
        key = transform(raw) if transform is not None else raw
        groups[str(key)].append(float(row["h3_minus_h2_tv"]))
    return {key: _summary(values) for key, values in sorted(groups.items())}


def _stack_geometry_key(row: dict) -> str:
    live_count = int(row["live_count"])
    stacks = [float(row[f"stack_rel{index}_bb"]) for index in range(live_count)]
    return "|".join(_stack_bin(value) for value in stacks)


def _context_row(row: dict) -> dict:
    keys = (
        "representation",
        "domain",
        "evaluation_seed",
        "state_index",
        "hand_index",
        "scenario_index",
        "actor",
        "street_name",
        "pot_bb",
        "to_call_bb",
        "current_bet_bb",
        "stack_rel0_bb",
        "stack_rel1_bb",
        "stack_rel2_bb",
        "live_count",
        "stack_spread_bb",
        "stack_geometry_bin",
        "spr",
        "history_len",
        "action_path_len",
        "forced_count",
        "nonforced_count",
        "unique_history_actors",
        "last_actor",
        "last_action_type",
        "action_composition",
        "history_paid_over_pot_mean",
        "history_paid_over_pot_std",
        "history_paid_over_pot_max",
        "history_commitment_over_pot_max",
        "v1_projection_exact_variants",
        "structured_projection_exact_variants",
        "legal_slots",
        "tv",
        "dominant_legal_delta_slot",
        "left_policy",
        "right_policy",
        "abs_delta_by_slot",
    )
    return {key: row.get(key) for key in keys}


def _find_heldout(root: Path, domain: str, evaluation_seed: int):
    from spincore.r7_5_representation_v3_referee_artifacts import load_heldout_v3_artifact

    matches = []
    for path in root.rglob("states.json.gz"):
        try:
            states = load_heldout_v3_artifact(
                path,
                expected_domain=domain,
                expected_evaluation_seed=int(evaluation_seed),
                expected_count=2048,
            )
        except Exception:
            continue
        if states:
            matches.append((path, states))
    if len(matches) != 1:
        raise RuntimeError(f"heldout identity mismatch for {domain}/{evaluation_seed}: {[str(x[0]) for x in matches]}")
    return matches[0]


def _heldout_geometry(heldout_root: Path, domains: Sequence[str], evaluation_seeds: Sequence[int]) -> dict:
    from spincore_nn.codec_v3 import decode_spnniv3

    geometry = {}
    provenance = {}
    for domain in domains:
        for evaluation_seed in evaluation_seeds:
            path, states = _find_heldout(heldout_root, str(domain), int(evaluation_seed))
            selected = states[:EXPECTED_POLICY_COUNT]
            if len(selected) != EXPECTED_POLICY_COUNT:
                raise RuntimeError("heldout artifact has insufficient states")
            for state in selected:
                decoded = decode_spnniv3(state.observation_v3)
                live_count = int(decoded.categorical[5])
                stacks = [float(decoded.numeric[3 + index]) for index in range(3)]
                live_stacks = stacks[:live_count]
                positive = [value for value in live_stacks if value > 0.0]
                stack_spread = (max(positive) - min(positive)) if positive else 0.0
                key = (str(domain), int(evaluation_seed), int(state.state_index))
                if key in geometry:
                    raise RuntimeError(f"duplicate heldout geometry identity {key}")
                geometry[key] = {
                    "stack_rel0_bb": stacks[0],
                    "stack_rel1_bb": stacks[1],
                    "stack_rel2_bb": stacks[2],
                    "stack_spread_bb": float(stack_spread),
                }
            provenance[f"{domain}|{int(evaluation_seed)}"] = str(path)
    return {"geometry": geometry, "provenance": provenance}


def _enrich_state_rows(raw_rows: Sequence[dict], heldout_root: Path, domains: Sequence[str], evaluation_seeds: Sequence[int]) -> tuple[list[dict], dict]:
    heldout = _heldout_geometry(heldout_root, domains, evaluation_seeds)
    geometry = heldout["geometry"]
    enriched = []
    seen_identities = set()
    for source in raw_rows:
        row = dict(source)
        identity = (
            str(row["representation"]),
            str(row["domain"]),
            int(row["evaluation_seed"]),
            int(row["state_index"]),
        )
        if identity in seen_identities:
            raise RuntimeError(f"duplicate raw state identity {identity}")
        seen_identities.add(identity)
        gkey = (identity[1], identity[2], identity[3])
        if gkey not in geometry:
            raise RuntimeError(f"raw state missing from heldout geometry {gkey}")
        row.update(geometry[gkey])
        row["legal_slot_count"] = len(row["legal_slots"])
        row["dominant_legal_delta_slot"] = _dominant_legal_delta_slot(
            row["legal_slots"], row["abs_delta_by_slot"], float(row["tv"])
        )
        row["action_composition"] = _action_composition_key(row)
        row["stack_geometry_bin"] = _stack_geometry_key(row)
        enriched.append(row)
    return enriched, heldout["provenance"]


def _slice_bundle(rows: Sequence[dict]) -> dict:
    return {
        "by_street": _grouped_tv(rows, "street_name"),
        "by_history_len": _grouped_tv(rows, "history_len", _history_count_bin),
        "by_action_path_len": _grouped_tv(rows, "action_path_len", _history_count_bin),
        "by_legal_slot_count": _grouped_tv(rows, "legal_slot_count"),
        "by_pot_bb": _grouped_tv(rows, "pot_bb", _pot_bin),
        "by_to_call_bb": _grouped_tv(rows, "to_call_bb", _call_bet_bin),
        "by_current_bet_bb": _grouped_tv(rows, "current_bet_bb", _call_bet_bin),
        "by_min_positive_stack_bb": _grouped_tv(rows, "min_positive_stack_bb", _stack_bin),
        "by_stack_spread_bb": _grouped_tv(rows, "stack_spread_bb", _stack_spread_bin),
        "by_stack_geometry": _grouped_tv(rows, "stack_geometry_bin"),
        "by_spr": _grouped_tv(rows, "spr", _spr_bin),
        "by_forced_count": _grouped_tv(rows, "forced_count", _history_count_bin),
        "by_nonforced_count": _grouped_tv(rows, "nonforced_count", _history_count_bin),
        "by_unique_history_actors": _grouped_tv(rows, "unique_history_actors"),
        "by_last_actor": _grouped_tv(rows, "last_actor"),
        "by_last_action_type": _grouped_tv(rows, "last_action_type"),
        "by_action_type_composition": _grouped_tv(rows, "action_composition"),
        "by_history_paid_over_pot_mean": _grouped_tv(rows, "history_paid_over_pot_mean", _ratio_bin),
        "by_history_paid_over_pot_std": _grouped_tv(rows, "history_paid_over_pot_std", _ratio_bin),
        "by_history_paid_over_pot_max": _grouped_tv(rows, "history_paid_over_pot_max", _ratio_bin),
        "by_history_commitment_over_pot_max": _grouped_tv(rows, "history_commitment_over_pot_max", _ratio_bin),
        "by_v1_projection_exact_variants": _grouped_tv(
            rows,
            "v1_projection_exact_variants",
            lambda value: "1" if int(value) <= 1 else "2" if int(value) == 2 else "3-4" if int(value) <= 4 else "5-8" if int(value) <= 8 else "9+",
        ),
        "by_structured_projection_exact_variants": _grouped_tv(
            rows,
            "structured_projection_exact_variants",
            lambda value: "1" if int(value) <= 1 else "2" if int(value) == 2 else "3-4" if int(value) <= 4 else "5-8" if int(value) <= 8 else "9+",
        ),
    }


def _row_level_summaries(rows: Sequence[dict], representations: Sequence[str], domains: Sequence[str], evaluation_seeds: Sequence[int]) -> list[dict]:
    output = []
    for representation in representations:
        for domain in domains:
            for evaluation_seed in evaluation_seeds:
                local = [
                    row
                    for row in rows
                    if row["representation"] == representation
                    and row["domain"] == domain
                    and int(row["evaluation_seed"]) == int(evaluation_seed)
                ]
                if len(local) != EXPECTED_POLICY_COUNT:
                    raise RuntimeError(
                        f"row count mismatch for {representation}/{domain}/{evaluation_seed}: {len(local)}"
                    )
                output.append(
                    {
                        "representation": representation,
                        "domain": domain,
                        "evaluation_seed": int(evaluation_seed),
                        "tv": _summary(row["tv"] for row in local),
                        "slices": _slice_bundle(local),
                        "top_25_high_tv_states_with_context": [
                            _context_row(row)
                            for row in sorted(local, key=lambda item: float(item["tv"]), reverse=True)[:25]
                        ],
                    }
                )
    return output


def _paired_h3_minus_h2(
    rows: Sequence[dict],
    domains: Sequence[str],
    evaluation_seeds: Sequence[int],
    h2_name: str = H2,
    h3_name: str = H3,
    policy_count: int = EXPECTED_POLICY_COUNT,
) -> list[dict]:
    by_identity = {}
    for row in rows:
        key = (
            str(row["representation"]),
            str(row["domain"]),
            int(row["evaluation_seed"]),
            int(row["state_index"]),
        )
        if key in by_identity:
            raise RuntimeError(f"duplicate paired identity {key}")
        by_identity[key] = row

    output = []
    for domain in domains:
        for evaluation_seed in evaluation_seeds:
            paired = []
            for state_index in range(int(policy_count)):
                h2_key = (h2_name, str(domain), int(evaluation_seed), state_index)
                h3_key = (h3_name, str(domain), int(evaluation_seed), state_index)
                if h2_key not in by_identity or h3_key not in by_identity:
                    raise RuntimeError(f"missing H2/H3 paired identity {domain}/{evaluation_seed}/{state_index}")
                h2 = by_identity[h2_key]
                h3 = by_identity[h3_key]
                invariant_fields = (
                    "street_name",
                    "history_len",
                    "action_path_len",
                    "pot_bb",
                    "to_call_bb",
                    "current_bet_bb",
                    "stack_geometry_bin",
                    "spr",
                    "forced_count",
                    "nonforced_count",
                    "unique_history_actors",
                    "last_actor",
                    "last_action_type",
                    "action_composition",
                    "v1_history_projection_sha256",
                    "structured_history_projection_sha256",
                    "exact_history_sha256",
                )
                for field in invariant_fields:
                    if h2.get(field) != h3.get(field):
                        raise RuntimeError(
                            f"heldout alignment drift at {domain}/{evaluation_seed}/{state_index} field={field}"
                        )
                row = {
                    "domain": str(domain),
                    "evaluation_seed": int(evaluation_seed),
                    "state_index": state_index,
                    "h2_tv": float(h2["tv"]),
                    "h3_tv": float(h3["tv"]),
                    "h3_minus_h2_tv": float(h3["tv"]) - float(h2["tv"]),
                }
                for field in invariant_fields:
                    row[field] = h2.get(field)
                row["history_paid_over_pot_mean"] = h2.get("history_paid_over_pot_mean")
                row["history_paid_over_pot_std"] = h2.get("history_paid_over_pot_std")
                row["history_paid_over_pot_max"] = h2.get("history_paid_over_pot_max")
                row["history_commitment_over_pot_max"] = h2.get("history_commitment_over_pot_max")
                paired.append(row)

            slices = {
                "by_street": _grouped_delta(paired, "street_name"),
                "by_history_len": _grouped_delta(paired, "history_len", _history_count_bin),
                "by_action_path_len": _grouped_delta(paired, "action_path_len", _history_count_bin),
                "by_pot_bb": _grouped_delta(paired, "pot_bb", _pot_bin),
                "by_to_call_bb": _grouped_delta(paired, "to_call_bb", _call_bet_bin),
                "by_current_bet_bb": _grouped_delta(paired, "current_bet_bb", _call_bet_bin),
                "by_stack_geometry": _grouped_delta(paired, "stack_geometry_bin"),
                "by_spr": _grouped_delta(paired, "spr", _spr_bin),
                "by_forced_count": _grouped_delta(paired, "forced_count", _history_count_bin),
                "by_unique_history_actors": _grouped_delta(paired, "unique_history_actors"),
                "by_last_action_type": _grouped_delta(paired, "last_action_type"),
                "by_action_type_composition": _grouped_delta(paired, "action_composition"),
                "by_history_paid_over_pot_max": _grouped_delta(paired, "history_paid_over_pot_max", _ratio_bin),
            }
            deltas = [row["h3_minus_h2_tv"] for row in paired]
            output.append(
                {
                    "domain": str(domain),
                    "evaluation_seed": int(evaluation_seed),
                    "meaning": "negative favors H3 stability; positive means H3 is less stable than H2 on the exact same heldout state",
                    "h3_minus_h2_tv": _summary(deltas),
                    "fraction_states_h3_more_stable": float(sum(value < 0.0 for value in deltas) / len(deltas)),
                    "fraction_states_h3_less_stable": float(sum(value > 0.0 for value in deltas) / len(deltas)),
                    "slices": slices,
                }
            )
    return output


def _action_slot_concentration(raw_action_slot: dict) -> dict:
    output = {}
    for key, row in sorted(raw_action_slot.items()):
        shares = [float(value) for value in row["share_of_total_l1_by_slot"]]
        ranked = sorted(enumerate(shares), key=lambda item: item[1], reverse=True)
        output[key] = {
            **row,
            "top_slot": int(ranked[0][0]) if ranked and ranked[0][1] > ZERO_TV_TOLERANCE else None,
            "top_slot_share": float(ranked[0][1]) if ranked else 0.0,
            "top_2_slot_share": float(sum(value for _slot, value in ranked[:2])),
            "top_3_slot_share": float(sum(value for _slot, value in ranked[:3])),
            "ranked_slots_by_l1_share": [
                {"slot": int(slot), "share": float(value)} for slot, value in ranked
            ],
        }
    return output


def _reservoir_pressure_summary(cells: Sequence[dict], overlaps: Sequence[dict]) -> dict:
    per_cell = []
    for row in cells:
        retained = int(row["retained"])
        unique_exact = int(row["unique_exact_observations"])
        per_cell.append(
            {
                "representation": row["representation"],
                "domain": row["domain"],
                "training_seed": int(row["training_seed"]),
                "memory": row["memory"],
                "saturation_factor_seen_over_capacity": row["saturation_factor_seen_over_capacity"],
                "retention_fraction_retained_over_seen": row["retention_fraction_retained_over_seen"],
                "unique_exact_fraction_of_retained": float(unique_exact / retained) if retained else 0.0,
                "exact_duplicate_fraction": row["exact_duplicate_fraction"],
                "unique_v1_history_projections": int(row["unique_v1_history_projections"]),
                "unique_structured_history_projections": int(row["unique_structured_history_projections"]),
            }
        )
    overlap_compact = []
    for row in overlaps:
        overlap_compact.append(
            {
                "representation": row["representation"],
                "domain": row["domain"],
                "memory": row["memory"],
                "exact_jaccard": row["exact_observation_overlap"]["jaccard"],
                "v1_history_projection_jaccard": row["v1_history_projection_overlap"]["jaccard"],
                "structured_history_projection_jaccard": row["structured_history_projection_overlap"]["jaccard"],
            }
        )
    return {"per_cell": per_cell, "cross_seed_overlap": overlap_compact}


def _validate_raw(raw: dict) -> None:
    if raw.get("schema") != RAW_SCHEMA:
        raise ValueError("wrong raw forensic schema")
    if raw.get("status") != "FORENSIC_READOUT_COMPLETE_NO_ARCHITECTURE_SELECTED":
        raise ValueError("raw forensic readout is not complete")
    if bool(raw.get("production_training_authorized")) or bool(raw.get("ready_for_tables")):
        raise ValueError("raw forensic output illegally authorizes production/table use")
    if int(raw.get("policy_count_per_representation_domain_evaluation_seed", -1)) != EXPECTED_POLICY_COUNT:
        raise ValueError("unexpected raw policy count")
    if len(raw.get("representations") or []) != 2:
        raise ValueError("expected exactly two representations")
    if len(raw.get("domains") or []) != 2:
        raise ValueError("expected exactly two domains")
    if len(raw.get("evaluation_seeds") or []) != 2:
        raise ValueError("expected exactly two evaluation seeds")
    expected_rows = 2 * 2 * 2 * EXPECTED_POLICY_COUNT
    if len(raw.get("state_rows") or []) != expected_rows:
        raise ValueError(f"raw state-row count mismatch: {len(raw.get('state_rows') or [])} != {expected_rows}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Enrich frozen R7.5.3D V1+ forensic JSON")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raw_path = args.raw.resolve()
    heldout_root = args.heldout_root.resolve()
    output = args.out.resolve()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    _validate_raw(raw)

    representations = [str(value) for value in raw["representations"]]
    domains = [str(value) for value in raw["domains"]]
    evaluation_seeds = [int(value) for value in raw["evaluation_seeds"]]
    if set(representations) != {H2, H3}:
        raise ValueError(f"unexpected representation identities: {representations}")

    rows, heldout_provenance = _enrich_state_rows(
        raw["state_rows"], heldout_root, domains, evaluation_seeds
    )
    summaries = _row_level_summaries(rows, representations, domains, evaluation_seeds)
    paired = _paired_h3_minus_h2(rows, domains, evaluation_seeds)
    action_slots = _action_slot_concentration(raw["action_slot_disagreement"])
    reservoir_pressure = _reservoir_pressure_summary(
        raw["reservoir_cells"], raw["reservoir_cross_seed_overlap"]
    )

    result = {
        "schema": SCHEMA,
        "status": "FORENSIC_ENRICHMENT_COMPLETE_NO_ARCHITECTURE_SELECTED",
        "source_raw": str(raw_path),
        "training_execution_sha": raw["training_execution_sha"],
        "heldout_provenance": heldout_provenance,
        "fixed_bins_contract": {
            "pot_bb": ["<=2", "(2,5]", "(5,10]", "(10,20]", ">20"],
            "to_call_current_bet_bb": ["0", "(0,1]", "(1,2]", "(2,5]", ">5"],
            "live_stack_bb": ["0", "(0,5]", "(5,10]", "(10,20]", "(20,40]", ">40"],
            "stack_spread_bb": ["0", "(0,5]", "(5,10]", "(10,20]", ">20"],
            "spr": ["<=1", "(1,2]", "(2,5]", "(5,10]", ">10", "NA"],
            "history_count": ["0", "1-2", "3-4", "5-8", "9-16", "17+"],
            "historical_ratio": ["0", "(0,.25]", "(.25,.5]", "(.5,1]", "(1,2]", ">2"],
        },
        "state_level_enriched_summaries": summaries,
        "h3_minus_h2_paired_enriched": paired,
        "action_slot_disagreement_enriched": action_slots,
        "reservoir_pressure_compact": reservoir_pressure,
        "state_rows_enriched": rows,
        "interpretation_guardrails": [
            "This enrichment is deterministic read-only analysis of already-paid x16 outputs.",
            "No architecture is selected by these slices.",
            "A slice or correlation is not causal evidence by itself.",
            "V1-like projection remains information-content-only, not byte-equivalent SPNNIV1 reconstruction.",
            "Stability remains an admission condition; strategic strength remains a separate selection condition.",
        ],
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "state_rows": len(rows),
                "summary_rows": len(summaries),
                "paired_rows": len(paired),
                "out": str(output),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
