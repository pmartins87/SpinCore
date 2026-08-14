from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from spincore.flop_abstraction import (
    RSCard,
    card_text,
    encode_spin_card_id,
    parse_flop_text,
    suit_isomorphic_key,
)

RepairMode = Literal["canonical_input", "majority_min_change"]


def _physical_key(text: str) -> tuple[int, int, int]:
    return tuple(sorted(encode_spin_card_id(card) for card in parse_flop_text(text)))


def _physical_text(key: tuple[int, int, int]) -> str:
    # Mapping outputs use deterministic SpinCore rank-major card order. The
    # historical Solver-V2 lookup itself was order-tolerant; new SpinCore code
    # should consume canonical exact-class keys rather than depend on card order.
    from spincore.flop_abstraction import decode_spin_card_id

    return "".join(card_text(decode_spin_card_id(card_id)) for card_id in key)


def _iso_text(key: tuple[tuple[int, int], ...]) -> str:
    return "".join(card_text(RSCard(rank=rank, suit=suit)) for rank, suit in key)


def load_legacy_mapping(path: str | Path) -> dict[tuple[int, int, int], str]:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("legacy 184 mapping must be a JSON object")

    out: dict[tuple[int, int, int], str] = {}
    for key_text, representative_text in raw.items():
        key = _physical_key(str(key_text))
        representative = "".join(card_text(card) for card in parse_flop_text(str(representative_text)))
        prior = out.get(key)
        if prior is not None and prior != representative:
            raise ValueError(f"conflicting legacy mapping for physical flop {key_text!r}")
        out[key] = representative
    return out


def exact_class_groups(
    physical_mapping: dict[tuple[int, int, int], str],
) -> dict[tuple[tuple[int, int], ...], list[tuple[tuple[int, int, int], str]]]:
    groups: defaultdict[
        tuple[tuple[int, int], ...],
        list[tuple[tuple[int, int, int], str]],
    ] = defaultdict(list)
    for physical, representative in physical_mapping.items():
        groups[suit_isomorphic_key(physical)].append((physical, representative))
    return dict(groups)


def derive_exact_class_mapping(
    physical_mapping: dict[tuple[int, int, int], str],
    *,
    mode: RepairMode,
) -> dict[str, str]:
    """Produce a suit-invariant exact-class -> legacy-bucket mapping.

    canonical_input:
      For each exact suit-isomorphic class, use the historical bucket assigned
      to the deterministic canonical physical suit spelling of that class.
      This keeps the historical 184 labels while making input suit spelling
      irrelevant by construction.

    majority_min_change:
      Assign the exact class to the historical representative used by the
      largest number of its physical suit spellings. This minimizes the number
      of physical historical assignments that would change. Exact ties use a
      stable lexical representative name.
    """
    if mode not in ("canonical_input", "majority_min_change"):
        raise ValueError(f"unknown repair mode: {mode}")

    groups = exact_class_groups(physical_mapping)
    out: dict[str, str] = {}
    for iso_key, members in groups.items():
        if mode == "canonical_input":
            canonical_physical = tuple(
                sorted(encode_spin_card_id(RSCard(rank=rank, suit=suit)) for rank, suit in iso_key)
            )
            if canonical_physical not in physical_mapping:
                raise ValueError(
                    "canonical-input repair requires the canonical physical suit spelling "
                    f"to exist in the legacy mapping: {_iso_text(iso_key)}"
                )
            representative = physical_mapping[canonical_physical]
        else:
            counts = Counter(representative for _, representative in members)
            max_count = max(counts.values())
            representative = min(rep for rep, count in counts.items() if count == max_count)
        out[_iso_text(iso_key)] = representative
    return dict(sorted(out.items()))


def expand_exact_class_mapping(
    physical_mapping: dict[tuple[int, int, int], str],
    exact_mapping: dict[str, str],
) -> dict[tuple[int, int, int], str]:
    out: dict[tuple[int, int, int], str] = {}
    for physical in physical_mapping:
        iso_text = _iso_text(suit_isomorphic_key(physical))
        out[physical] = exact_mapping[iso_text]
    return out


def _median(values: list[int]) -> float:
    if not values:
        raise ValueError("median of empty sequence")
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return float(ordered[n // 2])
    return 0.5 * float(ordered[n // 2 - 1] + ordered[n // 2])


def audit_descendant(path: str | Path, *, mode: RepairMode) -> dict[str, object]:
    path = Path(path)
    physical = load_legacy_mapping(path)
    exact = derive_exact_class_mapping(physical, mode=mode)
    expanded = expand_exact_class_mapping(physical, exact)

    changed = sum(1 for flop, representative in physical.items() if expanded[flop] != representative)
    bucket_counts = Counter(expanded.values())
    original_representatives = set(physical.values())
    repaired_representatives = set(expanded.values())
    bucket_sizes = list(bucket_counts.values())

    split_check: defaultdict[tuple[tuple[int, int], ...], set[str]] = defaultdict(set)
    for flop, representative in expanded.items():
        split_check[suit_isomorphic_key(flop)].add(representative)
    split_count = sum(1 for representatives in split_check.values() if len(representatives) > 1)

    return {
        "schema": "SPINCORE_R7_5_LEGACY184_DESCENDANT_AUDIT_V1",
        "mode": mode,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "physical_flops": len(physical),
        "exact_suit_isomorphic_classes": len(exact),
        "active_representatives": len(repaired_representatives),
        "physical_assignments_changed_from_legacy": changed,
        "physical_assignments_changed_fraction": changed / max(1, len(physical)),
        "suit_isomorphic_classes_split": split_count,
        "suit_permutation_invariance_pass": split_count == 0,
        "representatives_lost": sorted(original_representatives - repaired_representatives),
        "bucket_size_min": min(bucket_sizes),
        "bucket_size_median": _median(bucket_sizes),
        "bucket_size_mean": sum(bucket_sizes) / len(bucket_sizes),
        "bucket_size_max": max(bucket_sizes),
    }


def write_exact_class_mapping(
    source_path: str | Path,
    output_path: str | Path,
    *,
    mode: RepairMode,
) -> dict[str, object]:
    source_path = Path(source_path)
    output_path = Path(output_path)
    physical = load_legacy_mapping(source_path)
    exact = derive_exact_class_mapping(physical, mode=mode)
    payload = {
        "schema": "SPINCORE_R7_5_FLOP_EXACT_CLASS_TO_LEGACY_BUCKET_V1",
        "mode": mode,
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "exact_class_count": len(exact),
        "mapping": exact,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
