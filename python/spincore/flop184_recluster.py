from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from spincore.flop_abstraction import (
    RSCard,
    STRAIGHT_WINDOWS,
    all_physical_flops,
    card_text,
    decode_spin_card_id,
    encode_spin_card_id,
    suit_isomorphic_key,
)

H3_SCHEMA = "SPINCORE_R7_5_H3_RECLUSTERED_184_V1"
H3_FEATURE_SCHEMA = "SPINCORE_R7_5_H3_OBJECTIVE_FLOP_FEATURES_V1"
H3_TARGET_CLUSTERS = 184


@dataclass(frozen=True, order=True)
class HardStratum:
    suit_texture: str
    rank_shape: str
    max_straight_window_occupancy: int


def _iso_text(key: tuple[tuple[int, int], ...]) -> str:
    return "".join(card_text(RSCard(rank=rank, suit=suit)) for rank, suit in key)


def exact_iso_flop_keys() -> list[tuple[tuple[int, int], ...]]:
    """Return all 1,755 exact flop classes modulo global suit renaming."""
    return sorted({suit_isomorphic_key(flop) for flop in all_physical_flops()})


def hard_stratum(key: tuple[tuple[int, int], ...]) -> HardStratum:
    ranks = [rank for rank, _ in key]
    suits = [suit for _, suit in key]
    suit_max = max(Counter(suits).values())
    suit_texture = {1: "rainbow", 2: "two_tone", 3: "monotone"}[suit_max]
    distinct_ranks = len(set(ranks))
    rank_shape = {3: "unpaired", 2: "paired", 1: "trips"}[distinct_ranks]
    rank_set = set(ranks)
    max_occ = max(len(rank_set.intersection(window)) for window in STRAIGHT_WINDOWS)
    return HardStratum(suit_texture, rank_shape, max_occ)


def _feature_groups(key: tuple[tuple[int, int], ...]) -> tuple[list[float], ...]:
    """Objective flop-only features used by H3.

    No historical action, range, handlist or learned-policy result is included.
    The four groups are normalized independently so rank identity, suit
    structure, connectivity and future-turn dynamics each retain influence.
    """
    cards = list(key)
    ranks = [rank for rank, _ in cards]
    suits = [suit for _, suit in cards]
    rank_counts = Counter(ranks)
    distinct = sorted(rank_counts)

    sorted_ranks = sorted(ranks, reverse=True)
    rank_group: list[float] = [rank_counts.get(rank, 0) / 3.0 for rank in range(2, 15)]
    rank_group.extend(
        [
            sorted_ranks[0] / 14.0,
            sorted_ranks[1] / 14.0,
            sorted_ranks[2] / 14.0,
            (max(ranks) - min(ranks)) / 12.0,
            float(len(distinct) == 3),
            float(len(distinct) == 2),
            float(len(distinct) == 1),
            sum(rank >= 10 for rank in ranks) / 3.0,
            sum(rank == 14 for rank in ranks) / 3.0,
            sum(rank <= 6 for rank in ranks) / 3.0,
        ]
    )

    suit_counts = sorted(Counter(suits).values(), reverse=True)
    suit_group: list[float] = [
        float(suit_counts[0] == 1),
        float(suit_counts[0] == 2),
        float(suit_counts[0] == 3),
    ]
    for left, right in ((0, 1), (0, 2), (1, 2)):
        suit_group.append(float(cards[left][1] == cards[right][1]))
    suited_ranks = sorted(
        ranks[left] / 14.0
        for left, right in ((0, 1), (0, 2), (1, 2))
        if suits[left] == suits[right]
    )
    suited_ranks.extend([0.0] * (3 - len(suited_ranks)))
    suit_group.extend(suited_ranks[:3])

    rank_set = set(distinct)
    occupancies = [len(rank_set.intersection(window)) for window in STRAIGHT_WINDOWS]
    pairwise_gaps = [abs(left - right) for left, right in __import__("itertools").combinations(distinct, 2)]
    connectivity_group = [
        max(occupancies) / 5.0,
        sum(value == 3 for value in occupancies) / float(len(occupancies)),
        sum(value >= 2 for value in occupancies) / float(len(occupancies)),
        sum(gap == 1 for gap in pairwise_gaps) / 3.0,
        sum(gap == 2 for gap in pairwise_gaps) / 3.0,
        (min(pairwise_gaps) if pairwise_gaps else 0) / 12.0,
    ]

    used = set(cards)
    remaining = [
        (rank, suit)
        for rank in range(2, 15)
        for suit in range(4)
        if (rank, suit) not in used
    ]
    transition = Counter()
    for turn_rank, turn_suit in remaining:
        rr = ranks + [turn_rank]
        ss = suits + [turn_suit]
        rr_counts = Counter(rr)
        ss_counts = Counter(ss)
        rr_set = set(rr)
        max_window = max(len(rr_set.intersection(window)) for window in STRAIGHT_WINDOWS)
        if turn_rank in ranks:
            transition["pair_rank"] += 1
        if turn_rank > max(ranks):
            transition["over"] += 1
        if turn_rank < min(ranks):
            transition["under"] += 1
        if max(rr_counts.values()) >= 3:
            transition["tripsplus"] += 1
        if sum(value >= 2 for value in rr_counts.values()) >= 2:
            transition["two_pair_board"] += 1
        if max(ss_counts.values()) >= 3:
            transition["three_suit"] += 1
        if max(ss_counts.values()) >= 4:
            transition["four_suit"] += 1
        if max_window >= 4:
            transition["four_straight_window"] += 1
        if max_window >= 3:
            transition["three_straight_window"] += 1

    n_turns = float(len(remaining))
    future_group = [
        transition[name] / n_turns
        for name in (
            "pair_rank",
            "over",
            "under",
            "tripsplus",
            "two_pair_board",
            "three_suit",
            "four_suit",
            "four_straight_window",
            "three_straight_window",
        )
    ]
    return rank_group, suit_group, connectivity_group, future_group


def _feature_matrix(keys: list[tuple[tuple[int, int], ...]]) -> np.ndarray:
    grouped = [_feature_groups(key) for key in keys]
    normalized_groups: list[np.ndarray] = []
    for group_index in range(4):
        matrix = np.asarray([row[group_index] for row in grouped], dtype=np.float64)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < 1.0e-12, 1.0, std)
        z = (matrix - mean) / std
        z /= math.sqrt(float(matrix.shape[1]))
        normalized_groups.append(z)
    return np.concatenate(normalized_groups, axis=1)


def _allocate_clusters(stratum_counts: Counter[HardStratum], total_clusters: int) -> dict[HardStratum, int]:
    if total_clusters < len(stratum_counts):
        raise ValueError("cluster target smaller than hard-stratum count")
    if total_clusters > sum(stratum_counts.values()):
        raise ValueError("cluster target larger than exact-class count")

    allocation = {stratum: 1 for stratum in stratum_counts}
    remaining = total_clusters - len(stratum_counts)
    total = sum(stratum_counts.values())
    exact = {
        stratum: remaining * stratum_counts[stratum] / float(total)
        for stratum in stratum_counts
    }
    floors = {stratum: int(math.floor(value)) for stratum, value in exact.items()}
    for stratum in stratum_counts:
        allocation[stratum] += floors[stratum]
    left = total_clusters - sum(allocation.values())
    order = sorted(
        stratum_counts,
        key=lambda stratum: (-(exact[stratum] - floors[stratum]), stratum),
    )
    for stratum in order[:left]:
        allocation[stratum] += 1

    if any(allocation[stratum] > stratum_counts[stratum] for stratum in stratum_counts):
        raise ValueError("proportional allocation overfilled a hard stratum")
    return allocation


def _cluster_subset(
    global_indices: Iterable[int],
    cluster_count: int,
    *,
    keys: list[tuple[tuple[int, int], ...]],
    features: np.ndarray,
) -> tuple[list[int], dict[int, int]]:
    indices = np.asarray(sorted(global_indices, key=lambda index: keys[index]), dtype=np.int64)
    local = features[indices]
    n = int(indices.size)
    if cluster_count >= n:
        medoids = [int(index) for index in indices]
        return medoids, {int(index): int(index) for index in indices}

    center = local.mean(axis=0)
    d_center = ((local - center) ** 2).sum(axis=1)
    best_value = float(d_center.min())
    first_candidates = [
        position for position in range(n)
        if abs(float(d_center[position]) - best_value) <= 1.0e-12
    ]
    first = min(first_candidates, key=lambda position: keys[int(indices[position])])

    medoid_positions = [first]
    selected = {first}
    min_distance = ((local - local[first]) ** 2).sum(axis=1)

    # Deterministic farthest-first initialization.
    for _ in range(1, cluster_count):
        available = [position for position in range(n) if position not in selected]
        farthest = max(float(min_distance[position]) for position in available)
        candidates = [
            position for position in available
            if abs(float(min_distance[position]) - farthest) <= 1.0e-12
        ]
        chosen = min(candidates, key=lambda position: keys[int(indices[position])])
        medoid_positions.append(chosen)
        selected.add(chosen)
        distance = ((local - local[chosen]) ** 2).sum(axis=1)
        min_distance = np.minimum(min_distance, distance)

    medoid_positions = sorted(medoid_positions, key=lambda position: keys[int(indices[position])])

    # PAM-style within-cluster medoid refinement. Lexicographically ordered
    # medoids make nearest-medoid ties deterministic.
    for _ in range(32):
        medoid_matrix = local[np.asarray(medoid_positions, dtype=np.int64)]
        distances = ((local[:, None, :] - medoid_matrix[None, :, :]) ** 2).sum(axis=2)
        assignment = distances.argmin(axis=1)
        updated: list[int] = []
        for cluster_index in range(cluster_count):
            members = np.where(assignment == cluster_index)[0]
            if members.size == 0:
                raise RuntimeError("deterministic H3 produced an empty cluster")
            member_matrix = local[members]
            pairwise = ((member_matrix[:, None, :] - member_matrix[None, :, :]) ** 2).sum(axis=2)
            sums = pairwise.sum(axis=1)
            minimum = float(sums.min())
            candidates = members[np.where(np.abs(sums - minimum) <= 1.0e-12)[0]]
            updated.append(
                min((int(position) for position in candidates), key=lambda position: keys[int(indices[position])])
            )
        updated = sorted(set(updated), key=lambda position: keys[int(indices[position])])
        if len(updated) != cluster_count:
            raise RuntimeError("deterministic H3 medoid collapse")
        if updated == medoid_positions:
            break
        medoid_positions = updated

    medoid_matrix = local[np.asarray(medoid_positions, dtype=np.int64)]
    distances = ((local[:, None, :] - medoid_matrix[None, :, :]) ** 2).sum(axis=2)
    assignment = distances.argmin(axis=1)
    medoids = [int(indices[position]) for position in medoid_positions]
    mapped = {
        int(indices[position]): medoids[int(assignment[position])]
        for position in range(n)
    }
    return medoids, mapped


def build_h3_mapping(target_clusters: int = H3_TARGET_CLUSTERS) -> dict[str, str]:
    keys = exact_iso_flop_keys()
    features = _feature_matrix(keys)
    stratum_by_index = {index: hard_stratum(key) for index, key in enumerate(keys)}
    counts = Counter(stratum_by_index.values())
    allocation = _allocate_clusters(counts, target_clusters)

    mapping_indices: dict[int, int] = {}
    medoids: set[int] = set()
    for stratum in sorted(counts):
        members = [index for index in range(len(keys)) if stratum_by_index[index] == stratum]
        local_medoids, local_mapping = _cluster_subset(
            members,
            allocation[stratum],
            keys=keys,
            features=features,
        )
        medoids.update(local_medoids)
        mapping_indices.update(local_mapping)

    if len(mapping_indices) != len(keys):
        raise RuntimeError("H3 mapping does not cover every exact class")
    if len(medoids) != target_clusters:
        raise RuntimeError("H3 did not produce the requested medoid count")

    mapping = {
        _iso_text(keys[index]): _iso_text(keys[medoid])
        for index, medoid in mapping_indices.items()
    }
    return dict(sorted(mapping.items()))


def mapping_sha256(mapping: dict[str, str]) -> str:
    canonical = json.dumps(dict(sorted(mapping.items())), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return float(ordered[n // 2])
    return 0.5 * float(ordered[n // 2 - 1] + ordered[n // 2])


def audit_h3_mapping(mapping: dict[str, str]) -> dict[str, object]:
    keys = exact_iso_flop_keys()
    key_by_text = {_iso_text(key): key for key in keys}
    expected = set(key_by_text)
    actual = set(mapping)
    representatives = set(mapping.values())

    hard_stratum_mismatches = 0
    for source, representative in mapping.items():
        if source not in key_by_text or representative not in key_by_text:
            continue
        if hard_stratum(key_by_text[source]) != hard_stratum(key_by_text[representative]):
            hard_stratum_mismatches += 1

    exact_bucket_sizes = Counter(mapping.values())
    physical_bucket_sizes: Counter[str] = Counter()
    for physical in all_physical_flops():
        source = _iso_text(suit_isomorphic_key(physical))
        representative = mapping.get(source)
        if representative is not None:
            physical_bucket_sizes[representative] += 1

    exact_sizes = list(exact_bucket_sizes.values())
    physical_sizes = list(physical_bucket_sizes.values())
    return {
        "schema": H3_SCHEMA,
        "feature_schema": H3_FEATURE_SCHEMA,
        "mapping_sha256": mapping_sha256(mapping),
        "physical_flops": sum(physical_sizes),
        "exact_suit_isomorphic_classes": len(mapping),
        "active_representatives": len(representatives),
        "missing_exact_classes": len(expected - actual),
        "extra_exact_classes": len(actual - expected),
        "hard_stratum_mismatches": hard_stratum_mismatches,
        "suit_permutation_invariance_pass": actual == expected,
        "complete_1755_pass": len(mapping) == 1755 and actual == expected,
        "complete_22100_pass": sum(physical_sizes) == 22100,
        "representative_count_pass": len(representatives) == H3_TARGET_CLUSTERS,
        "hard_stratum_homogeneity_pass": hard_stratum_mismatches == 0,
        "exact_bucket_size_min": min(exact_sizes),
        "exact_bucket_size_median": _median(exact_sizes),
        "exact_bucket_size_mean": sum(exact_sizes) / len(exact_sizes),
        "exact_bucket_size_max": max(exact_sizes),
        "physical_bucket_size_min": min(physical_sizes),
        "physical_bucket_size_median": _median(physical_sizes),
        "physical_bucket_size_mean": sum(physical_sizes) / len(physical_sizes),
        "physical_bucket_size_max": max(physical_sizes),
    }


def build_h3_payload() -> dict[str, object]:
    mapping = build_h3_mapping()
    return {
        "schema": H3_SCHEMA,
        "feature_schema": H3_FEATURE_SCHEMA,
        "target_clusters": H3_TARGET_CLUSTERS,
        "audit": audit_h3_mapping(mapping),
        "mapping": mapping,
    }
