from __future__ import annotations

"""Legacy-derived realistic SpinGo scenario sampler.

This module restores the empirically weighted tournament-state distribution from
`Tentativas anteriores de SpinGo.zip` without importing the historical trainer.
It is intended to replace fixed-10/20 scenario generation for global strategic
training/evaluation.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .solver import Episode

_DATA = json.loads(Path(__file__).with_name("legacy_scenario_data.json").read_text(encoding="utf-8"))

BLIND_LEVELS = tuple(tuple(int(x) for x in row) for row in _DATA["BLIND_LEVELS"])
REAL_BLIND_WEIGHTS_3P = tuple(float(x) for x in _DATA["REAL_BLIND_WEIGHTS_3P"])
REAL_BLIND_WEIGHTS_HU = tuple(float(x) for x in _DATA["REAL_BLIND_WEIGHTS_HU"])
REAL_STACK_TABLE_3P = tuple(_DATA["REAL_STACK_TABLE_3P"])
REAL_STACK_TABLE_HU = tuple(_DATA["REAL_STACK_TABLE_HU"])


def _normalize(weights) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full(len(arr), 1.0 / len(arr), dtype=np.float64)
    return arr / total


def _sample_stack_from_table(rng: np.random.Generator, table, blind_key: str) -> int:
    weights = []
    ranges = []
    for row in table:
        lo, hi = (int(x) for x in row["range"])
        if lo < 0 or hi < lo:
            raise ValueError(f"invalid legacy stack range {row['range']!r}")
        weights.append(max(0.0, float(row.get("pct", {}).get(blind_key, 0.0) or 0.0)))
        ranges.append((lo, hi))
    probs = _normalize(weights)
    idx = int(rng.choice(len(ranges), p=probs))
    lo, hi = ranges[idx]
    return lo if lo == hi else int(rng.integers(lo, hi + 1))


@dataclass(frozen=True)
class LegacyScenarioConfig:
    total_chips: int = 1500
    heads_up_prob: float = 0.4548

    def __post_init__(self) -> None:
        if self.total_chips <= 2:
            raise ValueError("total_chips must be > 2")
        if not (0.0 <= self.heads_up_prob <= 1.0):
            raise ValueError("heads_up_prob must be in [0,1]")


class LegacyScenarioSampler:
    """Sample one-hand SpinGo states from the user's historical real-hand model.

    Preserved semantics:
    - 3H/HU mixture;
    - separate empirical blind weights by mode;
    - blind-conditioned stack distributions;
    - 1500 total chips by default;
    - random live-seat assignment and dealer;
    - one dead seat in true HU;
    - late 3H stack-table fallback to 60/120, matching the legacy code.
    """

    def __init__(self, *, seed: int = 0, config: LegacyScenarioConfig | None = None):
        self.config = config or LegacyScenarioConfig()
        self.rng = np.random.default_rng(int(seed))
        self._blind_probs_3p = _normalize(REAL_BLIND_WEIGHTS_3P)
        self._blind_probs_hu = _normalize(REAL_BLIND_WEIGHTS_HU)

    def _sample_blinds(self, game_is_hu: bool) -> tuple[int, int, int]:
        probs = self._blind_probs_hu if game_is_hu else self._blind_probs_3p
        idx = int(self.rng.choice(len(BLIND_LEVELS), p=probs))
        sb, bb = BLIND_LEVELS[idx]
        return idx, int(sb), int(bb)

    def _sample_hu_stacks(self, blind_key: str) -> tuple[tuple[int, int, int], tuple[int, ...]]:
        total = self.config.total_chips
        for _ in range(50):
            a = _sample_stack_from_table(self.rng, REAL_STACK_TABLE_HU, blind_key)
            if 1 <= a <= total - 1:
                b = total - a
                if self.rng.random() < 0.5:
                    a, b = b, a
                dead = int(self.rng.integers(0, 3))
                alive = [x for x in range(3) if x != dead]
                alive = [int(x) for x in self.rng.permutation(alive)]
                stacks = [0, 0, 0]
                stacks[alive[0]], stacks[alive[1]] = int(a), int(b)
                return tuple(stacks), (dead,)
        a = int(self.rng.integers(1, total))
        b = total - a
        dead = int(self.rng.integers(0, 3))
        alive = [x for x in range(3) if x != dead]
        if self.rng.random() < 0.5:
            a, b = b, a
        stacks = [0, 0, 0]
        stacks[alive[0]], stacks[alive[1]] = a, b
        return tuple(stacks), (dead,)

    def _sample_3p_stacks(self, blind_key: str, bb: int) -> tuple[int, int, int]:
        total = self.config.total_chips
        table_key = "60/120" if blind_key in {"80/160", "100/200"} else blind_key
        for _ in range(50):
            raw = [max(1, _sample_stack_from_table(self.rng, REAL_STACK_TABLE_3P, table_key)) for _ in range(3)]
            raw_sum = sum(raw)
            scaled = [int(round(x * total / raw_sum)) for x in raw]
            scaled[max(range(3), key=scaled.__getitem__)] += total - sum(scaled)
            for i, value in enumerate(scaled):
                if value <= 0:
                    donor = max(range(3), key=scaled.__getitem__)
                    if scaled[donor] > 1:
                        scaled[donor] -= 1
                        scaled[i] = 1
            if sum(scaled) == total and min(scaled) >= bb:
                return tuple(int(scaled[int(i)]) for i in self.rng.permutation(3))
        # Defensive fallback; should be rare. Keep total exact and all players alive.
        while True:
            a = int(self.rng.integers(bb, total - 2 * bb + 1))
            b = int(self.rng.integers(bb, total - a - bb + 1))
            c = total - a - b
            if c >= bb:
                vals = [a, b, c]
                return tuple(int(vals[int(i)]) for i in self.rng.permutation(3))

    def sample_episode(self, *, force_domain: str | None = None) -> Episode:
        if force_domain not in (None, "THREE_HANDED", "TRUE_HEADS_UP"):
            raise ValueError("force_domain must be THREE_HANDED, TRUE_HEADS_UP or None")
        if force_domain == "TRUE_HEADS_UP":
            game_is_hu = True
        elif force_domain == "THREE_HANDED":
            game_is_hu = False
        else:
            game_is_hu = bool(self.rng.random() < self.config.heads_up_prob)

        blind_index, sb, bb = self._sample_blinds(game_is_hu)
        blind_key = f"{sb}/{bb}"
        if game_is_hu:
            stacks, dead_players = self._sample_hu_stacks(blind_key)
            live = [i for i, stack in enumerate(stacks) if stack > 0]
        else:
            stacks = self._sample_3p_stacks(blind_key, bb)
            dead_players = ()
            live = [0, 1, 2]
        dealer = int(self.rng.choice(live))
        return Episode(
            total_chips=self.config.total_chips,
            game_is_hu=game_is_hu,
            blind_index=blind_index,
            small_blind=sb,
            big_blind=bb,
            stacks=stacks,
            dealer_id=dealer,
            dead_players=dead_players,
        )

    def sample(self, *, force_domain: str | None = None) -> dict[str, Any]:
        e = self.sample_episode(force_domain=force_domain)
        return {
            "total_chips": e.total_chips,
            "game_is_hu": e.game_is_hu,
            "blind_index": e.blind_index,
            "sb": e.small_blind,
            "bb": e.big_blind,
            "stacks": list(e.stacks),
            "dealer_id": e.dealer_id,
            "dead_players": list(e.dead_players),
        }
