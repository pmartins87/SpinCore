from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Sequence

from spincore_nn.codec import decode_spnniv1


_RANKS = "23456789TJQKA"
_HAND_RE = re.compile(r"^(22|33|44|55|66|77|88|99|TT|JJ|QQ|KK|AA|[2-9TJQKA]{2}[so])$")


def _card_from_token(token: int) -> tuple[int, int]:
    """Return (rank, suit) from SPNNIV1 token; token zero means no card."""
    token = int(token)
    if token <= 0 or token > 52:
        raise ValueError("hole-card token must be in 1..52")
    card_id = token - 1
    return 2 + card_id // 4, card_id % 4


def _rank_char(rank: int) -> str:
    if rank < 2 or rank > 14:
        raise ValueError("invalid card rank")
    return _RANKS[rank - 2]


def normalize_hand_class(value: str) -> str:
    raw = str(value).strip().upper()
    if len(raw) == 3:
        raw = raw[:2] + raw[2].lower()
    if not _HAND_RE.fullmatch(raw):
        raise ValueError(f"invalid Hold'em hand class: {value!r}")
    if len(raw) == 2:
        return raw
    r1 = _RANKS.index(raw[0])
    r2 = _RANKS.index(raw[1])
    if r1 == r2:
        raise ValueError("pair hand class must not carry suitedness suffix")
    if r1 < r2:
        raw = raw[1] + raw[0] + raw[2]
    return raw


def hole_class_from_neural_bytes(observation: bytes) -> str:
    decoded = decode_spnniv1(observation)
    r0, s0 = _card_from_token(decoded.cards[0])
    r1, s1 = _card_from_token(decoded.cards[1])
    c0, c1 = _rank_char(r0), _rank_char(r1)
    if r0 == r1:
        return c0 + c1
    if r0 < r1:
        r0, r1, c0, c1, s0, s1 = r1, r0, c1, c0, s1, s0
    return c0 + c1 + ("s" if s0 == s1 else "o")


@dataclass(frozen=True)
class SentinelStateCandidate:
    deck_seed: int
    actor: int
    hand_class: str
    action_prefix: tuple[int, ...]
    legal_actions: tuple[int, ...]
    observation_sha256: str
    observation: bytes

    def __post_init__(self) -> None:
        if self.deck_seed < 0:
            raise ValueError("deck_seed must be non-negative")
        object.__setattr__(self, "hand_class", normalize_hand_class(self.hand_class))
        if hashlib.sha256(self.observation).hexdigest() != self.observation_sha256:
            raise ValueError("sentinel observation SHA-256 mismatch")


def find_sentinel_state(
    solver,
    episode,
    *,
    target_hand_class: str,
    target_actor: int | None = None,
    action_prefix: Sequence[int] = (),
    seed_start: int = 0,
    seed_stop: int = 100_000,
) -> SentinelStateCandidate:
    """Find the first deterministic deck seed matching a named Hold'em hand class.

    The state identity is `(episode, action_prefix, deck_seed)`. No production
    policy is queried, so catalog construction cannot tune itself to policy
    outputs. The action prefix uses frozen abstract action IDs.
    """

    target = normalize_hand_class(target_hand_class)
    prefix = tuple(int(a) for a in action_prefix)
    if seed_start < 0 or seed_stop <= seed_start:
        raise ValueError("invalid deterministic seed interval")
    if target_actor is not None and target_actor not in (0, 1, 2):
        raise ValueError("target_actor must be seat 0, 1 or 2")
    if any(a < 0 or a >= 6 for a in prefix):
        raise ValueError("action_prefix contains action outside frozen abstraction")

    first_prefix_error: Exception | None = None
    for deck_seed in range(int(seed_start), int(seed_stop)):
        state = solver.create(episode, deck_seed)
        try:
            try:
                for action in prefix:
                    if action not in state.legal_actions():
                        raise ValueError(
                            f"action_prefix contains illegal action {action} at prefix state"
                        )
                    state.apply(action)
            except Exception as exc:
                first_prefix_error = exc
                break
            if state.terminal:
                raise ValueError("action_prefix resolves to a terminal state")
            actor = int(state.actor)
            if target_actor is not None and actor != int(target_actor):
                continue
            observation = state.neural_bytes()
            if hole_class_from_neural_bytes(observation) != target:
                continue
            legal = tuple(int(a) for a in state.legal_actions())
            return SentinelStateCandidate(
                deck_seed=int(deck_seed),
                actor=actor,
                hand_class=target,
                action_prefix=prefix,
                legal_actions=legal,
                observation_sha256=hashlib.sha256(observation).hexdigest(),
                observation=observation,
            )
        finally:
            state.close()

    if first_prefix_error is not None:
        raise ValueError("invalid sentinel action_prefix") from first_prefix_error
    raise LookupError(
        f"no {target} sentinel state found in deck seeds [{seed_start}, {seed_stop})"
    )
