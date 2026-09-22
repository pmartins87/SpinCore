from __future__ import annotations

"""Lossless observable-state view for the offline DeepCrusher oracle.

DeepCrusher/OpenPPL makes decisions from the current player's observable poker
state plus its own persistent user/memory variables.  SpinCore's SPNNIV3 carrier
already transports that public/private poker information losslessly up to true
suit/seat symmetries.  This module decodes it without changing strategy logic.
"""

from dataclasses import dataclass
import ctypes as C
import struct
from typing import Iterator

MAGIC = b"SPNNIV3\x00"
FIXED_BYTES = 120
EVENT_BYTES = 20

DOMAIN_THREE_HANDED = 0
DOMAIN_TRUE_HEADS_UP = 1
STREET_PREFLOP = 0
STREET_FLOP = 1
STREET_TURN = 2
STREET_RIVER = 3

ACTION_FOLD = 0
ACTION_CHECK = 1
ACTION_CALL = 2
ACTION_BET_TO = 3
ACTION_RAISE_TO = 4
ACTION_ALL_IN = 5

RANK_CHAR = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}


@dataclass(frozen=True)
class PublicActionEvent:
    actor_rel: int
    street: int
    action_type: int
    forced: bool
    paid_bb: float
    resulting_commitment_bb: float
    pot_before_bb: float
    pot_after_bb: float


@dataclass(frozen=True)
class DeepCrusherStateView:
    # Hero-relative current-state categories.
    domain: int
    street: int
    dealer_rel: int
    small_blind_rel: int
    big_blind_rel: int
    live_count: int
    visible_board: int
    statuses: tuple[int, int, int]

    # Ranks: hero hole 0/1, flop0/1/2, turn, river. 0 = unrevealed public.
    ranks: tuple[int, int, int, int, int, int, int]
    # Pairwise same-suit upper triangle over the seven rank slots.
    same_suit: tuple[int, ...]

    pot_bb: float
    to_call_bb: float
    current_bet_bb: float
    stacks_bb: tuple[float, float, float]
    street_commitments_bb: tuple[float, float, float]
    total_commitments_bb: tuple[float, float, float]
    small_blind_bb: float
    blind_index: int
    min_raise_to_bb: float
    max_raise_to_bb: float
    primitive_legal: tuple[bool, bool, bool, bool, bool, bool]
    history: tuple[PublicActionEvent, ...]
    # OpenHoldem suit ids (hearts=0, diamonds=1, clubs=2, spades=3) for
    # hero-hole0/1, flop0/1/2, turn, river. -1 = unrevealed.
    # SPNNIV3 itself intentionally stores only suit equivalence; state_view()
    # enriches this field from the solver's read-only deal snapshot when the
    # diagnostic ABI is available.
    exact_suits: tuple[int, int, int, int, int, int, int] | None = None

    @property
    def hero_hole_ranks(self) -> tuple[int, int]:
        return self.ranks[0], self.ranks[1]

    @property
    def board_ranks(self) -> tuple[int, ...]:
        return tuple(rank for rank in self.ranks[2:] if rank > 0)

    @property
    def hero_hand_class(self) -> str:
        """Canonical 169-class name used by OpenPPL list_* ranges."""
        left, right = self.hero_hole_ranks
        if left not in RANK_CHAR or right not in RANK_CHAR:
            raise ValueError(f"invalid hero hole ranks: {(left, right)}")
        hi, lo = max(left, right), min(left, right)
        if hi == lo:
            return RANK_CHAR[hi] + RANK_CHAR[lo]
        suffix = "s" if self.hole_suited else "o"
        return RANK_CHAR[hi] + RANK_CHAR[lo] + suffix


    @property
    def is_true_hu(self) -> bool:
        return self.domain == DOMAIN_TRUE_HEADS_UP

    @property
    def is_three_handed_origin(self) -> bool:
        return self.domain == DOMAIN_THREE_HANDED

    @property
    def hero_is_dealer(self) -> bool:
        return self.dealer_rel == 0

    @property
    def hero_is_small_blind(self) -> bool:
        return self.small_blind_rel == 0

    @property
    def hero_is_big_blind(self) -> bool:
        return self.big_blind_rel == 0

    @property
    def hero_stack_bb(self) -> float:
        return self.stacks_bb[0]

    @property
    def effective_stack_bb(self) -> float:
        live_opponents = [
            self.stacks_bb[rel] + self.total_commitments_bb[rel]
            for rel in (1, 2)
            if self.statuses[rel] != 2
        ]
        hero_total = self.stacks_bb[0] + self.total_commitments_bb[0]
        if not live_opponents:
            return hero_total
        return min(hero_total, max(live_opponents))

    @property
    def can_fold(self) -> bool:
        return self.primitive_legal[0]

    @property
    def can_check(self) -> bool:
        return self.primitive_legal[1]

    @property
    def can_call(self) -> bool:
        return self.primitive_legal[2]

    @property
    def can_bet(self) -> bool:
        return self.primitive_legal[3]

    @property
    def can_raise(self) -> bool:
        return self.primitive_legal[4]

    @property
    def can_all_in(self) -> bool:
        return self.primitive_legal[5]

    def same_suit_pair(self, left: int, right: int) -> bool:
        if not 0 <= left < right < 7:
            raise ValueError("same_suit_pair requires 0 <= left < right < 7")
        index = 0
        for i in range(7):
            for j in range(i + 1, 7):
                if i == left and j == right:
                    return bool(self.same_suit[index])
                index += 1
        raise AssertionError("unreachable pair index")

    @property
    def hole_suited(self) -> bool:
        return self.same_suit_pair(0, 1)

    @property
    def canonical_suits(self) -> tuple[int, int, int, int, int, int, int]:
        """Reconstruct suit-equivalence classes from SPNNIV3.

        SPNNIV3 intentionally removes absolute suit names but preserves every
        pairwise same-suit relation. OpenPPL strategy logic is suit-permutation
        invariant, so a deterministic canonical labelling is sufficient for
        suit counts, suited predicates and hand/board expressions.
        Unrevealed public-card slots use -1.
        """
        visible = [rank > 0 for rank in self.ranks]
        parent = list(range(7))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for left in range(7):
            if not visible[left]:
                continue
            for right in range(left + 1, 7):
                if visible[right] and self.same_suit_pair(left, right):
                    union(left, right)

        labels: dict[int, int] = {}
        out: list[int] = []
        for index in range(7):
            if not visible[index]:
                out.append(-1)
                continue
            root = find(index)
            if root not in labels:
                labels[root] = len(labels)
            out.append(labels[root])
        return tuple(out)  # type: ignore[return-value]

    @property
    def openholdem_suits(self) -> tuple[int, int, int, int, int, int, int]:
        """Best available OpenHoldem suit ids for the seven card slots."""
        if self.exact_suits is not None:
            return self.exact_suits
        # Canonical labels are only a fallback for unit/static contexts. The
        # canonical benchmark must use state_view() with the deal-snapshot ABI
        # so suit-tie semantics (for example tsuitcommon) are exact.
        return self.canonical_suits

    def voluntary_history(self, *, street: int | None = None) -> tuple[PublicActionEvent, ...]:
        return tuple(
            event
            for event in self.history
            if not event.forced and (street is None or event.street == street)
        )

    def hero_history(self, *, street: int | None = None) -> tuple[PublicActionEvent, ...]:
        return tuple(
            event
            for event in self.voluntary_history(street=street)
            if event.actor_rel == 0
        )

    def opponent_history(self, *, street: int | None = None) -> tuple[PublicActionEvent, ...]:
        return tuple(
            event
            for event in self.voluntary_history(street=street)
            if event.actor_rel != 0
        )

    def raise_count(self, *, street: int | None = None) -> int:
        return sum(
            1
            for event in self.voluntary_history(street=street)
            if event.action_type in (ACTION_BET_TO, ACTION_RAISE_TO, ACTION_ALL_IN)
        )



def decode_spnniv3(payload: bytes) -> DeepCrusherStateView:
    if len(payload) < FIXED_BYTES or payload[:8] != MAGIC:
        raise ValueError("bad SPNNIV3 payload")
    p = 8
    categorical = tuple(int(x) for x in payload[p : p + 10])
    p += 10
    ranks = tuple(int(x) for x in payload[p : p + 7])
    p += 7
    same_suit = tuple(int(x) for x in payload[p : p + 21])
    p += 21
    numeric = tuple(float(x) for x in struct.unpack_from("<16f", payload, p))
    p += 64
    primitive_legal_raw = tuple(int(x) for x in payload[p : p + 6])
    p += 6
    history_len = int(struct.unpack_from("<I", payload, p)[0])
    p += 4
    expected = FIXED_BYTES + history_len * EVENT_BYTES
    if len(payload) != expected or p != FIXED_BYTES:
        raise ValueError(f"SPNNIV3 size mismatch: got={len(payload)} expected={expected}")

    events: list[PublicActionEvent] = []
    for _ in range(history_len):
        cat = tuple(int(x) for x in payload[p : p + 4])
        p += 4
        values = tuple(float(x) for x in struct.unpack_from("<4f", payload, p))
        p += 16
        events.append(
            PublicActionEvent(
                actor_rel=cat[0],
                street=cat[1],
                action_type=cat[2],
                forced=bool(cat[3]),
                paid_bb=values[0],
                resulting_commitment_bb=values[1],
                pot_before_bb=values[2],
                pot_after_bb=values[3],
            )
        )

    if categorical[0] not in (DOMAIN_THREE_HANDED, DOMAIN_TRUE_HEADS_UP):
        raise ValueError("invalid strategy domain in SPNNIV3")
    if categorical[1] not in (STREET_PREFLOP, STREET_FLOP, STREET_TURN, STREET_RIVER):
        raise ValueError("invalid street in SPNNIV3")
    if categorical[5] not in (2, 3):
        raise ValueError("invalid live_count in SPNNIV3")
    if any(value not in (0, 1) for value in same_suit):
        raise ValueError("invalid same-suit relation in SPNNIV3")
    if any(value not in (0, 1) for value in primitive_legal_raw):
        raise ValueError("invalid primitive legal flag in SPNNIV3")

    return DeepCrusherStateView(
        domain=categorical[0],
        street=categorical[1],
        dealer_rel=categorical[2],
        small_blind_rel=categorical[3],
        big_blind_rel=categorical[4],
        live_count=categorical[5],
        visible_board=categorical[6],
        statuses=(categorical[7], categorical[8], categorical[9]),
        ranks=ranks,  # type: ignore[arg-type]
        same_suit=same_suit,
        pot_bb=numeric[0],
        to_call_bb=numeric[1],
        current_bet_bb=numeric[2],
        stacks_bb=(numeric[3], numeric[4], numeric[5]),
        street_commitments_bb=(numeric[6], numeric[7], numeric[8]),
        total_commitments_bb=(numeric[9], numeric[10], numeric[11]),
        small_blind_bb=numeric[12],
        blind_index=int(round(numeric[13])),
        min_raise_to_bb=numeric[14],
        max_raise_to_bb=numeric[15],
        primitive_legal=tuple(bool(x) for x in primitive_legal_raw),  # type: ignore[arg-type]
        history=tuple(events),
    )


def _configure_v3(owner) -> None:
    if getattr(owner, "_deepcrusher_v3_ready", False):
        return
    try:
        fn = owner.lib.spincore_solver_state_neural_input_v3
    except AttributeError as exc:
        raise RuntimeError("solver library predates SPNNIV3 observable-state API") from exc
    fn.argtypes = [C.c_void_p, C.POINTER(C.c_uint8), C.c_size_t]
    fn.restype = C.c_size_t
    owner._deepcrusher_v3_ready = True


def state_view(state) -> DeepCrusherStateView:
    """Return the current actor's lossless observable poker state."""
    if state.terminal:
        raise ValueError("terminal state has no DeepCrusher decision view")
    _configure_v3(state.owner)
    fn = state.owner.lib.spincore_solver_state_neural_input_v3
    n = int(fn(state._p(), None, 0))
    if n < FIXED_BYTES:
        raise RuntimeError(state.owner.error() or "no SPNNIV3 payload")
    buf = (C.c_uint8 * n)()
    got = int(fn(state._p(), buf, n))
    if got != n:
        raise RuntimeError(state.owner.error() or "SPNNIV3 payload size drift")
    view = decode_spnniv3(bytes(buf))

    # The benchmark solver exposes a read-only deal snapshot. Use it to restore
    # absolute suits without changing SpinCore's neural representation.
    if getattr(state.owner, "explicit_deal_available", False):
        deal = state.deal_snapshot()
        actor = int(state.actor)
        hero = deal.holes[actor]
        if any(card < 0 for card in hero):
            raise RuntimeError("live actor has missing hole cards in deal snapshot")

        # SpinCore Card ids are rank-major with suit order s,h,d,c.
        # OpenHoldem StdDeck suit ids are h=0,d=1,c=2,s=3.
        spin_to_oh = (3, 0, 1, 2)
        exact = [
            spin_to_oh[int(hero[0]) % 4],
            spin_to_oh[int(hero[1]) % 4],
        ]
        for index, card in enumerate(deal.board):
            if index < int(view.visible_board):
                exact.append(spin_to_oh[int(card) % 4])
            else:
                exact.append(-1)
        view = DeepCrusherStateView(
            **{**view.__dict__, "exact_suits": tuple(exact)}
        )
    return view
