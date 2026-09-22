from __future__ import annotations

"""Exact frozen-R8 preflop range-equity projection.

The compact table is derived from the user-supplied PreFlopEquities.txt:
1,624,350 exact disjoint two-hand matchups.  Each stored value is hero
showdown equity (prwin + 0.5 * prtie), averaged over every legal exact
opponent combo in DeepCrusher R8 list4/list6/list9/list12/list15.

This is sufficient for the frozen R8 because its two
vs$multiplex$f$backup_opp_allin_range$... symbols occur only together as
    prwin + prtie / 2
inside f$preflop_prob_win_BKP.
We intentionally do not pretend that the compact table separately preserves
prwin and prtie.
"""

from functools import lru_cache
import gzip
import hashlib
from pathlib import Path

from spincore.deepcrusher_state import DeepCrusherStateView, STREET_PREFLOP


SCHEMA = "SPINCORE_PREFLOP_RANGE_EQUITY_V1"
SOURCE_ZIP_SHA256 = "52a0a87174b0d7cabd5b16fe43387b0807a6abd036e5a61c1aafbc008ecf50c2"
SOURCE_TXT_SHA256 = "9dd539e2720010684d0006981207489e4f753b1d628f7e0443003b2c7f3e6c9f"
TABLE_GIT_BLOB_SHA1 = "d86110f6f0935b9b2929df2f71dc7868b8325805"
TABLE_TEXT_SHA256 = "10b42e456dac1f7fffeb79c3a8eacc80d868d5732d115b11bf8965b5dba597b0"
SUPPORTED_RANGE_IDS = frozenset({4, 6, 9, 12, 15})

DEFAULT_TABLE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "deepcrusher_r8_v22"
    / "preflop_range_equity_v1.tsv.gz"
)


class DeepCrusherPreflopEquityError(RuntimeError):
    pass


@lru_cache(maxsize=4)
def _load_table(path_text: str) -> dict[tuple[int, int], dict[int, float]]:
    path = Path(path_text)
    raw = path.read_bytes()
    # Gzip container bytes are repository-pinned by Git blob SHA-1. Runtime
    # validation intentionally hashes the canonical decompressed payload because
    # gzip metadata (mtime/header) is not semantic and may change when rebuilt.
    text_bytes = gzip.decompress(raw)
    if hashlib.sha256(text_bytes).hexdigest() != TABLE_TEXT_SHA256:
        raise DeepCrusherPreflopEquityError("preflop equity text SHA256 mismatch")

    lines = text_bytes.decode("ascii").splitlines()
    headers = [line for line in lines if line.startswith("#")]
    if not headers or SCHEMA not in headers[0]:
        raise DeepCrusherPreflopEquityError("preflop equity schema mismatch")

    table: dict[tuple[int, int], dict[int, float]] = {}
    for line in lines:
        if not line or line.startswith("#") or line.startswith("c0\t"):
            continue
        parts = line.split("\t")
        if len(parts) != 7:
            raise DeepCrusherPreflopEquityError(f"bad equity row: {line!r}")
        c0, c1 = int(parts[0]), int(parts[1])
        if not (0 <= c0 < c1 < 52):
            raise DeepCrusherPreflopEquityError(f"bad card ids: {(c0, c1)}")
        table[(c0, c1)] = {
            4: float(parts[2]),
            6: float(parts[3]),
            9: float(parts[4]),
            12: float(parts[5]),
            15: float(parts[6]),
        }
    if len(table) != 1326:
        raise DeepCrusherPreflopEquityError(
            f"expected 1326 hero combos, got {len(table)}"
        )
    return table


def _card_id(rank: int, openholdem_suit: int) -> int:
    if not 2 <= int(rank) <= 14:
        raise DeepCrusherPreflopEquityError(f"bad rank: {rank}")
    if not 0 <= int(openholdem_suit) <= 3:
        raise DeepCrusherPreflopEquityError(f"bad OpenHoldem suit: {openholdem_suit}")
    return (int(rank) - 2) * 4 + int(openholdem_suit)


def hero_combo_key(view: DeepCrusherStateView) -> tuple[int, int]:
    suits = view.openholdem_suits
    cards = sorted(
        (
            _card_id(view.ranks[0], suits[0]),
            _card_id(view.ranks[1], suits[1]),
        )
    )
    if cards[0] == cards[1]:
        raise DeepCrusherPreflopEquityError("duplicate hero card")
    return cards[0], cards[1]


def range_equity(
    view: DeepCrusherStateView,
    range_id: int,
    *,
    table_path: Path = DEFAULT_TABLE,
) -> float:
    if int(view.street) != STREET_PREFLOP:
        raise DeepCrusherPreflopEquityError("R8 compact range equity is preflop-only")
    rid = int(range_id)
    if rid not in SUPPORTED_RANGE_IDS:
        raise DeepCrusherPreflopEquityError(f"unsupported frozen R8 range id: {rid}")
    key = hero_combo_key(view)
    try:
        value = _load_table(str(table_path.resolve()))[key][rid]
    except KeyError as exc:
        raise DeepCrusherPreflopEquityError(
            f"missing range equity for hero={key} list{rid}"
        ) from exc
    if not 0.0 <= value <= 1.0:
        raise DeepCrusherPreflopEquityError(f"equity outside [0,1]: {value}")
    return value
