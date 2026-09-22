from __future__ import annotations

"""Frozen-R8 preflop range-equity projection.

The source dataset supplied with the project contains 1,624,350 exact disjoint
two-hand matchups.  For the five R8 all-in ranges (list4/list6/list9/list12/
list15), averaging legal opponent combos produces the same value for every
exact suit realization of a canonical 169 hand class.  The runtime fixture
therefore stores one row per canonical class instead of 1,326 redundant exact
hero combos.

Stored values are hero showdown equity (prwin + 0.5*prtie). Frozen R8 consumes
its two vs$multiplex$f$backup_opp_allin_range$... symbols only through exactly
that combined expression, so the compact projection preserves the decision
value needed by R8 without pretending to preserve the individual prwin/prtie
components.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

from spincore.deepcrusher_state import DeepCrusherStateView, STREET_PREFLOP


SCHEMA = "SPINCORE_PREFLOP_RANGE_EQUITY_CLASS_V1"
SOURCE_ZIP_SHA256 = "52a0a87174b0d7cabd5b16fe43387b0807a6abd036e5a61c1aafbc008ecf50c2"
SOURCE_TXT_SHA256 = "9dd539e2720010684d0006981207489e4f753b1d628f7e0443003b2c7f3e6c9f"
TABLE_TEXT_SHA256 = "114fd17d594fb63b5f46385687dc522f894c63bb6ccbf6a2d5c3c69ac2f34892"
SUPPORTED_RANGE_IDS = frozenset({4, 6, 9, 12, 15})

DEFAULT_TABLE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "deepcrusher_r8_v22"
    / "preflop_range_equity_class_v1.tsv"
)


class DeepCrusherPreflopEquityError(RuntimeError):
    pass


@lru_cache(maxsize=4)
def _load_table(path_text: str) -> dict[str, dict[int, float]]:
    path = Path(path_text)
    text_bytes = path.read_bytes()
    digest = hashlib.sha256(text_bytes).hexdigest()
    if digest != TABLE_TEXT_SHA256:
        raise DeepCrusherPreflopEquityError(
            "preflop class-equity SHA256 mismatch: "
            f"got={digest} expected={TABLE_TEXT_SHA256}"
        )

    lines = text_bytes.decode("ascii").splitlines()
    headers = [line for line in lines if line.startswith("#")]
    if not headers or SCHEMA not in headers[0]:
        raise DeepCrusherPreflopEquityError("preflop class-equity schema mismatch")

    table: dict[str, dict[int, float]] = {}
    for line in lines:
        if not line or line.startswith("#") or line.startswith("class\t"):
            continue
        parts = line.split("\t")
        if len(parts) != 6:
            raise DeepCrusherPreflopEquityError(f"bad class-equity row: {line!r}")
        hand_class = parts[0]
        if hand_class in table:
            raise DeepCrusherPreflopEquityError(
                f"duplicate class-equity row: {hand_class}"
            )
        table[hand_class] = {
            4: float(parts[1]),
            6: float(parts[2]),
            9: float(parts[3]),
            12: float(parts[4]),
            15: float(parts[5]),
        }

    if len(table) != 169:
        raise DeepCrusherPreflopEquityError(
            f"expected 169 canonical hand classes, got {len(table)}"
        )
    return table


def hero_class_key(view: DeepCrusherStateView) -> str:
    try:
        key = str(view.hero_hand_class)
    except Exception as exc:
        raise DeepCrusherPreflopEquityError(
            "unable to derive canonical hero hand class"
        ) from exc
    return key


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
        raise DeepCrusherPreflopEquityError(
            f"unsupported frozen R8 range id: {rid}"
        )
    key = hero_class_key(view)
    try:
        value = _load_table(str(table_path.resolve()))[key][rid]
    except KeyError as exc:
        raise DeepCrusherPreflopEquityError(
            f"missing range equity for hero_class={key} list{rid}"
        ) from exc
    if not 0.0 <= value <= 1.0:
        raise DeepCrusherPreflopEquityError(f"equity outside [0,1]: {value}")
    return value
