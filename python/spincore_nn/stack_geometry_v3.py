from __future__ import annotations

from dataclasses import dataclass

from spincore_nn.codec_v3 import DecodedInputV3


@dataclass(frozen=True)
class PairwiseStackGeometryV3:
    """Actor-relative pairwise stack geometry for the lossless SPNNIV3 wire.

    SPNNIV3 current-state numerics are normalized by the current big blind:
      numeric[0]      pot / BB
      numeric[3:6]    remaining stacks / BB for actor-relative seats 0..2
      numeric[6:9]    street commitments / BB
      numeric[9:12]   total commitments / BB
    categorical[7:10] stores status 0 active, 1 folded, 2 all-in.

    True HU is already canonicalized by the C++ V3 encoder as
    `[Hero, live opponent, absent]`. In 3H, rel1/rel2 retain positional identity;
    values are never sorted by stack size because doing so would erase position.
    """

    opponent_present: tuple[int, int]
    opponent_contesting: tuple[int, int]
    opponent_actionable: tuple[int, int]
    effective_remaining_bb: tuple[float, float]
    pairwise_spr: tuple[float, float]
    effective_total_cap_bb: tuple[float, float]
    commitment_gap_bb: tuple[float, float]


def derive_pairwise_stack_geometry_v3(item: DecodedInputV3) -> PairwiseStackGeometryV3:
    if len(item.numeric) != 16 or len(item.categorical) != 10:
        raise ValueError("unexpected SPNNIV3 state dimensions for stack geometry")

    domain = int(item.categorical[0])
    statuses = tuple(int(value) for value in item.categorical[7:10])
    if domain not in (0, 1):
        raise ValueError(f"unknown SPNNIV3 domain {domain}")

    pot_bb = float(item.numeric[0])
    hero_stack = float(item.numeric[3])
    hero_total = float(item.numeric[9])
    if pot_bb < 0.0 or hero_stack < 0.0 or hero_total < 0.0:
        raise ValueError("negative public chip geometry")

    present: list[int] = []
    contesting: list[int] = []
    actionable: list[int] = []
    effective_remaining: list[float] = []
    pairwise_spr: list[float] = []
    total_cap: list[float] = []
    commitment_gap: list[float] = []

    for rel in (1, 2):
        status = statuses[rel]
        stack = float(item.numeric[3 + rel])
        street_commitment = float(item.numeric[6 + rel])
        total = float(item.numeric[9 + rel])
        if status not in (0, 1, 2):
            raise ValueError(f"unknown player status {status}")
        if stack < 0.0 or total < 0.0 or street_commitment < 0.0:
            raise ValueError("negative opponent chip geometry")

        # True-HU rel2 is absent by the SPNNIV3 canonical contract. A real all-in
        # opponent is not inferred absent from stack==0; positive commitment and
        # its rel1 position keep it present/contesting.
        absent = domain == 1 and rel == 2
        is_present = not absent
        is_contesting = is_present and status != 1
        is_actionable = is_present and status == 0 and stack > 0.0

        eff = min(hero_stack, stack) if is_actionable else 0.0
        spr = eff / pot_bb if is_actionable and pot_bb > 0.0 else 0.0
        cap = min(hero_stack + hero_total, stack + total) if is_contesting else 0.0

        present.append(int(is_present))
        contesting.append(int(is_contesting))
        actionable.append(int(is_actionable))
        effective_remaining.append(float(eff))
        pairwise_spr.append(float(spr))
        total_cap.append(float(cap))
        commitment_gap.append(float(total - hero_total) if is_present else 0.0)

    return PairwiseStackGeometryV3(
        opponent_present=tuple(present),
        opponent_contesting=tuple(contesting),
        opponent_actionable=tuple(actionable),
        effective_remaining_bb=tuple(effective_remaining),
        pairwise_spr=tuple(pairwise_spr),
        effective_total_cap_bb=tuple(total_cap),
        commitment_gap_bb=tuple(commitment_gap),
    )
