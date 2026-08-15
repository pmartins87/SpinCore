from __future__ import annotations

import numpy as np

from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_phase2_eval import (
    H2,
    H3,
    INCONCLUSIVE,
    bootstrap_mean_ci,
    classify_local_deviation_ci,
    classify_pairwise_crossplay_ci,
    combine_domain_directions,
    equal_group_stratified_bootstrap_mean_ci,
    paired_two_seed_representation_difference,
)
from spincore.r7_5_representation_v3_stage_contract import DOMAINS, EVALUATION_SEEDS, TRAINING_SEEDS

MATERIAL_FLOOR = 0.001


def local_deviation_summary(heldout: list[dict]) -> dict:
    by_key = {(p["representation"], p["domain"], int(p["training_seed"]), int(p["evaluation_seed"])): p for p in heldout}
    cells, domain_directions, domain_groups = [], {}, {}
    for domain in DOMAINS:
        eval_directions, domain_values = {}, []
        for evaluation_seed in EVALUATION_SEEDS:
            h2 = [by_key[(H2_FINAL, domain, int(seed), int(evaluation_seed))]["local_deviation_gains"] for seed in TRAINING_SEEDS]
            h3 = [by_key[(H3_FINAL, domain, int(seed), int(evaluation_seed))]["local_deviation_gains"] for seed in TRAINING_SEEDS]
            differences = paired_two_seed_representation_difference(h2_seed_values=h2, h3_seed_values=h3)
            ci = bootstrap_mean_ci(differences, seed_parts=("SpinCore", "R7.5.3C", "PHASE2", "bootstrap", "localdev", domain, int(evaluation_seed), "H2|H3"))
            direction = classify_local_deviation_ci(ci["ci_low"], ci["ci_high"], material_floor=MATERIAL_FLOOR)
            eval_directions[str(evaluation_seed)] = direction
            domain_values.extend(differences)
            cells.append({"domain": domain, "evaluation_seed": int(evaluation_seed), "direction": direction, **ci})
        domain_directions[domain] = combine_domain_directions(eval_directions)
        domain_groups[domain] = tuple(domain_values)
    overall = combine_domain_directions(domain_directions) if all(v in (H2, H3, INCONCLUSIVE) for v in domain_directions.values()) else "DOMAIN_CONFLICT"
    pooled = equal_group_stratified_bootstrap_mean_ci(domain_groups, seed_parts=("SpinCore", "R7.5.3C", "PHASE2", "bootstrap", "localdev", "EQUAL_DOMAIN", "H2|H3"))
    return {
        "name": "ONE_STEP_SELF_CONTINUATION_DEVIATION_GAIN",
        "difference": "H3_MINUS_H2",
        "negative_favors": H3,
        "material_floor_icm": MATERIAL_FLOOR,
        "cells": cells,
        "domain_directions": domain_directions,
        "overall_direction": overall,
        "equal_domain_pooled_diagnostic": pooled,
        "called_exact_exploitability": False,
    }


def _mean_aligned(rows: list[list[float]]) -> list[float]:
    arrays = [np.asarray(row, dtype=np.float64) for row in rows]
    if not arrays or any(a.ndim != 1 for a in arrays) or any(a.shape != arrays[0].shape for a in arrays[1:]):
        raise RuntimeError("aligned score array shape mismatch")
    if not all(np.isfinite(a).all() for a in arrays):
        raise RuntimeError("non-finite aligned score")
    return [float(value) for value in np.stack(arrays, axis=0).mean(axis=0)]


def pairwise_summary(pairwise: list[dict]) -> dict:
    by_key = {(p["domain"], int(p["evaluation_seed"]), int(p["h2_training_seed"]), int(p["h3_training_seed"])): p for p in pairwise}
    cells, domain_directions, domain_groups = [], {}, {}
    for domain in DOMAINS:
        eval_directions, domain_values = {}, []
        seats = (1, 2) if domain == "TRUE_HEADS_UP" else (0, 1, 2)
        count = 20000 if domain == "TRUE_HEADS_UP" else 10000
        for evaluation_seed in EVALUATION_SEEDS:
            values = []
            for seat in seats:
                seedpairs = []
                for h2_seed in TRAINING_SEEDS:
                    for h3_seed in TRAINING_SEEDS:
                        cell = by_key[(domain, int(evaluation_seed), int(h2_seed), int(h3_seed))]
                        if int(cell["hands_per_candidate_seat"]) != count:
                            raise RuntimeError("pairwise hand-count drift")
                        row = cell["seats"].get(str(seat))
                        if row is None or len(row) != count:
                            raise RuntimeError("pairwise seat score-count drift")
                        seedpairs.append(row)
                values.extend(_mean_aligned(seedpairs))
            ci = bootstrap_mean_ci(values, seed_parts=("SpinCore", "R7.5.3C", "PHASE2", "bootstrap", "pairwise", domain, int(evaluation_seed), "H2|H3"))
            direction = classify_pairwise_crossplay_ci(ci["ci_low"], ci["ci_high"], material_floor=MATERIAL_FLOOR)
            eval_directions[str(evaluation_seed)] = direction
            domain_values.extend(values)
            cells.append({"domain": domain, "evaluation_seed": int(evaluation_seed), "direction": direction, **ci})
        domain_directions[domain] = combine_domain_directions(eval_directions)
        domain_groups[domain] = tuple(domain_values)
    overall = combine_domain_directions(domain_directions) if all(v in (H2, H3, INCONCLUSIVE) for v in domain_directions.values()) else "DOMAIN_CONFLICT"
    pooled = equal_group_stratified_bootstrap_mean_ci(domain_groups, seed_parts=("SpinCore", "R7.5.3C", "PHASE2", "bootstrap", "pairwise", "EQUAL_DOMAIN", "H2|H3"))
    return {
        "score_perspective": H3,
        "positive_favors": H3,
        "material_floor_icm": MATERIAL_FLOOR,
        "training_seed_pair_hand_values_averaged_before_bootstrap": True,
        "cells": cells,
        "domain_directions": domain_directions,
        "overall_direction": overall,
        "equal_domain_pooled_diagnostic": pooled,
    }


def common_reference_summary(commonref: list[dict]) -> dict:
    by_key = {(p["representation"], p["domain"], int(p["training_seed"]), int(p["evaluation_seed"])): p for p in commonref}
    cells = []
    for representation in (H2_FINAL, H3_FINAL):
        for domain in DOMAINS:
            seats = (1, 2) if domain == "TRUE_HEADS_UP" else (0, 1, 2)
            count = 20000 if domain == "TRUE_HEADS_UP" else 10000
            for evaluation_seed in EVALUATION_SEEDS:
                values = []
                for seat in seats:
                    seed_rows = []
                    for training_seed in TRAINING_SEEDS:
                        cell = by_key[(representation, domain, int(training_seed), int(evaluation_seed))]
                        if int(cell["hands_per_candidate_seat"]) != count:
                            raise RuntimeError("common-reference hand-count drift")
                        row = cell["seats"].get(str(seat))
                        if row is None or len(row) != count:
                            raise RuntimeError("common-reference seat score-count drift")
                        seed_rows.append(row)
                    values.extend(_mean_aligned(seed_rows))
                ci = bootstrap_mean_ci(values, seed_parts=("SpinCore", "R7.5.3C", "PHASE2", "bootstrap", "commonref", representation, domain, int(evaluation_seed)))
                cells.append({"representation": representation, "domain": domain, "evaluation_seed": int(evaluation_seed), **ci})
    return {
        "selection_role": "DIAGNOSTIC_ONLY",
        "selects_winner": False,
        "training_seed_hand_values_averaged_before_bootstrap": True,
        "cells": cells,
    }
