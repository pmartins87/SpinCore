from __future__ import annotations

import json

from spincore.r7_5_representation_v3 import H2_FINAL, H3_FINAL
from spincore.r7_5_representation_v3_stage_contract import DOMAINS, EVALUATION_SEEDS, TRAINING_SEEDS


def matrix() -> dict:
    include = []
    for representation in (H2_FINAL, H3_FINAL):
        for domain in DOMAINS:
            for training_seed in TRAINING_SEEDS:
                for evaluation_seed in EVALUATION_SEEDS:
                    for mode in ("heldout", "commonref"):
                        include.append({
                            "mode": mode,
                            "key": f"{mode}-{representation}-{domain}-{training_seed}-{evaluation_seed}",
                            "representation": representation,
                            "domain": domain,
                            "training_seed": int(training_seed),
                            "evaluation_seed": int(evaluation_seed),
                            "h2_training_seed": 0,
                            "h3_training_seed": 0,
                        })
    for domain in DOMAINS:
        for evaluation_seed in EVALUATION_SEEDS:
            for h2_training_seed in TRAINING_SEEDS:
                for h3_training_seed in TRAINING_SEEDS:
                    include.append({
                        "mode": "pairwise",
                        "key": f"pairwise-{domain}-{evaluation_seed}-{h2_training_seed}-{h3_training_seed}",
                        "representation": "PAIRWISE",
                        "domain": domain,
                        "training_seed": 0,
                        "evaluation_seed": int(evaluation_seed),
                        "h2_training_seed": int(h2_training_seed),
                        "h3_training_seed": int(h3_training_seed),
                    })
    if len(include) != 48 or len({row["key"] for row in include}) != 48:
        raise RuntimeError("Phase2 evaluator matrix is not exactly 48 unique cells")
    return {"include": include}


if __name__ == "__main__":
    print(json.dumps(matrix(), separators=(",", ":")))
