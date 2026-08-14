from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from spincore.r7_5_paired_corpus import (
    PairedSample,
    immutable_sample_identity,
    retention_key,
)
from spincore_nn.codec import decode_spnniv1
from spincore_nn.codec_v2 import decode_spnniv2

SCHEMA = "SPINCORE_R7_5_3_PAIRED_CORPUS_AUDIT_V1"
CORPUS_SCHEMA = "SPINCORE_R7_5_PAIRED_CORPUS_V1"
EXPECTED_DOMAINS = ("TRUE_HEADS_UP", "THREE_HANDED")
EXPECTED_SEEDS = (1202035427, 2078778133)
MIN_ADVANTAGE = 5000
MIN_STRATEGY = 2000


def _sample_from_row(row: dict) -> PairedSample:
    return PairedSample(
        kind=str(row["kind"]),
        domain=str(row["domain"]),
        corpus_seed=int(row["corpus_seed"]),
        observation_v1=bytes(row["observation_v1"]),
        observation_v2=bytes(row["observation_v2"]),
        legal=tuple(int(x) for x in row["legal"]),
        target=tuple(float(x) for x in row["target"]),
        weight=float(row["weight"]),
        iteration=int(row["iteration"]),
    )


def _semantic_tags(sample: PairedSample) -> tuple[str, ...]:
    decoded = decode_spnniv2(sample.observation_v2)
    c = decoded.categorical
    tags = {
        f"street:{c[1]}",
        f"preflop_lineage:{c[10]}",
        f"post_open:{c[17]}",
        f"post_facing:{c[18]}",
        f"post_attack:{c[19]}",
        f"raise_depth:{c[21]}",
        f"made:{c[27]}",
        f"pair_relation:{c[28]}",
        f"max_suit_count:{c[44]}",
        f"board_paired:{c[52]}",
    }
    flags = {
        32: "flush_draw",
        35: "backdoor_flush",
        37: "oesd",
        38: "gutshot",
        39: "double_gutshot",
        41: "backdoor_straight",
        57: "new_card_pairs_board",
        58: "new_card_overcard",
        59: "new_card_undercard",
        60: "new_card_three_suit",
        61: "new_card_four_suit",
        62: "new_card_more_connected",
        63: "new_card_four_to_straight",
        64: "new_card_board_straight",
    }
    for index, name in flags.items():
        if c[index]:
            tags.add(name)
    return tuple(sorted(tags))


def _load_samples(path: Path, expected_kind: str, domain: str, seed: int) -> list[PairedSample]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, list):
        raise ValueError(f"corpus file is not a list: {path}")
    out = []
    for row in raw:
        sample = _sample_from_row(row)
        if sample.kind != expected_kind or sample.domain != domain or sample.corpus_seed != int(seed):
            raise ValueError(f"sample provenance mismatch in {path}")
        v1 = decode_spnniv1(sample.observation_v1)
        v2 = decode_spnniv2(sample.observation_v2)
        if tuple(v1.legal) != sample.legal or tuple(v2.legal) != sample.legal:
            raise ValueError(f"legal-mask mismatch in {path}")
        out.append(sample)
    return out


def _identity_digest(samples: list[PairedSample]) -> str:
    """Reproduce BottomHashCorpus.state_summary() ordering exactly.

    The producer's canonical item order is retention_key order, with insertion
    sequence used only to break exact-key ties. Saved corpus files are emitted in
    that producer order. Sorting by immutable_sample_identity here was a distinct
    deterministic ordering and caused false hash mismatches on valid corpora.

    Exact retention-key ties imply byte-identical sample identities for this
    scheme, so their relative order cannot change the concatenated digest.
    """
    digest = hashlib.sha256()
    for sample in sorted(samples, key=retention_key):
        digest.update(immutable_sample_identity(sample))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def audit_corpus_dirs(corpus_dirs: list[Path]) -> dict:
    by_key: dict[tuple[str, int], Path] = {}
    for directory in corpus_dirs:
        report_path = directory / "report.json"
        if not report_path.exists():
            raise ValueError(f"missing corpus report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema") != CORPUS_SCHEMA:
            raise ValueError(f"wrong corpus report schema: {report_path}")
        key = (str(report["domain"]), int(report["corpus_seed"]))
        if key in by_key:
            raise ValueError(f"duplicate corpus directory for {key}")
        by_key[key] = directory

    expected = {(domain, seed) for domain in EXPECTED_DOMAINS for seed in EXPECTED_SEEDS}
    if set(by_key) != expected:
        raise ValueError(f"corpus matrix mismatch: expected {sorted(expected)}, got {sorted(by_key)}")

    rows = []
    global_tags = Counter()
    global_streets = Counter()
    all_coverage_pass = True

    for domain, seed in sorted(expected):
        directory = by_key[(domain, seed)]
        report_path = directory / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not bool(report.get("coverage_pass")):
            all_coverage_pass = False
        if bool(report.get("candidate_inference_used")):
            raise ValueError("candidate inference was used during paired corpus collection")
        if report.get("behavior_observation_wire") != "SPNNIV1" or report.get("paired_secondary_wire") != "SPNNIV2":
            raise ValueError("paired corpus wire contract mismatch")
        if bool(report.get("ready_for_tables")):
            raise ValueError("corpus worker illegally authorized table use")

        adv_path = directory / "advantage_pairs.pt"
        strategy_path = directory / "strategy_pairs.pt"
        advantage = _load_samples(adv_path, "advantage", domain, seed)
        strategy = _load_samples(strategy_path, "strategy", domain, seed)

        if len(advantage) < MIN_ADVANTAGE or len(strategy) < MIN_STRATEGY:
            all_coverage_pass = False
        if int(report["advantage"]["kept"]) != len(advantage):
            raise ValueError("advantage report count differs from corpus file")
        if int(report["strategy"]["kept"]) != len(strategy):
            raise ValueError("strategy report count differs from corpus file")
        if report["advantage"]["ordered_identity_sha256"] != _identity_digest(advantage):
            raise ValueError("advantage ordered identity hash mismatch")
        if report["strategy"]["ordered_identity_sha256"] != _identity_digest(strategy):
            raise ValueError("strategy ordered identity hash mismatch")

        kind_rows = {}
        for kind, samples, path in (
            ("advantage", advantage, adv_path),
            ("strategy", strategy, strategy_path),
        ):
            streets = Counter()
            tags = Counter()
            iterations = Counter()
            for sample in samples:
                decoded = decode_spnniv2(sample.observation_v2)
                street = int(decoded.categorical[1])
                streets[str(street)] += 1
                global_streets[f"{kind}/street:{street}"] += 1
                iterations[str(sample.iteration)] += 1
                for tag in _semantic_tags(sample):
                    tags[tag] += 1
                    global_tags[f"{kind}/{tag}"] += 1
            kind_rows[kind] = {
                "count": len(samples),
                "ordered_identity_sha256": _identity_digest(samples),
                "physical_file_sha256": _file_sha256(path),
                "street_counts": dict(sorted(streets.items())),
                "iteration_counts": dict(sorted(iterations.items())),
                "semantic_tag_counts": dict(sorted(tags.items())),
                "semantic_tags_ge_128": sorted(tag for tag, count in tags.items() if count >= 128),
            }

        rows.append(
            {
                "domain": domain,
                "corpus_seed": seed,
                "paired_roots": int(report["paired_roots"]),
                "scenario_counts_paired_phase": list(report["scenario_counts_paired_phase"]),
                "all_scenarios_exercised": bool(report["all_scenarios_exercised_paired_phase"]),
                "coverage_pass": bool(report["coverage_pass"]),
                "report_sha256": _file_sha256(report_path),
                "kinds": kind_rows,
            }
        )

    family_prefixes = (
        "preflop_lineage:",
        "post_open:",
        "post_facing:",
        "post_attack:",
        "raise_depth:",
        "made:",
        "pair_relation:",
        "max_suit_count:",
        "board_paired:",
    )
    family_coverage = {}
    for prefix in family_prefixes:
        count = sum(value for key, value in global_tags.items() if f"/{prefix}" in key)
        family_coverage[prefix[:-1]] = int(count)

    payload = {
        "schema": SCHEMA,
        "expected_domains": list(EXPECTED_DOMAINS),
        "expected_corpus_seeds": list(EXPECTED_SEEDS),
        "matrix_entries": rows,
        "minimum_samples_per_seed_domain": {
            "advantage": MIN_ADVANTAGE,
            "strategy": MIN_STRATEGY,
        },
        "global_street_counts": dict(sorted(global_streets.items())),
        "global_semantic_tag_counts": dict(sorted(global_tags.items())),
        "semantic_family_coverage": family_coverage,
        "coverage_pass": bool(all_coverage_pass),
        "candidate_inference_used": False,
        "identity_digest_order": "BOTTOM_HASH_RETENTION_KEY_CANONICAL_ORDER",
        "audit_correction_note": "Recomputes producer ordered_identity_sha256 in BottomHashCorpus retention-key order; prior immutable-identity sort was a validation-only ordering defect.",
        "strategic_gate_changed": False,
        "production_training_authorized": False,
        "ready_for_tables": False,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit R7.5.3 paired corpus provenance and semantic coverage")
    parser.add_argument("--corpus-dir", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = audit_corpus_dirs(list(args.corpus_dir))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if payload["coverage_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
