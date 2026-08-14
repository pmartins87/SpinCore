from __future__ import annotations

import argparse
import json
import math
import random
import resource
import time
from collections import defaultdict
from pathlib import Path

import torch

from spincore.r7_5_paired_corpus import PairedSample, immutable_sample_identity, split_items
from spincore_nn import AdvantageNet, AveragePolicyNet, NetworkConfig
from spincore_nn.codec import collate_inputs, decode_spnniv1
from spincore_nn.codec_v2 import collate_inputs_v2, decode_spnniv2
from spincore_nn.models_v2 import AdvantageNetV2, AveragePolicyNetV2, SemanticNetworkConfigV2
from spincore_nn.training import train_step

SCHEMA = "SPINCORE_R7_5_3_CANDIDATE_FIT_V1"
CANDIDATES = {
    "C0_V1_FROZEN_CONTROL": None,
    "C1_V2_NO_FLOP_TOKEN": "NONE",
    "C2_V2_H1_CANONICAL_184": "H1",
    "C3_V2_H2_MIN_CHANGE_181": "H2",
    "C4_V2_H3_RECLUSTERED_184": "H3",
    "C5_V2_H4_EXACT_1755": "H4",
}


def _load_samples(paths: list[Path], expected_kind: str, domain: str) -> list[PairedSample]:
    out: list[PairedSample] = []
    for path in paths:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(raw, list):
            raise ValueError(f"paired corpus is not a list: {path}")
        for row in raw:
            sample = PairedSample(
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
            if sample.kind != expected_kind or sample.domain != domain:
                raise ValueError("paired corpus kind/domain mismatch")
            out.append(sample)
    # Remove byte-identical duplicate paired samples deterministically.
    unique: dict[bytes, PairedSample] = {}
    for sample in out:
        unique.setdefault(immutable_sample_identity(sample), sample)
    return [unique[key] for key in sorted(unique)]


def _batch(samples: list[PairedSample], candidate: str, device: str):
    flop_candidate = CANDIDATES[candidate]
    if flop_candidate is None:
        batch = collate_inputs(
            [decode_spnniv1(sample.observation_v1) for sample in samples],
            device=device,
        )
    else:
        batch = collate_inputs_v2(
            [decode_spnniv2(sample.observation_v2) for sample in samples],
            device=device,
            flop_candidate=flop_candidate,
        )
    target = torch.tensor([sample.target for sample in samples], dtype=torch.float32, device=device)
    weights = torch.tensor([sample.weight for sample in samples], dtype=torch.float32, device=device)
    return batch, target, weights


def _models(candidate: str, device: str, fit_seed: int):
    torch.manual_seed(int(fit_seed))
    if CANDIDATES[candidate] is None:
        cfg = NetworkConfig()
        advantage = AdvantageNet(cfg).to(device)
    else:
        cfg = SemanticNetworkConfigV2()
        advantage = AdvantageNetV2(cfg).to(device)
    torch.manual_seed(int(fit_seed) ^ 0x5A17C0DE)
    if CANDIDATES[candidate] is None:
        policy = AveragePolicyNet(cfg).to(device)
    else:
        policy = AveragePolicyNetV2(cfg).to(device)
    return cfg, advantage, policy


def _train_fixed(
    *,
    model,
    samples: list[PairedSample],
    candidate: str,
    kind: str,
    fit_seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    rng = random.Random(int(fit_seed) ^ (0xA11D if kind == "advantage" else 0xB011C7))
    if not samples:
        raise ValueError(f"empty {kind} training set")
    losses: list[float] = []
    started = time.perf_counter()
    for _ in range(int(steps)):
        chosen = [samples[rng.randrange(len(samples))] for _ in range(int(batch_size))]
        batch, target, weights = _batch(chosen, candidate, device)
        losses.append(train_step(model, optimizer, batch, target, weights, kind))
    elapsed = time.perf_counter() - started
    return {
        "steps": int(steps),
        "elapsed_seconds": float(elapsed),
        "seconds_per_step": float(elapsed / max(1, int(steps))),
        "last_loss": float(losses[-1]),
        "mean_last_100_loss": float(sum(losses[-100:]) / min(100, len(losses))),
    }


def _sentinel_tags(sample: PairedSample) -> tuple[str, ...]:
    decoded = decode_spnniv2(sample.observation_v2)
    c = decoded.categorical
    tags = {
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
    named_flags = {
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
    for index, name in named_flags.items():
        if c[index]:
            tags.add(name)
    return tuple(sorted(tags))


def _evaluate_advantage(model, samples, candidate, device, min_tag_samples: int):
    numerator = denominator = 0.0
    tag_stats = defaultdict(lambda: [0, 0.0, 0.0])
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), 1024):
            chunk = samples[start : start + 1024]
            batch, target, weights = _batch(chunk, candidate, device)
            pred = model(batch)
            mask = batch["legal"].float()
            per_num = (((pred - target) ** 2) * mask).sum(1).detach().cpu()
            per_den = ((target ** 2) * mask).sum(1).detach().cpu()
            w = weights.detach().cpu()
            for index, sample in enumerate(chunk):
                n = float(per_num[index]) * float(w[index])
                d = float(per_den[index]) * float(w[index])
                numerator += n
                denominator += d
                for tag in _sentinel_tags(sample):
                    row = tag_stats[tag]
                    row[0] += 1
                    row[1] += n
                    row[2] += d
    elapsed = time.perf_counter() - started
    nrmse = math.sqrt(numerator / max(denominator, 1.0e-12))
    eligible = {
        tag: {
            "count": int(row[0]),
            "weighted_nrmse": math.sqrt(row[1] / max(row[2], 1.0e-12)),
        }
        for tag, row in sorted(tag_stats.items())
        if row[0] >= int(min_tag_samples)
    }
    macro = (
        sum(row["weighted_nrmse"] for row in eligible.values()) / len(eligible)
        if eligible
        else math.inf
    )
    return {
        "weighted_nrmse": float(nrmse),
        "sentinel_macro_weighted_nrmse": float(macro),
        "eligible_sentinel_count": len(eligible),
        "sentinels": eligible,
        "inference_seconds": float(elapsed),
        "inference_samples_per_second": float(len(samples) / max(elapsed, 1.0e-12)),
    }


def _evaluate_policy(model, samples, candidate, device):
    weighted_tv = weight_sum = 0.0
    predictions: list[list[float]] = []
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(samples), 1024):
            chunk = samples[start : start + 1024]
            batch, target, weights = _batch(chunk, candidate, device)
            pred = model.probabilities(batch)
            tv = (0.5 * torch.abs(pred - target).sum(1)).detach().cpu()
            w = weights.detach().cpu()
            weighted_tv += float((tv * w).sum())
            weight_sum += float(w.sum())
            predictions.extend(pred.detach().cpu().tolist())
    elapsed = time.perf_counter() - started
    return {
        "weighted_mean_tv": float(weighted_tv / max(weight_sum, 1.0e-12)),
        "inference_seconds": float(elapsed),
        "inference_samples_per_second": float(len(samples) / max(elapsed, 1.0e-12)),
        "heldout_predictions": predictions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="R7.5.3 offline paired representation fit")
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--domain", choices=["TRUE_HEADS_UP", "THREE_HANDED"], required=True)
    parser.add_argument("--fit-seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=1925930899)
    parser.add_argument("--advantage", type=Path, action="append", required=True)
    parser.add_argument("--strategy", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--advantage-steps", type=int, default=4096)
    parser.add_argument("--policy-steps", type=int, default=8192)
    parser.add_argument("--min-tag-samples", type=int, default=128)
    args = parser.parse_args()

    advantage = _load_samples(args.advantage, "advantage", args.domain)
    strategy = _load_samples(args.strategy, "strategy", args.domain)
    adv_train, adv_heldout = split_items(advantage, split_seed=int(args.split_seed))
    pol_train, pol_heldout = split_items(strategy, split_seed=int(args.split_seed))
    if not adv_heldout or not pol_heldout:
        raise RuntimeError("deterministic heldout split is empty")

    cfg, adv_model, pol_model = _models(args.candidate, args.device, int(args.fit_seed))
    adv_progress = _train_fixed(
        model=adv_model,
        samples=adv_train,
        candidate=args.candidate,
        kind="advantage",
        fit_seed=int(args.fit_seed),
        steps=int(args.advantage_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        device=args.device,
    )
    pol_progress = _train_fixed(
        model=pol_model,
        samples=pol_train,
        candidate=args.candidate,
        kind="strategy",
        fit_seed=int(args.fit_seed),
        steps=int(args.policy_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        device=args.device,
    )
    advantage_audit = _evaluate_advantage(
        adv_model,
        adv_heldout,
        args.candidate,
        args.device,
        int(args.min_tag_samples),
    )
    policy_audit = _evaluate_policy(pol_model, pol_heldout, args.candidate, args.device)
    predictions = policy_audit.pop("heldout_predictions")

    parameter_count = sum(parameter.numel() for parameter in adv_model.parameters())
    report = {
        "schema": SCHEMA,
        "candidate": args.candidate,
        "flop_candidate": CANDIDATES[args.candidate],
        "domain": args.domain,
        "fit_seed": int(args.fit_seed),
        "split_seed": int(args.split_seed),
        "counts": {
            "advantage_total": len(advantage),
            "advantage_train": len(adv_train),
            "advantage_heldout": len(adv_heldout),
            "strategy_total": len(strategy),
            "strategy_train": len(pol_train),
            "strategy_heldout": len(pol_heldout),
        },
        "parameter_count": int(parameter_count),
        "config": cfg.to_dict(),
        "advantage_fit": adv_progress,
        "policy_fit": pol_progress,
        "heldout_advantage": advantage_audit,
        "heldout_policy": policy_audit,
        "absolute_gates": {
            "advantage_weighted_nrmse_max": 0.75,
            "policy_weighted_mean_tv_max": 0.12,
            "advantage_pass": advantage_audit["weighted_nrmse"] <= 0.75,
            "policy_pass": policy_audit["weighted_mean_tv"] <= 0.12,
        },
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "ready_for_tables": false
    }
    report["absolute_gates"]["fit_pass"] = bool(
        report["absolute_gates"]["advantage_pass"]
        and report["absolute_gates"]["policy_pass"]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA,
            "report": report,
            "advantage_state": adv_model.state_dict(),
            "policy_state": pol_model.state_dict(),
            "policy_heldout_identity": [immutable_sample_identity(x) for x in pol_heldout],
            "policy_heldout_predictions": predictions,
        },
        args.out,
    )
    args.out.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["absolute_gates"]["fit_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
