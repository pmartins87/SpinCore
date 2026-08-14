from __future__ import annotations

import argparse
import json
import resource
from dataclasses import replace
from pathlib import Path

import torch

import r7_5_fit_representation_candidate as base
from spincore.card_symmetry_v1 import canonicalize_v1_input, encode_spnniv1
from spincore.r7_5_paired_corpus import PairedSample, immutable_sample_identity, split_items
from spincore_nn.codec import decode_spnniv1

SCHEMA = "SPINCORE_R7_5_3B_CARD_SYMMETRY_FIT_V1"
VARIANTS = ("S0_V1_FROZEN_CONTROL", "S1_V1_CARD_SYMMETRY_CANON")
BASE_CANDIDATE = "C0_V1_FROZEN_CONTROL"
EXPECTED_PARAMETER_COUNT = 152438


def _transform_sample(sample: PairedSample, variant: str) -> PairedSample:
    if variant == "S0_V1_FROZEN_CONTROL":
        return sample
    if variant != "S1_V1_CARD_SYMMETRY_CANON":
        raise ValueError(f"unknown variant: {variant}")
    decoded = decode_spnniv1(sample.observation_v1)
    canonical = canonicalize_v1_input(decoded)
    observation_v1 = encode_spnniv1(canonical)
    # The paired sample identity used for split/order remains the identity of the
    # source sample. Transformation happens strictly after that frozen split.
    return replace(sample, observation_v1=observation_v1)


def _transform_partition(samples: list[PairedSample], variant: str) -> list[PairedSample]:
    return [_transform_sample(sample, variant) for sample in samples]


def main() -> int:
    parser = argparse.ArgumentParser(description="R7.5.3B V1 card-symmetry paired fit")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
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

    advantage_source = base._load_samples(args.advantage, "advantage", args.domain)
    strategy_source = base._load_samples(args.strategy, "strategy", args.domain)
    adv_train_source, adv_heldout_source = split_items(
        advantage_source, split_seed=int(args.split_seed)
    )
    pol_train_source, pol_heldout_source = split_items(
        strategy_source, split_seed=int(args.split_seed)
    )
    if not adv_train_source or not pol_train_source or not adv_heldout_source or not pol_heldout_source:
        raise RuntimeError("deterministic train/heldout split contains an empty partition")

    # Transform only after the source split and preserve list order. Therefore
    # both variants consume exactly the same empirical rows in exactly the same
    # minibatch sequence for a given fit seed.
    adv_train = _transform_partition(adv_train_source, args.variant)
    adv_heldout = _transform_partition(adv_heldout_source, args.variant)
    pol_train = _transform_partition(pol_train_source, args.variant)
    pol_heldout = _transform_partition(pol_heldout_source, args.variant)

    cfg, adv_model, pol_model = base._models(BASE_CANDIDATE, args.device, int(args.fit_seed))
    parameter_count = sum(parameter.numel() for parameter in adv_model.parameters())
    if int(parameter_count) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"V1 parameter count drifted: expected {EXPECTED_PARAMETER_COUNT}, got {parameter_count}"
        )

    adv_progress = base._train_fixed(
        model=adv_model,
        samples=adv_train,
        candidate=BASE_CANDIDATE,
        kind="advantage",
        fit_seed=int(args.fit_seed),
        steps=int(args.advantage_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        device=args.device,
    )
    pol_progress = base._train_fixed(
        model=pol_model,
        samples=pol_train,
        candidate=BASE_CANDIDATE,
        kind="strategy",
        fit_seed=int(args.fit_seed),
        steps=int(args.policy_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        device=args.device,
    )
    advantage_audit = base._evaluate_advantage(
        adv_model,
        adv_heldout,
        BASE_CANDIDATE,
        args.device,
        int(args.min_tag_samples),
    )
    policy_audit = base._evaluate_policy(
        pol_model,
        pol_heldout,
        BASE_CANDIDATE,
        args.device,
    )
    predictions = policy_audit.pop("heldout_predictions")

    # Cross-fit comparison deliberately keys predictions by the immutable source
    # identities, not by transformed bytes. This proves the variants share the
    # same heldout observations.
    policy_identities = [immutable_sample_identity(sample) for sample in pol_heldout_source]

    source_corpus = {
        "advantage_ordered_identity_sha256": base._ordered_identity_sha256(advantage_source),
        "strategy_ordered_identity_sha256": base._ordered_identity_sha256(strategy_source),
        "advantage_train_identity_sha256": base._ordered_identity_sha256(adv_train_source),
        "advantage_heldout_identity_sha256": base._ordered_identity_sha256(adv_heldout_source),
        "strategy_train_identity_sha256": base._ordered_identity_sha256(pol_train_source),
        "strategy_heldout_identity_sha256": base._ordered_identity_sha256(pol_heldout_source),
    }

    changed_adv = sum(
        int(src.observation_v1 != dst.observation_v1)
        for src, dst in zip(advantage_source, _transform_partition(advantage_source, args.variant))
    )
    changed_pol = sum(
        int(src.observation_v1 != dst.observation_v1)
        for src, dst in zip(strategy_source, _transform_partition(strategy_source, args.variant))
    )

    report = {
        "schema": SCHEMA,
        "variant": args.variant,
        "base_candidate": BASE_CANDIDATE,
        "domain": args.domain,
        "fit_seed": int(args.fit_seed),
        "split_seed": int(args.split_seed),
        "counts": {
            "advantage_total": len(advantage_source),
            "advantage_train": len(adv_train_source),
            "advantage_heldout": len(adv_heldout_source),
            "strategy_total": len(strategy_source),
            "strategy_train": len(pol_train_source),
            "strategy_heldout": len(pol_heldout_source),
        },
        "source_corpus": source_corpus,
        "transformation": {
            "lossless_card_symmetry_only": args.variant == "S1_V1_CARD_SYMMETRY_CANON",
            "applied_after_frozen_split": True,
            "sample_order_preserved": True,
            "changed_advantage_observations": int(changed_adv),
            "changed_strategy_observations": int(changed_pol),
        },
        "fit_contract": {
            "optimizer": "Adam",
            "learning_rate": float(args.learning_rate),
            "batch_size": int(args.batch_size),
            "advantage_steps": int(args.advantage_steps),
            "policy_steps": int(args.policy_steps),
            "early_stopping": False,
            "sample_multiplicity_preserved": True,
        },
        "parameter_count": int(parameter_count),
        "serialized_observation_bytes": 126,
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
        "ready_for_tables": False,
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
            "policy_heldout_identity": policy_identities,
            "policy_heldout_predictions": predictions,
        },
        args.out,
    )
    args.out.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
