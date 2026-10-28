#!/usr/bin/env python3
from __future__ import annotations

"""Consolidate the pure-teacher 17315 sanity/root-cause audit.

PROJECT_CONTRACT_IDS:
DC-002,SAFE-001,SAFE-011,VALID-025,VALID-031,VALID-040,RNG-001,RNG-002,RNG-003

No model selection or policy mutation occurs here. This tool only validates and
compresses the three diagnostic reports into a human-reviewable summary.
"""

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def strip_top_rows(surface: dict) -> dict:
    return {
        k: v
        for k, v in surface.items()
        if k != "top_probability_rows"
    }


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sanity",type=Path,required=True)
    ap.add_argument("--raw",type=Path,required=True)
    ap.add_argument("--trend",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()

    sanity=load(args.sanity)
    raw=load(args.raw)
    trend=load(args.trend)

    if sanity.get("schema")!="SPINCORE_3H_TEACHER_SANITY_17315_V1":
        raise RuntimeError("wrong sanity schema")
    if raw.get("schema")!="SPINCORE_3H_TEACHER_RAW_ADVANTAGE_DIAGNOSTIC_17315_V1":
        raise RuntimeError("wrong raw diagnostic schema")
    if trend.get("schema")!="SPINCORE_3H_TEACHER_FLAGGED_CHECKPOINT_TREND_V1":
        raise RuntimeError("wrong checkpoint trend schema")
    if int(sanity.get("teacher_iteration",-1))!=17315:
        raise RuntimeError("sanity teacher iteration drift")
    if int(raw.get("teacher_iteration",-1))!=17315:
        raise RuntimeError("raw teacher iteration drift")
    if raw.get("status")!="PASS":
        raise RuntimeError("raw diagnostic replay is not PASS")
    if trend.get("status")!="PASS":
        raise RuntimeError("checkpoint trend replay is not PASS")
    if int(sanity.get("seed",-1))!=int(raw.get("seed",-2)):
        raise RuntimeError("sanity/raw seed mismatch")
    if int(sanity.get("seed",-1))!=int(trend.get("seed",-2)):
        raise RuntimeError("sanity/trend seed mismatch")

    policy={
        code:strip_top_rows(surface)
        for code,surface in sorted(
            (sanity.get("opportunity_policy") or {}).items()
        )
    }

    highlights=[]
    for row in raw.get("rows") or []:
        m=row.get("mechanism") or {}
        highlights.append({
            "scenario":row.get("scenario"),
            "lineup_index":row.get("lineup_index"),
            "decision_index":row.get("decision_index"),
            "hole":row.get("hole"),
            "board":row.get("board"),
            "pot":row.get("pot"),
            "to_call":row.get("to_call"),
            "effective_stack_bb":row.get("effective_stack_bb"),
            "flag_code":row.get("flag_code"),
            "policy_surface_codes":row.get("policy_surface_codes") or [],
            "audit_reasons":row.get("audit_reasons") or [],
            "diagnostic_target_action":row.get("diagnostic_target_action"),
            "actual_selected_action":(
                (row.get("policy_detail") or {}).get("selected_slot_name")
            ),
            "branch":(row.get("policy_detail") or {}).get("branch"),
            "ensemble_target_probability":m.get(
                "ensemble_selected_probability"
            ),
            "member_policy_mixture_target_probability":m.get(
                "member_policy_mixture_selected_probability"
            ),
            "ensemble_minus_member_mixture_probability":m.get(
                "ensemble_minus_member_mixture_probability"
            ),
            "member_target_positive_count":m.get(
                "member_selected_positive_count"
            ),
            "member_target_policy_argmax_count":m.get(
                "member_selected_policy_argmax_count"
            ),
            "sole_positive_mean_raw":m.get(
                "ensemble_selected_is_sole_positive_mean_raw"
            ),
            "extreme_ge_80pct":m.get(
                "ensemble_extreme_probability_ge_80pct"
            ),
            "raw_mean_amplification_ge_25pp":m.get(
                "raw_mean_amplification_ge_25pp"
            ),
            "extreme_with_no_member_majority":m.get(
                "extreme_with_no_member_majority"
            ),
        })

    highlights.sort(
        key=lambda x:(
            float(x["ensemble_target_probability"] or 0.0),
            float(x["ensemble_minus_member_mixture_probability"] or 0.0),
        ),
        reverse=True,
    )

    summary={
        "schema":"SPINCORE_3H_TEACHER_PURE_AUDIT_17315_SUMMARY_V1",
        "status":"COMPLETE",
        "teacher_iteration":17315,
        "seed":int(sanity["seed"]),
        "scenarios":int(sanity["scenarios"]),
        "balanced_games":int(sanity["balanced_games"]),
        "spin_decisions":int(sanity["spin_decisions"]),
        "sanity_status":sanity["status"],
        "sampled_flag_counts":dict(sanity.get("flag_counts") or {}),
        "sampled_deliberate_allin_flag_counts":dict(
            sanity.get("deliberate_allin_slot_flag_counts") or {}
        ),
        "opportunity_counts":dict(sanity.get("opportunity_counts") or {}),
        "opportunity_policy":policy,
        "raw_branch_counts":dict(raw.get("branch_counts") or {}),
        "raw_branch_by_flag":dict(raw.get("branch_by_flag") or {}),
        "raw_mechanism_by_flag":dict(raw.get("mechanism_by_flag") or {}),
        "raw_mechanism_by_reason":dict(raw.get("mechanism_by_reason") or {}),
        "checkpoint_trend_summary":dict(trend.get("summary") or {}),
        "raw_unique_target_count":int(raw.get("unique_target_count",0)),
        "top_root_cause_highlights":highlights[:50],
        "interpretation":{
            "sampled_flags":(
                "Actions actually sampled from teacher 17315 on the fresh audit."
            ),
            "opportunity_policy":(
                "Target-action probabilities on every eligible state, independent "
                "of sampled action RNG."
            ),
            "raw_mechanism":(
                "Compares regret matching after mean raw Advantages with the mean "
                "of member-level policies and counts member consensus."
            ),
            "checkpoint_trend":(
                "Counterfactual probabilities on the exact same flagged 17315 "
                "information states under teachers 10315/14815/15315/17315."
            ),
            "scope":(
                "Development-only pure-teacher diagnostic. No AveragePolicy, "
                "specialist, rebuild, model selection, or promotion claim."
            ),
        },
    }

    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n",
        encoding="utf-8",
    )
    print("=== Pure teacher 17315 audit summary ===")
    print("sanity_status="+str(summary["sanity_status"]))
    print(
        "sampled_flag_counts="
        +json.dumps(summary["sampled_flag_counts"],sort_keys=True)
    )
    print(
        "opportunity_policy="
        +json.dumps(summary["opportunity_policy"],sort_keys=True)
    )
    print(
        "raw_mechanism_by_reason="
        +json.dumps(summary["raw_mechanism_by_reason"],sort_keys=True)
    )
    print(f"summary={args.out.resolve()}")
    print("SPINCORE_3H_TEACHER_PURE_AUDIT_17315_SUMMARY_PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
