#!/usr/bin/env python3
from __future__ import annotations

"""Aggregate the frozen LT3 post-9105 development battery."""

import argparse
import json
from pathlib import Path


def load(path:Path):
    return json.loads(path.read_text(encoding="utf-8"))


def classify(stat:dict)->str:
    lo=float(stat["ci95_low"])
    hi=float(stat["ci95_high"])
    if lo>0.0:
        return "POSITIVE"
    if hi<0.0:
        return "NEGATIVE"
    return "INCONCLUSIVE"


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    return p.parse_args()


def main()->int:
    args=parse_args()
    d=args.dir.resolve()
    cp={
        "8100_to_8600":load(d/"crossplay_8100_8600.json"),
        "8600_to_9105":load(d/"crossplay_8600_9105.json"),
        "8100_to_9105":load(d/"crossplay_8100_9105.json"),
    }
    ens={
        "8100_to_8600":load(d/"ens8_8100_8600.json"),
        "8600_to_9105":load(d/"ens8_8600_9105.json"),
        "8100_to_9105":load(d/"ens8_8100_9105.json"),
    }
    drift={
        "8100_to_8600":load(d/"drift_8100_8600.json"),
        "8600_to_9105":load(d/"drift_8600_9105.json"),
        "8100_to_9105":load(d/"drift_8100_9105.json"),
    }
    quality={
        "8100":load(d/"quality_8100.json"),
        "8600":load(d/"quality_8600.json"),
        "9105":load(d/"quality_9105.json"),
    }
    finalization=load(d/"finalize_8600.json")

    avg={}
    for name,r in cp.items():
        avg[name]={}
        for domain in ("ALL","THREE_HANDED","TRUE_HEADS_UP"):
            stat=r["mixture_hero"][domain]["paired_delta_after_minus_before"]
            avg[name][domain]={**stat,"classification":classify(stat)}
        avg[name]["hu_direct_b_vs_a"]={
            **r["hu_direct_b_vs_a"],
            "classification":classify(r["hu_direct_b_vs_a"]),
        }
        inv=r["three_handed_invasion"]["difference_b_vs_aa_minus_a_vs_bb"]
        avg[name]["three_handed_invasion_difference"]={
            **inv,
            "classification":classify(inv),
        }

    hu={}
    for name,r in ens.items():
        stat=r["direct_after_vs_before"]
        hu[name]={
            "direct":{**stat,"classification":classify(stat)},
            "weak_baselines":{
                b:{
                    **x["delta_after_minus_before"],
                    "classification":classify(x["delta_after_minus_before"]),
                }
                for b,x in r["weak_baselines"].items()
            },
        }

    primary=avg["8100_to_9105"]["ALL"]
    domain_negative=any(
        avg["8100_to_9105"][dom]["classification"]=="NEGATIVE"
        for dom in ("THREE_HANDED","TRUE_HEADS_UP")
    )
    if primary["classification"]=="POSITIVE" and not domain_negative:
        avg_screen="DEV_FAVORS_9105_OVER_8100"
    elif primary["classification"]=="NEGATIVE":
        avg_screen="DEV_REJECTS_9105_VS_8100"
    else:
        avg_screen="DEV_INCONCLUSIVE_8100_VS_9105"

    hu_primary=hu["8100_to_9105"]["direct"]
    if hu_primary["classification"]=="POSITIVE":
        hu_screen="HU_CURRENT_FAVORS_9105_OVER_8100"
    elif hu_primary["classification"]=="NEGATIVE":
        hu_screen="HU_CURRENT_REJECTS_9105_VS_8100"
    else:
        hu_screen="HU_CURRENT_INCONCLUSIVE_8100_VS_9105"

    out={
        "schema":"SPINCORE_LT3_POST9105_DEVELOPMENT_BATTERY_V1",
        "status":"PASS",
        "average_policy_crossplay":avg,
        "hu_current_ens8":hu,
        "policy_drift":{
            k:{
                "overall":v["overall"],
                "by_domain":v["by_domain"],
            }
            for k,v in drift.items()
        },
        "weak_baseline_quality":{
            k:v["summary"]
            for k,v in quality.items()
        },
        "derived_8600_finalization":finalization,
        "development_screens":{
            "average_policy":avg_screen,
            "hu_current_behavior":hu_screen,
        },
        "holdout_touched":False,
        "promotion_authorized":False,
        "warning":"Development evidence only. Do not touch the LT3 sealed holdout or promote from this report alone.",
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== LT3 POST-9105 DEVELOPMENT BATTERY ===")
    print(f"AVERAGE_POLICY_SCREEN={avg_screen}")
    print(f"HU_CURRENT_SCREEN={hu_screen}")
    for name in ("8100_to_8600","8600_to_9105","8100_to_9105"):
        x=avg[name]["ALL"]
        print(f"AVG {name}: {x['mean']:+.3f} CI95=[{x['ci95_low']:+.3f},{x['ci95_high']:+.3f}] {x['classification']}")
    for name in ("8100_to_8600","8600_to_9105","8100_to_9105"):
        x=hu[name]["direct"]
        print(f"ENS8 {name}: {x['mean']:+.3f} CI95=[{x['ci95_low']:+.3f},{x['ci95_high']:+.3f}] {x['classification']}")
    print("LT3_POST9105_DEVELOPMENT_BATTERY_PASS")
    print(f"report={args.out.resolve()}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
