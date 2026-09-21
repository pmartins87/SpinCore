#!/usr/bin/env python3
from __future__ import annotations

"""Audit whether the frozen DeepCrusher R8 v22 oracle is ready for canonical DC1/DC2."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))

from spincore.deepcrusher_r8_dc0 import (
    OracleCapability,
    contract_summary,
    inspect_r8_source,
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--deepcrusher-source",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--oracle-capabilities",type=Path)
    return p.parse_args()


def _load_capabilities(path:Path|None)->OracleCapability:
    fields=OracleCapability.__dataclass_fields__
    raw={}
    if path is not None and path.is_file():
        data=json.loads(path.read_text(encoding="utf-8"))
        raw=dict(data.get("capabilities") or data)
    return OracleCapability(**{
        name:bool(raw.get(name,False))
        for name in fields
    })


def main()->int:
    args=parse_args()
    closure=inspect_r8_source(args.deepcrusher_source)
    capabilities=_load_capabilities(args.oracle_capabilities)
    blockers=list(capabilities.blockers())
    if closure.missing_function_dependencies:
        blockers.append("missing_reachable_f_functions")

    verdict="PASS" if not blockers else "BLOCKED"
    payload={
        "schema":"SPINCORE_DEEPCRUSHER_R8_DC0_READINESS_V1",
        "verdict":verdict,
        "contract":contract_summary(),
        "source":{
            "path":str(args.deepcrusher_source.resolve()),
            "sections":closure.sections,
            "function_sections":closure.function_sections,
            "list_sections":closure.list_sections,
            "when_lines":closure.when_lines,
            "reachable_function_count":len(closure.reachable_functions),
            "missing_function_dependencies":list(closure.missing_function_dependencies),
            "native_identifier_count":len(closure.native_identifiers),
            "environment_identifiers":list(closure.environment_identifiers),
        },
        "capabilities":asdict(capabilities),
        "blockers":blockers,
        "canonical_benchmark_authorized":verdict=="PASS",
    }
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")

    print("=== DeepCrusher R8 v22 DC0 readiness ===")
    print(f"sections={closure.sections}")
    print(f"reachable_functions={len(closure.reachable_functions)}")
    print(f"native_identifiers={len(closure.native_identifiers)}")
    print(f"environment_identifiers={len(closure.environment_identifiers)}")
    print("missing_f_functions="+str(len(closure.missing_function_dependencies)))
    print("blockers="+(",".join(blockers) if blockers else "NONE"))
    print(f"VERDICT={verdict}")
    print("DEEPC_RUSHER_R8_DC0_READINESS_COMPLETE")
    return 0 if verdict=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
