#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE}/checkpoint.pt"
HU="${SOURCE}/hu_ensemble_state.pt"
CANON_RUN="${ROOT}/runs/3h_semantic_research_canonical_10105_10115"
CANON_POLICY="${CANON_RUN}/semantic_tail_policy_10115.pt"
DIV_RUN="${ROOT}/runs/3h_semantic_strong_diversity_repair_10115"
DIV_POLICY="${DIV_RUN}/semantic_strong_diversity_tail_candidate.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_HU="8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"

[[ -x "${PY}" ]] || { echo "ERROR: missing Python venv" >&2; exit 2; }
for f in "${CHECKPOINT}" "${HU}" "${CANON_POLICY}" "${DIV_POLICY}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing artifact: ${f}" >&2; exit 2; }
done
[[ "$(sha256sum "${CHECKPOINT}"|awk '{print $1}')" == "${EXPECTED_CP}" ]] || { echo "ERROR: checkpoint SHA mismatch" >&2; exit 3; }
[[ "$(sha256sum "${HU}"|awk '{print $1}')" == "${EXPECTED_HU}" ]] || { echo "ERROR: HU SHA mismatch" >&2; exit 3; }

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"${PY}" -m py_compile   tools/evaluate_deepcrusher_dc1_semantic_candidate.py   tools/test_deepcrusher_nouts_native_collision.py

"${PY}" tools/test_deepcrusher_nouts_native_collision.py

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${ROOT}/runs/deepcrusher_dc1_semantic_diversity_1k/${STAMP}"
mkdir -p "${RUN}"

BASE_BUNDLE="${RUN}/spincore_hybrid_10105.pt"
BASE_REPORT="${RUN}/baseline_10105_report.json"
BASE_TRACES="${RUN}/baseline_10105_traces.jsonl"
BASE_SANITY="${RUN}/baseline_10105_sanity.json"

CANON_REPORT="${RUN}/canonical_10115_report.json"
CANON_TRACES="${RUN}/canonical_10115_traces.jsonl"
CANON_SANITY="${RUN}/canonical_10115_sanity.json"

DIV_REPORT="${RUN}/diversity_10115_report.json"
DIV_TRACES="${RUN}/diversity_10115_traces.jsonl"
DIV_SANITY="${RUN}/diversity_10115_sanity.json"

COMPARE="${RUN}/diversity_vs_canonical_vs_baseline_comparison.json"

"${PY}" tools/export_spincore_hybrid_inference.py   --checkpoint "${CHECKPOINT}" --ensemble "${HU}" --out "${BASE_BUNDLE}"

"${PY}" tools/evaluate_deepcrusher_dc1.py   --solver build/libspincore_solver_c.so   --spin-bundle "${BASE_BUNDLE}"   --scenarios 1000 --workers 8 --seed 20260923   --report "${BASE_REPORT}" --traces "${BASE_TRACES}" --max-decisions 200

"${PY}" tools/audit_benchmark_decisions.py   --traces "${BASE_TRACES}" --report "${BASE_SANITY}" --max-examples-per-code 50

"${PY}" tools/evaluate_deepcrusher_dc1_semantic_candidate.py   --solver build/libspincore_solver_c.so   --base-spin-bundle "${BASE_BUNDLE}"   --semantic-policy "${CANON_POLICY}"   --scenarios 1000 --workers 8 --seed 20260923   --report "${CANON_REPORT}" --traces "${CANON_TRACES}" --max-decisions 200

"${PY}" tools/audit_benchmark_decisions.py   --traces "${CANON_TRACES}" --report "${CANON_SANITY}" --max-examples-per-code 50

"${PY}" tools/evaluate_deepcrusher_dc1_semantic_candidate.py   --solver build/libspincore_solver_c.so   --base-spin-bundle "${BASE_BUNDLE}"   --semantic-policy "${DIV_POLICY}"   --scenarios 1000 --workers 8 --seed 20260923   --report "${DIV_REPORT}" --traces "${DIV_TRACES}" --max-decisions 200

"${PY}" tools/audit_benchmark_decisions.py   --traces "${DIV_TRACES}" --report "${DIV_SANITY}" --max-examples-per-code 50

"${PY}" -   "${BASE_REPORT}" "${CANON_REPORT}" "${DIV_REPORT}"   "${BASE_SANITY}" "${CANON_SANITY}" "${DIV_SANITY}" "${COMPARE}" <<'PY'
import json,math,statistics,sys
from pathlib import Path

bp,cp,dp,bs,cs,ds,out=map(Path,sys.argv[1:])
b=json.loads(bp.read_text())
c=json.loads(cp.read_text())
d=json.loads(dp.read_text())
bsa=json.loads(bs.read_text())
csa=json.loads(cs.read_text())
dsa=json.loads(ds.read_text())

def ordered(report):
    return sorted(report["scenario_rows"],key=lambda x:x["scenario"])

br,cr,dr=ordered(b),ordered(c),ordered(d)
if not (len(br)==len(cr)==len(dr)==1000):
    raise RuntimeError("scenario count mismatch")

for x,y,z in zip(br,cr,dr):
    ix=(x["scenario"],x["domain"],x["blind"],x["dealer_id"],x["deal_seed"])
    iy=(y["scenario"],y["domain"],y["blind"],y["dealer_id"],y["deal_seed"])
    iz=(z["scenario"],z["domain"],z["blind"],z["dealer_id"],z["deal_seed"])
    if not (ix==iy==iz):
        raise RuntimeError("scenario identity mismatch")

def ci(vals):
    vals=[float(x) for x in vals]
    n=len(vals)
    mean=statistics.fmean(vals) if vals else float("nan")
    sem=statistics.stdev(vals)/math.sqrt(n) if n>1 else 0.0
    return {
        "n":n,"mean":mean,"sem":sem,
        "ci95_low":mean-1.96*sem,"ci95_high":mean+1.96*sem,
        "median":statistics.median(vals) if vals else float("nan"),
        "positive":sum(x>0 for x in vals),
        "negative":sum(x<0 for x in vals),
        "zero":sum(x==0 for x in vals),
    }

def margin(row):
    return float(row["paired_spincore_minus_deepcrusher_chips_per_exposure"])

def paired_delta(left,right,domain=None):
    vals=[]
    for x,y in zip(left,right):
        if domain is None or x["domain"]==domain:
            vals.append(margin(y)-margin(x))
    return ci(vals)

def flag_count(sanity,code):
    return int((sanity.get("flag_counts") or {}).get(code,0))

def flag_rate(sanity,code):
    den=max(1,int(sanity.get("spincore_decisions",0)))
    return flag_count(sanity,code)/den

hu_exact_bc=all(
    margin(x)==margin(y)
    for x,y in zip(br,cr) if x["domain"]=="TRUE_HEADS_UP"
)
hu_exact_bd=all(
    margin(x)==margin(z)
    for x,z in zip(br,dr) if x["domain"]=="TRUE_HEADS_UP"
)
hu_exact_cd=all(
    margin(y)==margin(z)
    for y,z in zip(cr,dr) if y["domain"]=="TRUE_HEADS_UP"
)

codes=sorted(
    set((bsa.get("flag_counts") or {}))
    |set((csa.get("flag_counts") or {}))
    |set((dsa.get("flag_counts") or {}))
)
flags={
    code:{
        "baseline_count":flag_count(bsa,code),
        "canonical_count":flag_count(csa,code),
        "diversity_count":flag_count(dsa,code),
        "baseline_rate":flag_rate(bsa,code),
        "canonical_rate":flag_rate(csa,code),
        "diversity_rate":flag_rate(dsa,code),
    }
    for code in codes
}

bc3=paired_delta(br,cr,"THREE_HANDED")
bd3=paired_delta(br,dr,"THREE_HANDED")
cd3=paired_delta(cr,dr,"THREE_HANDED")

report={
    "schema":"SPINCORE_DC1_SEMANTIC_STRONG_DIVERSITY_1K_COMPARISON_V1",
    "scope":"DEVELOPMENT_ONLY_NO_CANONICAL_STRENGTH_CLAIM",
    "seed":20260923,
    "scenarios":1000,
    "domain_counts":b["domain_counts"],
    "hu_exact_scenario_margin_parity":{
        "baseline_vs_canonical":hu_exact_bc,
        "baseline_vs_diversity":hu_exact_bd,
        "canonical_vs_diversity":hu_exact_cd,
    },
    "absolute_3h_vs_deepcrusher":{
        "baseline_10105":b["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
        "canonical_10115":c["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
        "diversity_10115":d["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
    },
    "paired_scenario_cluster_delta":{
        "canonical_minus_baseline":{
            "ALL":paired_delta(br,cr),
            "THREE_HANDED":bc3,
        },
        "diversity_minus_baseline":{
            "ALL":paired_delta(br,dr),
            "THREE_HANDED":bd3,
        },
        "diversity_minus_canonical":{
            "ALL":paired_delta(cr,dr),
            "THREE_HANDED":cd3,
        },
    },
    "sanity_flags":flags,
    "repair_indicators":{
        "hu_parity_exact_all_three":hu_exact_bc and hu_exact_bd and hu_exact_cd,
        "diversity_trips_fold_no_higher_than_baseline":(
            flag_count(dsa,"POSTFLOP_TRIPS_PLUS_FOLD")
            <=flag_count(bsa,"POSTFLOP_TRIPS_PLUS_FOLD")
        ),
        "diversity_trips_fold_lower_than_or_equal_to_canonical":(
            flag_count(dsa,"POSTFLOP_TRIPS_PLUS_FOLD")
            <=flag_count(csa,"POSTFLOP_TRIPS_PLUS_FOLD")
        ),
        "diversity_high_card_jam_not_higher_than_canonical":(
            flag_rate(dsa,"POSTFLOP_DEEP_HIGH_CARD_JAM")
            <=flag_rate(csa,"POSTFLOP_DEEP_HIGH_CARD_JAM")
        ),
        "diversity_top_pair_fold_not_higher_than_canonical":(
            flag_count(dsa,"POSTFLOP_TOP_PAIR_FOLD")
            <=flag_count(csa,"POSTFLOP_TOP_PAIR_FOLD")
        ),
        "diversity_preflop_aa_fold_not_higher_than_canonical":(
            flag_count(dsa,"PREFLOP_AA_FOLD")
            <=flag_count(csa,"PREFLOP_AA_FOLD")
        ),
        "diversity_vs_canonical_3h_mean_nonnegative":cd3["mean"]>=0.0,
        "diversity_vs_baseline_3h_mean_positive":bd3["mean"]>0.0,
    },
    "interpretation":(
        "Development-only exact replay. The diversity candidate is compared with "
        "both the unchanged 10105 baseline and the canonical 10115 tail policy "
        "on identical scenario/deal seeds, with unchanged HU ENS8. DC0 real-"
        "OpenHoldem parity remains pending, so this cannot authorize a canonical "
        "strength claim."
    ),
}
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

print("=== DC1 semantic strong-diversity 1k ===")
print("canonical_minus_baseline_3h="+json.dumps(bc3,sort_keys=True))
print("diversity_minus_baseline_3h="+json.dumps(bd3,sort_keys=True))
print("diversity_minus_canonical_3h="+json.dumps(cd3,sort_keys=True))
print("repair_indicators="+json.dumps(report["repair_indicators"],sort_keys=True))
print(f"comparison={out}")
print("DC1_SEMANTIC_STRONG_DIVERSITY_1K_COMPARISON_COMPLETE")
PY

"${PY}" - "${RUN}" <<'PY'
from pathlib import Path
import sys,zipfile
run=Path(sys.argv[1])
out=run/"SpinCore_DC1_semantic_strong_diversity_1k_bundle.zip"
names=(
    "baseline_10105_report.json","baseline_10105_sanity.json",
    "canonical_10115_report.json","canonical_10115_sanity.json",
    "diversity_10115_report.json","diversity_10115_sanity.json",
    "diversity_vs_canonical_vs_baseline_comparison.json",
)
with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for name in names:
        p=run/name
        if p.is_file():
            z.write(p,arcname=name)
print(f"bundle={out}")
PY

ZIP="${RUN}/SpinCore_DC1_semantic_strong_diversity_1k_bundle.zip"
if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${ZIP}" "${DWSL}/SpinCore_DC1_semantic_strong_diversity_1k_bundle.zip"
    echo "desktop_bundle=${DWIN}\\SpinCore_DC1_semantic_strong_diversity_1k_bundle.zip"
  fi
fi

echo "DEEPC_RUSHER_DC1_SEMANTIC_STRONG_DIVERSITY_1K_COMPLETE"
