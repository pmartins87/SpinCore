#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE}/checkpoint.pt"
HU="${SOURCE}/hu_ensemble_state.pt"
STRAT_RUN="${ROOT}/runs/3h_semantic_stratified_strong_diversity_10115"
STRAT_POLICY="${STRAT_RUN}/semantic_stratified_strong_diversity_tail_candidate.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_HU="8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"

[[ -x "${PY}" ]] || { echo "ERROR: missing Python venv" >&2; exit 2; }
for f in "${CHECKPOINT}" "${HU}" "${STRAT_POLICY}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing artifact: ${f}" >&2; exit 2; }
done
[[ "$(sha256sum "${CHECKPOINT}"|awk '{print $1}')" == "${EXPECTED_CP}" ]] || { echo "ERROR: checkpoint SHA mismatch" >&2; exit 3; }
[[ "$(sha256sum "${HU}"|awk '{print $1}')" == "${EXPECTED_HU}" ]] || { echo "ERROR: HU SHA mismatch" >&2; exit 3; }

PREV="$(find "${ROOT}/runs/deepcrusher_dc1_semantic_diversity_1k" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
[[ -n "${PREV}" && -d "${PREV}" ]] || { echo "ERROR: prior exact three-arm DC1 1k run not found" >&2; exit 4; }

BASE_BUNDLE="${PREV}/spincore_hybrid_10105.pt"
BASE_REPORT="${PREV}/baseline_10105_report.json"
BASE_SANITY="${PREV}/baseline_10105_sanity.json"
CANON_REPORT="${PREV}/canonical_10115_report.json"
CANON_SANITY="${PREV}/canonical_10115_sanity.json"
DIV1_REPORT="${PREV}/diversity_10115_report.json"
DIV1_SANITY="${PREV}/diversity_10115_sanity.json"

for f in "${BASE_BUNDLE}" "${BASE_REPORT}" "${BASE_SANITY}" "${CANON_REPORT}" "${CANON_SANITY}" "${DIV1_REPORT}" "${DIV1_SANITY}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing prior DC1 artifact: ${f}" >&2; exit 4; }
done

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"${PY}" -m py_compile \
  tools/evaluate_deepcrusher_dc1_semantic_candidate.py \
  tools/test_deepcrusher_nouts_native_collision.py

"${PY}" tools/test_deepcrusher_nouts_native_collision.py

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${ROOT}/runs/deepcrusher_dc1_semantic_stratified_1k/${STAMP}"
mkdir -p "${RUN}"
STRAT_REPORT="${RUN}/stratified_10115_report.json"
STRAT_TRACES="${RUN}/stratified_10115_traces.jsonl"
STRAT_SANITY="${RUN}/stratified_10115_sanity.json"
COMPARE="${RUN}/stratified_vs_prior_1k_comparison.json"

"${PY}" tools/evaluate_deepcrusher_dc1_semantic_candidate.py \
  --solver build/libspincore_solver_c.so \
  --base-spin-bundle "${BASE_BUNDLE}" \
  --semantic-policy "${STRAT_POLICY}" \
  --scenarios 1000 --workers 8 --seed 20260923 \
  --report "${STRAT_REPORT}" --traces "${STRAT_TRACES}" --max-decisions 200

"${PY}" tools/audit_benchmark_decisions.py \
  --traces "${STRAT_TRACES}" --report "${STRAT_SANITY}" --max-examples-per-code 50

"${PY}" - \
  "${BASE_REPORT}" "${CANON_REPORT}" "${DIV1_REPORT}" "${STRAT_REPORT}" \
  "${BASE_SANITY}" "${CANON_SANITY}" "${DIV1_SANITY}" "${STRAT_SANITY}" \
  "${COMPARE}" <<'PY'
import json,math,statistics,sys
from pathlib import Path

bp,cp,dp,sp,bs,cs,ds,ss,out=map(Path,sys.argv[1:])
b=json.loads(bp.read_text())
c=json.loads(cp.read_text())
d=json.loads(dp.read_text())
s=json.loads(sp.read_text())
bsa=json.loads(bs.read_text())
csa=json.loads(cs.read_text())
dsa=json.loads(ds.read_text())
ssa=json.loads(ss.read_text())

reports=(b,c,d,s)
for r in reports:
    if int(r.get("seed",-1))!=20260923 or int(r.get("scenarios",-1))!=1000:
        raise RuntimeError("prior/new DC1 run contract drift")

def ordered(report):
    return sorted(report["scenario_rows"],key=lambda x:x["scenario"])

br,cr,dr,sr=[ordered(r) for r in reports]
if not all(len(x)==1000 for x in (br,cr,dr,sr)):
    raise RuntimeError("scenario count mismatch")

for rows in zip(br,cr,dr,sr):
    ids=[]
    for x in rows:
        ids.append((
            x["scenario"],x["domain"],x["blind"],x["dealer_id"],x["deal_seed"]
        ))
    if len(set(ids))!=1:
        raise RuntimeError("scenario identity mismatch")

def margin(row):
    return float(row["paired_spincore_minus_deepcrusher_chips_per_exposure"])

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

def paired(left,right,domain=None):
    vals=[]
    for x,y in zip(left,right):
        if domain is None or x["domain"]==domain:
            vals.append(margin(y)-margin(x))
    return ci(vals)

def fcount(sanity,code):
    return int((sanity.get("flag_counts") or {}).get(code,0))

def frate(sanity,code):
    den=max(1,int(sanity.get("spincore_decisions",0)))
    return fcount(sanity,code)/den

hu_exact=all(
    margin(x)==margin(y)==margin(z)==margin(w)
    for x,y,z,w in zip(br,cr,dr,sr)
    if x["domain"]=="TRUE_HEADS_UP"
)

codes=sorted(
    set((bsa.get("flag_counts") or {}))
    |set((csa.get("flag_counts") or {}))
    |set((dsa.get("flag_counts") or {}))
    |set((ssa.get("flag_counts") or {}))
)
flags={
    code:{
        "baseline_count":fcount(bsa,code),
        "canonical_count":fcount(csa,code),
        "diversity_v1_count":fcount(dsa,code),
        "stratified_count":fcount(ssa,code),
        "baseline_rate":frate(bsa,code),
        "canonical_rate":frate(csa,code),
        "diversity_v1_rate":frate(dsa,code),
        "stratified_rate":frate(ssa,code),
    }
    for code in codes
}

sb3=paired(br,sr,"THREE_HANDED")
sc3=paired(cr,sr,"THREE_HANDED")
sd3=paired(dr,sr,"THREE_HANDED")

trip="POSTFLOP_TRIPS_PLUS_FOLD"
hc="POSTFLOP_DEEP_HIGH_CARD_JAM"
tp="POSTFLOP_TOP_PAIR_FOLD"
aa="PREFLOP_AA_FOLD"
s72="PREFLOP_DEEP_72O_JAM"

repair={
    "hu_parity_exact_all_four":hu_exact,
    "stratified_trips_fold_no_higher_than_baseline":(
        fcount(ssa,trip)<=fcount(bsa,trip)
    ),
    "stratified_high_card_jam_rate_no_more_than_10pct_above_diversity_v1":(
        frate(ssa,hc)<=1.10*frate(dsa,hc)+1e-15
    ),
    "stratified_top_pair_fold_no_higher_than_diversity_v1":(
        fcount(ssa,tp)<=fcount(dsa,tp)
    ),
    "stratified_preflop_aa_fold_no_higher_than_diversity_v1":(
        fcount(ssa,aa)<=fcount(dsa,aa)
    ),
    "stratified_deep_72o_jam_no_higher_than_diversity_v1":(
        fcount(ssa,s72)<=fcount(dsa,s72)
    ),
    "stratified_vs_baseline_3h_mean_positive":sb3["mean"]>0.0,
}
repair_pass=all(repair.values())

report={
    "schema":"SPINCORE_DC1_SEMANTIC_STRATIFIED_STRONG_DIVERSITY_1K_COMPARISON_V1",
    "scope":"DEVELOPMENT_ONLY_NO_CANONICAL_STRENGTH_CLAIM",
    "seed":20260923,
    "scenarios":1000,
    "domain_counts":b["domain_counts"],
    "reused_prior_three_arm_run":str(bp.parent),
    "hu_exact_scenario_margin_parity":hu_exact,
    "absolute_3h_vs_deepcrusher":{
        "baseline_10105":b["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
        "canonical_10115":c["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
        "diversity_v1_10115":d["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
        "stratified_10115":s["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
    },
    "paired_scenario_cluster_delta":{
        "stratified_minus_baseline":{
            "ALL":paired(br,sr),
            "THREE_HANDED":sb3,
        },
        "stratified_minus_canonical":{
            "ALL":paired(cr,sr),
            "THREE_HANDED":sc3,
        },
        "stratified_minus_diversity_v1":{
            "ALL":paired(dr,sr),
            "THREE_HANDED":sd3,
        },
    },
    "sanity_flags":flags,
    "precommitted_scaleup_indicators":repair,
    "stratified_dc1_1k_scaleup_gate_pass":repair_pass,
    "interpretation":(
        "Development-only exact same-seed replay. PASS means the stratified "
        "candidate restores the targeted strong-hand sanity surface to no worse "
        "than baseline, avoids material regression on the frozen companion sanity "
        "surfaces, preserves exact HU parity, and retains a positive 3H mean delta "
        "versus baseline. PASS authorizes a 5k development scale-up only; DC0 "
        "real-OpenHoldem parity remains pending and no canonical strength or "
        "production/DC2 claim is authorized."
    ),
}
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

print("=== DC1 stratified strong-diversity 1k ===")
print("stratified_minus_baseline_3h="+json.dumps(sb3,sort_keys=True))
print("stratified_minus_canonical_3h="+json.dumps(sc3,sort_keys=True))
print("stratified_minus_diversity_v1_3h="+json.dumps(sd3,sort_keys=True))
print("scaleup_indicators="+json.dumps(repair,sort_keys=True))
print(f"stratified_dc1_1k_scaleup_gate_pass={repair_pass}")
print(f"comparison={out}")
print("DC1_STRATIFIED_STRONG_DIVERSITY_1K_COMPARISON_COMPLETE")
PY

"${PY}" - "${RUN}" "${PREV}" <<'PY'
from pathlib import Path
import sys,zipfile
run=Path(sys.argv[1]); prev=Path(sys.argv[2])
out=run/"SpinCore_DC1_semantic_stratified_1k_bundle.zip"
files=[
    (prev/"baseline_10105_report.json","baseline_10105_report.json"),
    (prev/"baseline_10105_sanity.json","baseline_10105_sanity.json"),
    (prev/"canonical_10115_report.json","canonical_10115_report.json"),
    (prev/"canonical_10115_sanity.json","canonical_10115_sanity.json"),
    (prev/"diversity_10115_report.json","diversity_v1_10115_report.json"),
    (prev/"diversity_10115_sanity.json","diversity_v1_10115_sanity.json"),
    (run/"stratified_10115_report.json","stratified_10115_report.json"),
    (run/"stratified_10115_sanity.json","stratified_10115_sanity.json"),
    (run/"stratified_vs_prior_1k_comparison.json","stratified_vs_prior_1k_comparison.json"),
]
with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for src,arc in files:
        if not src.is_file():
            raise RuntimeError(f"missing bundle source {src}")
        z.write(src,arcname=arc)
print(f"bundle={out}")
PY

ZIP="${RUN}/SpinCore_DC1_semantic_stratified_1k_bundle.zip"
if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${ZIP}" "${DWSL}/SpinCore_DC1_semantic_stratified_1k_bundle.zip"
    echo "desktop_bundle=${DWIN}\\SpinCore_DC1_semantic_stratified_1k_bundle.zip"
  fi
fi

echo "DEEPC_RUSHER_DC1_SEMANTIC_STRATIFIED_1K_COMPLETE"
