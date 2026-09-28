#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE}/checkpoint.pt"
HU="${SOURCE}/hu_ensemble_state.pt"
FULL_RUN="${ROOT}/runs/3h_semantic_fullpool_strong_diversity_10115"
FULL_AUDIT="${FULL_RUN}/3h_semantic_fullpool_strong_diversity_10115.json"
FULL_MODEL="${FULL_RUN}/semantic_fullpool_strong_diversity_tail_candidate.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_HU="8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"

for f in "${PY}" "${CHECKPOINT}" "${HU}" "${FULL_AUDIT}" "${FULL_MODEL}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing required artifact: ${f}" >&2; exit 2; }
done

[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${EXPECTED_CP}" ]] || { echo "ERROR: checkpoint SHA mismatch" >&2; exit 3; }
[[ "$(sha256sum "${HU}" | awk '{print $1}')" == "${EXPECTED_HU}" ]] || { echo "ERROR: HU SHA mismatch" >&2; exit 3; }

PASS="$("${PY}" - "${FULL_AUDIT}" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
ok=bool(r.get("semantic_fullpool_diversity_repair_pass"))
ok=ok and int(r.get("novel_strong_candidates",-1))==1039
print("1" if ok else "0")
PY
)"
[[ "${PASS}" == "1" ]] || { echo "ERROR: saved full-pool offline gate is not PASS" >&2; exit 4; }

PREV="$(find "${ROOT}/runs/deepcrusher_dc1_semantic_stratified_1k" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
[[ -n "${PREV}" && -d "${PREV}" ]] || { echo "ERROR: prior stratified DC1 run missing" >&2; exit 5; }
PREVCMP="${PREV}/stratified_vs_prior_1k_comparison.json"
STRAT_REPORT="${PREV}/stratified_10115_report.json"
STRAT_SANITY="${PREV}/stratified_10115_sanity.json"
for f in "${PREVCMP}" "${STRAT_REPORT}" "${STRAT_SANITY}"; do
  [[ -f "${f}" ]] || { echo "ERROR: prior stratified artifact missing: ${f}" >&2; exit 5; }
done

PRIOR3="$("${PY}" - "${PREVCMP}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["reused_prior_three_arm_run"])
PY
)"
BASE_BUNDLE="${PRIOR3}/spincore_hybrid_10105.pt"
BASE_REPORT="${PRIOR3}/baseline_10105_report.json"
BASE_SANITY="${PRIOR3}/baseline_10105_sanity.json"
for f in "${BASE_BUNDLE}" "${BASE_REPORT}" "${BASE_SANITY}"; do
  [[ -f "${f}" ]] || { echo "ERROR: prior baseline artifact missing: ${f}" >&2; exit 5; }
done

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"${PY}" -m py_compile tools/evaluate_deepcrusher_dc1_semantic_candidate.py
"${PY}" tools/test_deepcrusher_nouts_native_collision.py

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${ROOT}/runs/deepcrusher_dc1_semantic_fullpool_1k/${STAMP}"
mkdir -p "${RUN}"
REPORT="${RUN}/fullpool_10115_report.json"
TRACES="${RUN}/fullpool_10115_traces.jsonl"
SANITY="${RUN}/fullpool_10115_sanity.json"
COMPARE="${RUN}/fullpool_vs_stratified_vs_baseline_comparison.json"

"${PY}" tools/evaluate_deepcrusher_dc1_semantic_candidate.py \
  --solver build/libspincore_solver_c.so \
  --base-spin-bundle "${BASE_BUNDLE}" \
  --semantic-policy "${FULL_MODEL}" \
  --scenarios 1000 --workers 8 --seed 20260923 \
  --report "${REPORT}" --traces "${TRACES}" --max-decisions 200

"${PY}" tools/audit_benchmark_decisions.py \
  --traces "${TRACES}" --report "${SANITY}" --max-examples-per-code 50

"${PY}" - "${BASE_REPORT}" "${STRAT_REPORT}" "${REPORT}" "${BASE_SANITY}" "${STRAT_SANITY}" "${SANITY}" "${COMPARE}" <<'PY'
import json,math,statistics,sys
from pathlib import Path

bp,sp,fp,bs,ss,fs,out=map(Path,sys.argv[1:])
b=json.loads(bp.read_text())
s=json.loads(sp.read_text())
f=json.loads(fp.read_text())
bsa=json.loads(bs.read_text())
ssa=json.loads(ss.read_text())
fsa=json.loads(fs.read_text())

def rows(r):
    return sorted(r["scenario_rows"],key=lambda x:x["scenario"])

br,sr,fr=rows(b),rows(s),rows(f)
for x,y,z in zip(br,sr,fr):
    ids=[(q["scenario"],q["domain"],q["blind"],q["dealer_id"],q["deal_seed"]) for q in (x,y,z)]
    if len(set(ids))!=1:
        raise RuntimeError("scenario identity mismatch")

def margin(r):
    return float(r["paired_spincore_minus_deepcrusher_chips_per_exposure"])

def ci(vals):
    vals=[float(x) for x in vals]
    n=len(vals)
    mean=statistics.fmean(vals)
    sem=statistics.stdev(vals)/math.sqrt(n)
    return {
        "n":n,"mean":mean,"sem":sem,
        "ci95_low":mean-1.96*sem,"ci95_high":mean+1.96*sem,
        "median":statistics.median(vals),
        "positive":sum(x>0 for x in vals),
        "negative":sum(x<0 for x in vals),
        "zero":sum(x==0 for x in vals),
    }

def paired(a,b,domain=None):
    return ci([margin(y)-margin(x) for x,y in zip(a,b) if domain is None or x["domain"]==domain])

def count(sanity,code):
    return int((sanity.get("flag_counts") or {}).get(code,0))

def rate(sanity,code):
    return count(sanity,code)/max(1,int(sanity.get("spincore_decisions",0)))

trip="POSTFLOP_TRIPS_PLUS_FOLD"
hc="POSTFLOP_DEEP_HIGH_CARD_JAM"
tp="POSTFLOP_TOP_PAIR_FOLD"
aa="PREFLOP_AA_FOLD"
d72="PREFLOP_DEEP_72O_JAM"

fb3=paired(br,fr,"THREE_HANDED")
fs3=paired(sr,fr,"THREE_HANDED")
hu=all(
    margin(x)==margin(y)==margin(z)
    for x,y,z in zip(br,sr,fr)
    if x["domain"]=="TRUE_HEADS_UP"
)

indicators={
    "hu_parity_exact":hu,
    "fullpool_trips_fold_no_higher_than_baseline":count(fsa,trip)<=count(bsa,trip),
    "fullpool_high_card_jam_rate_no_more_than_10pct_above_stratified":rate(fsa,hc)<=1.10*rate(ssa,hc)+1e-15,
    "fullpool_top_pair_fold_no_higher_than_stratified":count(fsa,tp)<=count(ssa,tp),
    "fullpool_preflop_aa_fold_no_higher_than_stratified":count(fsa,aa)<=count(ssa,aa),
    "fullpool_deep_72o_jam_no_higher_than_stratified":count(fsa,d72)<=count(ssa,d72),
    "fullpool_vs_baseline_3h_mean_positive":fb3["mean"]>0,
}

codes=sorted(
    set((bsa.get("flag_counts") or {}))
    | set((ssa.get("flag_counts") or {}))
    | set((fsa.get("flag_counts") or {}))
)

report={
    "schema":"SPINCORE_DC1_SEMANTIC_FULLPOOL_STRONG_DIVERSITY_1K_COMPARISON_V1",
    "scope":"DEVELOPMENT_ONLY_NO_CANONICAL_STRENGTH_CLAIM",
    "seed":20260923,
    "scenarios":1000,
    "hu_exact_scenario_margin_parity":hu,
    "absolute_3h_vs_deepcrusher":{
        "baseline_10105":b["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
        "stratified_10115":s["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
        "fullpool_10115":f["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
    },
    "paired_scenario_cluster_delta":{
        "fullpool_minus_baseline":{"ALL":paired(br,fr),"THREE_HANDED":fb3},
        "fullpool_minus_stratified":{"ALL":paired(sr,fr),"THREE_HANDED":fs3},
    },
    "sanity_flags":{
        code:{
            "baseline_count":count(bsa,code),
            "stratified_count":count(ssa,code),
            "fullpool_count":count(fsa,code),
            "baseline_rate":rate(bsa,code),
            "stratified_rate":rate(ssa,code),
            "fullpool_rate":rate(fsa,code),
        }
        for code in codes
    },
    "precommitted_scaleup_indicators":indicators,
    "fullpool_dc1_1k_scaleup_gate_pass":all(indicators.values()),
    "interpretation":"PASS authorizes DC1 5k development scale-up only. DC0 parity remains pending; no production/DC2/canonical-strength claim is authorized.",
}
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print("=== fullpool DC1 1k resume ===")
print(json.dumps(report["paired_scenario_cluster_delta"],sort_keys=True))
print(json.dumps(indicators,sort_keys=True))
print(f"fullpool_dc1_1k_scaleup_gate_pass={report['fullpool_dc1_1k_scaleup_gate_pass']}")
print(f"comparison={out}")
PY

"${PY}" - "${FULL_AUDIT}" "${REPORT}" "${SANITY}" "${COMPARE}" "${RUN}" <<'PY'
import sys,zipfile
from pathlib import Path
audit,report,sanity,comparison,run=map(Path,sys.argv[1:])
out=run/"SpinCore_DC1_semantic_fullpool_1k_bundle.zip"
for p in (audit,report,sanity,comparison):
    if not p.is_file():
        raise RuntimeError(f"missing bundle source {p}")
with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for p in (audit,report,sanity,comparison):
        z.write(p,arcname=p.name)
print(f"bundle={out}")
PY

ZIP="${RUN}/SpinCore_DC1_semantic_fullpool_1k_bundle.zip"
if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${ZIP}" "${DWSL}/SpinCore_DC1_semantic_fullpool_1k_bundle.zip"
    echo "desktop_bundle=${DWIN}\\SpinCore_DC1_semantic_fullpool_1k_bundle.zip"
  fi
fi

echo "DEEPC_RUSHER_DC1_SEMANTIC_FULLPOOL_1K_RESUME_COMPLETE"
