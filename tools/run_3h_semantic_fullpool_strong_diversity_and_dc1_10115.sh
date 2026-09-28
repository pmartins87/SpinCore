#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE}/checkpoint.pt"
HU="${SOURCE}/hu_ensemble_state.pt"
CANON="${ROOT}/runs/3h_semantic_research_canonical_10105_10115"
ADV="${CANON}/semantic_ensemble_final_10115.pt"
CANON_POLICY="${CANON}/semantic_tail_policy_10115.pt"
STRAT="${ROOT}/runs/3h_semantic_stratified_strong_diversity_10115/semantic_stratified_strong_diversity_tail_candidate.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_HU="8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"

for f in "${PY}" "${CHECKPOINT}" "${HU}" "${ADV}" "${CANON_POLICY}" "${STRAT}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing artifact: ${f}" >&2; exit 2; }
done
[[ "$(sha256sum "${CHECKPOINT}"|awk '{print $1}')" == "${EXPECTED_CP}" ]] || { echo "ERROR: checkpoint SHA mismatch" >&2; exit 3; }
[[ "$(sha256sum "${HU}"|awk '{print $1}')" == "${EXPECTED_HU}" ]] || { echo "ERROR: HU SHA mismatch" >&2; exit 3; }

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

RUN="${ROOT}/runs/3h_semantic_fullpool_strong_diversity_10115"
mkdir -p "${RUN}"
REPORT="${RUN}/3h_semantic_fullpool_strong_diversity_10115.json"
MODEL="${RUN}/semantic_fullpool_strong_diversity_tail_candidate.pt"

"${PY}" -m py_compile tools/audit_3h_semantic_fullpool_strong_diversity_10115.py tools/evaluate_deepcrusher_dc1_semantic_candidate.py
"${PY}" tools/audit_3h_semantic_fullpool_strong_diversity_10115.py \
  --checkpoint "${CHECKPOINT}" --semantic-advantage "${ADV}" \
  --canonical-tail "${CANON_POLICY}" --stratified-tail "${STRAT}" \
  --solver build/libspincore_solver_c.so --report "${REPORT}" --out-model "${MODEL}" --threads 8

PASS="$("${PY}" - "${REPORT}" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print("1" if r.get("semantic_fullpool_diversity_repair_pass") else "0")
PY
)"
if [[ "${PASS}" != "1" ]]; then
  echo "FULLPOOL_REPAIR_GATE_FAILED_STOP_BEFORE_DC1"
  exit 5
fi

echo "FULLPOOL_REPAIR_GATE_PASS_STARTING_EXACT_DC1_1K"

PREV="$(find "${ROOT}/runs/deepcrusher_dc1_semantic_stratified_1k" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
[[ -n "${PREV}" && -d "${PREV}" ]] || { echo "ERROR: prior stratified DC1 run missing" >&2; exit 6; }
PREVCMP="${PREV}/stratified_vs_prior_1k_comparison.json"
STRAT_REPORT="${PREV}/stratified_10115_report.json"
STRAT_SANITY="${PREV}/stratified_10115_sanity.json"
[[ -f "${PREVCMP}" && -f "${STRAT_REPORT}" && -f "${STRAT_SANITY}" ]] || { echo "ERROR: prior stratified artifacts missing" >&2; exit 6; }

PRIOR3="$("${PY}" - "${PREVCMP}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["reused_prior_three_arm_run"])
PY
)"
BASE_BUNDLE="${PRIOR3}/spincore_hybrid_10105.pt"
BASE_REPORT="${PRIOR3}/baseline_10105_report.json"
BASE_SANITY="${PRIOR3}/baseline_10105_sanity.json"
for f in "${BASE_BUNDLE}" "${BASE_REPORT}" "${BASE_SANITY}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing original 1k artifact: ${f}" >&2; exit 6; }
done

"${PY}" tools/test_deepcrusher_nouts_native_collision.py
STAMP="$(date +%Y%m%d_%H%M%S)"
DC1="${ROOT}/runs/deepcrusher_dc1_semantic_fullpool_1k/${STAMP}"
mkdir -p "${DC1}"
FULL_REPORT="${DC1}/fullpool_10115_report.json"
FULL_TRACES="${DC1}/fullpool_10115_traces.jsonl"
FULL_SANITY="${DC1}/fullpool_10115_sanity.json"
COMPARE="${DC1}/fullpool_vs_stratified_vs_baseline_comparison.json"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "${PY}" tools/evaluate_deepcrusher_dc1_semantic_candidate.py \
  --solver build/libspincore_solver_c.so --base-spin-bundle "${BASE_BUNDLE}" \
  --semantic-policy "${MODEL}" --scenarios 1000 --workers 8 --seed 20260923 \
  --report "${FULL_REPORT}" --traces "${FULL_TRACES}" --max-decisions 200

"${PY}" tools/audit_benchmark_decisions.py --traces "${FULL_TRACES}" --report "${FULL_SANITY}" --max-examples-per-code 50

"${PY}" - "${BASE_REPORT}" "${STRAT_REPORT}" "${FULL_REPORT}" "${BASE_SANITY}" "${STRAT_SANITY}" "${FULL_SANITY}" "${COMPARE}" <<'PY'
import json,math,statistics,sys
from pathlib import Path
bp,sp,fp,bs,ss,fs,out=map(Path,sys.argv[1:])
b=json.loads(bp.read_text()); s=json.loads(sp.read_text()); f=json.loads(fp.read_text())
bsa=json.loads(bs.read_text()); ssa=json.loads(ss.read_text()); fsa=json.loads(fs.read_text())
def rows(r): return sorted(r["scenario_rows"],key=lambda x:x["scenario"])
br,sr,fr=rows(b),rows(s),rows(f)
for x,y,z in zip(br,sr,fr):
    ids=[(q["scenario"],q["domain"],q["blind"],q["dealer_id"],q["deal_seed"]) for q in (x,y,z)]
    if len(set(ids))!=1: raise RuntimeError("scenario identity mismatch")
def m(r): return float(r["paired_spincore_minus_deepcrusher_chips_per_exposure"])
def ci(vals):
    n=len(vals); mean=statistics.fmean(vals); sem=statistics.stdev(vals)/math.sqrt(n)
    return {"n":n,"mean":mean,"sem":sem,"ci95_low":mean-1.96*sem,"ci95_high":mean+1.96*sem,
            "median":statistics.median(vals),"positive":sum(x>0 for x in vals),"negative":sum(x<0 for x in vals),"zero":sum(x==0 for x in vals)}
def paired(a,b,domain=None):
    return ci([m(y)-m(x) for x,y in zip(a,b) if domain is None or x["domain"]==domain])
def fc(san,code): return int((san.get("flag_counts") or {}).get(code,0))
def rate(san,code): return fc(san,code)/max(1,int(san.get("spincore_decisions",0)))
trip="POSTFLOP_TRIPS_PLUS_FOLD"; hc="POSTFLOP_DEEP_HIGH_CARD_JAM"; tp="POSTFLOP_TOP_PAIR_FOLD"; aa="PREFLOP_AA_FOLD"; d72="PREFLOP_DEEP_72O_JAM"
fb3=paired(br,fr,"THREE_HANDED"); fs3=paired(sr,fr,"THREE_HANDED")
hu=all(m(x)==m(y)==m(z) for x,y,z in zip(br,sr,fr) if x["domain"]=="TRUE_HEADS_UP")
ind={
  "hu_parity_exact":hu,
  "fullpool_trips_fold_no_higher_than_baseline":fc(fsa,trip)<=fc(bsa,trip),
  "fullpool_high_card_jam_rate_no_more_than_10pct_above_stratified":rate(fsa,hc)<=1.10*rate(ssa,hc)+1e-15,
  "fullpool_top_pair_fold_no_higher_than_stratified":fc(fsa,tp)<=fc(ssa,tp),
  "fullpool_preflop_aa_fold_no_higher_than_stratified":fc(fsa,aa)<=fc(ssa,aa),
  "fullpool_deep_72o_jam_no_higher_than_stratified":fc(fsa,d72)<=fc(ssa,d72),
  "fullpool_vs_baseline_3h_mean_positive":fb3["mean"]>0,
}
report={
  "schema":"SPINCORE_DC1_SEMANTIC_FULLPOOL_STRONG_DIVERSITY_1K_COMPARISON_V1",
  "scope":"DEVELOPMENT_ONLY_NO_CANONICAL_STRENGTH_CLAIM","seed":20260923,"scenarios":1000,
  "hu_exact_scenario_margin_parity":hu,
  "absolute_3h_vs_deepcrusher":{
    "baseline_10105":b["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
    "stratified_10115":s["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
    "fullpool_10115":f["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
  },
  "paired_scenario_cluster_delta":{"fullpool_minus_baseline":{"ALL":paired(br,fr),"THREE_HANDED":fb3},"fullpool_minus_stratified":{"ALL":paired(sr,fr),"THREE_HANDED":fs3}},
  "sanity_flags":{code:{"baseline_count":fc(bsa,code),"stratified_count":fc(ssa,code),"fullpool_count":fc(fsa,code),"baseline_rate":rate(bsa,code),"stratified_rate":rate(ssa,code),"fullpool_rate":rate(fsa,code)} for code in sorted(set((bsa.get("flag_counts") or {}))|set((ssa.get("flag_counts") or {}))|set((fsa.get("flag_counts") or {})))},
  "precommitted_scaleup_indicators":ind,
  "fullpool_dc1_1k_scaleup_gate_pass":all(ind.values()),
  "interpretation":"PASS authorizes DC1 5k development scale-up only. DC0 parity remains pending; no production/DC2/canonical-strength claim is authorized."
}
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print("=== fullpool DC1 1k ==="); print(json.dumps(report["paired_scenario_cluster_delta"],sort_keys=True)); print(json.dumps(ind,sort_keys=True)); print(f"fullpool_dc1_1k_scaleup_gate_pass={report['fullpool_dc1_1k_scaleup_gate_pass']}")
PY

"${PY}" - "${REPORT}" "${FULL_REPORT}" "${FULL_SANITY}" "${COMPARE}" "${DC1}" <<'PY'
import sys,zipfile
from pathlib import Path
audit,rep,san,cmp,run=map(Path,sys.argv[1:])
out=run/"SpinCore_DC1_semantic_fullpool_1k_bundle.zip"
with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for p in (audit,rep,san,cmp): z.write(p,arcname=p.name)
print(f"bundle={out}")
PY

ZIP="${DC1}/SpinCore_DC1_semantic_fullpool_1k_bundle.zip"
if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${REPORT}" "${DWSL}/SpinCore_3H_semantic_fullpool_strong_diversity_10115.json"
    cp -f "${ZIP}" "${DWSL}/SpinCore_DC1_semantic_fullpool_1k_bundle.zip"
  fi
fi

echo "SEMANTIC_FULLPOOL_REPAIR_AND_DC1_1K_COMPLETE"
