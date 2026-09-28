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
PURE_RUN="${ROOT}/runs/3h_semantic_strong_specialist_moe_10115"
PURE_MODEL="${PURE_RUN}/semantic_strong_specialist_moe_10115.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_HU="8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"

for f in "${PY}" "${CHECKPOINT}" "${HU}" "${ADV}" "${PURE_MODEL}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing artifact: ${f}" >&2; exit 2; }
done
[[ "$(sha256sum "${CHECKPOINT}"|awk '{print $1}')" == "${EXPECTED_CP}" ]] || { echo "ERROR: checkpoint SHA mismatch" >&2; exit 3; }
[[ "$(sha256sum "${HU}"|awk '{print $1}')" == "${EXPECTED_HU}" ]] || { echo "ERROR: HU SHA mismatch" >&2; exit 3; }

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c
export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

"${PY}" -m py_compile   tools/audit_3h_semantic_confidence_gated_moe_10115.py   tools/evaluate_deepcrusher_dc1_semantic_candidate.py

RUN="${ROOT}/runs/3h_semantic_confidence_gated_moe_10115"
mkdir -p "${RUN}"
AUDIT="${RUN}/3h_semantic_confidence_gated_moe_10115.json"
MODEL="${RUN}/semantic_confidence_gated_moe_10115.pt"

"${PY}" tools/audit_3h_semantic_confidence_gated_moe_10115.py   --checkpoint "${CHECKPOINT}"   --semantic-advantage "${ADV}"   --pure-moe "${PURE_MODEL}"   --solver build/libspincore_solver_c.so   --report "${AUDIT}"   --out-model "${MODEL}"   --threads 8

if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${AUDIT}" "${DWSL}/SpinCore_3H_semantic_confidence_gated_moe_10115.json"
    echo "desktop_audit=${DWIN}\\SpinCore_3H_semantic_confidence_gated_moe_10115.json"
  fi
fi

PASS="$("${PY}" - "${AUDIT}" <<'PY'
import json,sys
print("1" if json.load(open(sys.argv[1])).get("semantic_confidence_gated_moe_pass") else "0")
PY
)"
if [[ "${PASS}" != "1" ]]; then
  echo "CONFIDENCE_GATED_MOE_OFFLINE_GATE_FAILED_STOP_BEFORE_DC1"
  exit 5
fi

echo "CONFIDENCE_GATED_MOE_OFFLINE_GATE_PASS_STARTING_DC1_1K"

FULLPREV="$(find "${ROOT}/runs/deepcrusher_dc1_semantic_fullpool_1k" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
[[ -n "${FULLPREV}" && -d "${FULLPREV}" ]] || { echo "ERROR: fullpool DC1 1k run missing" >&2; exit 6; }
FULL_REPORT="${FULLPREV}/fullpool_10115_report.json"
FULL_SANITY="${FULLPREV}/fullpool_10115_sanity.json"
FULL_COMPARE="${FULLPREV}/fullpool_vs_stratified_vs_baseline_comparison.json"
for f in "${FULL_REPORT}" "${FULL_SANITY}" "${FULL_COMPARE}"; do
  [[ -f "${f}" ]] || { echo "ERROR: prior fullpool artifact missing: ${f}" >&2; exit 6; }
done

STRATPREV="$(find "${ROOT}/runs/deepcrusher_dc1_semantic_stratified_1k" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | cut -d' ' -f2-)"
STRAT_COMPARE="${STRATPREV}/stratified_vs_prior_1k_comparison.json"
[[ -f "${STRAT_COMPARE}" ]] || { echo "ERROR: prior stratified comparison missing" >&2; exit 6; }

PRIOR3="$("${PY}" - "${STRAT_COMPARE}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["reused_prior_three_arm_run"])
PY
)"
BASE_BUNDLE="${PRIOR3}/spincore_hybrid_10105.pt"
BASE_REPORT="${PRIOR3}/baseline_10105_report.json"
BASE_SANITY="${PRIOR3}/baseline_10105_sanity.json"
for f in "${BASE_BUNDLE}" "${BASE_REPORT}" "${BASE_SANITY}"; do
  [[ -f "${f}" ]] || { echo "ERROR: baseline artifact missing: ${f}" >&2; exit 6; }
done

"${PY}" tools/test_deepcrusher_nouts_native_collision.py

STAMP="$(date +%Y%m%d_%H%M%S)"
DC1="${ROOT}/runs/deepcrusher_dc1_semantic_confidence_gated_moe_1k/${STAMP}"
mkdir -p "${DC1}"
REPORT="${DC1}/confidence_gated_moe_10115_report.json"
TRACES="${DC1}/confidence_gated_moe_10115_traces.jsonl"
SANITY="${DC1}/confidence_gated_moe_10115_sanity.json"
COMPARE="${DC1}/confidence_gated_moe_vs_fullpool_vs_baseline.json"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "${PY}" tools/evaluate_deepcrusher_dc1_semantic_candidate.py   --solver build/libspincore_solver_c.so   --base-spin-bundle "${BASE_BUNDLE}"   --semantic-policy "${MODEL}"   --scenarios 1000 --workers 8 --seed 20260923   --report "${REPORT}" --traces "${TRACES}" --max-decisions 200

"${PY}" tools/audit_benchmark_decisions.py   --traces "${TRACES}" --report "${SANITY}" --max-examples-per-code 50

"${PY}" -   "${BASE_REPORT}" "${FULL_REPORT}" "${REPORT}"   "${BASE_SANITY}" "${FULL_SANITY}" "${SANITY}" "${COMPARE}" <<'PY'
import json,math,statistics,sys
from pathlib import Path

bp,fp,gp,bs,fs,gs,out=map(Path,sys.argv[1:])
b=json.loads(bp.read_text())
f=json.loads(fp.read_text())
g=json.loads(gp.read_text())
bsa=json.loads(bs.read_text())
fsa=json.loads(fs.read_text())
gsa=json.loads(gs.read_text())

def rows(r):
    return sorted(r["scenario_rows"],key=lambda x:x["scenario"])

br,fr,gr=rows(b),rows(f),rows(g)
for x,y,z in zip(br,fr,gr):
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

def count(s,c):
    return int((s.get("flag_counts") or {}).get(c,0))

def rate(s,c):
    return count(s,c)/max(1,int(s.get("spincore_decisions",0)))

trip="POSTFLOP_TRIPS_PLUS_FOLD"
hc="POSTFLOP_DEEP_HIGH_CARD_JAM"
tp="POSTFLOP_TOP_PAIR_FOLD"
aa="PREFLOP_AA_FOLD"
d72="PREFLOP_DEEP_72O_JAM"

gb3=paired(br,gr,"THREE_HANDED")
gf3=paired(fr,gr,"THREE_HANDED")
hu=all(
    margin(x)==margin(y)==margin(z)
    for x,y,z in zip(br,fr,gr)
    if x["domain"]=="TRUE_HEADS_UP"
)

indicators={
    "hu_parity_exact":hu,
    "gated_moe_trips_fold_no_higher_than_baseline":count(gsa,trip)<=count(bsa,trip),
    "gated_moe_high_card_jam_rate_no_more_than_10pct_above_fullpool":rate(gsa,hc)<=1.10*rate(fsa,hc)+1e-15,
    "gated_moe_top_pair_fold_no_higher_than_fullpool":count(gsa,tp)<=count(fsa,tp),
    "gated_moe_preflop_aa_fold_no_higher_than_fullpool":count(gsa,aa)<=count(fsa,aa),
    "gated_moe_deep_72o_jam_no_higher_than_fullpool":count(gsa,d72)<=count(fsa,d72),
    "gated_moe_vs_baseline_3h_mean_positive":gb3["mean"]>0,
}

codes=sorted(
    set((bsa.get("flag_counts") or {}))
    | set((fsa.get("flag_counts") or {}))
    | set((gsa.get("flag_counts") or {}))
)

report={
    "schema":"SPINCORE_DC1_SEMANTIC_CONFIDENCE_GATED_STRONG_MOE_1K_COMPARISON_V1",
    "scope":"DEVELOPMENT_ONLY_NO_CANONICAL_STRENGTH_CLAIM",
    "seed":20260923,
    "scenarios":1000,
    "hu_exact_scenario_margin_parity":hu,
    "absolute_3h_vs_deepcrusher":{
        "baseline_10105":b["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
        "fullpool_10115":f["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
        "confidence_gated_moe_10115":g["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
    },
    "paired_scenario_cluster_delta":{
        "gated_moe_minus_baseline":{"ALL":paired(br,gr),"THREE_HANDED":gb3},
        "gated_moe_minus_fullpool":{"ALL":paired(fr,gr),"THREE_HANDED":gf3},
    },
    "sanity_flags":{
        c:{
            "baseline_count":count(bsa,c),
            "fullpool_count":count(fsa,c),
            "gated_moe_count":count(gsa,c),
            "baseline_rate":rate(bsa,c),
            "fullpool_rate":rate(fsa,c),
            "gated_moe_rate":rate(gsa,c),
        }
        for c in codes
    },
    "precommitted_scaleup_indicators":indicators,
    "confidence_gated_moe_dc1_1k_scaleup_gate_pass":all(indicators.values()),
    "interpretation":"PASS authorizes DC1 5k development scale-up only. DC0 parity remains pending; no production/DC2/canonical-strength claim is authorized.",
}
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print("=== confidence-gated MoE DC1 1k ===")
print(json.dumps(report["paired_scenario_cluster_delta"],sort_keys=True))
print(json.dumps(indicators,sort_keys=True))
print(f"confidence_gated_moe_dc1_1k_scaleup_gate_pass={report['confidence_gated_moe_dc1_1k_scaleup_gate_pass']}")
print(f"comparison={out}")
PY

"${PY}" - "${AUDIT}" "${REPORT}" "${SANITY}" "${COMPARE}" "${DC1}" <<'PY'
import sys,zipfile
from pathlib import Path
audit,report,sanity,comparison,run=map(Path,sys.argv[1:])
out=run/"SpinCore_DC1_semantic_confidence_gated_moe_1k_bundle.zip"
with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for p in (audit,report,sanity,comparison):
        if not p.is_file():
            raise RuntimeError(f"missing bundle source {p}")
        z.write(p,arcname=p.name)
print(f"bundle={out}")
PY

ZIP="${DC1}/SpinCore_DC1_semantic_confidence_gated_moe_1k_bundle.zip"
if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${ZIP}" "${DWSL}/SpinCore_DC1_semantic_confidence_gated_moe_1k_bundle.zip"
    echo "desktop_bundle=${DWIN}\\SpinCore_DC1_semantic_confidence_gated_moe_1k_bundle.zip"
  fi
fi

echo "SEMANTIC_CONFIDENCE_GATED_MOE_AND_DC1_1K_COMPLETE"
