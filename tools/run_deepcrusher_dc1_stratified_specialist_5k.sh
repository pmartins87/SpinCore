#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"

SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE}/checkpoint.pt"
HU="${SOURCE}/hu_ensemble_state.pt"
FULLPOOL="${ROOT}/runs/3h_semantic_fullpool_strong_diversity_10115/semantic_fullpool_strong_diversity_tail_candidate.pt"
SPECIALIST="${ROOT}/runs/3h_semantic_stratified_specialist_moe_10115/semantic_stratified_specialist_moe_10115.pt"

EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_HU="8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"
SCENARIOS=5000
SEED=20260929

for f in "${PY}" "${CHECKPOINT}" "${HU}" "${FULLPOOL}" "${SPECIALIST}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing artifact: ${f}" >&2; exit 2; }
done
[[ "$(sha256sum "${CHECKPOINT}"|awk '{print $1}')" == "${EXPECTED_CP}" ]] || { echo "ERROR: checkpoint SHA mismatch" >&2; exit 3; }
[[ "$(sha256sum "${HU}"|awk '{print $1}')" == "${EXPECTED_HU}" ]] || { echo "ERROR: HU SHA mismatch" >&2; exit 3; }

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c

export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"${PY}" -m py_compile   tools/evaluate_deepcrusher_dc1.py   tools/evaluate_deepcrusher_dc1_semantic_candidate.py   tools/audit_benchmark_decisions.py

"${PY}" tools/test_deepcrusher_nouts_native_collision.py

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${ROOT}/runs/deepcrusher_dc1_stratified_specialist_5k/${STAMP}"
mkdir -p "${RUN}"

BASE_BUNDLE="${RUN}/spincore_hybrid_10105.pt"

BASE_REPORT="${RUN}/baseline_10105_report.json"
BASE_TRACES="${RUN}/baseline_10105_traces.jsonl"
BASE_SANITY="${RUN}/baseline_10105_sanity.json"

FULL_REPORT="${RUN}/fullpool_10115_report.json"
FULL_TRACES="${RUN}/fullpool_10115_traces.jsonl"
FULL_SANITY="${RUN}/fullpool_10115_sanity.json"

SPEC_REPORT="${RUN}/stratified_specialist_10115_report.json"
SPEC_TRACES="${RUN}/stratified_specialist_10115_traces.jsonl"
SPEC_SANITY="${RUN}/stratified_specialist_10115_sanity.json"

COMPARE="${RUN}/stratified_specialist_5k_comparison.json"

"${PY}" tools/export_spincore_hybrid_inference.py   --checkpoint "${CHECKPOINT}"   --ensemble "${HU}"   --out "${BASE_BUNDLE}"

echo "DC1_5K_BASELINE_START"
"${PY}" tools/evaluate_deepcrusher_dc1.py   --solver build/libspincore_solver_c.so   --spin-bundle "${BASE_BUNDLE}"   --scenarios "${SCENARIOS}" --workers 8 --seed "${SEED}"   --report "${BASE_REPORT}" --traces "${BASE_TRACES}" --max-decisions 300
"${PY}" tools/audit_benchmark_decisions.py   --traces "${BASE_TRACES}" --report "${BASE_SANITY}" --max-examples-per-code 100
echo "DC1_5K_BASELINE_PASS"

echo "DC1_5K_FULLPOOL_START"
"${PY}" tools/evaluate_deepcrusher_dc1_semantic_candidate.py   --solver build/libspincore_solver_c.so   --base-spin-bundle "${BASE_BUNDLE}"   --semantic-policy "${FULLPOOL}"   --scenarios "${SCENARIOS}" --workers 8 --seed "${SEED}"   --report "${FULL_REPORT}" --traces "${FULL_TRACES}" --max-decisions 300
"${PY}" tools/audit_benchmark_decisions.py   --traces "${FULL_TRACES}" --report "${FULL_SANITY}" --max-examples-per-code 100
echo "DC1_5K_FULLPOOL_PASS"

echo "DC1_5K_SPECIALIST_START"
"${PY}" tools/evaluate_deepcrusher_dc1_semantic_candidate.py   --solver build/libspincore_solver_c.so   --base-spin-bundle "${BASE_BUNDLE}"   --semantic-policy "${SPECIALIST}"   --scenarios "${SCENARIOS}" --workers 8 --seed "${SEED}"   --report "${SPEC_REPORT}" --traces "${SPEC_TRACES}" --max-decisions 300
"${PY}" tools/audit_benchmark_decisions.py   --traces "${SPEC_TRACES}" --report "${SPEC_SANITY}" --max-examples-per-code 100
echo "DC1_5K_SPECIALIST_PASS"

"${PY}" -   "${BASE_REPORT}" "${FULL_REPORT}" "${SPEC_REPORT}"   "${BASE_SANITY}" "${FULL_SANITY}" "${SPEC_SANITY}"   "${COMPARE}" <<'PY'
import json,math,statistics,sys
from pathlib import Path

bp,fp,sp,bs,fs,ss,out=map(Path,sys.argv[1:])
b=json.loads(bp.read_text())
f=json.loads(fp.read_text())
s=json.loads(sp.read_text())
bsa=json.loads(bs.read_text())
fsa=json.loads(fs.read_text())
ssa=json.loads(ss.read_text())

EXPECTED_SCENARIOS=5000
EXPECTED_SEED=20260929

for name,r in (("baseline",b),("fullpool",f),("specialist",s)):
    if int(r.get("scenarios",-1))!=EXPECTED_SCENARIOS:
        raise RuntimeError(f"{name} scenario-count drift")
    if int(r.get("seed",-1))!=EXPECTED_SEED:
        raise RuntimeError(f"{name} seed drift")
    if str(r.get("status"))!="PASS":
        raise RuntimeError(f"{name} evaluator status not PASS: {r.get('status')}")

def rows(r):
    return sorted(r["scenario_rows"],key=lambda x:x["scenario"])

br,fr,sr=rows(b),rows(f),rows(s)
if not (len(br)==len(fr)==len(sr)==EXPECTED_SCENARIOS):
    raise RuntimeError("scenario row count mismatch")

for x,y,z in zip(br,fr,sr):
    ids=[
        (q["scenario"],q["domain"],q["blind"],q["dealer_id"],q["deal_seed"])
        for q in (x,y,z)
    ]
    if len(set(ids))!=1:
        raise RuntimeError("scenario identity mismatch")

def margin(r):
    return float(r["paired_spincore_minus_deepcrusher_chips_per_exposure"])

def ci(vals):
    vals=[float(x) for x in vals]
    n=len(vals)
    mean=statistics.fmean(vals)
    sem=statistics.stdev(vals)/math.sqrt(n) if n>1 else 0.0
    return {
        "n":n,
        "mean":mean,
        "sem":sem,
        "ci95_low":mean-1.96*sem,
        "ci95_high":mean+1.96*sem,
        "median":statistics.median(vals),
        "positive":sum(x>0 for x in vals),
        "negative":sum(x<0 for x in vals),
        "zero":sum(x==0 for x in vals),
    }

def paired(a,b,domain=None):
    return ci([
        margin(y)-margin(x)
        for x,y in zip(a,b)
        if domain is None or x["domain"]==domain
    ])

def count(sanity,code):
    return int((sanity.get("flag_counts") or {}).get(code,0))

def rate(sanity,code):
    return count(sanity,code)/max(1,int(sanity.get("spincore_decisions",0)))

trip="POSTFLOP_TRIPS_PLUS_FOLD"
hc="POSTFLOP_DEEP_HIGH_CARD_JAM"
tp="POSTFLOP_TOP_PAIR_FOLD"
aa="PREFLOP_AA_FOLD"
d72="PREFLOP_DEEP_72O_JAM"

sb3=paired(br,sr,"THREE_HANDED")
sf3=paired(fr,sr,"THREE_HANDED")
sb_all=paired(br,sr)
sf_all=paired(fr,sr)

hu=all(
    margin(x)==margin(y)==margin(z)
    for x,y,z in zip(br,fr,sr)
    if x["domain"]=="TRUE_HEADS_UP"
)

known_codes={trip,hc,tp,aa,d72}
specialist_flags=set((ssa.get("flag_counts") or {}).keys())
unexpected_specialist_codes=sorted(specialist_flags-known_codes)

criteria={
    "all_three_evaluators_passed":all(
        str(r.get("status"))=="PASS" for r in (b,f,s)
    ),
    "hu_parity_exact":hu,
    "specialist_vs_baseline_3h_mean_positive":sb3["mean"]>0.0,
    "specialist_vs_baseline_3h_ci95_low_positive":sb3["ci95_low"]>0.0,
    "specialist_trips_fold_rate_no_more_than_baseline_plus_0_01pp":
        rate(ssa,trip)<=rate(bsa,trip)+0.0001,
    "specialist_high_card_jam_rate_no_more_than_10pct_above_fullpool":
        rate(ssa,hc)<=1.10*rate(fsa,hc)+1e-15,
    "specialist_top_pair_fold_count_no_more_than_fullpool_plus_1":
        count(ssa,tp)<=count(fsa,tp)+1,
    "specialist_preflop_aa_fold_count_no_more_than_fullpool_plus_1":
        count(ssa,aa)<=count(fsa,aa)+1,
    "specialist_deep_72o_jam_count_no_more_than_fullpool_plus_1":
        count(ssa,d72)<=count(fsa,d72)+1,
    "no_unexpected_specialist_sanity_codes":
        len(unexpected_specialist_codes)==0,
}
passed=all(criteria.values())

codes=sorted(
    set((bsa.get("flag_counts") or {}))
    | set((fsa.get("flag_counts") or {}))
    | set((ssa.get("flag_counts") or {}))
)

report={
    "schema":"SPINCORE_DC1_STRATIFIED_SPECIALIST_5K_COMPARISON_V1",
    "scope":"DEVELOPMENT_ONLY_FRESH_SEED_SCALEUP_NO_CANONICAL_STRENGTH_CLAIM",
    "seed":EXPECTED_SEED,
    "scenarios":EXPECTED_SCENARIOS,
    "domain_counts":b.get("domain_counts"),
    "fresh_seed_relative_to_prior_1k":True,
    "hu_exact_scenario_margin_parity":hu,
    "absolute_3h_vs_deepcrusher":{
        "baseline_10105":
            b["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
        "fullpool_10115":
            f["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
        "stratified_specialist_10115":
            s["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
    },
    "paired_scenario_cluster_delta":{
        "specialist_minus_baseline":{
            "ALL":sb_all,
            "THREE_HANDED":sb3,
        },
        "specialist_minus_fullpool":{
            "ALL":sf_all,
            "THREE_HANDED":sf3,
        },
    },
    "sanity_flags":{
        code:{
            "baseline_count":count(bsa,code),
            "fullpool_count":count(fsa,code),
            "specialist_count":count(ssa,code),
            "baseline_rate":rate(bsa,code),
            "fullpool_rate":rate(fsa,code),
            "specialist_rate":rate(ssa,code),
        }
        for code in codes
    },
    "unexpected_specialist_sanity_codes":unexpected_specialist_codes,
    "precommitted_long_train_readiness_criteria":criteria,
    "dc1_5k_long_train_readiness_pass":passed,
    "interpretation":(
        "PASS means the unchanged multiseed-confirmed stratified specialist "
        "retains a statistically positive paired 3H improvement versus the "
        "frozen 10105 baseline on a fresh 5k development seed, while preserving "
        "the previously repaired sanity surfaces. PASS supports freezing the "
        "research architecture for the next longer training continuation. "
        "It does not authorize production, DC2, or a canonical strength claim; "
        "DC0 real-OpenHoldem action+sizing parity remains pending."
    ),
}
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")

print("=== DC1 fresh-seed stratified-specialist 5k ===")
print("specialist_minus_baseline_3h="+json.dumps(sb3,sort_keys=True))
print("specialist_minus_fullpool_3h="+json.dumps(sf3,sort_keys=True))
print("criteria="+json.dumps(criteria,sort_keys=True))
print(f"dc1_5k_long_train_readiness_pass={passed}")
print(f"comparison={out}")
print("DC1_STRATIFIED_SPECIALIST_5K_COMPARISON_COMPLETE")
PY

"${PY}" - "${RUN}" <<'PY'
from pathlib import Path
import sys,zipfile

run=Path(sys.argv[1])
out=run/"SpinCore_DC1_stratified_specialist_5k_bundle.zip"
names=(
    "baseline_10105_report.json",
    "baseline_10105_sanity.json",
    "fullpool_10115_report.json",
    "fullpool_10115_sanity.json",
    "stratified_specialist_10115_report.json",
    "stratified_specialist_10115_sanity.json",
    "stratified_specialist_5k_comparison.json",
)
with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for name in names:
        p=run/name
        if not p.is_file():
            raise RuntimeError(f"missing bundle source {p}")
        z.write(p,arcname=name)
print(f"bundle={out}")
PY

ZIP="${RUN}/SpinCore_DC1_stratified_specialist_5k_bundle.zip"
if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${ZIP}" "${DWSL}/SpinCore_DC1_stratified_specialist_5k_bundle.zip"
    echo "desktop_bundle=${DWIN}\\SpinCore_DC1_stratified_specialist_5k_bundle.zip"
  fi
fi

echo "DEEPC_RUSHER_DC1_STRATIFIED_SPECIALIST_5K_COMPLETE"
