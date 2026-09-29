#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"

# PROJECT_CONTRACT_IDS: BENCH-001,BENCH-002,BENCH-003,BENCH-004,BENCH-005,DC-001,DC-002,DC-004,PERF-001,VALID-001,VALID-023,VALID-024,SAFE-001,ART-001,SRC-003
"${PY}" tools/check_stage_manifest.py --manifest contracts/run_manifests/dc1_postlong_10315_5k.json --runner tools/run_deepcrusher_dc1_semantic_postlong_10315_5k.sh
SOLVER="${ROOT}/build/libspincore_solver_c.so"

SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE}/checkpoint.pt"
HU="${SOURCE}/hu_ensemble_state.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_HU="8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"

OLD_SPEC="${ROOT}/runs/3h_semantic_stratified_specialist_moe_10115/semantic_stratified_specialist_moe_10115.pt"
POST="${ROOT}/runs/3h_semantic_postlong_10315"
NEW_SPEC="${POST}/semantic_stratified_specialist_10315.pt"
POST_POLICY_REPORT="${POST}/postlong_policy_rebuild_10315.json"
POST_MULTI="${POST}/postlong_stratified_specialist_multiseed_10315.json"

SCENARIOS=5000
SEED=20261001

for f in "${PY}" "${CHECKPOINT}" "${HU}" "${OLD_SPEC}" "${NEW_SPEC}" "${POST_POLICY_REPORT}" "${POST_MULTI}"; do
  [[ -e "${f}" ]] || { echo "ERROR: missing prerequisite: ${f}" >&2; exit 2; }
done
[[ "$(sha256sum "${CHECKPOINT}" | awk '{print $1}')" == "${EXPECTED_CP}" ]] || { echo "ERROR: checkpoint SHA mismatch" >&2; exit 3; }
[[ "$(sha256sum "${HU}" | awk '{print $1}')" == "${EXPECTED_HU}" ]] || { echo "ERROR: HU SHA mismatch" >&2; exit 3; }

export PYTHONPATH="${ROOT}/python:${ROOT}/tools"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"${PY}" - "${POST_POLICY_REPORT}" "${POST_MULTI}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
m=json.load(open(sys.argv[2]))
assert p["postlong_policy_rebuild_pass"] is True
assert int(p["semantic_completed_iteration"])==10315
assert m["semantic_stratified_specialist_multiseed_pass"] is True
assert int(m["semantic_completed_iteration"])==10315
print("DC1_POSTLONG_10315_GATE_CONTRACT_PASS")
PY

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8 --target spincore_solver_c
"${PY}" -m py_compile \
  tools/evaluate_deepcrusher_dc1.py \
  tools/evaluate_deepcrusher_dc1_semantic_candidate.py \
  tools/evaluate_deepcrusher_dc1_semantic_candidate_10315.py \
  tools/audit_benchmark_decisions.py
"${PY}" tools/test_deepcrusher_nouts_native_collision.py

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN="${ROOT}/runs/deepcrusher_dc1_semantic_postlong_10315_5k/${STAMP}"
mkdir -p "${RUN}"

BASE_BUNDLE="${RUN}/spincore_hybrid_10105.pt"
BASE_REPORT="${RUN}/baseline_10105_report.json"
BASE_TRACES="${RUN}/baseline_10105_traces.jsonl"
BASE_SANITY="${RUN}/baseline_10105_sanity.json"

OLD_REPORT="${RUN}/specialist_10115_report.json"
OLD_TRACES="${RUN}/specialist_10115_traces.jsonl"
OLD_SANITY="${RUN}/specialist_10115_sanity.json"

NEW_REPORT="${RUN}/specialist_10315_report.json"
NEW_TRACES="${RUN}/specialist_10315_traces.jsonl"
NEW_SANITY="${RUN}/specialist_10315_sanity.json"

COMPARE="${RUN}/postlong_10315_5k_comparison.json"

"${PY}" tools/export_spincore_hybrid_inference.py \
  --checkpoint "${CHECKPOINT}" \
  --ensemble "${HU}" \
  --out "${BASE_BUNDLE}"

echo "DC1_POSTLONG_5K_BASELINE_START"
"${PY}" tools/evaluate_deepcrusher_dc1.py \
  --solver "${SOLVER}" \
  --spin-bundle "${BASE_BUNDLE}" \
  --scenarios "${SCENARIOS}" --workers 8 --seed "${SEED}" \
  --report "${BASE_REPORT}" --traces "${BASE_TRACES}" --max-decisions 300
"${PY}" tools/audit_benchmark_decisions.py \
  --traces "${BASE_TRACES}" --report "${BASE_SANITY}" --max-examples-per-code 100

echo "DC1_POSTLONG_5K_10115_START"
"${PY}" tools/evaluate_deepcrusher_dc1_semantic_candidate.py \
  --solver "${SOLVER}" \
  --base-spin-bundle "${BASE_BUNDLE}" \
  --semantic-policy "${OLD_SPEC}" \
  --scenarios "${SCENARIOS}" --workers 8 --seed "${SEED}" \
  --report "${OLD_REPORT}" --traces "${OLD_TRACES}" --max-decisions 300
"${PY}" tools/audit_benchmark_decisions.py \
  --traces "${OLD_TRACES}" --report "${OLD_SANITY}" --max-examples-per-code 100

echo "DC1_POSTLONG_5K_10315_START"
"${PY}" tools/evaluate_deepcrusher_dc1_semantic_candidate_10315.py \
  --solver "${SOLVER}" \
  --base-spin-bundle "${BASE_BUNDLE}" \
  --semantic-policy "${NEW_SPEC}" \
  --scenarios "${SCENARIOS}" --workers 8 --seed "${SEED}" \
  --report "${NEW_REPORT}" --traces "${NEW_TRACES}" --max-decisions 300
"${PY}" tools/audit_benchmark_decisions.py \
  --traces "${NEW_TRACES}" --report "${NEW_SANITY}" --max-examples-per-code 100

"${PY}" - \
  "${BASE_REPORT}" "${OLD_REPORT}" "${NEW_REPORT}" \
  "${BASE_SANITY}" "${OLD_SANITY}" "${NEW_SANITY}" \
  "${COMPARE}" <<'PY'
import json,math,statistics,sys
from pathlib import Path

bp,op,np,bs,os,ns,out=map(Path,sys.argv[1:])
b=json.loads(bp.read_text())
o=json.loads(op.read_text())
n=json.loads(np.read_text())
bsa=json.loads(bs.read_text())
osa=json.loads(os.read_text())
nsa=json.loads(ns.read_text())

EXPECTED_SCENARIOS=5000
EXPECTED_SEED=20261001

for name,r in (("baseline",b),("10115",o),("10315",n)):
    if int(r.get("scenarios",-1))!=EXPECTED_SCENARIOS:
        raise RuntimeError(f"{name} scenario-count drift")
    if int(r.get("seed",-1))!=EXPECTED_SEED:
        raise RuntimeError(f"{name} seed drift")
    if str(r.get("status"))!="PASS":
        raise RuntimeError(f"{name} evaluator status not PASS")

def rows(r):
    return sorted(r["scenario_rows"],key=lambda x:x["scenario"])

br,orr,nr=rows(b),rows(o),rows(n)
if not (len(br)==len(orr)==len(nr)==EXPECTED_SCENARIOS):
    raise RuntimeError("scenario row count mismatch")

for x,y,z in zip(br,orr,nr):
    ids=[(q["scenario"],q["domain"],q["blind"],q["dealer_id"],q["deal_seed"]) for q in (x,y,z)]
    if len(set(ids))!=1:
        raise RuntimeError("scenario identity mismatch")

def margin(r):
    return float(r["paired_spincore_minus_deepcrusher_chips_per_exposure"])

def ci(vals):
    vals=[float(x) for x in vals]
    mean=statistics.fmean(vals)
    sem=statistics.stdev(vals)/math.sqrt(len(vals)) if len(vals)>1 else 0.0
    return {
        "n":len(vals),
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
known={trip,hc,tp,aa,d72}

new_base_3h=paired(br,nr,"THREE_HANDED")
new_old_3h=paired(orr,nr,"THREE_HANDED")
new_base_all=paired(br,nr)
new_old_all=paired(orr,nr)

hu=all(
    margin(x)==margin(y)==margin(z)
    for x,y,z in zip(br,orr,nr)
    if x["domain"]=="TRUE_HEADS_UP"
)
unexpected=sorted(set((nsa.get("flag_counts") or {}))-known)

safety={
    "all_three_evaluators_passed":all(str(r.get("status"))=="PASS" for r in (b,o,n)),
    "hu_parity_exact":hu,
    "10315_vs_baseline_3h_mean_positive":new_base_3h["mean"]>0.0,
    "10315_vs_baseline_3h_ci95_low_positive":new_base_3h["ci95_low"]>0.0,
    "10315_trips_fold_rate_no_more_than_10115_plus_0_01pp":
        rate(nsa,trip)<=rate(osa,trip)+0.0001,
    "10315_high_card_jam_rate_no_more_than_10pct_above_10115":
        rate(nsa,hc)<=1.10*rate(osa,hc)+1e-15,
    "10315_top_pair_fold_no_more_than_10115_plus_1":
        count(nsa,tp)<=count(osa,tp)+1,
    "10315_aa_fold_no_more_than_10115_plus_1":
        count(nsa,aa)<=count(osa,aa)+1,
    "10315_deep_72o_jam_no_more_than_10115_plus_1":
        count(nsa,d72)<=count(osa,d72)+1,
    "no_unexpected_10315_sanity_codes":len(unexpected)==0,
}
safety_pass=all(safety.values())

if new_old_3h["mean"]>0.0 and new_old_3h["ci95_low"]>0.0:
    incremental="POSITIVE_RESOLVED"
elif new_old_3h["mean"]>0.0:
    incremental="POSITIVE_INCONCLUSIVE"
else:
    incremental="NO_POSITIVE_INCREMENT"

codes=sorted(
    set((bsa.get("flag_counts") or {}))
    | set((osa.get("flag_counts") or {}))
    | set((nsa.get("flag_counts") or {}))
)

report={
    "schema":"SPINCORE_DC1_SEMANTIC_POSTLONG_10315_5K_COMPARISON_V1",
    "scope":"DEVELOPMENT_ONLY_FRESH_SEED_NO_CANONICAL_STRENGTH_CLAIM",
    "seed":EXPECTED_SEED,
    "scenarios":EXPECTED_SCENARIOS,
    "domain_counts":b.get("domain_counts"),
    "hu_exact_scenario_margin_parity":hu,
    "absolute_3h_vs_deepcrusher":{
        "baseline_10105":b["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
        "specialist_10115":o["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
        "specialist_10315":n["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"],
    },
    "paired_scenario_cluster_delta":{
        "10315_minus_baseline":{
            "ALL":new_base_all,
            "THREE_HANDED":new_base_3h,
        },
        "10315_minus_10115":{
            "ALL":new_old_all,
            "THREE_HANDED":new_old_3h,
        },
    },
    "sanity_flags":{
        code:{
            "baseline_count":count(bsa,code),
            "specialist_10115_count":count(osa,code),
            "specialist_10315_count":count(nsa,code),
            "baseline_rate":rate(bsa,code),
            "specialist_10115_rate":rate(osa,code),
            "specialist_10315_rate":rate(nsa,code),
        }
        for code in codes
    },
    "unexpected_10315_sanity_codes":unexpected,
    "precommitted_safety_and_baseline_retention_criteria":safety,
    "postlong_10315_5k_safety_pass":safety_pass,
    "incremental_10315_vs_10115_status":incremental,
    "interpretation":(
        "Safety PASS means the rebuilt 10315 candidate retains the statistically "
        "resolved development improvement over frozen 10105 and does not regress "
        "the repaired sanity surfaces on this fresh 5k seed. The separate "
        "incremental status determines whether the extra 200 online iterations "
        "show resolved, inconclusive, or no positive development increment over "
        "the accepted 10115 architecture. DC0 real-OpenHoldem parity remains "
        "pending, so no production/DC2/canonical-strength claim is authorized."
    ),
}
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print("=== DC1 postlong 10315 fresh-seed 5k ===")
print("10315_minus_baseline_3h="+json.dumps(new_base_3h,sort_keys=True))
print("10315_minus_10115_3h="+json.dumps(new_old_3h,sort_keys=True))
print("safety="+json.dumps(safety,sort_keys=True))
print("incremental_status="+incremental)
print(f"postlong_10315_5k_safety_pass={safety_pass}")
print(f"comparison={out}")
PY

"${PY}" - "${RUN}" <<'PY'
from pathlib import Path
import sys,zipfile
run=Path(sys.argv[1])
out=run/"SpinCore_DC1_postlong_10315_5k_bundle.zip"
for name in (
    "baseline_10105_report.json","baseline_10105_sanity.json",
    "specialist_10115_report.json","specialist_10115_sanity.json",
    "specialist_10315_report.json","specialist_10315_sanity.json",
    "postlong_10315_5k_comparison.json",
):
    p=run/name
    if not p.is_file():
        raise RuntimeError(f"missing bundle source {p}")
with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for name in (
        "baseline_10105_report.json","baseline_10105_sanity.json",
        "specialist_10115_report.json","specialist_10115_sanity.json",
        "specialist_10315_report.json","specialist_10315_sanity.json",
        "postlong_10315_5k_comparison.json",
    ):
        z.write(run/name,arcname=name)
print(f"bundle={out}")
PY

ZIP="${RUN}/SpinCore_DC1_postlong_10315_5k_bundle.zip"
if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${ZIP}" "${DWSL}/SpinCore_DC1_postlong_10315_5k_bundle.zip"
    echo "desktop_bundle=${DWIN}\\SpinCore_DC1_postlong_10315_5k_bundle.zip"
  fi
fi

echo
echo "DEEPC_RUSHER_DC1_SEMANTIC_POSTLONG_10315_5K_COMPLETE"
echo "STOP HERE. Send SpinCore_DC1_postlong_10315_5k_bundle.zip to ChatGPT."
