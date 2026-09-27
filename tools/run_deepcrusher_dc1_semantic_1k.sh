#!/usr/bin/env bash
set -euo pipefail

ROOT="${HOME}/spincore_lean_functional"
cd "${ROOT}"
PY="${ROOT}/.venv_lean/bin/python"
SOURCE="${ROOT}/runs/lt3_parallel_9105_10105/20260922_132104"
CHECKPOINT="${SOURCE}/checkpoint.pt"
HU="${SOURCE}/hu_ensemble_state.pt"
SEM_RUN="${ROOT}/runs/3h_semantic_research_10105_10115"
SEM_POLICY="${SEM_RUN}/semantic_tail_policy_10115.pt"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"
EXPECTED_HU="8d11cb6bce172e24e903ced650c7a7d82aeb2b251c71c3cfc28e37d873ddb62d"

[[ -x "${PY}" ]] || { echo "ERROR: missing Python venv" >&2; exit 2; }
for f in "${CHECKPOINT}" "${HU}" "${SEM_POLICY}"; do
  [[ -f "${f}" ]] || { echo "ERROR: missing artifact: ${f}" >&2; exit 2; }
done
[[ "$(sha256sum "${CHECKPOINT}"|awk '{print $1}')" == "${EXPECTED_CP}" ]] || { echo "ERROR: checkpoint SHA mismatch" >&2; exit 3; }
[[ "$(sha256sum "${HU}"|awk '{print $1}')" == "${EXPECTED_HU}" ]] || { echo "ERROR: HU SHA mismatch" >&2; exit 3; }

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
RUN="${ROOT}/runs/deepcrusher_dc1_semantic_1k/${STAMP}"
mkdir -p "${RUN}"
BASE_BUNDLE="${RUN}/spincore_hybrid_10105.pt"
BASE_REPORT="${RUN}/baseline_10105_report.json"
BASE_TRACES="${RUN}/baseline_10105_traces.jsonl"
BASE_SANITY="${RUN}/baseline_10105_sanity.json"
SEM_REPORT="${RUN}/semantic_10115_report.json"
SEM_TRACES="${RUN}/semantic_10115_traces.jsonl"
SEM_SANITY="${RUN}/semantic_10115_sanity.json"
COMPARE="${RUN}/semantic_vs_baseline_comparison.json"

"${PY}" tools/export_spincore_hybrid_inference.py \
  --checkpoint "${CHECKPOINT}" --ensemble "${HU}" --out "${BASE_BUNDLE}"

"${PY}" tools/evaluate_deepcrusher_dc1.py \
  --solver build/libspincore_solver_c.so \
  --spin-bundle "${BASE_BUNDLE}" \
  --scenarios 1000 --workers 8 --seed 20260923 \
  --report "${BASE_REPORT}" --traces "${BASE_TRACES}" --max-decisions 200

"${PY}" tools/audit_benchmark_decisions.py \
  --traces "${BASE_TRACES}" --report "${BASE_SANITY}" --max-examples-per-code 30

"${PY}" tools/evaluate_deepcrusher_dc1_semantic_candidate.py \
  --solver build/libspincore_solver_c.so \
  --base-spin-bundle "${BASE_BUNDLE}" \
  --semantic-policy "${SEM_POLICY}" \
  --scenarios 1000 --workers 8 --seed 20260923 \
  --report "${SEM_REPORT}" --traces "${SEM_TRACES}" --max-decisions 200

"${PY}" tools/audit_benchmark_decisions.py \
  --traces "${SEM_TRACES}" --report "${SEM_SANITY}" --max-examples-per-code 30

"${PY}" - "${BASE_REPORT}" "${SEM_REPORT}" "${BASE_SANITY}" "${SEM_SANITY}" "${COMPARE}" <<'PY'
import json,math,statistics,sys
from pathlib import Path

bp,sp,bs,ss,out=map(Path,sys.argv[1:])
b=json.loads(bp.read_text())
s=json.loads(sp.read_text())
bsa=json.loads(bs.read_text())
ssa=json.loads(ss.read_text())

br=sorted(b["scenario_rows"],key=lambda x:x["scenario"])
sr=sorted(s["scenario_rows"],key=lambda x:x["scenario"])
if len(br)!=len(sr):
    raise RuntimeError("scenario count mismatch")

for x,y in zip(br,sr):
    ident=(x["scenario"],x["domain"],x["blind"],x["dealer_id"],x["deal_seed"])
    ident2=(y["scenario"],y["domain"],y["blind"],y["dealer_id"],y["deal_seed"])
    if ident!=ident2:
        raise RuntimeError("scenario identity mismatch")

def ci(vals):
    n=len(vals)
    mean=statistics.fmean(vals) if vals else float("nan")
    sem=statistics.stdev(vals)/math.sqrt(n) if n>1 else 0.0
    return {"n":n,"mean":mean,"sem":sem,"ci95_low":mean-1.96*sem,"ci95_high":mean+1.96*sem}

def margin(row):
    return float(row["paired_spincore_minus_deepcrusher_chips_per_exposure"])

all_delta=[margin(y)-margin(x) for x,y in zip(br,sr)]
three_delta=[margin(y)-margin(x) for x,y in zip(br,sr) if x["domain"]=="THREE_HANDED"]
hu_pairs=[(margin(x),margin(y)) for x,y in zip(br,sr) if x["domain"]=="TRUE_HEADS_UP"]
hu_exact=all(a==b for a,b in hu_pairs)

def rate(sanity,code):
    denom=max(1,int(sanity["spincore_decisions"]))
    return float((sanity.get("flag_counts") or {}).get(code,0))/denom

codes=sorted(set((bsa.get("flag_counts") or {}))|set((ssa.get("flag_counts") or {})))
flags={
    code:{
        "baseline_count":int((bsa.get("flag_counts") or {}).get(code,0)),
        "semantic_count":int((ssa.get("flag_counts") or {}).get(code,0)),
        "baseline_per_spincore_decision":rate(bsa,code),
        "semantic_per_spincore_decision":rate(ssa,code),
    }
    for code in codes
}

three_base=b["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"]
three_sem=s["summary"]["THREE_HANDED"]["paired_spincore_minus_deepcrusher_chips_per_policy_seat_hand"]

report={
    "schema":"SPINCORE_DC1_SEMANTIC_1K_PAIRED_COMPARISON_V1",
    "scope":"DEVELOPMENT_ONLY_NO_CANONICAL_STRENGTH_CLAIM",
    "seed":20260923,
    "scenarios":1000,
    "hu_exact_scenario_margin_parity":hu_exact,
    "baseline_10105_3h_vs_deepcrusher":three_base,
    "semantic_10115_3h_vs_deepcrusher":three_sem,
    "semantic_minus_baseline_scenario_cluster":{
        "ALL":ci(all_delta),
        "THREE_HANDED":ci(three_delta),
    },
    "sanity_flags":flags,
    "development_indicators":{
        "semantic_3h_delta_positive":ci(three_delta)["mean"]>0,
        "semantic_3h_delta_ci95_low_positive":ci(three_delta)["ci95_low"]>0,
        "hu_parity_exact":hu_exact,
        "high_card_jam_rate_not_higher":(
            rate(ssa,"POSTFLOP_DEEP_HIGH_CARD_JAM")
            <= rate(bsa,"POSTFLOP_DEEP_HIGH_CARD_JAM")
        ),
        "trips_fold_rate_not_higher":(
            rate(ssa,"POSTFLOP_TRIPS_PLUS_FOLD")
            <= rate(bsa,"POSTFLOP_TRIPS_PLUS_FOLD")
        ),
    },
    "interpretation":(
        "Relative 10115-semantic vs 10105-baseline evidence only. Both arms use "
        "the same DeepCrusher translation, same scenario/deal seeds and unchanged "
        "HU policy. DC0 real-OpenHoldem parity remains pending, so this is not a "
        "canonical strength claim."
    ),
}
out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print("=== semantic vs baseline DC1 1k ===")
print(json.dumps(report["semantic_minus_baseline_scenario_cluster"],sort_keys=True))
print(json.dumps(report["development_indicators"],sort_keys=True))
print(f"comparison={out}")
print("DC1_SEMANTIC_1K_COMPARISON_COMPLETE")
PY

"${PY}" - "${RUN}" <<'PY'
from pathlib import Path
import sys,zipfile
run=Path(sys.argv[1])
out=run/"SpinCore_DC1_semantic_1k_bundle.zip"
forname=(
    "baseline_10105_report.json","baseline_10105_sanity.json",
    "semantic_10115_report.json","semantic_10115_sanity.json",
    "semantic_vs_baseline_comparison.json",
)
with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for name in forname:
        p=run/name
        if p.is_file(): z.write(p,arcname=name)
print(f"bundle={out}")
PY

ZIP="${RUN}/SpinCore_DC1_semantic_1k_bundle.zip"
if command -v powershell.exe >/dev/null 2>&1; then
  DWIN="$(powershell.exe -NoProfile -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r' | tail -n1)"
  DWSL="$(wslpath -u "${DWIN}" 2>/dev/null || true)"
  if [[ -n "${DWSL}" && -d "${DWSL}" ]]; then
    cp -f "${ZIP}" "${DWSL}/SpinCore_DC1_semantic_1k_bundle.zip"
    echo "desktop_bundle=${DWIN}\\SpinCore_DC1_semantic_1k_bundle.zip"
  fi
fi

echo "DEEPC_RUSHER_DC1_SEMANTIC_1K_COMPLETE"
