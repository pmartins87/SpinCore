#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed alternating long-horizon SpinCore training orchestrator.

The plan starts only AFTER the currently-running HU migration/+469 block has
completed PASS. It then alternates 3H and HU fixed-root blocks. A stage is
considered complete only after its PASS report is present and internally
consistent. On interruption/reboot, rerunning this orchestrator revalidates
completed stages and resumes the first incomplete stage.

No git pull/update is performed here. The local source tree is intentionally
frozen for the duration of the plan.
"""

# PROJECT_CONTRACT_IDS: GOV-023,GOV-024,PERF-001,PERF-002,PERF-010,PERF-011,PERF-012,PERF-013,PERF-014,PERF-015,PERF-016,PERF-017,PERF-019,PERF-020,TRAIN-020,TRAIN-022,TRAIN-023,TRAIN-024,MODEL-020,MODEL-025,RNG-001,RNG-002,RNG-003,CKPT-001,CKPT-004,SAFE-001,SAFE-002,SAFE-010,VALID-025,VALID-034,VALID-035,ART-001,ART-015


import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PY=ROOT/".venv_lean"/"bin"/"python"
PLAN_DEFAULT=ROOT/"contracts"/"three_month_training_plan_v1.json"
STATE_DEFAULT=ROOT/"runs"/"three_month_training_v1"/"orchestrator_state.json"
EXPECTED_CP="f2058cae8a1b194e08295f1b726b43432544d668c86a4c3a4c9ee8724963faa0"


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path:Path,payload:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    os.replace(tmp,path)


def git_blob(path:str)->str:
    cp=subprocess.run(
        ["git","hash-object",path],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return cp.stdout.strip()


def check_support_blobs(plan:dict)->None:
    for path,expected in sorted((plan.get("support_blob_shas") or {}).items()):
        p=ROOT/path
        if not p.is_file():
            raise RuntimeError(f"support file missing: {path}")
        actual=git_blob(path)
        if actual!=str(expected):
            raise RuntimeError(
                f"support blob drift: {path} actual={actual} expected={expected}"
            )


def ensure_clean_repo()->None:
    for args in (["git","diff","--quiet"],["git","diff","--cached","--quiet"]):
        cp=subprocess.run(args,cwd=ROOT)
        if cp.returncode!=0:
            raise RuntimeError("tracked source/index changes present; refusing long plan")


def available_kib()->int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    return 0


def ensure_resources()->None:
    if available_kib()<12*1024*1024:
        raise RuntimeError("less than 12 GiB WSL MemAvailable")


def trainer_processes()->str:
    cp=subprocess.run(
        [
            "pgrep","-af",
            "run_3h_semantic_long_.*\\.py|run_hu_semantic_ens8_.*\\.py"
        ],
        text=True,
        capture_output=True,
    )
    return cp.stdout.strip() if cp.returncode==0 else ""


def ensure_no_external_trainer()->None:
    rows=trainer_processes()
    if rows:
        raise RuntimeError("another SpinCore semantic trainer is already running:\n"+rows)


def find_latest_file(parent:Path,filename:str)->Path:
    candidates=[
        p/filename
        for p in parent.iterdir()
        if p.is_dir() and (p/filename).is_file()
    ]
    if not candidates:
        raise RuntimeError(f"no {filename} found below {parent}")
    candidates.sort(key=lambda p:p.stat().st_mtime,reverse=True)
    return candidates[0].resolve()


def run_streamed(cmd:list[str],*,log_path:Path,env:dict|None=None)->int:
    log_path.parent.mkdir(parents=True,exist_ok=True)
    merged=os.environ.copy()
    if env:
        merged.update(env)
    with log_path.open("a",encoding="utf-8") as log:
        log.write("\n=== COMMAND ===\n"+" ".join(cmd)+"\n")
        log.flush()
        proc=subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=merged,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return int(proc.wait())


def validate_current_hu_prerequisite()->dict:
    d=ROOT/"runs"/"hu_semantic_ens8_continuation_10105_plus469"
    report=d/"hu_semantic_ens8_continuation_10105_plus469.json"
    resume=d/"resume_state.pt"
    ensemble=d/"hu_semantic_ensemble_final_plus469.pt"
    for p in (report,resume,ensemble):
        if not p.is_file():
            raise RuntimeError(f"current HU prerequisite not complete: missing {p}")
    r=json.loads(report.read_text(encoding="utf-8"))
    if r.get("status")!="PASS" or int(r.get("completed_updates",-1))!=469:
        raise RuntimeError("current HU prerequisite is not PASS @ cumulative 469")
    if int(r.get("new_hu_roots",-1))!=128037:
        raise RuntimeError("current HU prerequisite root accounting drift")
    if r.get("three_handed_training_performed") is not False:
        raise RuntimeError("current HU prerequisite records 3H training")
    if str(r.get("resume_state_sha256"))!=sha256(resume):
        raise RuntimeError("current HU prerequisite resume hash drift")
    if str(r.get("final_semantic_ensemble_sha256"))!=sha256(ensemble):
        raise RuntimeError("current HU prerequisite ensemble hash drift")
    return {
        "domain":"HU",
        "endpoint":469,
        "dir":str(d.resolve()),
        "report":str(report.resolve()),
        "resume":str(resume.resolve()),
        "ensemble":str(ensemble.resolve()),
    }


def validate_initial_3h_prerequisite()->dict:
    d=ROOT/"runs"/"3h_semantic_long_15315_17315"
    report=d/"3h_semantic_long_15315_17315.json"
    resume=d/"resume_state.pt"
    ensemble=d/"semantic_long_ensemble_final_17315.pt"
    for p in (report,resume,ensemble):
        if not p.is_file():
            raise RuntimeError(f"initial 3H prerequisite missing: {p}")
    r=json.loads(report.read_text(encoding="utf-8"))
    if r.get("status")!="PASS" or int(r.get("completed_iteration",-1))!=17315:
        raise RuntimeError("initial 3H prerequisite is not PASS @17315")
    if r.get("final_safety_guard_pass") is not True:
        raise RuntimeError("initial 3H prerequisite safety guard is not PASS")
    if str(r.get("resume_state_sha256"))!=sha256(resume):
        raise RuntimeError("initial 3H prerequisite resume hash drift")
    if str(r.get("final_ensemble_sha256"))!=sha256(ensemble):
        raise RuntimeError("initial 3H prerequisite ensemble hash drift")
    return {
        "domain":"3H",
        "endpoint":17315,
        "dir":str(d.resolve()),
        "report":str(report.resolve()),
        "resume":str(resume.resolve()),
        "ensemble":str(ensemble.resolve()),
    }


def validate_hu_endpoint(stage:dict,out_dir:Path)->dict:
    target=int(stage["target"])
    if stage.get("mode")=="specific":
        report=out_dir/"hu_semantic_ens8_long_block.json"
        # The first post-migration specific shell already uses the generic carrier.
    else:
        report=out_dir/"hu_semantic_ens8_long_block.json"
    resume=out_dir/"resume_state.pt"
    ensemble=out_dir/f"hu_semantic_ensemble_final_cum{target}.pt"
    for p in (report,resume,ensemble):
        if not p.is_file():
            raise RuntimeError(f"HU stage output missing after runner PASS: {p}")
    r=json.loads(report.read_text(encoding="utf-8"))
    if r.get("status")!="PASS" or int(r.get("cumulative_completed_updates",-1))!=target:
        raise RuntimeError("HU stage report endpoint mismatch")
    if int(r.get("block_new_hu_roots",-1))!=(target-int(stage["start"]))*273:
        raise RuntimeError("HU stage root accounting drift")
    if r.get("three_handed_training_performed") is not False:
        raise RuntimeError("HU stage unexpectedly records 3H training")
    if str(r.get("resume_state_sha256"))!=sha256(resume):
        raise RuntimeError("HU stage resume hash drift")
    if str(r.get("final_semantic_ensemble_sha256"))!=sha256(ensemble):
        raise RuntimeError("HU stage ensemble hash drift")
    return {
        "domain":"HU",
        "endpoint":target,
        "dir":str(out_dir.resolve()),
        "report":str(report.resolve()),
        "resume":str(resume.resolve()),
        "ensemble":str(ensemble.resolve()),
    }


def validate_3h_endpoint(stage:dict,out_dir:Path)->dict:
    target=int(stage["target"])
    if stage.get("mode")=="specific":
        report=out_dir/f"3h_semantic_long_{int(stage['start'])}_{target}.json"
    else:
        report=out_dir/"3h_semantic_long_block.json"
    resume=out_dir/"resume_state.pt"
    ensemble=out_dir/f"semantic_long_ensemble_final_{target}.pt"
    for p in (report,resume,ensemble):
        if not p.is_file():
            raise RuntimeError(f"3H stage output missing after runner PASS: {p}")
    r=json.loads(report.read_text(encoding="utf-8"))
    if r.get("status")!="PASS" or int(r.get("completed_iteration",-1))!=target:
        raise RuntimeError("3H stage report endpoint mismatch")
    if int(r.get("new_roots_recorded",-1))!=(target-int(stage["start"]))*64:
        raise RuntimeError("3H stage root accounting drift")
    if r.get("final_safety_guard_pass") is not True:
        raise RuntimeError("3H stage safety guard is not PASS")
    if r.get("hu_training_performed") is not False:
        raise RuntimeError("3H stage unexpectedly records HU training")
    if str(r.get("resume_state_sha256"))!=sha256(resume):
        raise RuntimeError("3H stage resume hash drift")
    if str(r.get("final_ensemble_sha256"))!=sha256(ensemble):
        raise RuntimeError("3H stage ensemble hash drift")
    return {
        "domain":"3H",
        "endpoint":target,
        "dir":str(out_dir.resolve()),
        "report":str(report.resolve()),
        "resume":str(resume.resolve()),
        "ensemble":str(ensemble.resolve()),
    }


def resolve_stage_out(stage:dict)->Path:
    if stage["id"]=="3h_block_01":
        return ROOT/"runs"/"3h_semantic_long_17315_19315"
    if stage["id"]=="hu_block_02":
        return ROOT/"runs"/"hu_semantic_ens8_long_cum469_938"
    return ROOT/"runs"/"three_month_training_v1"/"stages"/stage["id"]


def build_stage_command(stage:dict,last:dict,deps:dict)->tuple[list[str],dict|None,Path]:
    out=resolve_stage_out(stage)
    start=int(stage["start"])
    target=int(stage["target"])
    if stage["id"]=="3h_block_01":
        return ["bash","tools/run_3h_semantic_long_17315_19315.sh"],None,out
    if stage["id"]=="hu_block_02":
        return ["bash","tools/run_hu_semantic_ens8_long_cum469_938.sh"],None,out

    checkpoint=ROOT/"runs"/"lt3_parallel_9105_10105"/"20260922_132104"/"checkpoint.pt"
    solver=ROOT/"build"/"libspincore_solver_c.so"
    out.mkdir(parents=True,exist_ok=True)

    if stage["domain"]=="HU":
        cmd=[
            str(PY),"tools/run_hu_semantic_ens8_long_block.py",
            "--checkpoint",str(checkpoint),
            "--source-resume",last["HU"]["resume"],
            "--source-ensemble",last["HU"]["ensemble"],
            "--source-report",last["HU"]["report"],
            "--start-cumulative-updates",str(start),
            "--target-cumulative-updates",str(target),
            "--solver",str(solver),
            "--out-dir",str(out),
            "--workers","31",
            "--fit-concurrency","4",
            "--threads-per-member","8",
        ]
        return cmd,None,out

    pack=out/"parallel_semantic_pack"
    env={
        "PYTHONPATH":f"{ROOT/'python'}:{ROOT/'tools'}",
        "OMP_NUM_THREADS":"8",
        "MKL_NUM_THREADS":"8",
        "OPENBLAS_NUM_THREADS":"8",
        "SPINCORE_SEMANTIC_FIT_CONCURRENCY":"4",
        "SPINCORE_SEMANTIC_THREADS_PER_MEMBER":"8",
        "SPINCORE_SEMANTIC_PACK_DIR":str(pack),
    }
    cmd=[
        str(PY),"tools/run_3h_semantic_long_block_parallel.py",
        "--checkpoint",str(checkpoint),
        "--semantic-advantage-models",deps["adv_models"],
        "--probe",deps["probe"],
        "--solver",str(solver),
        "--spin-bundle",deps["spin_bundle"],
        "--source-resume",last["3H"]["resume"],
        "--source-final-ensemble",last["3H"]["ensemble"],
        "--source-report",last["3H"]["report"],
        "--start-iteration",str(start),
        "--target-iteration",str(target),
        "--run-dir",str(out),
        "--threads","8",
    ]
    return cmd,env,out


def resolve_dependencies(state:dict)->dict:
    existing=state.get("dependencies")
    if existing:
        for key,row in existing.items():
            p=Path(row["path"])
            if not p.is_file() or sha256(p)!=row["sha256"]:
                raise RuntimeError(f"frozen dependency drift: {key}")
        return {k:v["path"] for k,v in existing.items()}

    adv=find_latest_file(ROOT/"runs"/"3h_v1_semantic_shadow","3h_v1_semantic_shadow_models.pt")
    probe=find_latest_file(ROOT/"runs"/"3h_advantage_controlled_budget_curve","3h_advantage_controlled_split_probe.pt")
    spin=ROOT/"runs"/"3h_semantic_research_canonical_10105_10115"/"spincore_hybrid_10105.pt"
    for p in (adv,probe,spin):
        if not p.is_file():
            raise RuntimeError(f"frozen dependency missing: {p}")
    state["dependencies"]={
        "adv_models":{"path":str(adv),"sha256":sha256(adv)},
        "probe":{"path":str(probe),"sha256":sha256(probe)},
        "spin_bundle":{"path":str(spin.resolve()),"sha256":sha256(spin)},
    }
    return {k:v["path"] for k,v in state["dependencies"].items()}


def load_state(path:Path,plan_sha:str)->dict:
    if path.is_file():
        s=json.loads(path.read_text(encoding="utf-8"))
        if s.get("schema")!="SPINCORE_THREE_MONTH_ORCHESTRATOR_STATE_V1":
            raise RuntimeError("wrong orchestrator state schema")
        if str(s.get("plan_sha256"))!=plan_sha:
            raise RuntimeError("orchestrator plan changed after state creation")
        return s
    return {
        "schema":"SPINCORE_THREE_MONTH_ORCHESTRATOR_STATE_V1",
        "status":"PREPARED",
        "plan_sha256":plan_sha,
        "completed_stage_ids":[],
        "current_stage_id":None,
        "last_endpoints":{},
        "dependencies":{},
        "started_at_unix":None,
        "updated_at_unix":time.time(),
    }


def build_solver()->None:
    subprocess.run(
        ["cmake","-S",".","-B","build","-DCMAKE_BUILD_TYPE=Release"],
        cwd=ROOT,check=True
    )
    subprocess.run(
        ["cmake","--build","build","-j","8","--target","spincore_solver_c"],
        cwd=ROOT,check=True
    )


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan",type=Path,default=PLAN_DEFAULT)
    ap.add_argument("--state",type=Path,default=STATE_DEFAULT)
    ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args()

    plan_path=args.plan.resolve(strict=True)
    plan=json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema")!="SPINCORE_THREE_MONTH_TRAINING_PLAN_V1":
        raise RuntimeError("wrong three-month plan schema")
    if plan.get("status")!="READY_AFTER_CURRENT_HU_PASS":
        raise RuntimeError("three-month plan is not READY_AFTER_CURRENT_HU_PASS")
    plan_sha=sha256(plan_path)

    check_support_blobs(plan)
    ensure_clean_repo()

    state=load_state(args.state.resolve(),plan_sha)
    if state["started_at_unix"] is None:
        state["started_at_unix"]=time.time()

    hu0=validate_current_hu_prerequisite()
    h30=validate_initial_3h_prerequisite()
    last={"HU":hu0,"3H":h30}

    # Revalidate already-completed stages in plan order and reconstruct endpoints.
    completed=set(state.get("completed_stage_ids") or [])
    for stage in plan["stages"]:
        if stage["id"] not in completed:
            continue
        out=resolve_stage_out(stage)
        endpoint=validate_hu_endpoint(stage,out) if stage["domain"]=="HU" else validate_3h_endpoint(stage,out)
        last[stage["domain"]]=endpoint

    state["last_endpoints"]=last
    deps=resolve_dependencies(state)
    state["updated_at_unix"]=time.time()
    atomic_json(args.state.resolve(),state)

    if args.dry_run:
        print("THREE_MONTH_DRY_RUN_PASS")
        print(f"stages_total={len(plan['stages'])}")
        print(f"stages_completed={len(completed)}")
        for stage in plan["stages"]:
            marker="DONE" if stage["id"] in completed else "PENDING"
            print(f"{marker} {stage['id']} {stage['domain']} {stage['start']}->{stage['target']}")
        return 0

    ensure_no_external_trainer()
    ensure_resources()
    checkpoint=ROOT/"runs"/"lt3_parallel_9105_10105"/"20260922_132104"/"checkpoint.pt"
    if sha256(checkpoint)!=EXPECTED_CP:
        raise RuntimeError("frozen checkpoint hash drift")
    build_solver()

    state["status"]="RUNNING"
    atomic_json(args.state.resolve(),state)

    for stage in plan["stages"]:
        sid=stage["id"]
        if sid in completed:
            continue

        ensure_no_external_trainer()
        ensure_resources()
        check_support_blobs(plan)
        ensure_clean_repo()

        expected_last=int(stage["start"])
        actual_last=int(last[stage["domain"]]["endpoint"])
        if actual_last!=expected_last:
            raise RuntimeError(
                f"{sid}: source endpoint drift {actual_last} != expected {expected_last}"
            )

        cmd,env,out=build_stage_command(stage,last,deps)
        state["current_stage_id"]=sid
        state["status"]="RUNNING"
        state["updated_at_unix"]=time.time()
        atomic_json(args.state.resolve(),state)

        print(
            f"THREE_MONTH_STAGE_BEGIN id={sid} domain={stage['domain']} "
            f"start={stage['start']} target={stage['target']}",
            flush=True,
        )
        status=run_streamed(
            cmd,
            log_path=ROOT/"runs"/"three_month_training_v1"/"logs"/f"{sid}.log",
            env=env,
        )
        if status!=0:
            state["status"]="STOPPED_STAGE_FAILURE"
            state["last_failure"]={"stage_id":sid,"exit_status":status,"time":time.time()}
            state["updated_at_unix"]=time.time()
            atomic_json(args.state.resolve(),state)
            print(f"THREE_MONTH_STAGE_STOP id={sid} status={status}",flush=True)
            return status

        endpoint=validate_hu_endpoint(stage,out) if stage["domain"]=="HU" else validate_3h_endpoint(stage,out)
        last[stage["domain"]]=endpoint
        completed.add(sid)
        state["completed_stage_ids"]=[
            row["id"] for row in plan["stages"] if row["id"] in completed
        ]
        state["last_endpoints"]=last
        state["current_stage_id"]=None
        state["updated_at_unix"]=time.time()
        atomic_json(args.state.resolve(),state)
        print(
            f"THREE_MONTH_STAGE_PASS id={sid} endpoint={endpoint['endpoint']}",
            flush=True,
        )

    state["status"]="PASS"
    state["current_stage_id"]=None
    state["finished_at_unix"]=time.time()
    state["updated_at_unix"]=time.time()
    atomic_json(args.state.resolve(),state)
    print("SPINCORE_THREE_MONTH_TRAINING_PLAN_PASS")
    print(f"final_hu_cumulative_updates={last['HU']['endpoint']}")
    print(f"final_3h_iteration={last['3H']['endpoint']}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
