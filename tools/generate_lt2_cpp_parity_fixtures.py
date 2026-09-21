#!/usr/bin/env python3
from __future__ import annotations

"""Generate already-seen forensic inference fixtures for native C++ parity."""

import argparse
import hashlib
import random
import struct
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
sys.path.insert(0,str(ROOT/"tools"))

import torch

import audit_lt2_stage_a_b_first_divergence as fd
from spincore.lean_action_scope import FIRST_RELEASE_ACTION_SPEC
from spincore.lean_hybrid_deployment_agent import LeanHybridDeploymentAgent
from spincore.lean_solver_actions import apply_lean, lean_legal_actions
from spincore.legacy_scenario import LegacyScenarioConfig, LegacyScenarioSampler
from spincore.r7_5_action_cfr import legal_mask
from spincore.solver import SolverLibrary

MAGIC=b"SCLT2F1\x00"
VERSION=1
SEEDS=(20260920,20260921,20260922)


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def _sample(legal,probs,rng):
    x=rng.random()
    c=0.0
    for a in legal:
        c+=float(probs[a])
        if x<c:
            return int(a)
    return int(legal[-1])


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--solver",type=Path,required=True)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--scenarios-per-seed",type=int,default=700)
    p.add_argument("--max-decisions",type=int,default=20000)
    args=p.parse_args()

    if args.scenarios_per_seed<=0 or args.max_decisions<=0:
        raise SystemExit("positive fixture arguments required")

    torch.set_num_threads(8)
    solver=SolverLibrary(args.solver.resolve(strict=True))
    bundle=args.bundle.resolve(strict=True)
    agent=LeanHybridDeploymentAgent.from_bundle(bundle,device="cpu",seed=0)

    records=[]
    hands={"THREE_HANDED":0,"TRUE_HEADS_UP":0}
    decisions_by_domain={"THREE_HANDED":0,"TRUE_HEADS_UP":0}
    streets={0:0,1:0,2:0,3:0}

    for seed in SEEDS:
        sampler=LegacyScenarioSampler(
            seed=int(seed)^0x5CE0A710,
            config=LegacyScenarioConfig(),
        )
        for scenario in range(int(args.scenarios_per_seed)):
            if len(records)>=int(args.max_decisions):
                break
            episode=sampler.sample_episode()
            domain="TRUE_HEADS_UP" if episode.game_is_hu else "THREE_HANDED"
            hands[domain]+=1
            deal_seed=fd._mix64(int(seed),int(scenario),0xD34A1)
            state=solver.create(episode,int(deal_seed))
            rng=random.Random(fd._mix64(int(seed),int(scenario),0xC0FFEE))
            try:
                local=0
                while not state.terminal and len(records)<int(args.max_decisions):
                    active_mask,legal,probs=agent.distribution(state)
                    obs=bytes(state.neural_bytes())
                    if len(obs)!=126:
                        raise RuntimeError(f"unexpected SPNNIV1 size {len(obs)}")
                    mask=tuple(int(x) for x in legal_mask(legal))
                    street=int(state.neural_bytes_v2()[112])
                    records.append((
                        1 if episode.game_is_hu else 0,
                        street,
                        obs,
                        mask,
                        tuple(float(x) for x in probs),
                    ))
                    decisions_by_domain[domain]+=1
                    streets[street]+=1
                    slot=_sample(legal,probs,rng)
                    apply_lean(state,active_mask,slot)
                    local+=1
                    if local>200:
                        raise RuntimeError("fixture trajectory exceeded 200 decisions")
            finally:
                state.close()

    if not records:
        raise RuntimeError("no parity fixtures generated")
    if decisions_by_domain["THREE_HANDED"]==0 or decisions_by_domain["TRUE_HEADS_UP"]==0:
        raise RuntimeError("fixture set must cover both domains")
    if streets[0]==0 or sum(streets[s] for s in (1,2,3))==0:
        raise RuntimeError("fixture set must cover preflop and postflop")

    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II",VERSION,len(records)))
        for domain,street,obs,mask,probs in records:
            f.write(struct.pack("<BBH",domain,street,0))
            f.write(obs)
            f.write(bytes(mask))
            f.write(struct.pack("<10f",*probs))

    print("LT2_CPP_PARITY_FIXTURES_EXPORTED")
    print(f"bundle_sha256={sha256(bundle)}")
    print(f"out={args.out.resolve()}")
    print(f"out_sha256={sha256(args.out.resolve())}")
    print(f"records={len(records)}")
    print(f"hands={hands}")
    print(f"decisions_by_domain={decisions_by_domain}")
    print(f"streets={streets}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
