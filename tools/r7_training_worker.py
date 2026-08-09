from __future__ import annotations
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'python'))
from spincore.r7 import load_checkpoint,save_checkpoint
from spincore.deep_cfr import DeepCFRDomainSession,icm_delta_utility
from spincore.solver import SolverLibrary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--solver',type=Path,required=True);ap.add_argument('--kind',choices=['advantage','policy'],required=True);ap.add_argument('--steps',type=int,required=True);ap.add_argument('--batch-size',type=int,required=True);ap.add_argument('--payout',nargs=3,type=float,required=True);a=ap.parse_args();b,p,e=load_checkpoint(a.checkpoint);s=DeepCFRDomainSession(solver_library=SolverLibrary(a.solver),bundle=b,terminal_utility=icm_delta_utility(a.payout));loss=s.train_advantage(steps=a.steps,batch_size=a.batch_size) if a.kind=='advantage' else s.train_average_policy(steps=a.steps,batch_size=a.batch_size);p.adv_optimizer_step=b.counters['adv_optimizer_steps'];p.policy_optimizer_step=b.counters['policy_optimizer_steps'];e['last_worker']={'kind':a.kind,'steps':a.steps,'loss_last':loss[-1] if loss else None};save_checkpoint(a.checkpoint,b,p,e);return 0
if __name__=='__main__':raise SystemExit(main())
