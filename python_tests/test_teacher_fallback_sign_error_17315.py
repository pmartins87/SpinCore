from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))

import diagnose_teacher_fallback_sign_error_17315 as d


def rec(votes,fb,strength=1.0,tp=1.0,mean=-0.2,target=1.0,bprob=0.4,legal=3):
    return {
        "target_strength":strength,
        "target_best_policy_probability":tp,
        "target_best_positive_votes":votes,
        "individual_fallback_count":fb,
        "mean_raw_target_best":mean,
        "target_raw_target_best":target,
        "member_policy_mean_target_best_probability":bprob,
        "legal_count":legal,
    }


def test_summarize_records_zero_vote_and_fallback_rates():
    s=d.summarize_records([rec(0,8),rec(4,4),rec(8,0)])
    assert s["count"]==3
    assert abs(s["zero_positive_vote_fraction"]-1/3)<1e-12
    assert abs(s["at_least_four_positive_vote_fraction"]-2/3)<1e-12
    assert abs(s["all_members_individual_fallback_fraction"]-1/3)<1e-12
    assert abs(s["mean_individual_fallback_member_fraction"]-0.5)<1e-12


def test_quantile_edges_and_bins_are_stable():
    values=list(range(1,101))
    q=d.quantile_edges(values,4)
    assert len(q)==3
    assert d.bin_index(1,q)==0
    assert d.bin_index(100,q)==3


def test_rm_returns_normalized_policy():
    raw=[0.0]*10
    raw[0]=-1.0
    raw[1]=2.0
    raw[2]=-3.0
    p=d.rm(raw,(0,1,2))
    assert abs(float(p.sum())-1.0)<1e-12
    assert p[1]==1.0
