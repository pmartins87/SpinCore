from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))

import benchmark_teacher_architecture_17315 as bench


def test_current_raw_mean_can_be_extreme_while_policy_mean_is_not():
    raw=np.zeros((8,10),dtype=np.float64)
    legal=(1,9)
    for i in range(7):
        raw[i,1]=1.0
        raw[i,9]=-1.0
    raw[7,1]=-8.0
    raw[7,9]=8.0
    current=bench.candidate_policy(raw,legal,"A_CURRENT_RAW_MEAN_RM")
    pmean=bench.candidate_policy(raw,legal,"B_MEMBER_POLICY_MEAN")
    assert current[9]==1.0
    assert pmean[9] < 0.25
    assert pmean[1] > 0.75


def test_trimmed_mean_rejects_single_extreme_member():
    raw=np.zeros((8,10),dtype=np.float64)
    legal=(1,9)
    for i in range(7):
        raw[i,1]=1.0
        raw[i,9]=-1.0
    raw[7,1]=-100.0
    raw[7,9]=100.0
    trimmed=bench.candidate_policy(raw,legal,"D_TRIMMED_RAW_MEAN_RM")
    assert trimmed[1]==1.0
    assert trimmed[9]==0.0


def test_median_raw_rejects_minority_outliers():
    raw=np.zeros((8,10),dtype=np.float64)
    legal=(0,1,9)
    for i in range(6):
        raw[i,0]=-1.0
        raw[i,1]=2.0
        raw[i,9]=-2.0
    for i in range(6,8):
        raw[i,0]=-1.0
        raw[i,1]=-10.0
        raw[i,9]=10.0
    median=bench.candidate_policy(raw,legal,"C_MEDIAN_RAW_RM")
    assert median[1]==1.0
    assert median[9]==0.0


def test_member_policy_mean_is_normalized():
    rng=np.random.default_rng(123)
    raw=rng.normal(size=(8,10))
    legal=(0,1,3,5,7,8,9)
    p=bench.candidate_policy(raw,legal,"B_MEMBER_POLICY_MEAN")
    assert abs(float(p.sum())-1.0)<1e-12
    assert np.all(p>=0.0)
    assert all(p[a]==0.0 for a in set(range(10))-set(legal))


def test_target_branch_detects_positive_regret_and_fallback():
    raw=np.zeros(10,dtype=np.float64)
    raw[1]=-0.1;raw[9]=-0.2
    assert bench._target_branch(raw,(1,9))=="SOFTMAX_FALLBACK"
    raw[9]=0.01
    assert bench._target_branch(raw,(1,9))=="POSITIVE_REGRET_MATCHING"
