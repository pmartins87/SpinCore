from pathlib import Path
import pytest
from spincore.solver import Episode,SolverLibrary
ROOT=Path(__file__).resolve().parents[1]
LIB=ROOT/'build'/'libspincore_solver_c.so'
def lib():return SolverLibrary(LIB)
def hu():return Episode(1500,True,0,10,20,(0,750,750),1,(0,))
def three():return Episode(1500,False,0,10,20,(500,500,500),0,())
def test_abi_create_and_clone():
    L=lib();s=L.create(hu(),1);c=s.clone();assert s.actor==c.actor and s.neural_bytes()==c.neural_bytes();c.close();s.close()
def test_domain_separation():
    L=lib();a=L.create(hu(),1);b=L.create(three(),1);assert a.domain==1 and b.domain==0;a.close();b.close()
def test_frontier_contract():
    L=lib();s=L.create(hu(),4)
    with s.frontier_until_actor(2) as f:
        assert len(f)>0 and f.nodes_visited>=len(f)
        for i in range(len(f)):
            c=f.clone_state(i)
            assert c.terminal or c.actor==2
            c.close()
    s.close()
def test_terminal_icm_zero_sum_delta():
    L=lib();s=L.create(hu(),9)
    while not s.terminal:
        legal=s.legal_actions();a=1 if 1 in legal else legal[0];s.apply(a)
    d=s.terminal_icm_delta((.5,.3,.2));assert abs(sum(d))<1e-10;s.close()
def test_invalid_hu_fails_closed():
    L=lib()
    with pytest.raises(RuntimeError):L.create(Episode(1500,True,0,10,20,(500,500,500),0),1)
