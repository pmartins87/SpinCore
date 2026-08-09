import random
from spincore_nn import UniformReservoir
def test_algorithm_r_is_bounded_and_deterministic():
    a=UniformReservoir(10,7);b=UniformReservoir(10,7)
    for i in range(100):a.add(i);b.add(i)
    assert a.items==b.items and len(a.items)==10 and a.seen==100
def test_reservoir_state_roundtrip():
    a=UniformReservoir(5,1)
    for i in range(20):a.add(i)
    b=UniformReservoir.from_state_dict(a.state_dict());assert b.items==a.items and b.seen==a.seen
    for i in range(20,50):a.add(i);b.add(i)
    assert a.items==b.items
