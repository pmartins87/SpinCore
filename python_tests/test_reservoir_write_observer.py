from __future__ import annotations

import random
import unittest

from spincore_nn.reservoir import UniformReservoir


class UniformReservoirObserverTest(unittest.TestCase):
    def test_observer_is_semantically_inert(self):
        a=UniformReservoir(7,12345)
        b=UniformReservoir(7,12345)
        writes=[]

        b.set_write_observer(lambda index,item:writes.append((int(index),item)))

        for value in range(100):
            a.add(value)
            b.add(value)

        self.assertEqual(a.items,b.items)
        self.assertEqual(a.seen,b.seen)
        self.assertEqual(a.rng.getstate(),b.rng.getstate())
        self.assertEqual(a.state_dict(),b.state_dict())

        replay=[None]*7
        # Replay only the writes after initial fill is not enough to reconstruct
        # without the initial events, so use the complete observer stream.
        for index,item in writes:
            replay[index]=item
        self.assertEqual(replay,b.items)

    def test_observer_does_not_enter_checkpoint_state(self):
        r=UniformReservoir(3,77)
        r.set_write_observer(lambda _i,_x:None)
        for value in range(10):
            r.add(value)
        state=r.state_dict()
        self.assertNotIn("_write_observer",state)
        restored=UniformReservoir.from_state_dict(state)
        self.assertEqual(restored.state_dict(),state)
        self.assertIsNone(restored._write_observer)


if __name__=="__main__":
    unittest.main()
