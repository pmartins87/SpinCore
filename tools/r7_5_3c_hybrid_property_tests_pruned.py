from __future__ import annotations

# Reuse the already-audited property cases, but replace the superset prototype
# model with the pruned scientific candidate class so parameter/RAM accounting
# reflects only modules reachable by each candidate's forward path.
import r7_5_3c_hybrid_property_tests as tests
from spincore_nn.hybrid_v3_pruned import HybridCandidateNetV3


tests.HybridNetV3 = HybridCandidateNetV3


if __name__ == "__main__":
    raise SystemExit(tests.main())
