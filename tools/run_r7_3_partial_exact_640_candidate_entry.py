from __future__ import annotations

# Recovery-history metadata is not part of the generic R7.3 diagnostic API.
# Inject it before importing the shared replicated-candidate helpers, whose
# module-level import expects the historical parameter-count evidence field.
import run_r7_3_diagnostic as _diagnostic

_diagnostic.HISTORICAL_PARAMS_PER_NETWORK = 152_434

from run_r7_3_partial_exact_640_candidate import main


if __name__ == "__main__":
    raise SystemExit(main())
