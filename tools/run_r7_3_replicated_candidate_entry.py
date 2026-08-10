from __future__ import annotations

# The historical parameter count is recovery evidence metadata, not part of the
# runtime model contract. Keep it local to the candidate entry point instead of
# requiring the generic R7.3 diagnostic module to export recovery-history data.
import run_r7_3_diagnostic as _diagnostic

_diagnostic.HISTORICAL_PARAMS_PER_NETWORK = 152_434

from run_r7_3_replicated_640_candidate import main


if __name__ == "__main__":
    raise SystemExit(main())
