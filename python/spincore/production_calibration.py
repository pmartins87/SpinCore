from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .production_transaction_checkpoint import LoadedProductionTransaction


SCHEMA = "SPINCORE_R8_PRODUCTION_CALIBRATION_V1"


def transaction_stream_key(transaction: LoadedProductionTransaction) -> str:
    """Canonical independent-stream key for calibration comparisons."""
    identity = transaction.identity
    return f"{identity.profile_id}|{identity.domain}|{identity.algorithm_seed}"


def transaction_semantic_digest(transaction: LoadedProductionTransaction) -> str:
    """Return the already-validated integrated production generation identity.

    `load_current_production_transaction` / the transaction loader verifies the
    component hashes and semantic agreement among stream/model/RNG, scheduler
    and central Algorithm-R state before constructing this object.  Calibration
    therefore compares the integrated durable generation, not a caller-defined
    checksum from a log or performance harness.
    """
    manifest = dict(transaction.manifest)
    if manifest.get("semantic_consistency_validated") is not True:
        raise ValueError("calibration transaction lacks semantic-consistency validation")
    if manifest.get("ready_for_tables", False) is not False:
        raise ValueError("calibration transaction cannot authorize table use")
    if str(manifest.get("generation_id")) != str(transaction.generation_id):
        raise ValueError("calibration transaction generation identity mismatch")
    generation_id = str(transaction.generation_id)
    if not generation_id.startswith("spingen-v1-") or len(generation_id) != len("spingen-v1-") + 64:
        raise ValueError("calibration transaction generation id is malformed")
    return generation_id


def transaction_digest_map(
    transactions: Sequence[LoadedProductionTransaction],
) -> dict[str, str]:
    """Build a unique stream->integrated-generation map, fail closed."""
    if not transactions:
        raise ValueError("calibration transaction set cannot be empty")
    out: dict[str, str] = {}
    for transaction in transactions:
        key = transaction_stream_key(transaction)
        digest = transaction_semantic_digest(transaction)
        if key in out:
            raise ValueError(f"duplicate calibration production stream: {key}")
        out[key] = digest
    return out


@dataclass(frozen=True)
class CalibrationTrial:
    """One Ryzen production-calibration observation.

    `stream_state_digests` are authoritative semantic digests produced by the
    integrated production transaction for each independent stream.  A trial is
    eligible only when every digest is byte-for-byte identical to the serial
    reference for the same stream.
    """

    concurrency: int
    elapsed_seconds: float
    completed_work_units: int
    stream_state_digests: Mapping[str, str]
    peak_rss_mib: float | None = None
    mean_cpu_percent: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.concurrency <= 0:
            raise ValueError("calibration concurrency must be positive")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds <= 0.0:
            raise ValueError("elapsed_seconds must be finite and positive")
        if self.completed_work_units <= 0:
            raise ValueError("completed_work_units must be positive")
        if not self.stream_state_digests:
            raise ValueError("calibration trial must contain stream digests")
        if any(not str(k).strip() or not str(v).strip() for k, v in self.stream_state_digests.items()):
            raise ValueError("stream names and semantic digests must be non-empty")
        if self.peak_rss_mib is not None and (
            not math.isfinite(self.peak_rss_mib) or self.peak_rss_mib <= 0.0
        ):
            raise ValueError("peak_rss_mib must be finite and positive when present")
        if self.mean_cpu_percent is not None and (
            not math.isfinite(self.mean_cpu_percent) or self.mean_cpu_percent < 0.0
        ):
            raise ValueError("mean_cpu_percent must be finite and non-negative when present")

    @property
    def throughput(self) -> float:
        return float(self.completed_work_units) / float(self.elapsed_seconds)


def _digest_match(reference: Mapping[str, str], trial: CalibrationTrial) -> bool:
    return dict(trial.stream_state_digests) == dict(reference)


def select_calibration(
    *,
    reference_stream_state_digests: Mapping[str, str],
    trials: Sequence[CalibrationTrial],
) -> dict:
    """Select the fastest semantically exact concurrency, fail closed.

    Acceptance is deliberately independent of CPU utilization.  A concurrency
    level that is faster but changes any independent stream state is rejected.
    If throughput is exactly tied, the lower concurrency wins to reduce
    operational complexity and resource pressure.
    """

    reference = {str(k): str(v) for k, v in reference_stream_state_digests.items()}
    if not reference or any(not k.strip() or not v.strip() for k, v in reference.items()):
        raise ValueError("serial reference must contain non-empty stream digests")
    if not trials:
        raise ValueError("at least one calibration trial is required")

    seen: set[int] = set()
    rows: list[dict] = []
    eligible: list[CalibrationTrial] = []
    for trial in trials:
        if trial.concurrency in seen:
            raise ValueError(f"duplicate concurrency trial: {trial.concurrency}")
        seen.add(trial.concurrency)
        semantic_exact = _digest_match(reference, trial)
        error_free = trial.error is None
        accepted = bool(semantic_exact and error_free)
        if accepted:
            eligible.append(trial)
        rows.append(
            {
                "concurrency": int(trial.concurrency),
                "elapsed_seconds": float(trial.elapsed_seconds),
                "completed_work_units": int(trial.completed_work_units),
                "throughput_work_units_per_second": float(trial.throughput),
                "semantic_exact": semantic_exact,
                "error_free": error_free,
                "accepted": accepted,
                "peak_rss_mib": trial.peak_rss_mib,
                "mean_cpu_percent": trial.mean_cpu_percent,
                "error": trial.error,
            }
        )

    if not eligible:
        return {
            "schema": SCHEMA,
            "calibration_pass": False,
            "selected_concurrency": None,
            "selection_rule": "FASTEST_SEMANTICALLY_EXACT_ERROR_FREE_THEN_LOWEST_CONCURRENCY",
            "cpu_utilization_is_acceptance_gate": False,
            "minimum_speedup_required": False,
            "reference_stream_count": len(reference),
            "trials": sorted(rows, key=lambda r: r["concurrency"]),
            "ready_for_official_training": False,
            "ready_for_tables": False,
        }

    winner = min(eligible, key=lambda t: (-t.throughput, t.concurrency))
    return {
        "schema": SCHEMA,
        "calibration_pass": True,
        "selected_concurrency": int(winner.concurrency),
        "selected_throughput_work_units_per_second": float(winner.throughput),
        "selection_rule": "FASTEST_SEMANTICALLY_EXACT_ERROR_FREE_THEN_LOWEST_CONCURRENCY",
        "cpu_utilization_is_acceptance_gate": False,
        "minimum_speedup_required": False,
        "reference_stream_count": len(reference),
        "trials": sorted(rows, key=lambda r: r["concurrency"]),
        "ready_for_official_training": False,
        "ready_for_tables": False,
    }
