from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .production_stream_scheduler import IndependentStreamScheduler


SCHEMA = "SPINCORE_R8_SCHEDULER_DURABLE_CHECKPOINT_V1"


@dataclass(frozen=True)
class SchedulerCheckpointReceipt:
    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("scheduler checkpoint path is required")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("scheduler checkpoint SHA256 must be lowercase hex")
        if self.size_bytes <= 0:
            raise ValueError("scheduler checkpoint must contain bytes")


def _serialized_payload(scheduler: IndependentStreamScheduler) -> bytes:
    wrapper = {
        "schema": SCHEMA,
        "scheduler": scheduler.state_dict(),
        "ready_for_tables": False,
    }
    return (json.dumps(wrapper, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def save_scheduler_checkpoint_atomic(
    path: str | Path,
    scheduler: IndependentStreamScheduler,
) -> SchedulerCheckpointReceipt:
    """Persist scheduler state by fsync + same-directory atomic replacement."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialized_payload(scheduler)
    digest = hashlib.sha256(payload).hexdigest()

    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=target.name + ".tmp-",
            dir=target.parent,
            delete=False,
        ) as fh:
            tmp_name = fh.name
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
        tmp_name = None
        # Persist the directory entry where supported by the host filesystem.
        try:
            dir_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass

    return SchedulerCheckpointReceipt(str(target), digest, len(payload))


def load_scheduler_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    clear_active_leases: bool = True,
) -> IndependentStreamScheduler:
    target = Path(path)
    raw = target.read_bytes()
    if not raw:
        raise ValueError("scheduler checkpoint is empty")
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("scheduler checkpoint SHA256 mismatch")
    try:
        wrapper = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("scheduler checkpoint is not valid canonical JSON") from exc
    if wrapper.get("schema") != SCHEMA:
        raise ValueError("wrong durable scheduler checkpoint schema")
    if wrapper.get("ready_for_tables", False) is not False:
        raise ValueError("R8 scheduler checkpoint cannot authorize table use")
    return IndependentStreamScheduler.from_state_dict(
        wrapper["scheduler"],
        clear_active_leases=clear_active_leases,
    )
