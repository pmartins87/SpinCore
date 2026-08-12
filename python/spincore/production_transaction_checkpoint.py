from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping


SCHEMA = "SPINCORE_R8_PRODUCTION_TRANSACTION_V1"
POINTER_SCHEMA = "SPINCORE_R8_PRODUCTION_TRANSACTION_POINTER_V1"
REQUIRED_COMPONENTS = ("stream", "scheduler", "algorithm_r")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as f:
        os.fsync(f.fileno())


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, sort_keys=True, separators=(",", ":"))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
        _fsync_dir(path.parent)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


@dataclass(frozen=True)
class ProductionTransactionIdentity:
    profile_id: str
    domain: str
    algorithm_seed: int
    completed_iteration: int
    roots_per_iteration: int

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id is required")
        if self.domain not in ("TRUE_HEADS_UP", "THREE_HANDED"):
            raise ValueError("unsupported strategy domain")
        if self.algorithm_seed <= 0:
            raise ValueError("algorithm_seed must be positive")
        if self.completed_iteration < 0:
            raise ValueError("completed_iteration must be non-negative")
        if self.roots_per_iteration <= 0:
            raise ValueError("roots_per_iteration must be positive")

    def as_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "domain": self.domain,
            "algorithm_seed": self.algorithm_seed,
            "completed_iteration": self.completed_iteration,
            "roots_per_iteration": self.roots_per_iteration,
        }


@dataclass(frozen=True)
class LoadedProductionTransaction:
    identity: ProductionTransactionIdentity
    generation_id: str
    generation_dir: Path
    component_paths: Mapping[str, Path]
    manifest: dict


def _generation_id(identity: ProductionTransactionIdentity, components: Mapping[str, dict]) -> str:
    canonical = json.dumps(
        {"identity": identity.as_dict(), "components": components},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "spingen-v1-" + hashlib.sha256(canonical).hexdigest()


def _load_torch_mapping(path: Path) -> dict:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - production runtime includes torch
        raise RuntimeError("torch is required to validate production checkpoints") from exc
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a mapping: {path.name}")
    return payload


def _load_algorithm_r_mapping(path: Path) -> dict:
    try:
        raw = path.read_bytes()
        if raw[:1] in (b"{", b"["):
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Algorithm-R JSON checkpoint is not a mapping")
            return payload
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    return _load_torch_mapping(path)


def _semantic_validate_components(
    *,
    identity: ProductionTransactionIdentity,
    stream_path: Path,
    scheduler_path: Path,
    algorithm_r_path: Path,
) -> None:
    """Prove that all checkpoint components describe one exact logical state."""
    stream = _load_torch_mapping(stream_path)
    if stream.get("schema") != "SPINCORE_R7_CHECKPOINT_V2":
        raise ValueError("wrong production stream checkpoint schema")
    if str(stream.get("domain")) != identity.domain:
        raise ValueError("stream checkpoint domain mismatch")
    if int(stream.get("seed", -1)) != identity.algorithm_seed:
        raise ValueError("stream checkpoint algorithm-seed mismatch")
    progress = stream.get("progress") or {}
    if int(progress.get("iteration", -1)) != identity.completed_iteration:
        raise ValueError("stream checkpoint completed-iteration mismatch")

    stream_sha = _sha256_file(stream_path)
    stream_size = int(stream_path.stat().st_size)

    try:
        wrapper = json.loads(scheduler_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("scheduler checkpoint is not canonical JSON") from exc
    if wrapper.get("schema") != "SPINCORE_R8_SCHEDULER_DURABLE_CHECKPOINT_V1":
        raise ValueError("wrong durable scheduler checkpoint schema")
    if wrapper.get("ready_for_tables", False) is not False:
        raise ValueError("scheduler checkpoint cannot authorize table use")
    scheduler = wrapper.get("scheduler") or {}
    if scheduler.get("schema") != "SPINCORE_R8_INDEPENDENT_STREAM_SCHEDULER_V2":
        raise ValueError("wrong production scheduler state schema")
    matches = [
        row for row in list(scheduler.get("streams") or [])
        if str(row.get("profile_id")) == identity.profile_id
        and str(row.get("domain")) == identity.domain
        and int(row.get("algorithm_seed", -1)) == identity.algorithm_seed
    ]
    if len(matches) != 1:
        raise ValueError("scheduler does not contain exactly one matching production stream")
    row = matches[0]
    if row.get("active_lease_id") is not None:
        raise ValueError("scheduler transaction cannot publish an active lease")
    if int(row.get("next_iteration", -1)) != identity.completed_iteration + 1:
        raise ValueError("scheduler completed-iteration mismatch")
    if identity.completed_iteration == 0:
        if row.get("last_checkpoint_sha256") is not None:
            raise ValueError("genesis scheduler unexpectedly names a stream checkpoint")
    else:
        if str(row.get("last_checkpoint_sha256")) != stream_sha:
            raise ValueError("scheduler stream checkpoint SHA does not match transaction stream bytes")
        if int(row.get("last_checkpoint_size_bytes", -1)) != stream_size:
            raise ValueError("scheduler stream checkpoint size does not match transaction stream bytes")

    algorithm_r = _load_algorithm_r_mapping(algorithm_r_path)
    if algorithm_r.get("schema") != "SPINCORE_R8_CENTRAL_ALGORITHM_R_V2":
        raise ValueError("wrong central Algorithm-R checkpoint schema")
    if algorithm_r.get("ready_for_tables", False) is not False:
        raise ValueError("Algorithm-R checkpoint cannot authorize table use")
    if str(algorithm_r.get("profile_id")) != identity.profile_id:
        raise ValueError("Algorithm-R production profile mismatch")
    if str(algorithm_r.get("domain")) != identity.domain:
        raise ValueError("Algorithm-R domain mismatch")
    if int(algorithm_r.get("algorithm_seed", -1)) != identity.algorithm_seed:
        raise ValueError("Algorithm-R algorithm-seed mismatch")
    if int(algorithm_r.get("roots_per_iteration", -1)) != identity.roots_per_iteration:
        raise ValueError("Algorithm-R roots-per-iteration mismatch")
    expected_root = identity.completed_iteration * identity.roots_per_iteration
    if int(algorithm_r.get("next_global_root", -1)) != expected_root:
        raise ValueError("Algorithm-R global-root/iteration mismatch")
    if int(algorithm_r.get("committed_roots", -1)) != expected_root:
        raise ValueError("Algorithm-R committed-root count mismatch")
    if list(algorithm_r.get("pending") or []):
        raise ValueError("Algorithm-R transaction cannot publish pending root gaps")


def publish_production_transaction(
    root: Path,
    *,
    identity: ProductionTransactionIdentity,
    stream_checkpoint: Path,
    scheduler_checkpoint: Path,
    algorithm_r_checkpoint: Path,
) -> str:
    """Publish one semantically consistent, all-or-nothing durable generation."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    generations = root / "generations"
    generations.mkdir(parents=True, exist_ok=True)

    inputs = {
        "stream": Path(stream_checkpoint),
        "scheduler": Path(scheduler_checkpoint),
        "algorithm_r": Path(algorithm_r_checkpoint),
    }
    for name, src in inputs.items():
        if not src.is_file():
            raise FileNotFoundError(f"missing {name} checkpoint: {src}")
        if src.stat().st_size <= 0:
            raise ValueError(f"empty {name} checkpoint")

    _semantic_validate_components(
        identity=identity,
        stream_path=inputs["stream"],
        scheduler_path=inputs["scheduler"],
        algorithm_r_path=inputs["algorithm_r"],
    )

    component_meta = {
        name: {
            "filename": name + src.suffix if src.suffix else name + ".bin",
            "sha256": _sha256_file(src),
            "size_bytes": int(src.stat().st_size),
        }
        for name, src in inputs.items()
    }
    generation_id = _generation_id(identity, component_meta)
    final_dir = generations / generation_id

    if final_dir.exists():
        loaded = _load_generation(final_dir)
        if loaded.identity != identity:
            raise RuntimeError("existing generation identity mismatch")
    else:
        stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=generations))
        try:
            for name, src in inputs.items():
                dst = stage / component_meta[name]["filename"]
                shutil.copyfile(src, dst)
                _fsync_file(dst)
                if _sha256_file(dst) != component_meta[name]["sha256"]:
                    raise RuntimeError(f"copied {name} checkpoint hash mismatch")

            _semantic_validate_components(
                identity=identity,
                stream_path=stage / component_meta["stream"]["filename"],
                scheduler_path=stage / component_meta["scheduler"]["filename"],
                algorithm_r_path=stage / component_meta["algorithm_r"]["filename"],
            )
            manifest = {
                "schema": SCHEMA,
                "generation_id": generation_id,
                "identity": identity.as_dict(),
                "components": component_meta,
                "semantic_consistency_validated": True,
                "ready_for_tables": False,
            }
            _atomic_json(stage / "manifest.json", manifest)
            _fsync_dir(stage)
            os.replace(stage, final_dir)
            _fsync_dir(generations)
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)

    pointer = {
        "schema": POINTER_SCHEMA,
        "generation_id": generation_id,
        "manifest_sha256": _sha256_file(final_dir / "manifest.json"),
        "ready_for_tables": False,
    }
    _atomic_json(root / "CURRENT.json", pointer)
    return generation_id


def _identity_from_dict(row: dict) -> ProductionTransactionIdentity:
    return ProductionTransactionIdentity(
        profile_id=str(row["profile_id"]),
        domain=str(row["domain"]),
        algorithm_seed=int(row["algorithm_seed"]),
        completed_iteration=int(row["completed_iteration"]),
        roots_per_iteration=int(row["roots_per_iteration"]),
    )


def _load_generation(generation_dir: Path) -> LoadedProductionTransaction:
    manifest_path = generation_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("wrong production transaction schema")
    if manifest.get("ready_for_tables", False) is not False:
        raise ValueError("R8 checkpoint cannot authorize table use")
    if manifest.get("semantic_consistency_validated") is not True:
        raise ValueError("production transaction lacks semantic-consistency certification")
    if manifest.get("generation_id") != generation_dir.name:
        raise ValueError("generation directory/manifest mismatch")

    identity = _identity_from_dict(manifest["identity"])
    components = manifest.get("components") or {}
    if tuple(sorted(components)) != tuple(sorted(REQUIRED_COMPONENTS)):
        raise ValueError("production transaction component set mismatch")

    paths: dict[str, Path] = {}
    for name in REQUIRED_COMPONENTS:
        meta = components[name]
        path = generation_dir / str(meta["filename"])
        if not path.is_file():
            raise FileNotFoundError(f"missing transaction component {name}")
        if int(path.stat().st_size) != int(meta["size_bytes"]):
            raise ValueError(f"transaction component size mismatch: {name}")
        if _sha256_file(path) != str(meta["sha256"]):
            raise ValueError(f"transaction component hash mismatch: {name}")
        paths[name] = path

    expected_id = _generation_id(identity, components)
    if expected_id != generation_dir.name:
        raise ValueError("production generation identity hash mismatch")
    _semantic_validate_components(
        identity=identity,
        stream_path=paths["stream"],
        scheduler_path=paths["scheduler"],
        algorithm_r_path=paths["algorithm_r"],
    )
    return LoadedProductionTransaction(identity, generation_dir.name, generation_dir, paths, manifest)


def load_current_production_transaction(root: Path) -> LoadedProductionTransaction:
    root = Path(root)
    pointer_path = root / "CURRENT.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("schema") != POINTER_SCHEMA:
        raise ValueError("wrong production transaction pointer schema")
    if pointer.get("ready_for_tables", False) is not False:
        raise ValueError("R8 transaction pointer cannot authorize table use")
    generation_id = str(pointer["generation_id"])
    generation_dir = root / "generations" / generation_id
    manifest_path = generation_dir / "manifest.json"
    if _sha256_file(manifest_path) != str(pointer["manifest_sha256"]):
        raise ValueError("CURRENT manifest hash mismatch")
    return _load_generation(generation_dir)
