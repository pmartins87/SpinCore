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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def publish_production_transaction(
    root: Path,
    *,
    identity: ProductionTransactionIdentity,
    stream_checkpoint: Path,
    scheduler_checkpoint: Path,
    algorithm_r_checkpoint: Path,
) -> str:
    """Publish one all-or-nothing durable production checkpoint generation.

    Components are copied into an immutable generation directory and fsynced.
    The generation manifest is written only after all components are durable.
    Finally CURRENT.json is atomically replaced. A crash before CURRENT replace
    leaves the previous generation authoritative; a crash after it sees a fully
    materialized generation whose component hashes are verified on load.
    """
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

            manifest = {
                "schema": SCHEMA,
                "generation_id": generation_id,
                "identity": identity.as_dict(),
                "components": component_meta,
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
