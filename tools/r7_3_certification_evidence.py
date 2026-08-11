from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Callable


Validator = Callable[[dict], None]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_and_validate(data: bytes, validator: Validator) -> dict:
    obj = json.loads(data.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("certification evidence must be a JSON object")
    validator(obj)
    return obj


def _git(*args: str, cwd: Path) -> bytes:
    return subprocess.check_output(["git", *args], cwd=cwd)


def resolve_valid_json(
    path: str | Path,
    *,
    validator: Validator,
    repo_root: str | Path = ".",
    history_limit: int = 200,
) -> tuple[dict, dict]:
    """Resolve the newest provenance-valid version of a certification JSON file.

    Concurrent/legacy GitHub Actions may write the same canonical evidence path
    after a corrected certification run.  Certification must not silently trust
    whichever bytes happen to be at HEAD.  This helper first validates the
    current working-tree bytes, then walks the file's Git history newest-first
    and returns the first version that satisfies the caller's provenance
    validator.  Validators are expected to bind the file to the immutable
    semantic freeze (source head, evidence commit/hash, thread contract, etc.).

    The returned metadata records exactly which bytes and commit were consumed.
    Failure is fail-closed if no valid version exists.
    """
    root = Path(repo_root).resolve()
    p = Path(path)
    absolute = p if p.is_absolute() else root / p
    rel = absolute.relative_to(root).as_posix()
    failures: list[dict[str, str]] = []

    if absolute.is_file():
        data = absolute.read_bytes()
        try:
            obj = _parse_and_validate(data, validator)
            return obj, {
                "path": rel,
                "origin": "WORKING_TREE_HEAD",
                "commit_sha": None,
                "sha256": _sha256_bytes(data),
                "history_fallback_used": False,
            }
        except Exception as exc:
            failures.append({"origin": "WORKING_TREE_HEAD", "error": str(exc)})

    try:
        raw_log = _git("log", f"-{int(history_limit)}", "--format=%H", "--", rel, cwd=root)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"cannot inspect Git history for {rel}") from exc

    for commit in raw_log.decode("utf-8").splitlines():
        commit = commit.strip()
        if not commit:
            continue
        try:
            data = _git("show", f"{commit}:{rel}", cwd=root)
        except subprocess.CalledProcessError as exc:
            failures.append({"origin": commit, "error": f"git show failed: {exc}"})
            continue
        try:
            obj = _parse_and_validate(data, validator)
            return obj, {
                "path": rel,
                "origin": "GIT_HISTORY",
                "commit_sha": commit,
                "sha256": _sha256_bytes(data),
                "history_fallback_used": True,
                "rejected_newer_versions": failures,
            }
        except Exception as exc:
            failures.append({"origin": commit, "error": str(exc)})

    raise ValueError(
        f"no provenance-valid certification evidence found for {rel}; "
        f"checked current bytes plus {len(failures) - (1 if absolute.is_file() else 0)} historical versions"
    )
