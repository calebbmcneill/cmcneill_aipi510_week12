"""Run provenance: tie every model to the code, data and config that produced it."""
from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def environment() -> dict:
    libs = {}
    for name in ("numpy", "pandas", "scikit-learn", "joblib"):
        try:
            libs[name] = version(name)
        except PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "libraries": libs}
