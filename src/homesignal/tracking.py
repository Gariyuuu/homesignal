"""Minimal experiment tracking: one JSON object per run appended to ``models/experiments.jsonl``.

Chosen over mlflow to keep the project dependency-light; every run records config, git commit,
features, per-fold metrics and timing so results are auditable.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from homesignal.config import ROOT

LOG_PATH = ROOT / "models" / "experiments.jsonl"


def git_commit() -> str | None:
    """Return the current git commit hash, if available."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def log_run(record: dict[str, Any], path: Path = LOG_PATH) -> None:
    """Append a run record with a timestamp and commit hash."""
    record = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        **record,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def read_runs(path: Path = LOG_PATH) -> list[dict[str, Any]]:
    """Read all run records."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
