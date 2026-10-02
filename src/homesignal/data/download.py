"""Cached, idempotent HTTP downloads with retries and a JSON sidecar recording provenance."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)


def _sidecar(dest: Path) -> Path:
    return dest.with_name(dest.name + ".meta.json")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(
    url: str,
    dest: Path,
    *,
    user_agent: str,
    retries: int = 5,
    timeout: int = 300,
    force: bool = False,
    method: str = "GET",
    json_body: dict[str, Any] | None = None,
) -> Path:
    """Download ``url`` to ``dest`` unless a cached copy already exists.

    A sidecar ``<name>.meta.json`` records the URL, download timestamp, size and SHA-256 so
    the exact raw inputs of a run can be audited. Retries use exponential backoff.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        log.debug("cache hit %s", dest.name)
        return dest
    headers = {"User-Agent": user_agent}
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.request(
                method,
                url,
                headers=headers,
                timeout=timeout,
                stream=True,
                json=json_body,
            ) as resp:
                resp.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                tmp.replace(dest)
            size = dest.stat().st_size
            meta = {
                "url": url,
                "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "size_bytes": size,
                "sha256": _sha256(dest),
            }
            _sidecar(dest).write_text(json.dumps(meta, indent=2))
            log.info("downloaded %s (%.1f MB)", dest.name, size / 1e6)
            return dest
        except (requests.RequestException, OSError) as err:  # pragma: no cover - network
            last_err = err
            wait = 2**attempt
            log.warning(
                "attempt %d/%d failed for %s: %s (retry in %ss)",
                attempt + 1,
                retries,
                url,
                err,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url}") from last_err


def read_meta(dest: Path) -> dict[str, Any] | None:
    """Return the provenance sidecar for a cached file, if present."""
    p = _sidecar(dest)
    if not p.exists():
        return None
    data: dict[str, Any] = json.loads(p.read_text())
    return data
