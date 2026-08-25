"""Disk-backed HTTP caching.

Two flavours are needed:

* ``fetch_json`` — async, small JSON payloads from the Sleeper API. Some of
  these (live draft picks) want a 5 second TTL, others (the player dump) a day.
* ``fetch_file`` — sync, multi-megabyte CSVs from nflverse. These are fed
  straight to pandas, so we keep them on disk and hand back a path.

Both share one on-disk layout: ``<cache>/<sha1-of-url>.<ext>`` plus a sidecar
``.meta.json`` holding the fetch timestamp. Stale-while-error is deliberate: if
a refresh fails but we hold an expired copy, the expired copy is returned rather
than raising. Losing a live draft board because Sleeper hiccuped is worse than
showing five-second-old picks.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from .config import get_settings

log = logging.getLogger(__name__)

_USER_AGENT = "FantasyPicker/0.1 (+https://github.com/that-coder-guy/fantasypicker)"


class FetchError(RuntimeError):
    """Raised when a resource could not be fetched and no cached copy exists."""


def _paths(url: str, ext: str) -> tuple[Path, Path]:
    settings = get_settings()
    settings.ensure_dirs()
    digest = hashlib.sha1(url.encode()).hexdigest()
    return (
        settings.cache_dir / f"{digest}{ext}",
        settings.cache_dir / f"{digest}.meta.json",
    )


def _fresh(meta_path: Path, ttl: float) -> bool:
    if ttl <= 0 or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (time.time() - float(meta.get("fetched_at", 0))) < ttl


def _stamp(meta_path: Path, url: str) -> None:
    meta_path.write_text(
        json.dumps({"url": url, "fetched_at": time.time()}), encoding="utf-8"
    )


def cache_age(url: str, ext: str = ".json") -> float | None:
    """Seconds since ``url`` was last fetched, or None if never."""
    _, meta_path = _paths(url, ext)
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return time.time() - float(meta.get("fetched_at", 0))


# --------------------------------------------------------------------------- #
# async JSON
# --------------------------------------------------------------------------- #


async def fetch_json(
    url: str,
    ttl: float,
    *,
    client: httpx.AsyncClient | None = None,
    allow_404: bool = False,
) -> Any:
    """GET ``url`` as JSON, honouring a disk cache with ``ttl`` seconds.

    ``allow_404`` returns ``None`` instead of raising for endpoints that
    legitimately 404 (a league with no draft, a week with no matchups).
    """
    data_path, meta_path = _paths(url, ".json")

    if _fresh(meta_path, ttl) and data_path.exists():
        try:
            return json.loads(data_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass  # corrupt cache entry; fall through and refetch

    owned = client is None
    client = client or httpx.AsyncClient(
        timeout=get_settings().http_timeout, headers={"User-Agent": _USER_AGENT}
    )
    try:
        payload = await _get_json_with_retry(client, url, allow_404=allow_404)
    except FetchError:
        if data_path.exists():
            log.warning("using stale cache for %s", url)
            return json.loads(data_path.read_text(encoding="utf-8"))
        raise
    finally:
        if owned:
            await client.aclose()

    if payload is None and allow_404:
        return None

    data_path.write_text(json.dumps(payload), encoding="utf-8")
    _stamp(meta_path, url)
    return payload


async def _get_json_with_retry(
    client: httpx.AsyncClient, url: str, *, allow_404: bool
) -> Any:
    settings = get_settings()
    last: Exception | None = None
    for attempt in range(settings.max_retries):
        try:
            resp = await client.get(url, follow_redirects=True)
            if resp.status_code == 404 and allow_404:
                return None
            if resp.status_code == 429 or resp.status_code >= 500:
                raise httpx.HTTPStatusError(
                    f"{resp.status_code} from {url}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            # Sleeper answers "null" (valid JSON) for some empty resources.
            return resp.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < settings.max_retries - 1:
                await asyncio.sleep(2 ** (attempt + 1))
    raise FetchError(f"could not fetch {url}: {last}") from last


# --------------------------------------------------------------------------- #
# sync files
# --------------------------------------------------------------------------- #


def fetch_file(url: str, ttl: float, *, ext: str = ".csv", required: bool = True) -> Path | None:
    """Download ``url`` to the cache and return the local path.

    Returns ``None`` when the resource is missing and ``required`` is False —
    nflverse has not published every dataset for every season, and an
    in-progress season's files appear mid-week.
    """
    data_path, meta_path = _paths(url, ext)
    if _fresh(meta_path, ttl) and data_path.exists() and data_path.stat().st_size > 0:
        return data_path

    settings = get_settings()
    last: Exception | None = None
    for attempt in range(settings.max_retries):
        try:
            with httpx.Client(
                timeout=settings.http_timeout, headers={"User-Agent": _USER_AGENT}
            ) as client:
                with client.stream("GET", url, follow_redirects=True) as resp:
                    if resp.status_code == 404:
                        if required:
                            raise FetchError(f"404 for {url}")
                        return data_path if data_path.exists() else None
                    resp.raise_for_status()
                    tmp = data_path.with_suffix(data_path.suffix + ".part")
                    with tmp.open("wb") as fh:
                        for chunk in resp.iter_bytes(chunk_size=1 << 20):
                            fh.write(chunk)
                    tmp.replace(data_path)
            _stamp(meta_path, url)
            return data_path
        except FetchError:
            raise
        except httpx.HTTPError as exc:
            last = exc
            if attempt < settings.max_retries - 1:
                time.sleep(2 ** (attempt + 1))

    if data_path.exists():
        log.warning("using stale cache for %s (%s)", url, last)
        return data_path
    if required:
        raise FetchError(f"could not fetch {url}: {last}")
    return None
