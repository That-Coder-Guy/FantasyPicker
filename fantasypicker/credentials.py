"""Storage for the cookies a private ESPN league needs.

``espn_s2`` and ``SWID`` are session credentials for the user's own ESPN
account. Anyone holding them can act as that user on ESPN, so they get handled
differently from everything else the app remembers:

* they live in their own file, not in ``state.json``, so that file stays
  harmless to copy, paste into an issue, or sync;
* the file is created ``0600`` and the directory ``0700``, so other accounts on
  a shared machine cannot read them;
* they are never logged, never included in any API response, and never sent
  anywhere but ``espn.com``. :func:`redact` exists for the times something does
  need to be displayed.

They are stored rather than asked for each time because ESPN's cookies last for
weeks and re-copying them out of developer tools mid-draft is not something
anyone should have to do.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from .config import get_settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EspnCredentials:
    espn_s2: str
    swid: str

    @property
    def valid(self) -> bool:
        return bool(self.espn_s2.strip() and self.swid.strip())

    def normalised(self) -> "EspnCredentials":
        """Trim whitespace and put SWID back in the braces ESPN expects."""
        swid = self.swid.strip()
        if swid and not swid.startswith("{"):
            swid = "{" + swid.strip("{}") + "}"
        return EspnCredentials(espn_s2=self.espn_s2.strip(), swid=swid)


def redact(value: str | None) -> str:
    """A recognisable stub, for confirming *which* credential is stored."""
    if not value:
        return "(none)"
    text = str(value).strip()
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}…{text[-4:]} ({len(text)} chars)"


def _path() -> Path:
    return get_settings().home / "credentials.json"


def _read_all() -> dict:
    path = _path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("could not read stored credentials (%s); ignoring them", exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def load_credentials(league_id: str) -> EspnCredentials | None:
    """The stored cookies for one league, if any."""
    row = _read_all().get(str(league_id))
    if not isinstance(row, dict):
        return None
    creds = EspnCredentials(
        espn_s2=str(row.get("espn_s2") or ""), swid=str(row.get("swid") or "")
    ).normalised()
    return creds if creds.valid else None


def save_credentials(league_id: str, credentials: EspnCredentials) -> None:
    """Store cookies for one league, readable only by this user."""
    credentials = credentials.normalised()
    if not credentials.valid:
        return
    settings = get_settings()
    settings.ensure_dirs()
    try:
        os.chmod(settings.home, stat.S_IRWXU)  # 0700
    except OSError:  # pragma: no cover - Windows and odd filesystems
        pass

    payload = _read_all()
    payload[str(league_id)] = {
        "espn_s2": credentials.espn_s2,
        "swid": credentials.swid,
    }
    path = _path()
    tmp = path.with_suffix(".json.tmp")
    try:
        # Create with restrictive permissions *before* writing, so the secret is
        # never briefly world-readable.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        tmp.replace(path)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    except OSError as exc:
        log.warning("could not save ESPN credentials: %s", exc)


def forget_credentials(league_id: str) -> bool:
    payload = _read_all()
    if str(league_id) not in payload:
        return False
    payload.pop(str(league_id), None)
    path = _path()
    try:
        if payload:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        else:
            path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("could not remove ESPN credentials: %s", exc)
        return False
    return True


def describe(league_id: str) -> dict[str, str | bool]:
    """What is stored, safe to show in a UI or a diagnostic."""
    creds = load_credentials(league_id)
    if creds is None:
        return {"stored": False, "espn_s2": "(none)", "swid": "(none)"}
    return {
        "stored": True,
        "espn_s2": redact(creds.espn_s2),
        "swid": redact(creds.swid),
    }
