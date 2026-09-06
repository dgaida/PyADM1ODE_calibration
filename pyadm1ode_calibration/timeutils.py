"""Timestamp helpers.

The persisted ``DateTime`` columns are timezone-naive, so every writer has to agree
on one zone or the stored values stop being comparable. That zone is UTC, and
:func:`utc_now` is how a timestamp should enter the database or any record that
ends up there.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """The current UTC time, without a ``tzinfo``.

    Same value as the deprecated ``datetime.utcnow()``, which Python 3.12 warns about.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)
