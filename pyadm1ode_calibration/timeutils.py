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


def normalize_freq(freq: str) -> str:
    """Translate the deprecated lowercase day alias in a pandas frequency string.

    ``"1d"`` becomes ``"1D"``. Pandas warns about the lowercase form today and drops it
    in version 4, so schema files and call sites that use it keep working.

    Only the day alias is translated. Pandas renamed other aliases in both directions,
    ``H`` to ``h`` and ``M`` to ``ME`` among them, so a blanket case change would break
    more than it fixes.

    Args:
        freq: A pandas frequency string, for example ``"1d"``, ``"15min"`` or ``"1h"``.

    Returns:
        str: The same string with a trailing lowercase ``d`` upper-cased.
    """
    return freq[:-1] + "D" if freq.endswith("d") and not freq.endswith(("ed", "id")) else freq
