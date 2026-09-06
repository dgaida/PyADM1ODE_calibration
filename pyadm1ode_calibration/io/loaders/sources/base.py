"""
DataSource protocol.

A ``DataSource`` is anything that can deliver time-series measurement data
for a biogas plant: a directory of CSV exports, an OPC UA server, a SQL
historian, an MQTT feed, … Every adapter implements the same minimal
interface so the rest of the pipeline (schema mapping, calibration) does
not care where the data comes from.

The protocol is intentionally narrow:

- ``list_tags()``  — cheap; should not load full data
- ``time_range()`` — best effort; may return ``(None, None)`` if unknown
  without reading
- ``read()``       — the actual data load
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

import pandas as pd

TimeRange = tuple[datetime | None, datetime | None]


@runtime_checkable
class DataSource(Protocol):
    """Uniform interface for time-series measurement sources.

    Attributes:
        name: Unique, human-readable identifier for the source. Used for
            logging and for namespacing tags when several sources are
            combined.
    """

    name: str

    def list_tags(self) -> list[str]:
        """Return all tag names available from this source.

        Should be cheap (header-only or cached). Callers may invoke this
        repeatedly to discover what is available before deciding what to
        read.
        """
        ...

    def time_range(self) -> TimeRange:
        """Return ``(earliest, latest)`` timestamps available from this source.

        Implementations that cannot determine the range without a full
        read should return ``(None, None)``.
        """
        ...

    def read(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        tags: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load data into a DataFrame.

        Args:
            start: Lower time bound, inclusive. ``None`` means "from the
                earliest available record".
            end: Upper time bound, inclusive. ``None`` means "to the
                latest available record".
            tags: Restrict the result to these tag names. ``None`` means
                "all available tags". Unknown tag names should raise
                ``KeyError``.

        Returns:
            DataFrame with:

            - a ``DatetimeIndex`` named ``"timestamp"`` (timezone-aware,
              UTC),
            - one column per tag, with numeric or boolean dtype as
              appropriate for the tag.

            An empty DataFrame (with the index but no rows) is returned
            when the requested window contains no data.
        """
        ...
