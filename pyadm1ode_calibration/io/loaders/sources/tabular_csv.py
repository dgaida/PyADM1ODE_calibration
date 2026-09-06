"""
Tabular CSV DataSource.

Reads tabular CSV exports where each file contains a fixed set of tags
(one column per tag) and rows are timestamped readings. The exact file
layout — encoding, separator, decimal mark, header row count, date
format, tag-name extraction — is fully configurable per file via
:class:`FileSpec`. Several files can be combined into one source; tags
are namespaced by an optional prefix (typically the file stem) so that
tags with the same name in different files do not collide.

The same code therefore handles, among others:

- ABB Aspect / 800xA TRD CSV exports (UTF-16, decimal comma, "Spur N: …"
  headers),
- plain ISO-format CSVs from custom exporters,
- sparse laboratory CSVs (one row per sample, irregular timestamps).

Add a new vendor format by adding a factory to :data:`PRESETS` rather
than writing a new class.

Example:
    >>> from pathlib import Path
    >>> source = TabularCSVSource.from_directory(
    ...     name="plant",
    ...     directory=Path("archive_export"),
    ...     preset="abb",
    ... )
    >>> "BHKW_TR::Wirkleistung" in source.list_tags()
    True
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .base import TimeRange

# ----------------------------------------------------------------------------
# Sentinels
# ----------------------------------------------------------------------------

# Float32 max (~3.4028235e38) is emitted by some historians (notably ABB) for
# tags that have no valid reading. Values within ``DEFAULT_SENTINEL_RTOL``
# relative tolerance of any entry here are mapped to NaN.
DEFAULT_SENTINELS: tuple[float, ...] = (3.4028235e38,)
DEFAULT_SENTINEL_RTOL: float = 1e-3


# ----------------------------------------------------------------------------
# Per-file configuration
# ----------------------------------------------------------------------------


@dataclass
class FileSpec:
    """Parsing configuration for a single CSV file.

    Defaults are tuned for the ABB TRD export format observed in this
    project (UTF-16, semicolon-separated, decimal comma, three metadata
    rows above the column header, German two-digit year). Override
    individual fields for other vendors via either constructor arguments
    or one of the :data:`PRESETS` factories.

    Attributes:
        path: File path; either absolute or relative to the source's
            ``base_dir``.
        encoding: File encoding (e.g. ``"utf-16"``, ``"utf-8"``,
            ``"cp1252"``).
        separator: Field separator.
        decimal: Decimal mark.
        skip_rows: Number of metadata rows to skip before the column
            header. Set to ``0`` if the file starts directly with the
            header.
        date_column: Name of the column carrying the date part of the
            timestamp.
        time_column: Name of the column carrying the time part. Leave
            empty if the date column already contains a full timestamp.
        date_format: ``strptime`` format string for the combined
            ``"date time"`` string.
        tag_pattern: Regular expression with one capture group that
            extracts the tag name from each column header. Columns whose
            header does not match this pattern are dropped (except for
            the timestamp and ``drop_columns`` ones, which are handled
            separately). The default pattern captures the ``Name:`` part
            of ABB ``"Spur N: Typ: T  Name: X"`` headers.
        tag_prefix: Prefix prepended to every tag read from this file,
            so that tags from different files do not collide. Typically
            the file stem.
        prefix_separator: String inserted between ``tag_prefix`` and the
            raw tag name.
        drop_columns: Headers to drop unconditionally (e.g. row IDs,
            quality flags). The values are matched against the *raw*
            column headers, before tag-pattern extraction.
        sentinel_values: Numeric values to replace with NaN. See
            :data:`DEFAULT_SENTINELS`.
        sentinel_rtol: Relative tolerance used when matching values
            against ``sentinel_values``.
        bool_true_values, bool_false_values: Strings that are treated as
            booleans during type coercion. A column that contains *any*
            of these (and no other numeric values) becomes a boolean
            column.
    """

    path: Path
    encoding: str = "utf-16"
    separator: str = ";"
    decimal: str = ","
    skip_rows: int = 3
    date_column: str = "Datum"
    time_column: str = "Zeit"
    date_format: str = "%d.%m.%y %H:%M:%S.%f"
    tag_pattern: str = r"Spur\s+\d+:\s*Typ:\s*\w+\s+Name:\s*(.+?)\s*$"
    tag_prefix: str | None = None
    prefix_separator: str = "::"
    drop_columns: tuple[str, ...] = ("EintragID", "Status")
    sentinel_values: tuple[float, ...] = DEFAULT_SENTINELS
    sentinel_rtol: float = DEFAULT_SENTINEL_RTOL
    bool_true_values: tuple[str, ...] = ("TRUE", "True", "true", "1")
    bool_false_values: tuple[str, ...] = ("FALSE", "False", "false", "0")

    def qualified_tag(self, raw_tag: str) -> str:
        """Return the tag name with the file-specific prefix applied."""
        if self.tag_prefix is None:
            return raw_tag
        return f"{self.tag_prefix}{self.prefix_separator}{raw_tag}"


# ----------------------------------------------------------------------------
# Source
# ----------------------------------------------------------------------------


class TabularCSVSource:
    """A :class:`DataSource` backed by one or more tabular CSV files.

    Each file is described by its own :class:`FileSpec`. The source
    discovers available tags at construction time by parsing only the
    header row of each file, so :meth:`list_tags` is cheap.

    Tags from different files are kept on independent time axes and
    outer-joined on the index when :meth:`read` is called.

    Args:
        name: Unique identifier for this source.
        files: Per-file parsing configurations.
        base_dir: Optional directory used to resolve relative
            :attr:`FileSpec.path` entries.
        timezone: IANA timezone name (e.g. ``"Europe/Berlin"``) used to
            interpret the naive timestamps stored in the files. The
            output index is always converted to UTC. Pass ``None`` to
            keep timestamps naive (interpreted as already UTC).
    """

    def __init__(
        self,
        name: str,
        files: list[FileSpec],
        base_dir: Path | None = None,
        timezone: str | None = "Europe/Berlin",
    ):
        self.name = name
        self.files = list(files)
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self.timezone = timezone

        # Discover tag layout up-front; surfaces config errors early.
        self._tag_index: dict[str, FileSpec] = {}
        self._raw_tags_by_file: dict[int, list[str]] = {}
        self._build_tag_index()

    # ----- DataSource protocol ----------------------------------------------

    def list_tags(self) -> list[str]:
        return sorted(self._tag_index.keys())

    def time_range(self) -> TimeRange:
        # We do not parse metadata headers here because the format is
        # vendor-specific. Callers that need the range without reading
        # all data can override this in a subclass.
        return (None, None)

    def read(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        tags: list[str] | None = None,
    ) -> pd.DataFrame:
        files_to_read = self._files_for_tags(tags)
        if not files_to_read:
            return self._empty_frame()

        frames: list[pd.DataFrame] = []
        for spec, raw_tags in files_to_read:
            df = self._read_one(spec, start, end, raw_tags)
            if not df.empty:
                frames.append(df)

        if not frames:
            return self._empty_frame()

        merged = pd.concat(frames, axis=1, join="outer").sort_index()

        # Preserve the caller-requested ordering when given.
        if tags is not None:
            merged = merged.reindex(columns=tags)

        merged.index.name = "timestamp"
        return merged

    # ----- Construction helpers ---------------------------------------------

    @classmethod
    def from_directory(
        cls,
        name: str,
        directory: str | Path,
        preset: str = "abb",
        pattern: str = "*.csv",
        prefix_from_stem: bool = True,
        timezone: str | None = "Europe/Berlin",
        **spec_overrides,
    ) -> TabularCSVSource:
        """Build a source by scanning a directory.

        Every file matching ``pattern`` becomes a :class:`FileSpec`
        configured with the named ``preset``. The file stem is used as
        the tag prefix unless ``prefix_from_stem`` is False.

        Additional keyword arguments are forwarded to the preset
        factory, which lets callers override individual fields (for
        example ``encoding="utf-8"``) for all files in the directory at
        once.
        """
        directory = Path(directory)
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: '{preset}'. Available: {sorted(PRESETS)}")
        factory = PRESETS[preset]

        files: list[FileSpec] = []
        for path in sorted(directory.glob(pattern)):
            # Store the basename and let base_dir handle resolution, so
            # the source remains relocatable.
            spec = factory(path=Path(path.name), **spec_overrides)
            if prefix_from_stem:
                spec.tag_prefix = path.stem
            files.append(spec)

        if not files:
            raise FileNotFoundError(f"No files matching '{pattern}' found in {directory!s}")

        return cls(name=name, files=files, base_dir=directory, timezone=timezone)

    # ----- Internals --------------------------------------------------------

    def _resolve_path(self, p: Path) -> Path:
        if p.is_absolute() or self.base_dir is None:
            return p
        return self.base_dir / p

    def _build_tag_index(self) -> None:
        for idx, spec in enumerate(self.files):
            raw_tags = self._read_header_tags(spec)
            self._raw_tags_by_file[idx] = raw_tags
            for raw_tag in raw_tags:
                qualified = spec.qualified_tag(raw_tag)
                if qualified in self._tag_index:
                    other = self._tag_index[qualified].path
                    raise ValueError(
                        f"Tag collision: '{qualified}' appears in both "
                        f"{other} and {spec.path}. Use distinct tag_prefix "
                        f"values (or rename one of the columns) to avoid this."
                    )
                self._tag_index[qualified] = spec

    def _read_header_tags(self, spec: FileSpec) -> list[str]:
        """Parse only the header row of ``spec.path`` and extract tag names."""
        path = self._resolve_path(spec.path)
        with open(path, encoding=spec.encoding) as f:
            for _ in range(spec.skip_rows):
                f.readline()
            header_line = f.readline().rstrip("\r\n")

        columns = header_line.split(spec.separator)
        tag_re = re.compile(spec.tag_pattern)
        tags: list[str] = []
        for col in columns:
            m = tag_re.match(col.strip())
            if m:
                tags.append(m.group(1).strip())
        return tags

    def _files_for_tags(self, tags: list[str] | None) -> list[tuple[FileSpec, list[str] | None]]:
        """Map requested tags to the files they live in.

        Returns a list of ``(spec, raw_tag_names_or_None)`` pairs in the
        same order as :attr:`files`. ``None`` for the second element
        means "load every tag in this file".
        """
        if tags is None:
            return [(spec, None) for spec in self.files]

        # Collect requested raw tags per file (by object identity).
        by_id: dict[int, list[str]] = {}
        for tag in tags:
            spec = self._tag_index.get(tag)
            if spec is None:
                raise KeyError(f"Unknown tag '{tag}'. Available tags via list_tags().")
            raw = self._strip_prefix(tag, spec)
            by_id.setdefault(id(spec), []).append(raw)

        result: list[tuple[FileSpec, list[str] | None]] = []
        for spec in self.files:
            if id(spec) in by_id:
                result.append((spec, by_id[id(spec)]))
        return result

    @staticmethod
    def _strip_prefix(tag: str, spec: FileSpec) -> str:
        if spec.tag_prefix is None:
            return tag
        head = f"{spec.tag_prefix}{spec.prefix_separator}"
        return tag.removeprefix(head)

    def _read_one(
        self,
        spec: FileSpec,
        start: datetime | None,
        end: datetime | None,
        raw_tags: list[str] | None,
    ) -> pd.DataFrame:
        path = self._resolve_path(spec.path)

        # Read everything as string first; we coerce to numeric / bool
        # ourselves so we can apply the file-specific decimal mark,
        # sentinel handling, and BOOL parsing uniformly. The Python
        # engine is used because the C engine is brittle with UTF-16.
        df = pd.read_csv(
            path,
            sep=spec.separator,
            encoding=spec.encoding,
            skiprows=spec.skip_rows,
            header=0,
            dtype=str,
            engine="python",
            on_bad_lines="skip",
        )

        # Rename tag columns from their raw header to the extracted name.
        tag_re = re.compile(spec.tag_pattern)
        rename_map: dict[str, str] = {}
        for col in df.columns:
            m = tag_re.match(col.strip())
            if m:
                rename_map[col] = m.group(1).strip()
        df = df.rename(columns=rename_map)

        # Drop ignored columns.
        df = df.drop(
            columns=[c for c in spec.drop_columns if c in df.columns],
            errors="ignore",
        )

        # Build the timestamp index.
        df = self._attach_timestamp(df, spec)

        # Subset to the requested tags (in the file's own tag order).
        all_raw = self._raw_tags_by_file_for(spec)
        cols = raw_tags if raw_tags is not None else all_raw
        df = df[[c for c in cols if c in df.columns]]

        # Coerce types (numeric, bool, sentinel masking).
        df = self._coerce_types(df, spec)

        # Apply optional time filter (loc supports both ends being None).
        if start is not None or end is not None:
            df = df.loc[self._to_index_ts(start) : self._to_index_ts(end)]

        # Apply the namespacing prefix on output.
        if spec.tag_prefix is not None:
            df = df.rename(columns={c: spec.qualified_tag(c) for c in df.columns})

        return df

    def _raw_tags_by_file_for(self, spec: FileSpec) -> list[str]:
        for idx, s in enumerate(self.files):
            if s is spec:
                return self._raw_tags_by_file[idx]
        # Fallback for FileSpec instances created outside __init__.
        return self._read_header_tags(spec)

    def _attach_timestamp(self, df: pd.DataFrame, spec: FileSpec) -> pd.DataFrame:
        if spec.date_column not in df.columns:
            raise ValueError(
                f"Date column '{spec.date_column}' not found in " f"{spec.path}. Available columns: {list(df.columns)}"
            )

        if spec.time_column and spec.time_column in df.columns:
            ts_str = df[spec.date_column].astype(str) + " " + df[spec.time_column].astype(str)
        else:
            ts_str = df[spec.date_column].astype(str)

        timestamp = pd.to_datetime(ts_str, format=spec.date_format, errors="coerce")

        df = df.drop(columns=[c for c in (spec.date_column, spec.time_column) if c in df.columns and c])
        df.index = pd.DatetimeIndex(timestamp, name="timestamp")
        df = df[~df.index.isna()]

        if self.timezone is not None:
            # Tolerate DST oddities by marking ambiguous / non-existent
            # stamps as NaT and dropping them, rather than failing.
            df.index = df.index.tz_localize(self.timezone, ambiguous="NaT", nonexistent="NaT")
            df = df[~df.index.isna()]
            df.index = df.index.tz_convert("UTC")
        else:
            df.index = df.index.tz_localize("UTC")

        return df

    def _coerce_types(self, df: pd.DataFrame, spec: FileSpec) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in df.columns:
            raw = df[col].astype(str).str.strip()

            # Try numeric: convert the decimal mark, then to_numeric.
            if spec.decimal != ".":
                numeric_input = raw.str.replace(spec.decimal, ".", regex=False)
            else:
                numeric_input = raw
            numeric = pd.to_numeric(numeric_input, errors="coerce")

            if numeric.notna().any():
                # Apply sentinel masking on the numeric values.
                for sentinel in spec.sentinel_values:
                    is_sentinel = np.isclose(
                        numeric.to_numpy(dtype=float),
                        sentinel,
                        rtol=spec.sentinel_rtol,
                        equal_nan=False,
                    )
                    if is_sentinel.any():
                        numeric = numeric.mask(pd.Series(is_sentinel, index=numeric.index), np.nan)
                out[col] = numeric
            else:
                # Treat as boolean. Cells matching neither true nor
                # false strings become NaN, so dtype stays object.
                mask_true = raw.isin(spec.bool_true_values)
                mask_false = raw.isin(spec.bool_false_values)
                bool_col = pd.Series(np.nan, index=raw.index, dtype="object")
                bool_col[mask_true] = True
                bool_col[mask_false] = False
                out[col] = bool_col
        return out

    @staticmethod
    def _to_index_ts(ts: datetime | None) -> pd.Timestamp | None:
        """Convert a user-supplied datetime to a UTC pandas Timestamp.

        Naive datetimes are assumed to already be in UTC. A ``None``
        bound is preserved.
        """
        if ts is None:
            return None
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        return t

    def _empty_frame(self) -> pd.DataFrame:
        idx = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        return pd.DataFrame(index=idx)


# ----------------------------------------------------------------------------
# Presets
# ----------------------------------------------------------------------------
#
# A preset is just a factory ``(path, **overrides) -> FileSpec``. To add a
# new vendor format, copy one of the existing presets, adjust the defaults,
# and register it in the PRESETS dict below.


PresetFactory = Callable[..., FileSpec]


def _abb_preset(path: Path, **overrides) -> FileSpec:
    """ABB Aspect / 800xA TRD CSV export.

    UTF-16 LE with BOM, three metadata rows above the column header,
    semicolon separator, decimal comma, ``"Spur N: Typ: T  Name: X"``
    headers, German two-digit-year timestamp split across two columns.
    """
    defaults = {
        "encoding": "utf-16",
        "separator": ";",
        "decimal": ",",
        "skip_rows": 3,
        "date_column": "Datum",
        "time_column": "Zeit",
        "date_format": "%d.%m.%y %H:%M:%S.%f",
        "tag_pattern": r"Spur\s+\d+:\s*Typ:\s*\w+\s+Name:\s*(.+?)\s*$",
        "drop_columns": ("EintragID", "Status"),
    }
    defaults.update(overrides)
    return FileSpec(path=Path(path), **defaults)


def _plain_csv_preset(path: Path, **overrides) -> FileSpec:
    """Plain UTF-8 CSV with a single ISO-format ``timestamp`` column."""
    defaults = {
        "encoding": "utf-8",
        "separator": ",",
        "decimal": ".",
        "skip_rows": 0,
        "date_column": "timestamp",
        "time_column": "",
        "date_format": "%Y-%m-%dT%H:%M:%S",
        "tag_pattern": r"(.+)",
        "drop_columns": (),
    }
    defaults.update(overrides)
    return FileSpec(path=Path(path), **defaults)


def _lab_csv_preset(path: Path, **overrides) -> FileSpec:
    """Sparse laboratory CSV: one row per sample, date-only timestamps.

    Designed for irregular lab analyses (pH, VFA, TAC, NH4-N, …) that
    arrive on lab-report dates. Free-form column names are kept as-is;
    add ``drop_columns`` to suppress administrative fields such as
    sample IDs.
    """
    defaults = {
        "encoding": "utf-8",
        "separator": ",",
        "decimal": ".",
        "skip_rows": 0,
        "date_column": "date",
        "time_column": "",
        "date_format": "%Y-%m-%d",
        "tag_pattern": r"(.+)",
        "drop_columns": (),
    }
    defaults.update(overrides)
    return FileSpec(path=Path(path), **defaults)


PRESETS: dict[str, PresetFactory] = {
    "abb": _abb_preset,
    "plain": _plain_csv_preset,
    "lab": _lab_csv_preset,
}
