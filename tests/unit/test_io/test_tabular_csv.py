"""Tests for :mod:`pyadm1ode_calibration.io.loaders.sources.tabular_csv`."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyadm1ode_calibration.io.loaders.sources import (
    PRESETS,
    DataSource,
    FileSpec,
    TabularCSVSource,
)

# ----------------------------------------------------------------------------
# Synthetic fixture writers
# ----------------------------------------------------------------------------


def _write_abb_file(
    path: Path,
    track_specs: Sequence[tuple[str, str]],
    rows: Iterable[tuple[str, str, Sequence[str]]],
) -> None:
    """Write a synthetic ABB-format UTF-16 CSV.

    Args:
        path: Output file path.
        track_specs: Sequence of ``(track_type, track_name)`` pairs,
            e.g. ``[("REAL", "Power"), ("BOOL", "Running")]``.
        rows: Sequence of ``(date, time, values)`` rows. ``values`` has
            the same length as ``track_specs``; each entry is the
            string-encoded cell value.
    """
    header_cols = ["EintragID", "Status", "Datum", "Zeit"]
    header_cols += [f"Spur {i + 1}: Typ: {t}  Name: {n}" for i, (t, n) in enumerate(track_specs)]

    lines = [
        "Projektname = test;Bildname = Synth;Archivtyp = TRD;",
        "",
        ";Neuester Eintrag = X;Ältester Eintrag = 0;",
        ";".join(header_cols),
    ]
    for i, (d, t, values) in enumerate(rows):
        cells = [str(i), "", d, t, *values]
        lines.append(";".join(cells))

    # Use ``\n`` join here; on Windows the underlying text-mode write
    # translates that to the platform's native line ending.
    path.write_text("\n".join(lines) + "\n", encoding="utf-16")


@pytest.fixture
def abb_like_dir(tmp_path: Path) -> Path:
    """Two synthetic files in the ABB TRD export layout."""
    _write_abb_file(
        tmp_path / "BHKW_TR.csv",
        track_specs=[
            ("REAL", "Sollwert"),
            ("REAL", "Wirkleistung"),
            ("REAL", "Gasverbrauch"),
        ],
        rows=[
            ("01.06.25", "00:00:00.000", ("250,0", "248,5", "115,1")),
            ("01.06.25", "00:00:05.000", ("250,0", "249,2", "115,6")),
            ("01.06.25", "00:00:10.000", ("250,0", "3,4028235e38", "115,3")),
            ("01.06.25", "00:00:15.000", ("250,0", "251,0", "112,8")),
        ],
    )
    _write_abb_file(
        tmp_path / "Sickers_TR.csv",
        track_specs=[
            ("BOOL", "Min Schalter"),
            ("BOOL", "Pumpe Ein"),
        ],
        rows=[
            ("01.06.25", "00:00:00.000", ("FALSE", "FALSE")),
            ("01.06.25", "00:00:05.000", ("FALSE", "TRUE")),
            ("01.06.25", "00:00:10.000", ("TRUE", "TRUE")),
            ("01.06.25", "00:00:15.000", ("FALSE", "FALSE")),
        ],
    )
    return tmp_path


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


class TestProtocolConformance:
    def test_source_satisfies_protocol(self, abb_like_dir: Path) -> None:
        src = TabularCSVSource.from_directory("plant", abb_like_dir, preset="abb")
        assert isinstance(src, DataSource)


class TestTagDiscovery:
    def test_lists_all_tags_with_prefix(self, abb_like_dir: Path) -> None:
        src = TabularCSVSource.from_directory("plant", abb_like_dir, preset="abb")
        tags = src.list_tags()
        assert "BHKW_TR::Wirkleistung" in tags
        assert "BHKW_TR::Sollwert" in tags
        assert "BHKW_TR::Gasverbrauch" in tags
        assert "Sickers_TR::Min Schalter" in tags
        assert "Sickers_TR::Pumpe Ein" in tags

    def test_unknown_tag_raises(self, abb_like_dir: Path) -> None:
        src = TabularCSVSource.from_directory("plant", abb_like_dir, preset="abb")
        with pytest.raises(KeyError):
            src.read(tags=["does_not_exist"])

    def test_tag_collision_is_detected(self, tmp_path: Path) -> None:
        # Two files exposing the same raw tag but no namespacing.
        _write_abb_file(
            tmp_path / "a.csv",
            track_specs=[("REAL", "X")],
            rows=[("01.06.25", "00:00:00.000", ("1,0",))],
        )
        _write_abb_file(
            tmp_path / "b.csv",
            track_specs=[("REAL", "X")],
            rows=[("01.06.25", "00:00:00.000", ("2,0",))],
        )
        with pytest.raises(ValueError, match="Tag collision"):
            TabularCSVSource.from_directory("test", tmp_path, preset="abb", prefix_from_stem=False)


class TestRead:
    def test_returns_utc_indexed_frame(self, abb_like_dir: Path) -> None:
        src = TabularCSVSource.from_directory("plant", abb_like_dir, preset="abb")
        df = src.read()
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "timestamp"
        assert str(df.index.tz) == "UTC"
        assert len(df) == 4

    def test_numeric_values_parsed_with_comma_decimal(self, abb_like_dir: Path) -> None:
        src = TabularCSVSource.from_directory("plant", abb_like_dir, preset="abb")
        df = src.read(tags=["BHKW_TR::Sollwert"])
        assert df["BHKW_TR::Sollwert"].iloc[0] == pytest.approx(250.0)

    def test_sentinel_value_becomes_nan(self, abb_like_dir: Path) -> None:
        src = TabularCSVSource.from_directory("plant", abb_like_dir, preset="abb")
        df = src.read(tags=["BHKW_TR::Wirkleistung"])
        # Row index 2 in our fixture contains the float32-max sentinel.
        wirk = df["BHKW_TR::Wirkleistung"].to_numpy()
        assert np.isnan(wirk[2])
        # Surrounding rows are intact.
        assert wirk[0] == pytest.approx(248.5)
        assert wirk[1] == pytest.approx(249.2)

    def test_bool_values_are_parsed(self, abb_like_dir: Path) -> None:
        src = TabularCSVSource.from_directory("plant", abb_like_dir, preset="abb")
        df = src.read(tags=["Sickers_TR::Pumpe Ein"])
        col = df["Sickers_TR::Pumpe Ein"].tolist()
        assert col == [False, True, True, False]

    def test_tag_filter_returns_only_requested_columns(self, abb_like_dir: Path) -> None:
        src = TabularCSVSource.from_directory("plant", abb_like_dir, preset="abb")
        df = src.read(tags=["BHKW_TR::Sollwert", "Sickers_TR::Pumpe Ein"])
        assert list(df.columns) == ["BHKW_TR::Sollwert", "Sickers_TR::Pumpe Ein"]

    def test_time_filter_restricts_rows(self, abb_like_dir: Path) -> None:
        src = TabularCSVSource.from_directory("plant", abb_like_dir, preset="abb", timezone=None)
        df = src.read(
            start=datetime(2025, 6, 1, 0, 0, 5),
            end=datetime(2025, 6, 1, 0, 0, 10),
        )
        assert len(df) == 2

    def test_empty_window_returns_empty_frame(self, abb_like_dir: Path) -> None:
        src = TabularCSVSource.from_directory("plant", abb_like_dir, preset="abb", timezone=None)
        df = src.read(
            start=datetime(2030, 1, 1),
            end=datetime(2030, 1, 2),
        )
        assert df.empty
        assert df.index.tz is not None

    def test_multi_file_outer_join(self, abb_like_dir: Path) -> None:
        src = TabularCSVSource.from_directory("plant", abb_like_dir, preset="abb")
        df = src.read()
        # Tags from both files share the same 4 timestamps.
        assert len(df) == 4
        for tag in (
            "BHKW_TR::Wirkleistung",
            "Sickers_TR::Pumpe Ein",
        ):
            assert tag in df.columns


class TestTimezone:
    def test_naive_localized_to_configured_zone(self, abb_like_dir: Path) -> None:
        # Berlin in summer is UTC+2, so 00:00:00 local -> 22:00:00 UTC
        # of the previous day.
        src = TabularCSVSource.from_directory("plant", abb_like_dir, preset="abb", timezone="Europe/Berlin")
        df = src.read(tags=["BHKW_TR::Sollwert"])
        first = df.index[0]
        assert first == pd.Timestamp("2025-05-31 22:00:00", tz="UTC")

    def test_timezone_none_treats_input_as_utc(self, abb_like_dir: Path) -> None:
        src = TabularCSVSource.from_directory("plant", abb_like_dir, preset="abb", timezone=None)
        df = src.read(tags=["BHKW_TR::Sollwert"])
        first = df.index[0]
        assert first == pd.Timestamp("2025-06-01 00:00:00", tz="UTC")


class TestPresets:
    def test_plain_csv_preset(self, tmp_path: Path) -> None:
        path = tmp_path / "telemetry.csv"
        path.write_text(
            "timestamp,Q_gas,T_digester\n" "2025-06-01T00:00:00,123.4,310.5\n" "2025-06-01T01:00:00,128.1,310.6\n",
            encoding="utf-8",
        )
        spec = PRESETS["plain"](path=path, tag_prefix="plant1")
        src = TabularCSVSource("test", [spec], timezone=None)
        df = src.read()
        assert list(df.columns) == ["plant1::Q_gas", "plant1::T_digester"]
        assert df["plant1::Q_gas"].iloc[0] == pytest.approx(123.4)
        assert len(df) == 2

    def test_lab_csv_preset(self, tmp_path: Path) -> None:
        path = tmp_path / "lab.csv"
        path.write_text(
            "date,pH,VFA,TAC,NH4_N\n" "2025-06-01,7.8,2.1,12.5,1.2\n" "2025-06-08,7.6,3.4,11.8,1.3\n",
            encoding="utf-8",
        )
        spec = PRESETS["lab"](path=path, tag_prefix="lab")
        src = TabularCSVSource("test", [spec], timezone=None)
        tags = src.list_tags()
        assert "lab::pH" in tags
        assert "lab::VFA" in tags

        df = src.read()
        assert len(df) == 2
        assert df["lab::pH"].iloc[0] == pytest.approx(7.8)


class TestFileSpec:
    def test_qualified_tag_no_prefix(self) -> None:
        spec = FileSpec(path=Path("x.csv"))
        assert spec.qualified_tag("Power") == "Power"

    def test_qualified_tag_with_prefix(self) -> None:
        spec = FileSpec(path=Path("x.csv"), tag_prefix="CHP1")
        assert spec.qualified_tag("Power") == "CHP1::Power"

    def test_custom_separator(self) -> None:
        spec = FileSpec(path=Path("x.csv"), tag_prefix="CHP1", prefix_separator=".")
        assert spec.qualified_tag("Power") == "CHP1.Power"


class TestConstruction:
    def test_directory_glob_filter_excludes_files(self, abb_like_dir: Path) -> None:
        # Only load BHKW_TR.csv, skip Sickers_TR.csv.
        src = TabularCSVSource.from_directory(
            "plant",
            abb_like_dir,
            preset="abb",
            pattern="BHKW_TR.csv",
        )
        tags = src.list_tags()
        assert any(t.startswith("BHKW_TR::") for t in tags)
        assert not any(t.startswith("Sickers_TR::") for t in tags)

    def test_unknown_preset_raises(self, abb_like_dir: Path) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            TabularCSVSource.from_directory("plant", abb_like_dir, preset="not_a_preset")

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            TabularCSVSource.from_directory("empty", tmp_path, preset="abb")
