"""Tests for :mod:`pyadm1ode_calibration.io.loaders.builder`.

These tests exercise the full pipeline: TabularCSVSource (Step 1) ->
PlantSchema (Step 2) -> MeasurementBuilder (Step 3) -> MeasurementData.
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from pyadm1ode_calibration.io.loaders import (
    SOURCE_FACTORIES,
    MeasurementBuilder,
    MeasurementData,
    PlantSchema,
    register_source_type,
)
from pyadm1ode_calibration.io.loaders.sources import DataSource
from tests.unit.test_io.test_tabular_csv import _write_abb_file

# ----------------------------------------------------------------------------
# Shared fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def abb_dir(tmp_path: Path) -> Path:
    """Two synthetic ABB files: one with REAL temperatures, one with bools."""
    _write_abb_file(
        tmp_path / "Temp_PCS.csv",
        track_specs=[("REAL", "Durchschnitt F1"), ("REAL", "Nachgärer Temp.")],
        rows=[
            ("01.06.25", "00:00:00.000", ("38,0", "37,5")),
            ("01.06.25", "00:15:00.000", ("38,2", "37,6")),
            ("01.06.25", "00:30:00.000", ("38,4", "37,7")),
            ("01.06.25", "00:45:00.000", ("38,3", "37,6")),
            ("01.06.25", "01:00:00.000", ("38,1", "37,5")),
            ("01.06.25", "01:15:00.000", ("80,0", "37,4")),  # out of range
            ("01.06.25", "01:30:00.000", ("38,0", "37,4")),
            ("01.06.25", "01:45:00.000", ("37,9", "37,3")),
        ],
    )
    _write_abb_file(
        tmp_path / "Level_Gas.csv",
        track_specs=[("BOOL", "Fackel Ein")],
        rows=[
            ("01.06.25", "00:00:00.000", ("FALSE",)),
            ("01.06.25", "00:15:00.000", ("FALSE",)),
            ("01.06.25", "00:30:00.000", ("TRUE",)),
            ("01.06.25", "00:45:00.000", ("FALSE",)),
            ("01.06.25", "01:00:00.000", ("FALSE",)),
            ("01.06.25", "01:15:00.000", ("FALSE",)),
            ("01.06.25", "01:30:00.000", ("FALSE",)),
            ("01.06.25", "01:45:00.000", ("FALSE",)),
        ],
    )
    return tmp_path


def _write_schema(tmp_path: Path, abb_dir: Path, body: str) -> Path:
    yaml_path = tmp_path / "schema.yaml"
    yaml_path.write_text(
        textwrap.dedent(body).format(abb_dir=abb_dir.as_posix()).lstrip("\n"),
        encoding="utf-8",
    )
    return yaml_path


@pytest.fixture
def basic_schema_yaml(tmp_path: Path, abb_dir: Path) -> Path:
    return _write_schema(
        tmp_path,
        abb_dir,
        """
        plant_id: test
        sources:
          abb:
            type: tabular_csv
            preset: abb
            directory: {abb_dir}
            timezone: null   # treat raw timestamps as UTC for easier asserts
        variables:
          T_digester:
            source: abb
            tag: "Temp_PCS::Durchschnitt F1"
            unit: degC
            to: K
            resample: 1h
            valid_range: [20, 55]
          flare_on:
            source: abb
            tag: "Level_Gas::Fackel Ein"
            unit: bool
            resample:
              freq: 1h
              agg: max
    """,
    )


# ----------------------------------------------------------------------------
# Construction & source management
# ----------------------------------------------------------------------------


class TestSourceManagement:
    def test_auto_instantiates_source_from_schema(self, basic_schema_yaml: Path) -> None:
        schema = PlantSchema.from_yaml(basic_schema_yaml)
        builder = MeasurementBuilder(schema)
        source = builder.get_source("abb")
        assert isinstance(source, DataSource)
        assert source.name == "abb"

    def test_register_source_overrides_schema(self, basic_schema_yaml: Path) -> None:
        schema = PlantSchema.from_yaml(basic_schema_yaml)
        builder = MeasurementBuilder(schema)

        class DummySource:
            name = "abb"

            def list_tags(self):
                return []

            def time_range(self):
                return (None, None)

            def read(self, start=None, end=None, tags=None):
                idx = pd.DatetimeIndex([], name="timestamp", tz="UTC")
                return pd.DataFrame(index=idx)

        dummy = DummySource()
        builder.register_source("abb", dummy)
        assert builder.get_source("abb") is dummy

    def test_register_unknown_source_raises(self, basic_schema_yaml: Path) -> None:
        schema = PlantSchema.from_yaml(basic_schema_yaml)
        builder = MeasurementBuilder(schema)

        class DummySource:
            name = "x"

        with pytest.raises(KeyError, match="not declared"):
            builder.register_source("nonexistent", DummySource())

    def test_unknown_source_type_raises(self, tmp_path: Path) -> None:
        yaml_path = tmp_path / "s.yaml"
        yaml_path.write_text(
            textwrap.dedent("""
            plant_id: x
            sources:
              weird:
                type: never_seen_before
                foo: bar
            variables:
              v:
                source: weird
                tag: t
                unit: K
        """).lstrip("\n"),
            encoding="utf-8",
        )
        schema = PlantSchema.from_yaml(yaml_path)
        builder = MeasurementBuilder(schema)
        with pytest.raises(ValueError, match="No factory"):
            builder.get_source("weird")


class TestRegisterSourceType:
    def test_custom_source_type(self, tmp_path: Path) -> None:
        class InMemorySource:
            def __init__(self, name: str, **_):
                self.name = name

            def list_tags(self):
                return ["x"]

            def time_range(self):
                return (None, None)

            def read(self, start=None, end=None, tags=None):
                idx = pd.date_range("2025-06-01", periods=3, freq="1h", tz="UTC")
                return pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=idx)

        register_source_type("in_memory", InMemorySource)
        try:
            yaml_path = tmp_path / "s.yaml"
            yaml_path.write_text(
                textwrap.dedent("""
                plant_id: x
                sources:
                  m:
                    type: in_memory
                variables:
                  Y:
                    source: m
                    tag: x
                    unit: K
                    resample: 1h
            """).lstrip("\n"),
                encoding="utf-8",
            )
            schema = PlantSchema.from_yaml(yaml_path)
            builder = MeasurementBuilder(schema)
            data = builder.build()
            assert "Y" in data.data.columns
            assert len(data.data) == 3
        finally:
            SOURCE_FACTORIES.pop("in_memory", None)


# ----------------------------------------------------------------------------
# Build behavior
# ----------------------------------------------------------------------------


class TestBuild:
    def test_returns_measurement_data(self, basic_schema_yaml: Path) -> None:
        schema = PlantSchema.from_yaml(basic_schema_yaml)
        builder = MeasurementBuilder(schema)
        data = builder.build()
        assert isinstance(data, MeasurementData)
        assert "T_digester" in data.data.columns
        assert "flare_on" in data.data.columns

    def test_unit_conversion_applied(self, basic_schema_yaml: Path) -> None:
        schema = PlantSchema.from_yaml(basic_schema_yaml)
        builder = MeasurementBuilder(schema)
        data = builder.build(variables=["T_digester"])
        t = data.data["T_digester"]
        # 38.0 degC -> 311.15 K. Resampled to 1h (mean of in-range values).
        assert t.iloc[0] == pytest.approx(38.225 + 273.15, abs=0.01)

    def test_valid_range_filter(self, basic_schema_yaml: Path) -> None:
        schema = PlantSchema.from_yaml(basic_schema_yaml)
        builder = MeasurementBuilder(schema)
        data = builder.build(variables=["T_digester"])
        # The synthetic data has one out-of-range value (80 degC) in the
        # second hour. It must be excluded from the hourly mean.
        # Remaining values 38.1, 38.0, 37.9 in degC -> mean 38.0 -> 311.15 K.
        t_hour2 = data.data["T_digester"].iloc[1]
        assert t_hour2 == pytest.approx(38.0 + 273.15, abs=0.01)

    def test_bool_resamples_with_max(self, basic_schema_yaml: Path) -> None:
        schema = PlantSchema.from_yaml(basic_schema_yaml)
        builder = MeasurementBuilder(schema)
        data = builder.build(variables=["flare_on"])
        flare = data.data["flare_on"]
        # First hour has one TRUE (at 00:30). max should be 1.0.
        assert flare.iloc[0] == pytest.approx(1.0)
        # Second hour has only FALSE. max should be 0.0.
        assert flare.iloc[1] == pytest.approx(0.0)

    def test_resample_frequency(self, basic_schema_yaml: Path) -> None:
        schema = PlantSchema.from_yaml(basic_schema_yaml)
        builder = MeasurementBuilder(schema)
        data = builder.build()
        # 1h resampling of 2h of data -> 2 rows.
        assert len(data.data) == 2

    def test_variables_subset(self, basic_schema_yaml: Path) -> None:
        schema = PlantSchema.from_yaml(basic_schema_yaml)
        builder = MeasurementBuilder(schema)
        data = builder.build(variables=["T_digester"])
        assert list(data.data.columns) == ["T_digester"]

    def test_unknown_variable_raises(self, basic_schema_yaml: Path) -> None:
        schema = PlantSchema.from_yaml(basic_schema_yaml)
        builder = MeasurementBuilder(schema)
        with pytest.raises(KeyError, match="not declared"):
            builder.build(variables=["bogus"])

    def test_time_filter(self, basic_schema_yaml: Path) -> None:
        schema = PlantSchema.from_yaml(basic_schema_yaml)
        builder = MeasurementBuilder(schema)
        data = builder.build(
            start=datetime(2025, 6, 1, 1, 0),
            end=datetime(2025, 6, 1, 2, 0),
            variables=["T_digester"],
        )
        # Only the second hour remains.
        assert len(data.data) == 1

    def test_metadata_populated(self, basic_schema_yaml: Path) -> None:
        schema = PlantSchema.from_yaml(basic_schema_yaml)
        builder = MeasurementBuilder(schema)
        data = builder.build()
        meta = data.metadata
        assert meta["plant_id"] == "test"
        assert meta["sources"] == ["abb"]
        assert set(meta["variables"]) == {"T_digester", "flare_on"}
        assert meta["time_range"][0] is not None
        assert meta["build_time"] is not None

    def test_column_order_follows_schema(self, basic_schema_yaml: Path) -> None:
        schema = PlantSchema.from_yaml(basic_schema_yaml)
        builder = MeasurementBuilder(schema)
        data = builder.build(variables=["flare_on", "T_digester"])
        assert list(data.data.columns) == ["flare_on", "T_digester"]


# ----------------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------------


class TestEdgeCases:
    def test_missing_tag_raises_by_default(self, tmp_path: Path, abb_dir: Path) -> None:
        yaml_path = _write_schema(
            tmp_path,
            abb_dir,
            """
            plant_id: x
            sources:
              abb:
                type: tabular_csv
                preset: abb
                directory: {abb_dir}
                timezone: null
            variables:
              missing:
                source: abb
                tag: "Temp_PCS::Does Not Exist"
                unit: K
        """,
        )
        schema = PlantSchema.from_yaml(yaml_path)
        builder = MeasurementBuilder(schema)
        with pytest.raises(KeyError, match="does not expose"):
            builder.build()

    def test_missing_tag_skipped_with_flag(self, tmp_path: Path, abb_dir: Path) -> None:
        yaml_path = _write_schema(
            tmp_path,
            abb_dir,
            """
            plant_id: x
            sources:
              abb:
                type: tabular_csv
                preset: abb
                directory: {abb_dir}
                timezone: null
            variables:
              missing:
                source: abb
                tag: "Temp_PCS::Does Not Exist"
                unit: K
              present:
                source: abb
                tag: "Temp_PCS::Durchschnitt F1"
                unit: degC
                to: K
                resample: 1h
        """,
        )
        schema = PlantSchema.from_yaml(yaml_path)
        builder = MeasurementBuilder(schema)
        with pytest.warns(UserWarning, match="missing tags"):
            data = builder.build(skip_missing=True)
        assert "present" in data.data.columns
        assert "missing" not in data.data.columns

    def test_ignored_tag_skipped_with_warning(self, tmp_path: Path, abb_dir: Path) -> None:
        yaml_path = _write_schema(
            tmp_path,
            abb_dir,
            """
            plant_id: x
            sources:
              abb:
                type: tabular_csv
                preset: abb
                directory: {abb_dir}
                timezone: null
            variables:
              T_digester:
                source: abb
                tag: "Temp_PCS::Durchschnitt F1"
                unit: degC
                to: K
                resample: 1h
            ignored_tags:
              "abb::Temp_PCS::Durchschnitt F1": "Suspect sensor"
        """,
        )
        schema = PlantSchema.from_yaml(yaml_path)
        builder = MeasurementBuilder(schema)
        with pytest.warns(UserWarning, match="ignored tag"):
            data = builder.build()
        assert "T_digester" not in data.data.columns

    def test_empty_variable_list(self, basic_schema_yaml: Path) -> None:
        schema = PlantSchema.from_yaml(basic_schema_yaml)
        builder = MeasurementBuilder(schema)
        data = builder.build(variables=[])
        assert data.data.empty
        assert data.metadata["plant_id"] == "test"


# ----------------------------------------------------------------------------
# Multi-source build
# ----------------------------------------------------------------------------


class TestMultiSource:
    def test_combines_two_sources(self, tmp_path: Path) -> None:
        # Source A: ABB file
        abb_dir = tmp_path / "abb"
        abb_dir.mkdir()
        _write_abb_file(
            abb_dir / "Temp.csv",
            track_specs=[("REAL", "F1")],
            rows=[
                ("01.06.25", "00:00:00.000", ("38,0",)),
                ("01.06.25", "00:30:00.000", ("38,5",)),
                ("01.06.25", "01:00:00.000", ("38,2",)),
            ],
        )

        # Source B: plain ISO CSV (e.g. a lab-style daily report)
        lab_dir = tmp_path / "lab"
        lab_dir.mkdir()
        (lab_dir / "lab.csv").write_text("date,pH\n2025-06-01,7.8\n", encoding="utf-8")

        yaml_path = tmp_path / "s.yaml"
        yaml_path.write_text(
            textwrap.dedent(f"""
            plant_id: combo
            sources:
              abb:
                type: tabular_csv
                preset: abb
                directory: {abb_dir.as_posix()}
                timezone: null
              lab:
                type: tabular_csv
                preset: lab
                directory: {lab_dir.as_posix()}
                timezone: null
            variables:
              T_digester:
                source: abb
                tag: "Temp::F1"
                unit: degC
                to: K
                resample: 1h
              pH:
                source: lab
                tag: "lab::pH"
                unit: dimensionless
                resample: 1d
        """).lstrip("\n"),
            encoding="utf-8",
        )
        schema = PlantSchema.from_yaml(yaml_path)
        builder = MeasurementBuilder(schema)
        data = builder.build()
        assert "T_digester" in data.data.columns
        assert "pH" in data.data.columns
        assert sorted(data.metadata["sources"]) == ["abb", "lab"]
