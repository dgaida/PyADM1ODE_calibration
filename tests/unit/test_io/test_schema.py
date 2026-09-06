"""Tests for :mod:`pyadm1ode_calibration.io.loaders.schema`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pyadm1ode_calibration.io.loaders.schema import (
    PlantSchema,
    ResampleSpec,
    SourceSpec,
    SubstrateMixEntry,
    VariableSpec,
)


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


class TestFromYaml:
    def test_minimal_schema(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: testplant
            sources:
              abb:
                type: tabular_csv
                directory: /tmp/data
            variables:
              T_digester:
                source: abb
                tag: "Temp::Avg"
                unit: degC
                to: K
        """,
        )
        schema = PlantSchema.from_yaml(p)
        assert schema.plant_id == "testplant"
        assert "abb" in schema.sources
        assert schema.sources["abb"].type == "tabular_csv"
        assert schema.sources["abb"].config["directory"] == "/tmp/data"
        var = schema.variables["T_digester"]
        assert var.source == "abb"
        assert var.unit == "degC"
        assert var.to_unit == "K"
        assert var.canonical_unit == "K"

    def test_canonical_unit_defaults_to_unit(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables:
              v:
                source: s1
                tag: t
                unit: kW
        """,
        )
        schema = PlantSchema.from_yaml(p)
        assert schema.variables["v"].canonical_unit == "kW"

    def test_resample_shorthand(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables:
              v:
                source: s1
                tag: t
                unit: K
                resample: 15min
        """,
        )
        schema = PlantSchema.from_yaml(p)
        assert schema.variables["v"].resample.freq == "15min"
        assert schema.variables["v"].resample.agg == "mean"

    def test_resample_full(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables:
              v:
                source: s1
                tag: t
                unit: bool
                resample:
                  freq: 1h
                  agg: max
        """,
        )
        schema = PlantSchema.from_yaml(p)
        assert schema.variables["v"].resample.agg == "max"

    def test_resample_missing_defaults(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables:
              v:
                source: s1
                tag: t
                unit: K
        """,
        )
        schema = PlantSchema.from_yaml(p)
        rs = schema.variables["v"].resample
        assert rs.freq == "1h"
        assert rs.agg == "mean"

    def test_ignored_tags(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables: {}
            ignored_tags:
              "s1::bad_tag": "Sensor broken"
        """,
        )
        schema = PlantSchema.from_yaml(p)
        assert schema.is_tag_ignored("s1", "bad_tag")
        assert not schema.is_tag_ignored("s1", "good_tag")
        assert schema.ignored_tags["s1::bad_tag"] == "Sensor broken"

    def test_valid_range_parsed_as_tuple(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables:
              v:
                source: s1
                tag: t
                unit: K
                valid_range: [273.15, 333.15]
        """,
        )
        schema = PlantSchema.from_yaml(p)
        assert schema.variables["v"].valid_range == (273.15, 333.15)


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------


class TestValidation:
    def test_missing_plant_id_raises(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            sources:
              s1: {type: tabular_csv}
            variables: {}
        """,
        )
        with pytest.raises(ValueError, match="plant_id"):
            PlantSchema.from_yaml(p)

    def test_unknown_source_raises(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables:
              v:
                source: missing
                tag: t
                unit: K
        """,
        )
        with pytest.raises(ValueError, match="unknown source"):
            PlantSchema.from_yaml(p)

    def test_unknown_unit_raises(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables:
              v:
                source: s1
                tag: t
                unit: unobtainium
        """,
        )
        with pytest.raises(ValueError, match="unknown unit"):
            PlantSchema.from_yaml(p)

    def test_unconvertible_unit_raises(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables:
              v:
                source: s1
                tag: t
                unit: K
                to: kW
        """,
        )
        with pytest.raises(ValueError, match="no registered conversion"):
            PlantSchema.from_yaml(p)

    def test_bad_resample_agg_raises(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables:
              v:
                source: s1
                tag: t
                unit: K
                resample:
                  freq: 1h
                  agg: argmin
        """,
        )
        with pytest.raises(ValueError, match="unsupported resample agg"):
            PlantSchema.from_yaml(p)

    def test_inverted_valid_range_raises(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables:
              v:
                source: s1
                tag: t
                unit: K
                valid_range: [10, 5]
        """,
        )
        with pytest.raises(ValueError, match="lower bound"):
            PlantSchema.from_yaml(p)

    def test_required_variable_field_raises(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables:
              v:
                source: s1
                unit: K
        """,
        )
        with pytest.raises(ValueError, match="missing required key 'tag'"):
            PlantSchema.from_yaml(p)

    def test_source_missing_type_raises(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {directory: /tmp}
            variables: {}
        """,
        )
        with pytest.raises(ValueError, match="'type'"):
            PlantSchema.from_yaml(p)


# ----------------------------------------------------------------------------
# Convenience queries
# ----------------------------------------------------------------------------


class TestQueries:
    def test_variables_for_source(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
              s2: {type: tabular_csv}
            variables:
              a:
                source: s1
                tag: t1
                unit: K
              b:
                source: s2
                tag: t2
                unit: K
              c:
                source: s1
                tag: t3
                unit: K
        """,
        )
        schema = PlantSchema.from_yaml(p)
        s1_names = {v.name for v in schema.variables_for_source("s1")}
        assert s1_names == {"a", "c"}
        assert schema.variables_for_source("nonexistent") == []

    def test_tags_for_source(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables:
              a:
                source: s1
                tag: alpha
                unit: K
              b:
                source: s1
                tag: beta
                unit: K
        """,
        )
        schema = PlantSchema.from_yaml(p)
        assert set(schema.tags_for_source("s1")) == {"alpha", "beta"}


# ----------------------------------------------------------------------------
# Bundled configs
# ----------------------------------------------------------------------------


class TestBundledConfigs:
    """Sanity check the YAML files shipped under configs/plants/."""

    @staticmethod
    def _configs_dir() -> Path:
        return Path(__file__).resolve().parents[3] / "configs" / "plants"

    def test_template_yaml_validates(self) -> None:
        path = self._configs_dir() / "_template.yaml"
        if not path.exists():
            pytest.skip(f"{path} not present")
        schema = PlantSchema.from_yaml(path)
        assert schema.plant_id == "my_plant"

    def test_every_shipped_config_validates(self) -> None:
        """Whatever plant configs are present must parse and be self-consistent.

        Configs of real plants are kept outside the repository, so this test
        adapts to what is actually there instead of naming a specific file.
        """
        paths = sorted(p for p in self._configs_dir().glob("*.yaml") if p.name != "_template.yaml")
        if not paths:
            pytest.skip("no plant configs present")
        for path in paths:
            schema = PlantSchema.from_yaml(path)
            assert schema.plant_id, f"{path.name}: plant_id missing"
            assert schema.sources, f"{path.name}: no source defined"
            assert schema.variables, f"{path.name}: no variable mapped"

    def test_shipped_template_is_valid(self):
        """The template is what users copy, so it has to parse.

        The test above skips when no real plant config is present, which is the normal
        state of the repository. Without this one the only schema the project actually
        ships would never be loaded by the suite.
        """
        schema = PlantSchema.from_yaml(self._configs_dir() / "_template.yaml")

        assert schema.plant_id
        assert schema.sources, "the template should demonstrate at least one source"
        assert schema.variables, "the template should demonstrate at least one variable"
        for name, spec in schema.variables.items():
            assert spec.source in schema.sources, f"{name} refers to an undeclared source"


# ----------------------------------------------------------------------------
# Direct construction (no YAML)
# ----------------------------------------------------------------------------


class TestProgrammaticConstruction:
    def test_construct_from_dataclasses(self) -> None:
        schema = PlantSchema(
            plant_id="prog",
            sources={"s1": SourceSpec(name="s1", type="tabular_csv")},
            variables={
                "T": VariableSpec(
                    name="T",
                    source="s1",
                    tag="x",
                    unit="degC",
                    to_unit="K",
                    resample=ResampleSpec(freq="1h", agg="mean"),
                ),
            },
        )
        assert schema.variables["T"].canonical_unit == "K"

    def test_validation_runs_on_direct_construction(self) -> None:
        with pytest.raises(ValueError, match="unknown source"):
            PlantSchema(
                plant_id="prog",
                sources={"s1": SourceSpec(name="s1", type="tabular_csv")},
                variables={
                    "T": VariableSpec(name="T", source="wrong", tag="x", unit="K"),
                },
            )


# ----------------------------------------------------------------------------
# Variable meta + substrate mix (Step 4 extensions)
# ----------------------------------------------------------------------------


class TestVariableMeta:
    def test_meta_default_is_empty_dict(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables:
              v:
                source: s1
                tag: t
                unit: K
        """,
        )
        schema = PlantSchema.from_yaml(p)
        assert schema.variables["v"].meta == {}

    def test_meta_parsed(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables:
              feed_total:
                source: s1
                tag: t
                unit: kg
                meta:
                  density_kg_per_m3: 600
                  note: "maize-typical"
        """,
        )
        schema = PlantSchema.from_yaml(p)
        meta = schema.variables["feed_total"].meta
        assert meta["density_kg_per_m3"] == 600
        assert meta["note"] == "maize-typical"

    def test_meta_must_be_mapping(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables:
              v:
                source: s1
                tag: t
                unit: K
                meta: "not a mapping"
        """,
        )
        with pytest.raises(ValueError, match="must be a mapping"):
            PlantSchema.from_yaml(p)


class TestSubstrates:
    def test_substrates_parsed(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables: {}
            substrates:
              - name: maize_silage_milk_ripeness
                fraction: 0.6
              - name: cattle_manure
                fraction: 0.4
                description: "Rindergülle"
        """,
        )
        schema = PlantSchema.from_yaml(p)
        assert len(schema.substrates) == 2
        assert schema.substrates[0].name == "maize_silage_milk_ripeness"
        assert schema.substrates[0].fraction == 0.6
        assert schema.substrates[1].description == "Rindergülle"

    def test_substrates_default_empty(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables: {}
        """,
        )
        schema = PlantSchema.from_yaml(p)
        assert schema.substrates == []

    def test_fractions_must_sum_to_one(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables: {}
            substrates:
              - name: a
                fraction: 0.3
              - name: b
                fraction: 0.3
        """,
        )
        with pytest.raises(ValueError, match="Substrate fractions sum"):
            PlantSchema.from_yaml(p)

    def test_fraction_out_of_range(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables: {}
            substrates:
              - name: a
                fraction: 1.5
        """,
        )
        with pytest.raises(ValueError, match="within \\[0, 1\\]"):
            PlantSchema.from_yaml(p)

    def test_missing_required_field(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables: {}
            substrates:
              - name: a
        """,
        )
        with pytest.raises(ValueError, match="missing required key 'fraction'"):
            PlantSchema.from_yaml(p)

    def test_programmatic_substrate_entry(self) -> None:
        entry = SubstrateMixEntry(name="maize", fraction=0.5)
        assert entry.description is None
        assert entry.fraction == 0.5


class TestTagLabels:
    def test_tag_labels_parsed(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables: {}
            tag_labels:
              "s1::Foo::Bar": "Foo Display"
              "s1::Other": "Anderes Signal"
        """,
        )
        schema = PlantSchema.from_yaml(p)
        assert schema.tag_labels["s1::Foo::Bar"] == "Foo Display"
        assert schema.tag_labels["s1::Other"] == "Anderes Signal"

    def test_tag_labels_default_empty(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables: {}
        """,
        )
        schema = PlantSchema.from_yaml(p)
        assert schema.tag_labels == {}

    def test_tag_labels_must_be_mapping(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables: {}
            tag_labels:
              - "not"
              - "a mapping"
        """,
        )
        with pytest.raises(ValueError, match="tag_labels"):
            PlantSchema.from_yaml(p)

    def test_tag_units_parsed(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables: {}
            tag_units:
              "s1::Mixer::Strom": "A"
              "s1::Pump::Pressure": "bar"
        """,
        )
        schema = PlantSchema.from_yaml(p)
        assert schema.tag_units["s1::Mixer::Strom"] == "A"
        assert schema.tag_units["s1::Pump::Pressure"] == "bar"

    def test_tag_units_default_empty(self, tmp_path: Path) -> None:
        p = _write_yaml(
            tmp_path / "s.yaml",
            """
            plant_id: x
            sources:
              s1: {type: tabular_csv}
            variables: {}
        """,
        )
        schema = PlantSchema.from_yaml(p)
        assert schema.tag_units == {}
