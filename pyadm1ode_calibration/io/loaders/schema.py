"""
Plant schema: declarative description of one plant's data layout.

A :class:`PlantSchema` describes, in one file, where a plant's
measurements come from and how to interpret them — without prescribing
how they are loaded or used downstream. It separates the *what*
(mapping raw tag names to canonical variables, units, quality bounds)
from the *how* (file parsing, optimization, validation), so the same
calibration code can be run against any plant by swapping the YAML.

Schemas are usually authored as YAML files under ``configs/plants/``,
but can be constructed programmatically too.

Construction performs internal validation:

- every variable's ``source`` must refer to a declared source,
- every unit must be known to :mod:`.units`,
- if a ``to`` unit is requested, the conversion must be registered,
- the resample aggregation must be one of :data:`ALLOWED_AGGS`,
- ``valid_range`` lower bound must be < upper bound.

Tag-name existence is *not* checked here, since that requires
instantiating the source adapter. The Builder (next layer) handles
that lookup at read time.

Example:
    >>> schema = PlantSchema.from_yaml("configs/plants/my_plant.yaml")
    >>> schema.variables["T_digester"].source
    'abb'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import units

# Resampling aggregations supported by the schema. These map directly to
# pandas Resampler methods, so adding one is a matter of listing it here.
ALLOWED_AGGS = frozenset({"mean", "sum", "first", "last", "min", "max", "median"})


@dataclass
class ResampleSpec:
    """How to aggregate a variable onto a uniform time grid.

    Attributes:
        freq: pandas frequency string (e.g. ``"1h"``, ``"15min"``, ``"1d"``).
        agg: Aggregation method. One of :data:`ALLOWED_AGGS`.
    """

    freq: str = "1h"
    agg: str = "mean"

    @classmethod
    def from_yaml(cls, value: str | dict[str, Any] | None) -> ResampleSpec:
        """Build from YAML shorthand.

        Accepts:

        - ``None`` → default ``1h`` mean,
        - a bare string ``"15min"`` → frequency only, default agg,
        - a mapping ``{freq: 1h, agg: max}`` → full form.
        """
        if value is None:
            return cls()
        if isinstance(value, str):
            return cls(freq=value)
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"resample must be a string, mapping, or null; got {type(value).__name__}")


@dataclass
class VariableSpec:
    """Mapping from one canonical variable to a raw tag in a source.

    Attributes:
        name: Canonical name (used everywhere downstream, e.g.
            ``"T_digester"``).
        source: Name of the data source where the tag lives.
        tag: Tag name as exposed by that source. Use
            ``source.list_tags()`` to inspect available names.
        unit: Unit of the value in the source data (e.g. ``"degC"``).
        to_unit: Optional target unit (e.g. ``"K"``). ``None`` means no
            conversion.
        resample: How to aggregate the tag onto the analysis time grid.
        valid_range: ``(lo, hi)`` range of plausible values, expressed
            in ``unit`` (i.e. before conversion). Values outside this
            range should be set to NaN by downstream code.
        description: Optional free-form note.
        meta: Free-form variable-level metadata. Plant-specific
            attributes (substrate densities, sensor model numbers,
            installation dates, …) live here so the schema does not
            need a new field for every domain quirk.
    """

    name: str
    source: str
    tag: str
    unit: str
    to_unit: str | None = None
    resample: ResampleSpec = field(default_factory=ResampleSpec)
    valid_range: tuple[float, float] | None = None
    description: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def canonical_unit(self) -> str:
        """The unit the variable is expressed in after conversion."""
        return self.to_unit if self.to_unit is not None else self.unit


@dataclass
class SubstrateMixEntry:
    """One entry in a plant's substrate mix.

    Attributes:
        name: Substrate identifier; must match a substrate that
            :class:`pyadm1.Feedstock` can resolve at plant-build time
            (typically a file stem under PyADM1ODE's substrate library,
            e.g. ``"maize_silage_milk_ripeness"``).
        fraction: Mass or volume fraction of the total dosed amount that
            this substrate represents, in ``[0, 1]``. The fractions of
            all entries must sum to ~1.0.
        description: Optional human-readable note.
        meta: Free-form metadata, e.g. ``{"source": "hopper"}`` to mark
            which physical feed path the substrate travels through.
            Consumed by the calibration runner to split Q_total into
            the correct per-substrate flows.
    """

    name: str
    fraction: float
    description: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceSpec:
    """Configuration of one named data source.

    The ``config`` mapping is opaque to the schema and is forwarded as
    keyword arguments to the source adapter constructor (e.g. for
    :class:`~.sources.TabularCSVSource`: ``directory``, ``preset``,
    ``timezone``, ``pattern``, …).

    Attributes:
        name: Identifier used by variables to refer to this source.
        type: Adapter type. Currently supported: ``"tabular_csv"``.
        config: Type-specific adapter arguments.
    """

    name: str
    type: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlantSchema:
    """Declarative description of one plant's data layout.

    Attributes:
        plant_id: Stable identifier (used in DB rows, log lines).
        sources: Source configurations, keyed by source name.
        variables: Variable mappings, keyed by canonical name.
        description: Free-form description of the plant.
        ignored_tags: Map ``"source::tag"`` → reason. Variables that
            accidentally reference these are flagged downstream.
        substrates: Substrate mix in order of declaration. Fractions
            should sum to ~1.0. Used by plant-build helpers to construct
            the :class:`pyadm1.Feedstock` and by derived-variable helpers
            to split aggregate dosing by substrate.
        tag_labels: Map ``"source::tag"`` → human-readable display name.
            Used by visualisation helpers for tags that are *not*
            mapped to a canonical :class:`VariableSpec` but still
            appear in plots (e.g. an "unused tags" view in a plotting
            script). For mapped variables, ``meta.label`` on the
            spec itself takes precedence.
        tag_units: Map ``"source::tag"`` → unit string for the same
            unmapped tags. Used as the y-axis label in plots. Allows
            documenting plausible units that the ABB CSV export does
            not carry (it ships only types like REAL/BOOL, no units).
    """

    plant_id: str
    sources: dict[str, SourceSpec]
    variables: dict[str, VariableSpec]
    description: str | None = None
    ignored_tags: dict[str, str] = field(default_factory=dict)
    substrates: list[SubstrateMixEntry] = field(default_factory=list)
    tag_labels: dict[str, str] = field(default_factory=dict)
    tag_units: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    # ----- Construction -----------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> PlantSchema:
        """Load a schema from a YAML file on disk."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a mapping at the top level of {path}, " f"got {type(data).__name__}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlantSchema:
        """Build a schema from an already-parsed dictionary (e.g. JSON)."""
        if "plant_id" not in d:
            raise ValueError("Schema is missing required key 'plant_id'.")

        sources_d = d.get("sources") or {}
        if not isinstance(sources_d, dict):
            raise ValueError("'sources' must be a mapping of name -> config.")
        sources: dict[str, SourceSpec] = {}
        for name, cfg in sources_d.items():
            if not isinstance(cfg, dict) or "type" not in cfg:
                raise ValueError(f"Source '{name}' must be a mapping with at least a 'type' field.")
            sources[name] = SourceSpec(
                name=name,
                type=cfg["type"],
                config={k: v for k, v in cfg.items() if k != "type"},
            )

        variables_d = d.get("variables") or {}
        if not isinstance(variables_d, dict):
            raise ValueError("'variables' must be a mapping of name -> config.")
        variables: dict[str, VariableSpec] = {}
        for name, cfg in variables_d.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"Variable '{name}' must be a mapping.")
            for required in ("source", "tag", "unit"):
                if required not in cfg:
                    raise ValueError(f"Variable '{name}' is missing required key '{required}'.")
            valid_range = cfg.get("valid_range")
            if valid_range is not None:
                if not (isinstance(valid_range, (list, tuple)) and len(valid_range) == 2):
                    raise ValueError(f"Variable '{name}': valid_range must be a [min, max] list.")
                valid_range = (float(valid_range[0]), float(valid_range[1]))

            meta = cfg.get("meta") or {}
            if not isinstance(meta, dict):
                raise ValueError(f"Variable '{name}': 'meta' must be a mapping if present.")

            variables[name] = VariableSpec(
                name=name,
                source=cfg["source"],
                tag=cfg["tag"],
                unit=cfg["unit"],
                to_unit=cfg.get("to"),
                resample=ResampleSpec.from_yaml(cfg.get("resample")),
                valid_range=valid_range,
                description=cfg.get("description"),
                meta=dict(meta),
            )

        ignored_raw = d.get("ignored_tags") or {}
        if not isinstance(ignored_raw, dict):
            raise ValueError("'ignored_tags' must be a mapping of 'source::tag' -> reason.")

        tag_labels_raw = d.get("tag_labels") or {}
        if not isinstance(tag_labels_raw, dict):
            raise ValueError("'tag_labels' must be a mapping of 'source::tag' -> display name.")

        tag_units_raw = d.get("tag_units") or {}
        if not isinstance(tag_units_raw, dict):
            raise ValueError("'tag_units' must be a mapping of 'source::tag' -> unit string.")

        substrates_raw = d.get("substrates") or []
        if not isinstance(substrates_raw, list):
            raise ValueError("'substrates' must be a list of mix entries.")
        substrates: list[SubstrateMixEntry] = []
        for i, entry in enumerate(substrates_raw):
            if not isinstance(entry, dict):
                raise ValueError(f"substrates[{i}] must be a mapping with 'name' and 'fraction'.")
            for required in ("name", "fraction"):
                if required not in entry:
                    raise ValueError(f"substrates[{i}] is missing required key '{required}'.")
            sub_meta = entry.get("meta") or {}
            if not isinstance(sub_meta, dict):
                raise ValueError(f"substrates[{i}]: 'meta' must be a mapping if present.")
            substrates.append(
                SubstrateMixEntry(
                    name=str(entry["name"]),
                    fraction=float(entry["fraction"]),
                    description=entry.get("description"),
                    meta=dict(sub_meta),
                )
            )

        return cls(
            plant_id=d["plant_id"],
            sources=sources,
            variables=variables,
            description=d.get("description"),
            ignored_tags={str(k): str(v) for k, v in ignored_raw.items()},
            substrates=substrates,
            tag_labels={str(k): str(v) for k, v in tag_labels_raw.items()},
            tag_units={str(k): str(v) for k, v in tag_units_raw.items()},
        )

    # ----- Validation -------------------------------------------------------

    def _validate(self) -> None:
        errors: list[str] = []

        for var_name, var in self.variables.items():
            # Source reference
            if var.source not in self.sources:
                errors.append(
                    f"Variable '{var_name}' references unknown source "
                    f"'{var.source}'. Known sources: {sorted(self.sources)}."
                )

            # Units
            if not units.is_known_unit(var.unit):
                errors.append(f"Variable '{var_name}' uses unknown unit '{var.unit}'.")
            if var.to_unit is not None:
                if not units.is_known_unit(var.to_unit):
                    errors.append(f"Variable '{var_name}' uses unknown target unit " f"'{var.to_unit}'.")
                elif not units.can_convert(var.unit, var.to_unit):
                    errors.append(
                        f"Variable '{var_name}' has no registered conversion " f"from '{var.unit}' to '{var.to_unit}'."
                    )

            # Resample
            if var.resample.agg not in ALLOWED_AGGS:
                errors.append(
                    f"Variable '{var_name}' uses unsupported resample agg "
                    f"'{var.resample.agg}'. Allowed: {sorted(ALLOWED_AGGS)}."
                )

            # Valid range
            if var.valid_range is not None:
                lo, hi = var.valid_range
                if not lo < hi:
                    errors.append(
                        f"Variable '{var_name}' has invalid valid_range " f"[{lo}, {hi}]: lower bound must be < upper bound."
                    )

        # Substrate mix: fractions must sum to ~1 (tolerance 1e-3) and each
        # entry must be in [0, 1]. An empty list is fine — not every schema
        # needs substrates declared.
        if self.substrates:
            total = sum(s.fraction for s in self.substrates)
            if abs(total - 1.0) > 1e-3:
                errors.append(f"Substrate fractions sum to {total:.4f}; expected 1.0 " f"(tolerance 1e-3).")
            for s in self.substrates:
                if not 0.0 <= s.fraction <= 1.0:
                    errors.append(f"Substrate '{s.name}' has fraction {s.fraction}; " f"must be within [0, 1].")

        if errors:
            raise ValueError("Schema validation failed:\n  - " + "\n  - ".join(errors))

    # ----- Convenience queries ----------------------------------------------

    def variables_for_source(self, source: str) -> list[VariableSpec]:
        """All variables that read from the given source."""
        return [v for v in self.variables.values() if v.source == source]

    def tags_for_source(self, source: str) -> list[str]:
        """All raw tag names referenced for the given source."""
        return [v.tag for v in self.variables.values() if v.source == source]

    def is_tag_ignored(self, source: str, tag: str) -> bool:
        """True if ``"{source}::{tag}"`` is listed in ``ignored_tags``."""
        return f"{source}::{tag}" in self.ignored_tags
