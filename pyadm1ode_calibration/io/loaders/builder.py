"""
MeasurementBuilder: turn a :class:`PlantSchema` plus :class:`DataSource`
adapters into a unified :class:`MeasurementData` ready for calibration.

This is the layer that ties the previous two layers together:

1. The schema declares *what* the plant produces and how to interpret it
   (raw tags, units, valid ranges, resampling).
2. The sources supply the *raw bytes*.
3. The builder loads, cleans, converts, resamples, and joins everything
   onto one DataFrame with canonical column names.

Source instantiation
--------------------

By default the builder creates source adapters from the
:class:`SourceSpec` entries in the schema, dispatching on
``source.type`` via :data:`SOURCE_FACTORIES`. Adding a new source type
(OPC UA, SQL, MQTT, …) is a one-liner via :func:`register_source_type`.

Callers can also inject pre-built sources via
:meth:`MeasurementBuilder.register_source`, which is the recommended
way to substitute mocks in tests or use a custom adapter that does not
fit the schema YAML.

Example:
    >>> schema = PlantSchema.from_yaml("configs/plants/my_plant.yaml")
    >>> builder = MeasurementBuilder(schema)
    >>> data = builder.build(
    ...     start="2025-06-01",
    ...     end="2025-06-08",
    ...     variables=["T_digester", "gas_consumption_chp1"],
    ... )
    >>> data.data.head()
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ...timeutils import normalize_freq
from . import units
from .measurement_data import MeasurementData
from .schema import PlantSchema, VariableSpec
from .sources import DataSource, TabularCSVSource

# ----------------------------------------------------------------------------
# Source factory registry
# ----------------------------------------------------------------------------


def _make_tabular_csv_source(name: str, **config: Any) -> DataSource:
    """Build a :class:`TabularCSVSource` from a schema entry.

    Accepts either ``directory`` (uses :meth:`TabularCSVSource.from_directory`)
    or ``files`` (constructs the source directly with explicit file specs).
    """
    if "directory" in config:
        return TabularCSVSource.from_directory(name=name, **config)
    if "files" in config:
        return TabularCSVSource(name=name, **config)
    raise ValueError(f"tabular_csv source '{name}' must provide either 'directory' " f"or 'files' in its config.")


SOURCE_FACTORIES: dict[str, Callable[..., DataSource]] = {
    "tabular_csv": _make_tabular_csv_source,
}


def register_source_type(type_name: str, factory: Callable[..., DataSource]) -> None:
    """Make a new source type available to schemas.

    The ``factory`` is called as ``factory(name=..., **schema_config)``
    and must return a :class:`DataSource` instance.
    """
    SOURCE_FACTORIES[type_name] = factory


# ----------------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------------


class MeasurementBuilder:
    """Compose a :class:`MeasurementData` from a schema and its sources.

    Args:
        schema: The plant schema describing canonical variables and
            their source mapping.
    """

    def __init__(self, schema: PlantSchema):
        self.schema = schema
        self._sources: dict[str, DataSource] = {}

    # ----- Source management -------------------------------------------------

    def register_source(self, name: str, source: DataSource) -> None:
        """Inject a pre-built source adapter.

        Overrides the schema's :class:`SourceSpec` for ``name``. Useful
        for tests (mock adapters) or for adapters that need custom
        construction the YAML can't express.
        """
        if name not in self.schema.sources:
            raise KeyError(f"Source '{name}' is not declared in the schema. " f"Known sources: {sorted(self.schema.sources)}.")
        self._sources[name] = source

    def get_source(self, name: str) -> DataSource:
        """Return the adapter for ``name``, instantiating it if needed."""
        if name in self._sources:
            return self._sources[name]

        spec = self.schema.sources.get(name)
        if spec is None:
            raise KeyError(f"Unknown source '{name}'. " f"Known sources: {sorted(self.schema.sources)}.")
        factory = SOURCE_FACTORIES.get(spec.type)
        if factory is None:
            raise ValueError(
                f"No factory registered for source type '{spec.type}'. "
                f"Available types: {sorted(SOURCE_FACTORIES)}. "
                f"Use register_source_type() to add one."
            )
        source = factory(name=name, **spec.config)
        self._sources[name] = source
        return source

    # ----- Build -------------------------------------------------------------

    def build(
        self,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        variables: list[str] | None = None,
        skip_missing: bool = False,
    ) -> MeasurementData:
        """Load and process measurements into a :class:`MeasurementData`.

        Args:
            start: Lower bound on the time window. Strings are parsed by
                pandas. Naive datetimes are interpreted as UTC. ``None``
                loads from the earliest available record (may be slow).
            end: Upper bound. ``None`` loads to the latest record.
            variables: Restrict the build to these canonical variable
                names. ``None`` builds every variable in the schema.
            skip_missing: When True, variables whose tag is not present
                in the underlying source are skipped with a warning
                instead of raising.

        Returns:
            A :class:`MeasurementData` with one column per canonical
            variable, in canonical units, resampled per the schema, and
            outer-joined on the time index. The result's metadata
            contains the plant id, the sources used, the variable list,
            and the build timestamp.
        """
        selected = self._select_variables(variables)
        selected = self._drop_ignored(selected)
        if not selected:
            return self._empty_result(selected, start, end)

        start_ts = self._normalize_ts(start)
        end_ts = self._normalize_ts(end)

        # Group variables by source so each source is read only once.
        vars_by_source: dict[str, list[VariableSpec]] = {}
        for var in selected:
            vars_by_source.setdefault(var.source, []).append(var)

        frames: list[pd.DataFrame] = []
        for src_name, vars_in_src in vars_by_source.items():
            source = self.get_source(src_name)
            present, missing = self._partition_by_availability(source, vars_in_src)
            if missing:
                self._handle_missing_tags(src_name, missing, skip_missing)

            if not present:
                continue

            raw = source.read(
                start=start_ts,
                end=end_ts,
                tags=[v.tag for v in present],
            )
            for var in present:
                if var.tag not in raw.columns:
                    # The source returned no data for this tag in the
                    # requested window; skip silently.
                    continue
                processed = self._process_column(raw[var.tag], var)
                frames.append(processed.to_frame(name=var.name))

        if not frames:
            return self._empty_result(selected, start, end)

        merged = pd.concat(frames, axis=1, join="outer").sort_index()
        merged.index.name = "timestamp"
        merged = merged.reindex(columns=[v.name for v in selected if v.name in merged.columns])

        return MeasurementData(
            data=merged,
            metadata=self._build_metadata(merged, selected, vars_by_source),
        )

    # ----- Internals ---------------------------------------------------------

    def _select_variables(self, requested: list[str] | None) -> list[VariableSpec]:
        """The variable specs to build, defaulting to every one the schema declares.

        Args:
            requested: Variable names to build, or ``None`` for all of them.

        Returns:
            list[VariableSpec]: The matching specs, in the requested order.

        Raises:
            KeyError: If a requested name is not declared in the schema. Failing here
                is deliberate: a typo would otherwise show up as a silently missing
                column much later.
        """
        if requested is None:
            return list(self.schema.variables.values())
        result: list[VariableSpec] = []
        for name in requested:
            if name not in self.schema.variables:
                raise KeyError(f"Variable '{name}' is not declared in the schema.")
            result.append(self.schema.variables[name])
        return result

    def _drop_ignored(self, vars_: list[VariableSpec]) -> list[VariableSpec]:
        """Remove variables whose tag the schema marks as ignored.

        A tag lands in ``ignored_tags`` when it is known to be broken or meaningless,
        so a variable pointing at one is a mistake worth a warning rather than a
        column of rubbish.

        Args:
            vars_: The candidate specs.

        Returns:
            list[VariableSpec]: Those not referencing an ignored tag.
        """
        kept: list[VariableSpec] = []
        for var in vars_:
            if self.schema.is_tag_ignored(var.source, var.tag):
                reason = self.schema.ignored_tags.get(f"{var.source}::{var.tag}", "no reason given")
                warnings.warn(
                    f"Variable '{var.name}' references ignored tag " f"'{var.source}::{var.tag}' ({reason}); skipping.",
                    stacklevel=3,
                )
                continue
            kept.append(var)
        return kept

    def _partition_by_availability(
        self, source: DataSource, vars_in_src: list[VariableSpec]
    ) -> tuple[list[VariableSpec], list[VariableSpec]]:
        """Split the variables into those the source can deliver and those it cannot.

        Args:
            source: The data source to interrogate.
            vars_in_src: Variables declared against that source.

        Returns:
            tuple: ``(present, missing)``, both in the input order.
        """
        available = set(source.list_tags())
        present = [v for v in vars_in_src if v.tag in available]
        missing = [v for v in vars_in_src if v.tag not in available]
        return present, missing

    @staticmethod
    def _handle_missing_tags(
        src_name: str,
        missing: list[VariableSpec],
        skip_missing: bool,
    ) -> None:
        """Warn about or reject the tags a source does not expose.

        Args:
            src_name: Name of the source, for the message.
            missing: Variables whose tags are absent.
            skip_missing: Warn and carry on when true, raise when false.

        Raises:
            KeyError: If ``skip_missing`` is false. Building a frame that quietly
                lacks columns the caller asked for is the worse failure mode.
        """
        names = ", ".join(f"'{v.name}'->'{v.tag}'" for v in missing)
        if skip_missing:
            warnings.warn(
                f"Source '{src_name}' is missing tags for variables: " f"{names}. Skipping.",
                stacklevel=4,
            )
        else:
            raise KeyError(
                f"Source '{src_name}' does not expose required tags: " f"{names}. Pass skip_missing=True to skip them."
            )

    @staticmethod
    def _process_column(series: pd.Series, var: VariableSpec) -> pd.Series:
        """Apply valid_range, unit conversion, and resampling to one column."""
        # Object dtype (mixed bool/NaN from TabularCSVSource) resamples
        # poorly. Cast to numeric: True->1.0, False->0.0, NaN->NaN.
        if series.dtype == object:
            series = pd.to_numeric(series, errors="coerce")

        # 1. Quality gate: drop out-of-range values BEFORE unit conversion,
        #    so the bounds in the schema stay in source units.
        if var.valid_range is not None:
            lo, hi = var.valid_range
            series = series.where((series >= lo) & (series <= hi))

        # 2. Unit conversion.
        if var.to_unit is not None and var.to_unit != var.unit:
            series = units.convert(series, var.unit, var.to_unit)

        # 3. Resample.
        resampled = series.resample(normalize_freq(var.resample.freq)).agg(var.resample.agg)
        return resampled

    @staticmethod
    def _normalize_ts(ts: str | datetime | None) -> pd.Timestamp | None:
        """Turn a window bound into a UTC timestamp, or pass ``None`` through.

        A naive input is read as UTC rather than as local time, so the same schema
        gives the same window on every machine.

        Args:
            ts: The bound, as a string, a datetime, or ``None`` for open-ended.

        Returns:
            pd.Timestamp | None: The bound in UTC.
        """
        if ts is None:
            return None
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t

    def _empty_result(
        self,
        selected: list[VariableSpec],
        start: str | datetime | None,
        end: str | datetime | None,
    ) -> MeasurementData:
        """An empty result that still carries the requested columns and metadata.

        Returned when no source yields a single row. Keeping the columns means callers
        can go on addressing them instead of guarding every access.

        Args:
            selected: The variables that were requested.
            start: Start of the requested window, kept for the metadata.
            end: End of the requested window, kept for the metadata.

        Returns:
            MeasurementData: A frame with zero rows and the expected columns.
        """
        idx = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        df = pd.DataFrame(index=idx, columns=[v.name for v in selected])
        return MeasurementData(
            data=df,
            metadata=self._build_metadata(df, selected, {}),
        )

    def _build_metadata(
        self,
        df: pd.DataFrame,
        selected: list[VariableSpec],
        vars_by_source: dict[str, list[VariableSpec]],
    ) -> dict[str, Any]:
        """Describe what was built, so a stored result can be traced back later.

        Args:
            df: The assembled frame, read for its time range.
            selected: The variables that went into it.
            vars_by_source: Which source contributed which variables.

        Returns:
            dict[str, Any]: Plant id, sources, variables, time range and build time.
        """
        if len(df) > 0:
            time_range = (df.index.min().isoformat(), df.index.max().isoformat())
        else:
            time_range = (None, None)
        return {
            "plant_id": self.schema.plant_id,
            "sources": sorted(vars_by_source.keys()),
            "variables": [v.name for v in selected],
            "time_range": time_range,
            "build_time": datetime.now(timezone.utc).isoformat(),
        }
