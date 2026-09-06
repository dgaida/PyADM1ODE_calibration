"""
Data source adapters for measurement input.

Each adapter converts one specific transport or format (CSV files,
databases, live feeds, …) into the uniform ``DataSource`` interface,
so downstream code (schema mapping, calibration) is independent of
where the data comes from.

To add support for a new format:

1. Implement the ``DataSource`` protocol from :mod:`.base`.
2. If the format is CSV-based, prefer adding a new preset to
   :data:`.tabular_csv.PRESETS` rather than writing a new class.
3. Export the new symbol from this ``__init__``.
"""

from .base import DataSource
from .tabular_csv import PRESETS, FileSpec, TabularCSVSource

__all__ = [
    "PRESETS",
    "DataSource",
    "FileSpec",
    "TabularCSVSource",
]
