"""
Unit conversion for plant measurements.

Conversions are affine: ``y_target = scale * x_source + offset``.
Temperature is the only common case that needs a non-zero offset; all
other registered conversions are purely multiplicative.

The table below covers the units that occur in biogas plant historians.
Add new conversions via :func:`register_conversion`; both directions
must be registered explicitly, which keeps the table self-documenting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Numeric = float | int | np.ndarray | pd.Series


class UnknownUnitError(ValueError):
    """Raised when a requested unit conversion is not registered."""


# (source_unit, target_unit) -> (scale, offset)
# y_target = scale * x_source + offset
_CONVERSIONS: dict[tuple[str, str], tuple[float, float]] = {
    # Temperature
    ("degC", "K"): (1.0, 273.15),
    ("K", "degC"): (1.0, -273.15),
    # Electrical / thermal power
    ("W", "kW"): (1e-3, 0.0),
    ("kW", "W"): (1e3, 0.0),
    ("kW", "MW"): (1e-3, 0.0),
    ("MW", "kW"): (1e3, 0.0),
    # Pressure
    ("Pa", "kPa"): (1e-3, 0.0),
    ("kPa", "Pa"): (1e3, 0.0),
    ("Pa", "mbar"): (1e-2, 0.0),
    ("mbar", "Pa"): (1e2, 0.0),
    ("bar", "mbar"): (1e3, 0.0),
    ("mbar", "bar"): (1e-3, 0.0),
    ("bar", "Pa"): (1e5, 0.0),
    ("Pa", "bar"): (1e-5, 0.0),
    # Volumetric flow
    ("m3_per_h", "m3_per_d"): (24.0, 0.0),
    ("m3_per_d", "m3_per_h"): (1.0 / 24.0, 0.0),
    ("m3_per_s", "m3_per_h"): (3600.0, 0.0),
    ("m3_per_h", "m3_per_s"): (1.0 / 3600.0, 0.0),
    ("L_per_h", "m3_per_h"): (1e-3, 0.0),
    ("m3_per_h", "L_per_h"): (1e3, 0.0),
    # Mass
    ("g", "kg"): (1e-3, 0.0),
    ("kg", "g"): (1e3, 0.0),
    ("t", "kg"): (1e3, 0.0),
    ("kg", "t"): (1e-3, 0.0),
    # Dimensionless
    ("%", "fraction"): (1e-2, 0.0),
    ("fraction", "%"): (1e2, 0.0),
}


# Units that pass through without conversion. Listed here so validation
# accepts them even when no conversion is requested.
_KNOWN_UNITS: set[str] = {
    "bool",
    "dimensionless",
    "count",
    # Temperature
    "K",
    "degC",
    # Power
    "W",
    "kW",
    "MW",
    # Pressure
    "Pa",
    "kPa",
    "mbar",
    "bar",
    # Geometry
    "m",
    "m2",
    "m3",
    # Mass
    "g",
    "kg",
    "t",
    # Time
    "s",
    "min",
    "h",
    "d",
    # Electrical
    "A",
    "V",
    # Dimensionless
    "fraction",
    "%",
    # Flow
    "m3_per_h",
    "m3_per_d",
    "m3_per_s",
    "L_per_h",
}


def convert(value: Numeric, source_unit: str, target_unit: str) -> Numeric:
    """Convert ``value`` from ``source_unit`` to ``target_unit``.

    Args:
        value: Scalar, NumPy array, or pandas Series. Returned as the
            same type.
        source_unit: Unit of the input value.
        target_unit: Desired output unit.

    Returns:
        Value in ``target_unit``.

    Raises:
        UnknownUnitError: When the conversion is not registered.
    """
    if source_unit == target_unit:
        return value
    key = (source_unit, target_unit)
    if key not in _CONVERSIONS:
        raise UnknownUnitError(
            f"No conversion registered from '{source_unit}' to " f"'{target_unit}'. Use register_conversion() to add one."
        )
    scale, offset = _CONVERSIONS[key]
    return value * scale + offset


def is_known_unit(unit: str) -> bool:
    """True if ``unit`` is documented as a valid unit name."""
    return unit in _KNOWN_UNITS


def can_convert(source_unit: str, target_unit: str) -> bool:
    """True if a conversion from ``source_unit`` to ``target_unit`` exists."""
    if source_unit == target_unit:
        return True
    return (source_unit, target_unit) in _CONVERSIONS


def register_conversion(
    source_unit: str,
    target_unit: str,
    scale: float,
    offset: float = 0.0,
) -> None:
    """Register an affine conversion ``y = scale * x + offset``.

    Conversions are directional: register both directions if both are
    needed. Both unit names are added to the set of known units.
    """
    _CONVERSIONS[(source_unit, target_unit)] = (scale, offset)
    _KNOWN_UNITS.add(source_unit)
    _KNOWN_UNITS.add(target_unit)
