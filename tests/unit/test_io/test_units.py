"""Tests for :mod:`pyadm1ode_calibration.io.loaders.units`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyadm1ode_calibration.io.loaders import units


class TestConvert:
    def test_celsius_to_kelvin(self) -> None:
        assert units.convert(25.0, "degC", "K") == pytest.approx(298.15)

    def test_kelvin_to_celsius(self) -> None:
        assert units.convert(298.15, "K", "degC") == pytest.approx(25.0)

    def test_kw_to_w(self) -> None:
        assert units.convert(2.5, "kW", "W") == pytest.approx(2500.0)

    def test_percent_to_fraction(self) -> None:
        assert units.convert(50.0, "%", "fraction") == pytest.approx(0.5)

    def test_mbar_to_pa(self) -> None:
        assert units.convert(1013.25, "mbar", "Pa") == pytest.approx(101325.0)

    def test_same_unit_passthrough(self) -> None:
        assert units.convert(42.0, "K", "K") == 42.0

    def test_unknown_conversion_raises(self) -> None:
        with pytest.raises(units.UnknownUnitError, match="No conversion"):
            units.convert(1.0, "K", "unobtainium")

    def test_numpy_array_roundtrip(self) -> None:
        x = np.array([0.0, 25.0, 100.0])
        y = units.convert(x, "degC", "K")
        np.testing.assert_allclose(y, [273.15, 298.15, 373.15])
        back = units.convert(y, "K", "degC")
        np.testing.assert_allclose(back, x)

    def test_pandas_series(self) -> None:
        s = pd.Series([0.0, 25.0])
        y = units.convert(s, "degC", "K")
        pd.testing.assert_series_equal(y, pd.Series([273.15, 298.15]))


class TestKnownUnit:
    def test_temperature_units(self) -> None:
        assert units.is_known_unit("K")
        assert units.is_known_unit("degC")

    def test_power_units(self) -> None:
        assert units.is_known_unit("kW")
        assert units.is_known_unit("W")
        assert units.is_known_unit("MW")

    def test_bool_dimensionless(self) -> None:
        assert units.is_known_unit("bool")
        assert units.is_known_unit("dimensionless")
        assert units.is_known_unit("fraction")

    def test_unknown(self) -> None:
        assert not units.is_known_unit("unobtainium")


class TestCanConvert:
    def test_same_unit_always_convertible(self) -> None:
        assert units.can_convert("anything", "anything")

    def test_known_pair(self) -> None:
        assert units.can_convert("degC", "K")
        assert units.can_convert("kW", "W")

    def test_unknown_pair(self) -> None:
        assert not units.can_convert("K", "foobar")


class TestRegisterConversion:
    def test_registered_pair_becomes_convertible(self) -> None:
        units.register_conversion("foo_unit", "bar_unit", scale=2.0, offset=1.0)
        try:
            assert units.can_convert("foo_unit", "bar_unit")
            assert units.is_known_unit("foo_unit")
            assert units.is_known_unit("bar_unit")
            assert units.convert(3.0, "foo_unit", "bar_unit") == pytest.approx(7.0)
        finally:
            # Clean up the global state so tests stay independent.
            units._CONVERSIONS.pop(("foo_unit", "bar_unit"), None)
            units._KNOWN_UNITS.discard("foo_unit")
            units._KNOWN_UNITS.discard("bar_unit")
