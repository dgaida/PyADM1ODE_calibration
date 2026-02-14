# -*- coding: utf-8 -*-
"""
Unit tests

- Test CSV loading
- Test validation
- Test outlier detection
- Test gap filling
"""

import pandas as pd
import numpy as np
import pytest
from pyadm1ode_calibration import (
    MeasurementData,
    OutlierDetector,
)


def test_from_csv_loading(tmp_path):
    # Create sample CSV
    csv = tmp_path / "test.csv"
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="h"), "Q_ch4": [1, 2, 3, 4, 5]})
    df.to_csv(csv, index=False)

    data = MeasurementData.from_csv(str(csv))

    assert isinstance(data, MeasurementData)
    assert "Q_ch4" in data.data.columns
    assert isinstance(data.data.index, pd.DatetimeIndex)
    assert len(data.data) == 5


def test_validation_detects_missing_values():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="h"), "Q_ch4": [1, np.nan, 3, np.nan, 5]})

    data = MeasurementData(df)
    result = data.validate()

    assert not result.is_valid
    assert "Q_ch4" in result.missing_data
    assert result.missing_data["Q_ch4"] > 0


def test_outlier_detection_iqr():
    s = pd.Series([1, 2, 3, 100])  # 100 is an outlier
    outliers = OutlierDetector.detect_iqr(s, multiplier=1.5)

    assert outliers.iloc[-1]
    assert outliers.sum() == 1


def test_outlier_detection_moving_window():
    s = pd.Series([1, 2, 3, 4, 80, 2, 4])  # entry at index 4 is outlier
    outliers = OutlierDetector.detect_moving_window(s, window=5, threshold=1.5)
    assert outliers.iloc[4]


def test_remove_outliers_zscore():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="h"),
            "Q_ch4": [1, 2, 1000, 3, 4, 3],
        }
    )
    data = MeasurementData(df)

    removed = data.remove_outliers(method="zscore", threshold=2.0)

    assert removed == 1
    assert data.data["Q_ch4"].isna().sum() == 1


def test_fill_gaps_interpolate():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
            "Q_ch4": [1, np.nan, np.nan, 4, 5],
        }
    )
    data = MeasurementData(df)

    data.fill_gaps(method="interpolate", limit=2)

    assert data.data["Q_ch4"].isna().sum() == 0
    assert np.isclose(data.data["Q_ch4"].iloc[1], 2.0, atol=0.1)


def test_get_measurement():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
            "pH": [7.0, 7.1, 7.2],
        }
    )
    data = MeasurementData(df)

    pH_series = data.get_measurement("pH")

    assert len(pH_series) == 3
    assert list(pH_series.values) == [7.0, 7.1, 7.2]


def test_get_time_window():
    timestamps = pd.date_range("2024-01-01", periods=10, freq="h")
    df = pd.DataFrame({"timestamp": timestamps, "Q_ch4": range(10)})
    data = MeasurementData(df)

    window = data.get_time_window(timestamps[3], timestamps[6])

    assert len(window.data) == 4
    assert window.data["Q_ch4"].iloc[0] == 3

def test_measurement_data_extra():
    df = pd.DataFrame({
        "Q_sub1": [1, 2, 3],
        "Q_sub2": [4, 5, 6],
        "other": [7, 8, 9]
    }, index=pd.date_range("2024-01-01", periods=3, freq="h"))
    data = MeasurementData(df)

    feeds = data.get_substrate_feeds()
    assert feeds.shape == (3, 2)

    data.resample("1d", aggregation="sum")
    assert len(data.data) == 1

    sum_df = data.summary()
    assert "Q_sub1" in sum_df.columns
    assert "MeasurementData" in repr(data)

def test_fill_gaps_methods():
    df = pd.DataFrame({
        "A": [1, np.nan, 3]
    }, index=pd.date_range("2024-01-01", periods=3, freq="h"))

    data = MeasurementData(df.copy())
    data.fill_gaps(method="mean")
    assert data.data["A"].iloc[1] == 2.0

    data = MeasurementData(df.copy())
    data.fill_gaps(method="median")
    assert data.data["A"].iloc[1] == 2.0

    data = MeasurementData(df.copy())
    data.fill_gaps(method="forward")
    assert data.data["A"].iloc[1] == 1.0

def test_to_csv(tmp_path):
    csv = tmp_path / "out.csv"
    df = pd.DataFrame({"A": [1, 2]}, index=pd.date_range("2024-01-01", periods=2, freq="h"))
    data = MeasurementData(df)
    data.to_csv(str(csv))
    assert csv.exists()
