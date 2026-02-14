import pytest
import pandas as pd
import numpy as np
from pyadm1ode_calibration.io.loaders.measurement_data import MeasurementData
import os

def test_measurement_data_init_with_string_timestamp():
    df = pd.DataFrame({"timestamp": ["2024-01-01", "2024-01-02"], "val": [1, 2]})
    data = MeasurementData(df)
    assert isinstance(data.data.index, pd.DatetimeIndex)

def test_from_csv_custom_timestamp(tmp_path):
    csv = tmp_path / "test.csv"
    df = pd.DataFrame({"time": ["2024-01-01 00:00", "2024-01-01 01:00"], "val": [1, 2]})
    df.to_csv(csv, index=False)
    data = MeasurementData.from_csv(str(csv), timestamp_column="time")
    assert isinstance(data.data.index, pd.DatetimeIndex)

def test_outlier_methods():
    df = pd.DataFrame({"A": [1, 2, 100, 4, 5]}, index=pd.date_range("2024-01-01", periods=5, freq="h"))
    data = MeasurementData(df)

    # IQR
    data_iqr = MeasurementData(df.copy())
    n = data_iqr.remove_outliers(method="iqr", threshold=1.5)
    assert n > 0

    # Moving window
    data_mw = MeasurementData(df.copy())
    n = data_mw.remove_outliers(method="moving_window", threshold=1.0, window=3)
    assert n > 0

    with pytest.raises(ValueError, match="Unknown outlier detection method"):
        data.remove_outliers(method="invalid")

def test_fill_gaps_more_methods():
    df = pd.DataFrame({"A": [1, np.nan, 3]}, index=pd.date_range("2024-01-01", periods=3, freq="h"))

    # Backward
    data = MeasurementData(df.copy())
    data.fill_gaps(method="backward")
    assert data.data["A"].iloc[1] == 3.0

    with pytest.raises(ValueError, match="Unknown fill method"):
        data.fill_gaps(method="invalid")

def test_resample_aggregations():
    df = pd.DataFrame({"A": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3, freq="h"))
    data = MeasurementData(df)

    data_sum = MeasurementData(df.copy())
    data_sum.resample("3h", aggregation="sum")
    assert data_sum.data["A"].iloc[0] == 6

    data_first = MeasurementData(df.copy())
    data_first.resample("3h", aggregation="first")
    assert data_first.data["A"].iloc[0] == 1

    data_last = MeasurementData(df.copy())
    data_last.resample("3h", aggregation="last")
    assert data_last.data["A"].iloc[0] == 3

    with pytest.raises(ValueError, match="Unknown aggregation method"):
        data.resample("3h", aggregation="invalid")

def test_get_measurement_with_time():
    df = pd.DataFrame({"A": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3, freq="h"))
    data = MeasurementData(df)
    series = data.get_measurement("A", start_time="2024-01-01 01:00")
    assert len(series) == 2

def test_get_substrate_feeds_error():
    data = MeasurementData(pd.DataFrame({"A": [1]}))
    with pytest.raises(ValueError, match="No substrate columns found"):
        data.get_substrate_feeds()

def test_repr_empty():
    data = MeasurementData(pd.DataFrame())
    assert "empty" in repr(data)
