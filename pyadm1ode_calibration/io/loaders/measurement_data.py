# pyadm1ode_calibration/io/loaders/measurement_data.py
"""
Measurement Data Management for Biogas Plant Calibration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from ..validation.validators import DataValidator, OutlierDetector, ValidationResult


class MeasurementData:
    """
    Container for biogas plant measurement data.
    """

    def __init__(self, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize measurement data.

        Args:
            data: DataFrame with measurements
            metadata: Optional metadata dictionary
        """
        self.data = data
        self.metadata = metadata or {}

        if "timestamp" in self.data.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.data["timestamp"]):
                self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
            self.data = self.data.set_index("timestamp").sort_index()

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        timestamp_column: str = "timestamp",
        sep: str = ",",
        parse_dates: bool = True,
        resample: Optional[str] = None,
        **kwargs: Any,
    ) -> "MeasurementData":
        """Load measurement data from CSV file."""
        data = pd.read_csv(filepath, sep=sep, **kwargs)
        if timestamp_column in data.columns:
            data["timestamp"] = pd.to_datetime(data[timestamp_column])
            if timestamp_column != "timestamp":
                data = data.drop(columns=[timestamp_column])
        instance = cls(data)
        if resample is not None:
            instance.resample(resample)
        return instance

    def validate(
        self,
        required_columns: Optional[List[str]] = None,
        expected_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> ValidationResult:
        """Validate measurement data."""
        if expected_ranges is None:
            expected_ranges = {
                "pH": (5.0, 9.0),
                "VFA": (0.0, 20.0),
                "TAC": (0.0, 50.0),
                "Q_gas": (0.0, 5000.0),
                "Q_ch4": (0.0, 3000.0),
                "T_digester": (273.15, 333.15),
            }
        return DataValidator.validate(self.data, required_columns=required_columns, expected_ranges=expected_ranges)

    def remove_outliers(
        self, columns: Optional[List[str]] = None, method: str = "zscore", threshold: float = 3.0, **kwargs: Any
    ) -> int:
        """Remove outliers from specified columns."""
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        n_outliers = 0
        for col in columns:
            if col not in self.data.columns:
                continue
            if method == "zscore":
                is_outlier = OutlierDetector.detect_zscore(self.data[col], threshold=threshold)
            elif method == "iqr":
                is_outlier = OutlierDetector.detect_iqr(self.data[col], multiplier=threshold)
            elif method == "moving_window":
                window = kwargs.get("window", 5)
                is_outlier = OutlierDetector.detect_moving_window(self.data[col], window=window, threshold=threshold)
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")

            n_col_outliers = int(is_outlier.sum())
            self.data.loc[is_outlier, col] = np.nan
            n_outliers += n_col_outliers
        return n_outliers

    def fill_gaps(self, columns: Optional[List[str]] = None, method: str = "interpolate", **kwargs: Any) -> None:
        """Fill missing values in time series."""
        if columns is None:
            columns = self.data.columns.tolist()

        for col in columns:
            if col not in self.data.columns:
                continue
            if method == "interpolate":
                limit = kwargs.get("limit", None)
                self.data[col] = self.data[col].interpolate(method="linear", limit=limit)
            elif method == "forward":
                limit = kwargs.get("limit", None)
                self.data[col] = self.data[col].ffill(limit=limit)
            elif method == "backward":
                limit = kwargs.get("limit", None)
                self.data[col] = self.data[col].bfill(limit=limit)
            elif method == "mean":
                self.data[col] = self.data[col].fillna(self.data[col].mean())
            elif method == "median":
                self.data[col] = self.data[col].fillna(self.data[col].median())
            else:
                raise ValueError(f"Unknown fill method: {method}")

    def resample(self, freq: str, aggregation: str = "mean") -> None:
        """Resample time series."""
        resampler = self.data.resample(freq)
        if aggregation == "mean": self.data = resampler.mean()
        elif aggregation == "sum": self.data = resampler.sum()
        elif aggregation == "first": self.data = resampler.first()
        elif aggregation == "last": self.data = resampler.last()
        else: raise ValueError(f"Unknown aggregation method: {aggregation}")

    def get_measurement(
        self, column: str, start_time: Optional[Union[str, datetime]] = None, end_time: Optional[Union[str, datetime]] = None
    ) -> pd.Series:
        """Get measurement time series."""
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found")
        series = self.data[column]
        if start_time is not None or end_time is not None:
            series = series.loc[start_time:end_time] # type: ignore
        return series

    def get_substrate_feeds(self, substrate_columns: Optional[List[str]] = None) -> np.ndarray:
        """Get substrate feed rates as array."""
        if substrate_columns is None:
            substrate_columns = [col for col in self.data.columns if col.startswith("Q_sub")]
        if not substrate_columns:
            raise ValueError("No substrate columns found")
        return self.data[substrate_columns].values

    def get_time_window(self, start_time: Union[str, datetime], end_time: Union[str, datetime]) -> "MeasurementData":
        """Get data for specific time window."""
        windowed_data = self.data.loc[start_time:end_time].copy() # type: ignore
        return MeasurementData(windowed_data, metadata=self.metadata.copy())

    def summary(self) -> pd.DataFrame:
        """Get statistical summary of measurements."""
        return self.data.describe()

    def to_csv(self, filepath: str, **kwargs: Any) -> None:
        """Save measurement data to CSV."""
        self.data.to_csv(filepath, **kwargs)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        try:
            time_range = f"{self.data.index[0]} to {self.data.index[-1]}"
        except Exception:
            time_range = "empty"
        return f"MeasurementData(n_rows={len(self.data)}, n_columns={len(self.data.columns)}, time_range={time_range})"
