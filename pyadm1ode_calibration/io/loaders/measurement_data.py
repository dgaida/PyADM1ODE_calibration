"""Measurement data module."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from ..validation.validators import DataValidator, OutlierDetector, ValidationResult


class MeasurementData:
    """
    Container for biogas plant measurement data.

    This class manages time-series data from biogas plants, providing
    methods for loading, validation, cleaning (outlier removal),
    and pre-processing (gap filling, resampling).

    Args:
        data (pd.DataFrame): DataFrame containing measurements. If a 'timestamp'
            column exists, it will be converted to datetime and used as the index.
        metadata (Optional[Dict[str, Any]]): Optional dictionary containing
            contextual information (e.g., plant ID, location).
    """

    def __init__(self, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None):
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
        """
        Load measurement data from a CSV file.

        Args:
            filepath (str): Path to the CSV file.
            timestamp_column (str): Name of the column containing time information.
            sep (str): CSV delimiter. Defaults to ','.
            parse_dates (bool): Whether to parse dates. Defaults to True.
            resample (Optional[str]): Frequency string to resample to (e.g., '1h').
            **kwargs (Any): Additional arguments passed to pd.read_csv.

        Returns:
            MeasurementData: A new instance with the loaded data.
        """
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
        self, required_columns: Optional[List[str]] = None, expected_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> ValidationResult:
        """
        Validate measurement data against schema and range expectations.

        Args:
            required_columns (Optional[List[str]]): Columns that must be present.
            expected_ranges (Optional[Dict[str, Tuple[float, float]]]):
                Mapping of column names to (min, max) range tuples.

        Returns:
            ValidationResult: Result of the validation checks.
        """
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
        """
        Detect and remove outliers from specified columns.

        Outliers are replaced with NaN.

        Args:
            columns (Optional[List[str]]): Columns to check. Defaults to all numeric.
            method (str): Detection method ('zscore', 'iqr', 'moving_window').
            threshold (float): Threshold for outlier detection.
            **kwargs (Any): Additional arguments for the detection method.

        Returns:
            int: Total number of outliers removed.
        """
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
        """
        Fill missing values (NaNs) in the data.

        Args:
            columns (Optional[List[str]]): Columns to fill.
            method (str): Fill method ('interpolate', 'forward', 'backward', 'mean', 'median').
            **kwargs (Any): Additional arguments for filling (e.g., 'limit').
        """
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
        """
        Resample the time series data to a new frequency.

        Args:
            freq (str): Frequency string (e.g., '1h', '1d').
            aggregation (str): Aggregation function ('mean', 'sum', 'first', 'last').
        """
        resampler = self.data.resample(freq)
        if aggregation == "mean":
            self.data = resampler.mean()
        elif aggregation == "sum":
            self.data = resampler.sum()
        elif aggregation == "first":
            self.data = resampler.first()
        elif aggregation == "last":
            self.data = resampler.last()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    def get_measurement(
        self, column: str, start_time: Optional[Union[str, datetime]] = None, end_time: Optional[Union[str, datetime]] = None
    ) -> pd.Series:
        """
        Get a specific measurement series, optionally windowed.

        Args:
            column (str): Name of the measurement column.
            start_time (Optional[datetime]): Start of window.
            end_time (Optional[datetime]): End of window.

        Returns:
            pd.Series: The requested time series.
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found")
        series = self.data[column]
        if start_time is not None or end_time is not None:
            series = series.loc[start_time:end_time]  # type: ignore
        return series

    def get_substrate_feeds(self, substrate_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract substrate feed rates as a 2D numpy array.

        Args:
            substrate_columns (Optional[List[str]]): Column names for substrates.

        Returns:
            np.ndarray: Matrix of feed rates.
        """
        if substrate_columns is None:
            substrate_columns = [col for col in self.data.columns if col.startswith("Q_sub")]
        if not substrate_columns:
            raise ValueError("No substrate columns found")
        return self.data[substrate_columns].values

    def get_time_window(self, start_time: Union[str, datetime], end_time: Union[str, datetime]) -> "MeasurementData":
        """
        Create a new MeasurementData instance for a specific time window.

        Args:
            start_time (Union[str, datetime]): Start timestamp.
            end_time (Union[str, datetime]): End timestamp.

        Returns:
            MeasurementData: A subset of the data.
        """
        windowed_data = self.data.loc[start_time:end_time].copy()  # type: ignore
        return MeasurementData(windowed_data, metadata=self.metadata.copy())

    def summary(self) -> pd.DataFrame:
        """
        Get a statistical summary of all measurement columns.

        Returns:
            pd.DataFrame: Descriptive statistics.
        """
        return self.data.describe()

    def to_csv(self, filepath: str, **kwargs: Any) -> None:
        """
        Save the current data to a CSV file.

        Args:
            filepath (str): Destination path.
            **kwargs (Any): Passed to pd.DataFrame.to_csv.
        """
        self.data.to_csv(filepath, **kwargs)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        try:
            time_range = f"{self.data.index[0]} to {self.data.index[-1]}"
        except Exception:
            time_range = "empty"
        return f"MeasurementData(n_rows={len(self.data)}, n_columns={len(self.data.columns)}, time_range={time_range})"
