import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """
    Result from data validation.

    Attributes:
        is_valid: Overall validation status
        quality_score: Overall quality score (0-1)
        issues: List of identified issues
        warnings: List of warnings
        statistics: Dictionary of data statistics
        missing_data: Dictionary of missing data percentages per column
    """

    is_valid: bool
    quality_score: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    missing_data: Dict[str, float] = field(default_factory=dict)

    def print_report(self) -> None:
        """Print validation report."""
        print("=" * 70)
        print("Data Validation Report")
        print("=" * 70)
        print(f"Status: {'✓ Valid' if self.is_valid else '✗ Invalid'}")
        print(f"Quality Score: {self.quality_score:.2f}")

        if self.issues:
            print(f"\nIssues ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  - {issue}")

        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if self.missing_data:
            print("\nMissing Data:")
            for col, pct in self.missing_data.items():
                if pct > 0:
                    print(f"  {col}: {pct:.1f}%")

        print("=" * 70)


class DataValidator:
    """
    Validator for biogas plant measurement data.

    Checks data quality, identifies issues, and provides statistics.
    """

    @staticmethod
    def validate(
        data: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        expected_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> ValidationResult:
        """
        Validate measurement data.

        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            expected_ranges: Dictionary mapping columns to (min, max) tuples

        Returns:
            ValidationResult object
        """
        issues = []
        warnings_list = []

        # Check for required columns
        if required_columns:
            missing_cols = set(required_columns) - set(data.columns)
            if missing_cols:
                issues.append(f"Missing required columns: {missing_cols}")

        # Calculate missing data percentages
        missing_data = {}
        for col in data.columns:
            pct_missing = (data[col].isna().sum() / len(data)) * 100
            missing_data[col] = pct_missing

            if pct_missing > 30:
                issues.append(f"Column '{col}' has {pct_missing:.1f}% missing data")
            elif pct_missing > 5:
                warnings_list.append(f"Column '{col}' has {pct_missing:.1f}% missing data")

        # Check for expected ranges
        if expected_ranges:
            for col, (min_val, max_val) in expected_ranges.items():
                if col in data.columns:
                    values = data[col].dropna()
                    if len(values) > 0:
                        actual_min = values.min()
                        actual_max = values.max()

                        if actual_min < min_val or actual_max > max_val:
                            warnings_list.append(
                                f"Column '{col}' has values outside expected range "
                                f"[{min_val}, {max_val}]: actual [{actual_min:.2f}, {actual_max:.2f}]"
                            )

        # Check for duplicate timestamps
        if "timestamp" in data.columns:
            duplicates = data["timestamp"].duplicated().sum()
            if duplicates > 0:
                warnings_list.append(f"Found {duplicates} duplicate timestamps")

        # Calculate statistics
        statistics = {
            "n_rows": len(data),
            "n_columns": len(data.columns),
            "total_missing": data.isna().sum().sum(),
            "pct_missing": (data.isna().sum().sum() / (len(data) * len(data.columns))) * 100,
        }

        # Calculate quality score
        quality_score = DataValidator._calculate_quality_score(data, len(issues), len(warnings_list), statistics)

        is_valid = len(issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            warnings=warnings_list,
            statistics=statistics,
            missing_data=missing_data,
        )

    @staticmethod
    def _calculate_quality_score(data: pd.DataFrame, n_issues: int, n_warnings: int, statistics: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-1)."""
        score = 1.0

        # Penalize for issues
        score -= min(0.5, n_issues * 0.1)

        # Penalize for warnings
        score -= min(0.3, n_warnings * 0.05)

        # Penalize for missing data
        pct_missing = statistics["pct_missing"]
        score -= min(0.2, pct_missing / 100 * 0.5)

        return max(0.0, score)


class OutlierDetector:
    """
    Outlier detection for time series data.

    Supports multiple detection methods for identifying anomalous values.
    """

    @staticmethod
    def detect_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers using z-score method.

        Args:
            series: Pandas Series
            threshold: Z-score threshold

        Returns:
            Boolean Series indicating outliers
        """
        if len(series.dropna()) < 2:
            return pd.Series([False] * len(series), index=series.index)

        mean = series.mean()
        std = series.std()

        if std == 0:
            return pd.Series([False] * len(series), index=series.index)

        z_scores = np.abs((series - mean) / std)
        return z_scores > threshold

    @staticmethod
    def detect_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """
        Detect outliers using IQR (Interquartile Range) method.

        Args:
            series: Pandas Series
            multiplier: IQR multiplier (1.5 for outliers, 3.0 for extreme)

        Returns:
            Boolean Series indicating outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        return (series < lower_bound) | (series > upper_bound)

    @staticmethod
    def detect_moving_window(series: pd.Series, window: int = 5, threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers using moving window method.

        Args:
            series: Pandas Series
            window: Window size for rolling statistics
            threshold: Number of standard deviations

        Returns:
            Boolean Series indicating outliers
        """
        rolling_mean = series.rolling(window=window, center=True).mean()
        rolling_std = series.rolling(window=window, center=True).std()

        z_scores = np.abs((series - rolling_mean) / rolling_std)
        return z_scores > threshold
