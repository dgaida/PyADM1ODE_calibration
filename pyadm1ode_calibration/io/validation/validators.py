"""Validators module."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """
    Result of data validation.

    Attributes:
        is_valid: Overall validation status
        quality_score: Score from 0 to 1
        issues: List of issues found
        warnings: List of minor warnings
        missing_data: Mapping of column names to missing count
        statistics: Summary statistics
    """

    is_valid: bool
    quality_score: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_data: Dict[str, int] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def print_report(self) -> None:
        """Print formatted validation report."""
        print("\n" + "=" * 40)
        print("DATA VALIDATION REPORT")
        print("=" * 40)
        print(f"Status: {'VALID' if self.is_valid else 'INVALID'}")
        print(f"Quality Score: {self.quality_score:.2f}")

        if self.issues:
            print("\nIssues:")
            for issue in self.issues:
                print(f"  - {issue}")

        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  - {warning}")

        print("=" * 40)


class DataValidator:
    """
    Validates measurement data quality and consistency.
    """

    @staticmethod
    def validate(
        data: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        expected_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> ValidationResult:
        """
        Perform comprehensive data validation.

        Args:
            data: DataFrame to validate
            required_columns: Columns that must be present
            expected_ranges: Mapping of column names to (min, max)

        Returns:
            ValidationResult object
        """
        issues = []
        warnings = []
        missing_counts = data.isnull().sum().to_dict()
        missing_pct = data.isnull().mean()

        # 1. Check required columns
        if required_columns:
            for col in required_columns:
                if col not in data.columns:
                    issues.append(f"Required column '{col}' is missing")

        # 2. Check for empty data
        if data.empty:
            issues.append("Dataset is empty")
            return ValidationResult(is_valid=False, quality_score=0.0, issues=issues, missing_data=missing_counts)

        # 3. Check for missing values (NaN)
        for col, pct in missing_pct.items():
            if pct > 0.3:
                issues.append(f"Column '{col}' has {pct*100:.1f}% missing values")
            elif pct > 0:
                warnings.append(f"Column '{col}' has {pct*100:.1f}% missing values")

        # 4. Check expected ranges
        if expected_ranges:
            for col, (vmin, vmax) in expected_ranges.items():
                if col in data.columns:
                    out_of_range = (data[col] < vmin) | (data[col] > vmax)
                    n_out = out_of_range.sum()
                    if n_out > 0:
                        pct = n_out / len(data)
                        if pct > 0.2:
                            issues.append(f"Column '{col}' has {n_out} values outside range [{vmin}, {vmax}]")
                        else:
                            warnings.append(f"Column '{col}' has {n_out} values outside range [{vmin}, {vmax}]")

        # 5. Check for duplicates in index (timestamps)
        if data.index.duplicated().any():
            issues.append("Dataset has duplicate timestamps")

        # Calculate quality score
        quality_score = DataValidator._calculate_quality_score(len(issues), len(warnings), missing_pct.mean())

        return ValidationResult(
            is_valid=len(issues) == 0,
            quality_score=quality_score,
            issues=issues,
            warnings=warnings,
            missing_data=missing_counts,
            statistics={"missing_pct_avg": float(missing_pct.mean())},
        )

    @staticmethod
    def _calculate_quality_score(n_issues: int, n_warnings: int, avg_missing: float) -> float:
        """Calculate quality score from 0 to 1."""
        score = 1.0 - (n_issues * 0.2) - (n_warnings * 0.05) - (avg_missing * 0.5)
        return float(max(0.0, min(1.0, score)))


class OutlierDetector:
    """
    Detects outliers in measurement data using various methods.
    """

    @staticmethod
    def detect_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using Z-score."""
        z = np.abs((series - series.mean()) / (series.std() + 1e-10))
        return z > threshold

    @staticmethod
    def detect_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """Detect outliers using Interquartile Range."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        return (series < lower) | (series > upper)

    @staticmethod
    def detect_moving_window(series: pd.Series, window: int = 5, threshold: float = 3.0) -> pd.Series:
        """Detect outliers using a moving window median."""
        median = series.rolling(window=window, center=True).median()
        std = series.rolling(window=window, center=True).std()
        z = np.abs((series - median) / (std + 1e-10))
        return z > threshold
