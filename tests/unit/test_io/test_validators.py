import pytest
import pandas as pd
import numpy as np
from pyadm1ode_calibration.io.validation.validators import (
    DataValidator, OutlierDetector, ValidationResult
)

class TestDataValidator:
    def test_validate(self):
        df = pd.DataFrame({
            "pH": [7.0, 7.1, np.nan, 7.2],
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h")
        })
        res = DataValidator.validate(df, required_columns=["pH"])
        assert res.is_valid

class TestOutlierDetector:
    def test_detect_zscore(self):
        s = pd.Series([1.0] * 10 + [100.0])
        outliers = OutlierDetector.detect_zscore(s, threshold=2.0)
        assert outliers[10] == True
