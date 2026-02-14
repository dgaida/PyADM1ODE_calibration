import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from pyadm1ode_calibration.calibration.validation import (
    CalibrationValidator,
    ValidationMetrics,
    ResidualAnalysis,
    ParameterCorrelation
)
from pyadm1ode_calibration.io.loaders.measurement_data import MeasurementData
from pyadm1ode_calibration.exceptions import DataValidationError

@pytest.fixture
def mock_plant():
    plant = MagicMock()
    plant.simulate.return_value = [
        {
            "components": {
                "digester1": {"Q_ch4": 1.0, "pH": 7.0, "VFA": 100.0, "TAC": 5.0}
            }
        },
        {
            "components": {
                "digester1": {"Q_ch4": 1.1, "pH": 7.1, "VFA": 110.0, "TAC": 5.1}
            }
        }
    ]
    comp = MagicMock()
    comp.component_type.value = "digester"
    plant.components = {"digester1": comp}
    return plant

@pytest.fixture
def sample_measurements():
    df = pd.DataFrame({
        "Q_ch4": [1.05, 1.08],
        "pH": [7.05, 7.08],
        "VFA": [105, 108]
    }, index=pd.date_range("2024-01-01", periods=2, freq="h"))
    return MeasurementData(df)

class TestCalibrationValidator:
    def test_init(self, mock_plant):
        validator = CalibrationValidator(mock_plant, verbose=False)
        assert validator.plant == mock_plant

    def test_validate(self, mock_plant, sample_measurements):
        validator = CalibrationValidator(mock_plant, verbose=False)
        parameters = {"k_dis": 0.5}
        metrics = validator.validate(parameters, sample_measurements, objectives=["Q_ch4", "pH"])
        assert "Q_ch4" in metrics
        assert "pH" in metrics

    def test_validate_missing_objective(self, mock_plant, sample_measurements):
        validator = CalibrationValidator(mock_plant, verbose=False)
        with pytest.warns(UserWarning, match="Objective 'MissingObj' not in simulation outputs"):
            metrics = validator.validate({"k_dis": 0.5}, sample_measurements, objectives=["MissingObj"])
            assert "MissingObj" not in metrics

    def test_analyze_residuals(self, mock_plant, sample_measurements):
        validator = CalibrationValidator(mock_plant, verbose=False)
        simulated = {
            "Q_ch4": np.array([1.0, 1.1, 1.2]),
            "pH": np.array([7.0, 7.1, 7.2])
        }
        df = pd.DataFrame({
            "Q_ch4": [1.05, 1.08, 1.15],
            "pH": [7.05, 7.08, 7.15]
        }, index=pd.date_range("2024-01-01", periods=3, freq="h"))
        measurements = MeasurementData(df)
        results = validator.analyze_residuals(measurements, simulated)
        assert "Q_ch4" in results

    def test_cross_validate(self, mock_plant):
        validator = CalibrationValidator(mock_plant, verbose=False)
        df = pd.DataFrame({
            "Q_ch4": np.random.rand(10),
            "pH": np.random.rand(10),
            "VFA": np.random.rand(10)
        }, index=pd.date_range("2024-01-01", periods=10, freq="h"))
        measurements = MeasurementData(df)
        mock_plant.simulate.return_value = [
            {"components": {"d": {"Q_ch4": 0.5, "pH": 7.0, "VFA": 100}}}
        ] * 5
        results = validator.cross_validate({"k_dis": 0.5}, measurements, n_folds=2)
        assert "Q_ch4" in results
        assert len(results["Q_ch4"]) == 2

    def test_extract_measurements_error(self, mock_plant):
        validator = CalibrationValidator(mock_plant, verbose=False)
        measurements = MeasurementData(pd.DataFrame({"A": [1, 2]}))
        with pytest.raises(DataValidationError, match="Objective 'B' not found in measurements"):
            validator._extract_measurements(measurements, "B")

    def test_to_dict(self):
        m = ValidationMetrics(
            objective="test", n_samples=1, rmse=0.1, mae=0.1, r2=0.9, nse=0.9,
            pbias=1.0, correlation=0.99, mape=5.0, me=0.01,
            observations_mean=1.0, observations_std=0.1,
            predictions_mean=1.0, predictions_std=0.1
        )
        d = m.to_dict()
        assert d["objective"] == "test"

class TestParameterCorrelation:
    def test_parameter_correlation(self):
        matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
        names = ["p1", "p2"]
        corr = ParameterCorrelation(matrix, names)
        assert corr.get_correlation("p1", "p2") == 0.8
