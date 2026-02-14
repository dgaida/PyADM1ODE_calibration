import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from pyadm1ode_calibration.calibration.methods.initial import InitialCalibrator
from pyadm1ode_calibration.io.loaders.measurement_data import MeasurementData
from pyadm1ode_calibration.calibration.core.result import CalibrationResult


@pytest.fixture
def mock_plant():
    plant = MagicMock()
    comp = MagicMock()
    comp.component_type.value = "digester"
    plant.components = {"d1": comp}
    return plant


@pytest.fixture
def sample_measurements():
    df = pd.DataFrame(
        {"Q_ch4": np.random.rand(10), "pH": np.random.rand(10) + 6.0}, index=pd.date_range("2024-01-01", periods=10, freq="h")
    )
    return MeasurementData(df)


class TestInitialCalibrator:
    def test_calibrate_full(self, mock_plant, sample_measurements):
        cal = InitialCalibrator(mock_plant, verbose=False)
        cal.simulator = MagicMock()
        cal.simulator.simulate_with_parameters.return_value = {"Q_ch4": np.array([1.0] * 10)}
        cal.validator = MagicMock()
        cal.validator.validate.return_value = {}

        # Test with weights and constraints
        res = cal.calibrate(
            sample_measurements,
            parameters=["k_dis"],
            objectives=["Q_ch4"],
            weights={"Q_ch4": 1.0},
            use_constraints=True,
            max_iterations=1,
            method="nelder_mead",
        )
        assert isinstance(res, CalibrationResult)

    def test_analysis_methods(self, mock_plant, sample_measurements):
        cal = InitialCalibrator(mock_plant, verbose=False)
        cal.simulator = MagicMock()
        cal.simulator.simulate_with_parameters.return_value = {"Q_ch4": np.array([1.0] * 10)}

        sens = cal.sensitivity_analysis({"k_dis": 0.5}, sample_measurements)
        assert "k_dis" in sens

        ident = cal.identifiability_analysis({"k_dis": 0.5}, sample_measurements)
        assert "k_dis" in ident

    def test_setup_bounds_edge_cases(self, mock_plant):
        cal = InitialCalibrator(mock_plant, verbose=False)
        # Parameter in default bounds
        bounds = cal._setup_bounds(["k_dis"], None)
        assert "k_dis" in bounds

        # Custom bounds
        bounds = cal._setup_bounds(["k_dis"], {"k_dis": (0.1, 0.2)})
        assert bounds["k_dis"] == (0.1, 0.2)
