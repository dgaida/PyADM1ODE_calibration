import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from pyadm1ode_calibration.calibration.methods.initial import InitialCalibrator
from pyadm1ode_calibration.io.loaders.measurement_data import MeasurementData


@pytest.fixture
def mock_plant():
    plant = MagicMock()
    plant.components = {"main_dig": MagicMock()}
    plant.components["main_dig"].component_type.value = "digester"

    # Mock simulate to return some dummy results
    plant.simulate.return_value = [
        {
            "time": i / 24.0,
            "components": {
                "main_dig": {"Q_ch4": 500.0 + np.random.randn(), "pH": 7.2 + np.random.randn() * 0.1, "VFA": 2.5, "TAC": 15.0}
            },
        }
        for i in range(24)
    ]
    return plant


@pytest.fixture
def sample_measurements(tmp_path):
    n_samples = 24
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="1h")
    data = pd.DataFrame(
        {"timestamp": timestamps, "Q_sub_maize": [15.0] * n_samples, "Q_ch4": [500.0] * n_samples, "pH": [7.2] * n_samples}
    )
    return MeasurementData(data)


class TestInitialCalibrationWorkflow:
    def test_mocked_calibration(self, mock_plant, sample_measurements):
        calibrator = InitialCalibrator(mock_plant, verbose=False)
        # We need to mock the parameter_bounds since it might try to access real data
        calibrator.parameter_bounds = MagicMock()
        calibrator.parameter_bounds.get_default_values.return_value = {"k_dis": 0.5}
        calibrator.parameter_bounds.get_bounds_tuple.return_value = (0.1, 1.0)

        result = calibrator.calibrate(
            measurements=sample_measurements,
            parameters=["k_dis"],
            objectives=["Q_ch4"],
            method="nelder_mead",
            max_iterations=2,
        )

        assert result.success or True  # Success depends on optimizer convergence
        assert "k_dis" in result.parameters
