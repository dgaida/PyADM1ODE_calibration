import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from datetime import datetime, timedelta
from pyadm1ode_calibration.calibration.methods.online import (
    OnlineCalibrator, OnlineCalibrationTrigger, OnlineState
)
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
    df = pd.DataFrame({
        "Q_ch4": np.random.rand(24 * 7),
        "pH": np.random.rand(24 * 7) + 6.0
    }, index=pd.date_range("2024-01-01", periods=24 * 7, freq="h"))
    return MeasurementData(df)

class TestOnlineCalibrator:
    def test_calibrate_full(self, mock_plant, sample_measurements):
        cal = OnlineCalibrator(mock_plant, verbose=False)
        cal.simulator = MagicMock()
        cal.simulator.simulate_with_parameters.return_value = {
            "Q_ch4": np.random.rand(24 * 7),
            "pH": np.random.rand(24 * 7) + 6.0
        }
        cal.validator = MagicMock()
        cal.validator.validate.return_value = {}

        # Test with constraints and history
        res = cal.calibrate(
            sample_measurements,
            parameters=["k_dis"],
            current_parameters={"k_dis": 0.5}, # Explicitly provide current_parameters
            use_constraints=True,
            max_iterations=1
        )
        assert isinstance(res, CalibrationResult)

        # Test calling without parameters (should use history)
        cal._get_current_parameters = lambda: {"k_dis": 0.5}
        res2 = cal.calibrate(sample_measurements, max_iterations=1)
        assert res2.parameters is not None

    def test_should_recalibrate_logic(self, mock_plant, sample_measurements):
        cal = OnlineCalibrator(mock_plant, verbose=False)
        cal.simulator = MagicMock()
        cal.simulator.simulate_with_parameters.return_value = {"Q_ch4": np.zeros(24*7)}

        cal.trigger.variance_threshold = 0.001
        cal.trigger.consecutive_violations = 2
        cal.state.last_calibration_time = datetime.now() - timedelta(days=2)

        # Provide current parameters to avoid MagicMock issues
        with MagicMock() as mock_get:
            cal._get_current_parameters = lambda: {"k_dis": 0.5}
            should, _ = cal.should_recalibrate(sample_measurements)
            assert should is False # 1st violation
            should, _ = cal.should_recalibrate(sample_measurements)
            assert should is True # 2nd violation

    def test_calculate_prediction_variance_branches(self, mock_plant, sample_measurements):
        cal = OnlineCalibrator(mock_plant, verbose=False)
        cal.simulator = MagicMock()
        # Mock outputs with some NaNs
        out = {"Q_ch4": np.array([np.nan, 1.0])}
        cal.simulator.simulate_with_parameters.return_value = out

        var = cal._calculate_prediction_variance(sample_measurements, {"k_dis": 0.5}, ["Q_ch4"])
        assert isinstance(var, float)
