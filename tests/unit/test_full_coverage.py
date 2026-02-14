import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from datetime import datetime, timedelta
from pyadm1ode_calibration.calibration.analysis.sensitivity import SensitivityAnalyzer
from pyadm1ode_calibration.calibration.analysis.identifiability import IdentifiabilityAnalyzer
from pyadm1ode_calibration.calibration.validation import CalibrationValidator
from pyadm1ode_calibration.calibration.core.simulator import PlantSimulator
from pyadm1ode_calibration.calibration.methods.initial import InitialCalibrator
from pyadm1ode_calibration.calibration.methods.online import OnlineCalibrator
from pyadm1ode_calibration.calibration.parameter_bounds import ParameterBounds, BoundType
from pyadm1ode_calibration.io.loaders.measurement_data import MeasurementData
from pyadm1ode_calibration.io.validation.validators import DataValidator, OutlierDetector

@pytest.fixture
def mock_plant():
    plant = MagicMock()
    comp = MagicMock()
    comp.component_type.value = "digester"
    comp._calibration_params = {"k_dis": 0.5}
    plant.components = {"d1": comp}
    plant.simulate.return_value = [{"components": {"d1": {"Q_ch4": 1.0, "pH": 7.0, "VFA": 1.0, "TAC": 1.0}}}]
    return plant

@pytest.fixture
def sample_measurements():
    df = pd.DataFrame({
        "Q_ch4": np.random.rand(10),
        "pH": np.random.rand(10) + 6.0,
        "VFA": np.random.rand(10),
        "TAC": np.random.rand(10),
        "Q_sub1": [10]*10
    }, index=pd.date_range("2024-01-01", periods=10, freq="h"))
    return MeasurementData(df)

def test_full_coverage_suite(mock_plant, sample_measurements):
    # Simulator
    sim = PlantSimulator(mock_plant, verbose=False)
    sim.simulate_with_parameters({"k_dis": 0.6}, sample_measurements)

    # Analysis
    sens_analyzer = SensitivityAnalyzer(mock_plant, sim, verbose=False)
    sens_analyzer.analyze({"k_dis": 0.5}, sample_measurements)
    ident_analyzer = IdentifiabilityAnalyzer(mock_plant, sens_analyzer, verbose=False)
    ident_analyzer.analyze({"k_dis": 0.5}, sample_measurements)

    # Validation
    val = CalibrationValidator(mock_plant, verbose=False)
    val.validate({"k_dis": 0.5}, sample_measurements)

    # Initial Calibrator
    init_cal = InitialCalibrator(mock_plant, verbose=False)
    init_cal.calibrate(sample_measurements, parameters=["k_dis"], max_iterations=1)

    # Online Calibrator
    online_cal = OnlineCalibrator(mock_plant, verbose=False)
    online_cal.calibrate(sample_measurements, parameters=["k_dis"], current_parameters={"k_dis": 0.5}, max_iterations=1)

def test_parameter_bounds_extra():
    pm = ParameterBounds()
    pm.add_bound("p1", 0, 1, 0.5, bound_type=BoundType.SOFT)
    assert pm.calculate_penalty("p1", 1.5, penalty_type="linear") > 0
    assert pm.calculate_penalty("p1", 1.5, penalty_type="quadratic") > 0
    assert pm.calculate_penalty("p1", 0.5) == 0.0

def test_io_validators():
    df = pd.DataFrame({"pH": [7, 8, 9], "timestamp": pd.date_range("2024-01-01", periods=3, freq="h")})
    res = DataValidator.validate(df, expected_ranges={"pH": (0, 14)})
    assert res.is_valid

    s = pd.Series([1]*10 + [100])
    assert OutlierDetector.detect_zscore(s, threshold=2).any()
