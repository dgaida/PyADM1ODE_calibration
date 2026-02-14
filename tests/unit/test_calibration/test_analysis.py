import pytest
import numpy as np
from unittest.mock import MagicMock
from pyadm1ode_calibration.calibration.analysis.sensitivity import SensitivityAnalyzer, SensitivityResult
from pyadm1ode_calibration.calibration.analysis.identifiability import IdentifiabilityAnalyzer


@pytest.fixture
def mock_plant():
    return MagicMock()


@pytest.fixture
def mock_simulator():
    simulator = MagicMock()
    # Mock simulation output
    simulator.simulate_with_parameters.return_value = {
        "Q_ch4": np.array([1.0, 1.1, 0.9]),
        "pH": np.array([7.0, 7.1, 6.9]),
        "VFA": np.array([100, 110, 90]),
    }
    return simulator


class TestSensitivityAnalyzer:
    def test_init(self, mock_plant):
        analyzer = SensitivityAnalyzer(mock_plant, verbose=False)
        assert analyzer.plant == mock_plant
        assert analyzer.verbose is False

    def test_analyze(self, mock_plant, mock_simulator):
        analyzer = SensitivityAnalyzer(mock_plant, simulator=mock_simulator, verbose=False)
        parameters = {"k_dis": 0.5, "Y_su": 0.1}
        measurements = MagicMock()

        # We need to make the simulator return different values for different parameters to have non-zero gradients
        def side_effect(params, meas):
            if params.get("k_dis") > 0.5:
                return {"Q_ch4": np.array([1.2, 1.3, 1.1])}
            elif params.get("k_dis") < 0.5:
                return {"Q_ch4": np.array([0.8, 0.9, 0.7])}
            elif params.get("Y_su") > 0.1:
                return {"Q_ch4": np.array([1.1, 1.2, 1.0])}
            elif params.get("Y_su") < 0.1:
                return {"Q_ch4": np.array([0.9, 1.0, 0.8])}
            return {"Q_ch4": np.array([1.0, 1.1, 0.9])}

        mock_simulator.simulate_with_parameters.side_effect = side_effect

        results = analyzer.analyze(parameters, measurements, perturbation=0.1)

        assert "k_dis" in results
        assert "Y_su" in results
        assert isinstance(results["k_dis"], SensitivityResult)
        assert results["k_dis"].parameter == "k_dis"
        assert results["k_dis"].base_value == 0.5
        assert "Q_ch4" in results["k_dis"].local_gradient
        assert results["k_dis"].local_gradient["Q_ch4"] > 0
        assert results["k_dis"].variance_contribution > 0

    def test_analyze_zero_base_value(self, mock_plant, mock_simulator):
        analyzer = SensitivityAnalyzer(mock_plant, simulator=mock_simulator, verbose=False)
        parameters = {"zero_param": 0.0}
        measurements = MagicMock()

        results = analyzer.analyze(parameters, measurements)
        assert "zero_param" in results
        assert results["zero_param"].base_value == 0.0


class TestIdentifiabilityAnalyzer:
    def test_init(self, mock_plant):
        analyzer = IdentifiabilityAnalyzer(mock_plant, verbose=False)
        assert analyzer.plant == mock_plant

    def test_analyze(self, mock_plant):
        mock_sensitivity_analyzer = MagicMock()
        mock_sensitivity_analyzer.analyze.return_value = {
            "k_dis": SensitivityResult(
                parameter="k_dis",
                base_value=0.5,
                sensitivity_indices={"Q_ch4": 2.0},
                local_gradient={"Q_ch4": 1.0},
                normalized_sensitivity={"Q_ch4": 0.5},
                variance_contribution=4.0,
            ),
            "unidentifiable": SensitivityResult(
                parameter="unidentifiable",
                base_value=0.1,
                sensitivity_indices={"Q_ch4": 1e-10},
                local_gradient={"Q_ch4": 1e-11},
                normalized_sensitivity={"Q_ch4": 1e-12},
                variance_contribution=1e-20,
            ),
        }

        analyzer = IdentifiabilityAnalyzer(mock_plant, sensitivity_analyzer=mock_sensitivity_analyzer, verbose=False)
        parameters = {"k_dis": 0.5, "unidentifiable": 0.1}
        measurements = MagicMock()

        results = analyzer.analyze(parameters, measurements)

        assert results["k_dis"].is_identifiable is True
        assert results["unidentifiable"].is_identifiable is False
        assert results["k_dis"].objective_sensitivity == 2.0
        assert len(results["k_dis"].confidence_interval) == 2
