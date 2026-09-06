from unittest.mock import MagicMock

import numpy as np
import pytest

from pyadm1ode_calibration.calibration.analysis.identifiability import (
    MAX_COLLINEARITY_INDEX,
    IdentifiabilityAnalyzer,
    IdentifiabilityResult,
    collinearity_index,
)
from pyadm1ode_calibration.calibration.analysis.sensitivity import SensitivityAnalyzer, SensitivityResult


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

    def test_weak_sensitivity_is_practically_unidentifiable(self, mock_plant):
        """A parameter can move the outputs and still not be pinned down.

        The old criterion only asked whether the sensitivity was above 1e-6, so a
        confidence interval spanning several times the estimate still counted as
        identifiable. Practical non-identifiability in the sense of Raue et al. 2009
        is exactly this case.
        """
        mock_sensitivity_analyzer = MagicMock()
        mock_sensitivity_analyzer.analyze.return_value = {
            # 0.046 is the value measured for k_hyd_ch on the example plant.
            "weak": SensitivityResult(
                parameter="weak",
                base_value=2.5,
                sensitivity_indices={"Q_ch4": 0.046},
                local_gradient={"Q_ch4": 0.02},
                normalized_sensitivity={"Q_ch4": 0.01},
                variance_contribution=0.002,
            ),
            "strong": SensitivityResult(
                parameter="strong",
                base_value=6.0,
                sensitivity_indices={"Q_ch4": 3.14},
                local_gradient={"Q_ch4": 1.5},
                normalized_sensitivity={"Q_ch4": 0.8},
                variance_contribution=9.9,
            ),
        }
        analyzer = IdentifiabilityAnalyzer(mock_plant, sensitivity_analyzer=mock_sensitivity_analyzer, verbose=False)

        results = analyzer.analyze({"weak": 2.5, "strong": 6.0}, MagicMock())

        assert results["strong"].is_identifiable is True
        assert results["weak"].is_identifiable is False
        assert "practically non-identifiable" in results["weak"].reason
        # The interval reported is the one that was computed, not a placeholder.
        low, high = results["weak"].confidence_interval
        assert high - low > 2 * 2.5

    def test_zero_sensitivity_is_structurally_unidentifiable(self, mock_plant):
        """No effect at all is a different failure, and says so."""
        mock_sensitivity_analyzer = MagicMock()
        mock_sensitivity_analyzer.analyze.return_value = {
            "dead": SensitivityResult(
                parameter="dead",
                base_value=0.5,
                sensitivity_indices={"Q_ch4": 0.0},
                local_gradient={"Q_ch4": 0.0},
                normalized_sensitivity={"Q_ch4": 0.0},
                variance_contribution=0.0,
            )
        }
        analyzer = IdentifiabilityAnalyzer(mock_plant, sensitivity_analyzer=mock_sensitivity_analyzer, verbose=False)

        result = analyzer.analyze({"dead": 0.5}, MagicMock())["dead"]

        assert result.is_identifiable is False
        assert "structurally non-identifiable" in result.reason


class TestCollinearityIndex:
    """Brun's index over the columns of the sensitivity matrix."""

    def test_orthogonal_columns_give_one(self):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        assert collinearity_index(matrix) == pytest.approx(1.0)

    def test_near_duplicate_columns_exceed_the_threshold(self):
        """Two parameters doing almost the same thing to the model.

        For two columns the index is (1 - |cos|)^-0.5, so the customary limit of 20
        sits at cos = 0.9975. These two are past that.
        """
        base = np.linspace(1.0, 2.0, 200)
        matrix = np.column_stack([base, base + 1e-3 * np.sin(np.linspace(0, 3, 200))])
        assert collinearity_index(matrix) > MAX_COLLINEARITY_INDEX

    def test_two_columns_match_the_closed_form(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)  # cos = 1/sqrt(2)
        expected = 1.0 / np.sqrt(1.0 - abs(float(np.dot(a, b))))
        assert collinearity_index(np.column_stack([a, b])) == pytest.approx(expected, rel=1e-6)

    def test_zero_column_is_infinite(self):
        matrix = np.column_stack([np.array([1.0, 2.0, 3.0]), np.zeros(3)])
        assert collinearity_index(matrix) == float("inf")

    def test_single_column_is_one(self):
        assert collinearity_index(np.array([[1.0], [2.0]])) == pytest.approx(1.0)


class TestSubsetVerdict:
    """The set as a whole, screened before it is scored."""

    @staticmethod
    def _sensitivity(columns):
        return {
            name: SensitivityResult(
                parameter=name,
                base_value=1.0,
                sensitivity_indices={"Q_ch4": 1.0},
                local_gradient={"Q_ch4": 1.0},
                normalized_sensitivity={"Q_ch4": 1.0},
                variance_contribution=1.0,
                sensitivity_column=np.asarray(column, dtype=float),
            )
            for name, column in columns.items()
        }

    @staticmethod
    def _verdicts(names, identifiable):
        return {
            name: IdentifiabilityResult(
                parameter=name,
                is_identifiable=name in identifiable,
                confidence_interval=(0.0, 1.0),
                correlation_with={},
                objective_sensitivity=1.0,
                reason="",
            )
            for name in names
        }

    def test_weak_member_sinks_the_set_and_gamma_covers_the_rest(self):
        """Stage one removes the weak parameter, stage two scores what is left."""
        sensitivity = self._sensitivity({"strong_a": [1.0, 0.0, 0.0], "strong_b": [0.0, 1.0, 0.0], "weak": [1e-9, 0.0, 1e-9]})
        verdict = IdentifiabilityAnalyzer.subset_verdict(sensitivity, self._verdicts(sensitivity, {"strong_a", "strong_b"}))

        assert verdict.is_identifiable is False
        assert "weak" in verdict.reason
        # The two survivors are orthogonal, so the index describes them, not the set.
        assert verdict.collinearity_index == pytest.approx(1.0)

    def test_collinear_survivors_are_rejected_with_the_number(self):
        base = np.linspace(1.0, 2.0, 200)
        sensitivity = self._sensitivity({"a": base, "b": base + 1e-3 * np.sin(np.linspace(0, 3, 200))})
        verdict = IdentifiabilityAnalyzer.subset_verdict(sensitivity, self._verdicts(sensitivity, {"a", "b"}))

        assert verdict.is_identifiable is False
        assert verdict.collinearity_index > MAX_COLLINEARITY_INDEX
        assert "compensate" in verdict.reason

    def test_independent_set_passes(self):
        sensitivity = self._sensitivity({"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0]})
        verdict = IdentifiabilityAnalyzer.subset_verdict(sensitivity, self._verdicts(sensitivity, {"a", "b"}))

        assert verdict.is_identifiable is True
        assert verdict.collinearity_index == pytest.approx(1.0)
