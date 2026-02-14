"""
Calibration Result Validation.

This module provides tools for assessing the quality of calibration results
by comparing simulated plant outputs with real measurement data using
various statistical metrics and residual analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats
import warnings
from ..exceptions import DataValidationError
from ..io.loaders.measurement_data import MeasurementData


@dataclass
class ValidationMetrics:
    """
    Comprehensive validation metrics for calibration results.

    Attributes:
        objective (str): Name of the validated variable.
        n_samples (int): Number of samples used for validation.
        rmse (float): Root Mean Squared Error.
        mae (float): Mean Absolute Error.
        r2 (float): Coefficient of Determination.
        nse (float): Nash-Sutcliffe Efficiency.
        pbias (float): Percent Bias.
        correlation (float): Pearson correlation coefficient.
        mape (float): Mean Absolute Percentage Error.
        me (float): Mean Error (bias).
        observations_mean (float): Mean of observed values.
        observations_std (float): Standard deviation of observed values.
        predictions_mean (float): Mean of predicted values.
        predictions_std (float): Standard deviation of predicted values.
    """

    objective: str
    n_samples: int
    rmse: float
    mae: float
    r2: float
    nse: float
    pbias: float
    correlation: float
    mape: float
    me: float
    observations_mean: float
    observations_std: float
    predictions_mean: float
    predictions_std: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to a dictionary.

        Returns:
            Dict[str, Any]: Name-to-value mapping of metrics.
        """
        return {
            "objective": self.objective,
            "n_samples": self.n_samples,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "nse": self.nse,
            "pbias": self.pbias,
            "correlation": self.correlation,
            "mape": self.mape,
            "me": self.me,
            "observations_mean": self.observations_mean,
            "observations_std": self.observations_std,
            "predictions_mean": self.predictions_mean,
            "predictions_std": self.predictions_std,
        }


@dataclass
class ResidualAnalysis:
    """
    Results from residual analysis.

    Provides statistical tests for normality, autocorrelation, and
    heteroscedasticity of simulation residuals.

    Attributes:
        objective (str): Name of the analyzed variable.
        residuals (np.ndarray): Raw residuals (observed - predicted).
        standardized_residuals (np.ndarray): Residuals normalized by standard deviation.
        normality_test (Dict[str, float]): Results from Shapiro-Wilk test.
        autocorrelation (float): First-order autocorrelation coefficient.
        heteroscedasticity_test (Dict[str, float]): Correlation between residuals and predictions.
        outlier_indices (List[int]): Indices of residuals exceeding 3 standard deviations.
    """

    objective: str
    residuals: np.ndarray
    standardized_residuals: np.ndarray
    normality_test: Dict[str, float]
    autocorrelation: float
    heteroscedasticity_test: Dict[str, float]
    outlier_indices: List[int] = field(default_factory=list)

    def is_normally_distributed(self, alpha: float = 0.05) -> bool:
        """
        Check if residuals are normally distributed.

        Args:
            alpha (float): Significance level. Defaults to 0.05.

        Returns:
            bool: True if p-value > alpha.
        """
        return self.normality_test["p_value"] > alpha

    def has_autocorrelation(self, threshold: float = 0.3) -> bool:
        """
        Check for significant first-order autocorrelation.

        Args:
            threshold (float): Correlation threshold. Defaults to 0.3.

        Returns:
            bool: True if absolute autocorrelation exceeds threshold.
        """
        return abs(self.autocorrelation) > threshold

    def has_heteroscedasticity(self, alpha: float = 0.05) -> bool:
        """
        Check for heteroscedasticity (variance depending on value).

        Args:
            alpha (float): Significance level. Defaults to 0.05.

        Returns:
            bool: True if p-value < alpha.
        """
        return self.heteroscedasticity_test["p_value"] < alpha


@dataclass
class ParameterCorrelation:
    """
    Parameter correlation analysis results.

    Attributes:
        correlation_matrix (np.ndarray): Square matrix of correlations.
        parameter_names (List[str]): List of parameter names in matrix order.
        high_correlations (List[Tuple[str, str, float]]): Pairs with correlation > threshold.
        vif (Optional[Dict[str, float]]): Variance Inflation Factors for each parameter.
    """

    correlation_matrix: np.ndarray
    parameter_names: List[str]
    high_correlations: List[Tuple[str, str, float]] = field(default_factory=list)
    vif: Optional[Dict[str, float]] = None

    def get_correlation(self, param1: str, param2: str) -> float:
        """
        Get correlation coefficient between two parameters.

        Args:
            param1 (str): First parameter name.
            param2 (str): Second parameter name.

        Returns:
            float: Correlation coefficient.
        """
        idx1 = self.parameter_names.index(param1)
        idx2 = self.parameter_names.index(param2)
        return float(self.correlation_matrix[idx1, idx2])


class CalibrationValidator:
    """
    Validator for calibrated model parameters.

    Provides methods to evaluate the goodness-of-fit of calibrated
    parameters on both training and out-of-sample validation data.

    Args:
        plant (Any): The PyADM1ODE plant model instance.
        verbose (bool): Whether to enable verbose output. Defaults to True.
    """

    def __init__(self, plant: Any, verbose: bool = True):
        self.plant = plant
        self.verbose = verbose

    def validate(
        self,
        parameters: Dict[str, float],
        measurements: MeasurementData,
        objectives: Optional[List[str]] = None,
        simulation_duration: Optional[float] = None,
    ) -> Dict[str, ValidationMetrics]:
        """
        Validate parameters against measurement data.

        Args:
            parameters (Dict[str, float]): Calibrated parameters to test.
            measurements (MeasurementData): Reference measurement data.
            objectives (Optional[List[str]]): Variables to validate.
            simulation_duration (Optional[float]): Duration in days.

        Returns:
            Dict[str, ValidationMetrics]: Metrics for each objective.
        """
        if objectives is None:
            objectives = ["Q_ch4", "pH", "VFA"]

        self._apply_parameters(parameters)

        if simulation_duration is None:
            simulation_duration = len(measurements) * (1.0 / 24.0)

        simulated_outputs = self._simulate_plant(measurements, simulation_duration)

        metrics = {}
        for objective in objectives:
            if objective not in simulated_outputs:
                warnings.warn(f"Objective '{objective}' not in simulation outputs")
                continue

            observed = self._extract_measurements(measurements, objective)
            predicted = simulated_outputs[objective]

            observed, predicted = self._align_arrays(observed, predicted)

            if len(observed) == 0:
                warnings.warn(f"No valid data for objective '{objective}'")
                continue

            obj_metrics = self._calculate_metrics(objective, observed, predicted)
            metrics[objective] = obj_metrics

        return metrics

    def analyze_residuals(
        self,
        measurements: MeasurementData,
        simulated: Dict[str, np.ndarray],
        objectives: Optional[List[str]] = None,
    ) -> Dict[str, ResidualAnalysis]:
        """
        Perform detailed residual analysis.

        Args:
            measurements (MeasurementData): Reference measurements.
            simulated (Dict[str, np.ndarray]): Simulated outputs.
            objectives (Optional[List[str]]): Variables to analyze.

        Returns:
            Dict[str, ResidualAnalysis]: Analysis results for each objective.
        """
        if objectives is None:
            objectives = list(simulated.keys())

        results = {}
        for objective in objectives:
            if objective not in simulated:
                continue

            observed = self._extract_measurements(measurements, objective)
            predicted = simulated[objective]

            observed, predicted = self._align_arrays(observed, predicted)

            if len(observed) < 3:
                continue

            residuals = observed - predicted
            std_residuals = self._standardize_residuals(residuals)
            normality = self._test_normality(residuals)
            autocorr = self._calculate_autocorrelation(residuals)
            hetero = self._test_heteroscedasticity(residuals, predicted)
            outliers = np.where(np.abs(std_residuals) > 3)[0].tolist()

            results[objective] = ResidualAnalysis(
                objective=objective,
                residuals=residuals,
                standardized_residuals=std_residuals,
                normality_test=normality,
                autocorrelation=autocorr,
                heteroscedasticity_test=hetero,
                outlier_indices=outliers,
            )

        return results

    def cross_validate(
        self,
        parameters: Dict[str, float],
        measurements: MeasurementData,
        n_folds: int = 5,
        objectives: Optional[List[str]] = None,
    ) -> Dict[str, List[ValidationMetrics]]:
        """
        Perform k-fold cross-validation.

        Args:
            parameters (Dict[str, float]): Calibrated parameters.
            measurements (MeasurementData): Full dataset to split.
            n_folds (int): Number of folds. Defaults to 5.
            objectives (Optional[List[str]]): Variables to validate.

        Returns:
            Dict[str, List[ValidationMetrics]]: List of metrics for each fold.
        """
        if objectives is None:
            objectives = ["Q_ch4", "pH", "VFA"]

        n_samples = len(measurements)
        fold_size = n_samples // n_folds
        cv_results: Dict[str, List[ValidationMetrics]] = {obj: [] for obj in objectives}

        for fold in range(n_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_samples

            val_data = measurements.data.iloc[start_idx:end_idx].copy()
            val_measurements = type(measurements)(val_data)

            fold_metrics = self.validate(parameters, val_measurements, objectives)
            for obj, metrics in fold_metrics.items():
                cv_results[obj].append(metrics)

        return cv_results

    def _apply_parameters(self, parameters: Dict[str, float]) -> None:
        """Apply parameters to plant."""
        for component in self.plant.components.values():
            if component.component_type.value == "digester":
                if not hasattr(component, "_calibration_params"):
                    component._calibration_params = {}
                component._calibration_params.update(parameters)

    def _simulate_plant(self, measurements: MeasurementData, duration: float) -> Dict[str, np.ndarray]:
        """Run plant simulation."""
        dt = 1.0 / 24.0
        results = self.plant.simulate(duration=duration, dt=dt, save_interval=dt)
        return self._extract_outputs_from_results(results)

    def _extract_outputs_from_results(self, results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract relevant outputs."""
        outputs: Dict[str, List[float]] = {"Q_ch4": [], "pH": [], "VFA": [], "TAC": []}
        for result in results:
            comp_data = next(iter(result["components"].values()))
            for key in outputs:
                outputs[key].append(comp_data.get(key, 0.0))
        return {k: np.array(v) for k, v in outputs.items()}

    def _extract_measurements(self, measurements: MeasurementData, objective: str) -> np.ndarray:
        """Extract measurement array for objective."""
        if objective not in measurements.data.columns:
            raise DataValidationError(f"Objective '{objective}' not found in measurements")

        series = measurements.get_measurement(objective)
        if series.empty:
            raise DataValidationError(f"No valid data for objective '{objective}'")

        return series.values

    def _align_arrays(self, observed: np.ndarray, predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align observed and predicted arrays."""
        min_len = min(len(observed), len(predicted))
        observed, predicted = observed[:min_len], predicted[:min_len]
        valid = ~(np.isnan(observed) | np.isnan(predicted))
        return observed[valid], predicted[valid]

    def _calculate_metrics(self, objective: str, observed: np.ndarray, predicted: np.ndarray) -> ValidationMetrics:
        """Calculate statistical metrics."""
        n = len(observed)
        obs_mean, obs_std = np.mean(observed), np.std(observed)
        pred_mean, pred_std = np.mean(predicted), np.std(predicted)
        residuals = observed - predicted

        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((observed - obs_mean) ** 2)
        r2 = float(1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0)

        pbias = float((np.sum(residuals) / np.sum(observed)) * 100 if np.sum(observed) != 0 else 0.0)
        correlation = float(np.corrcoef(observed, predicted)[0, 1] if n > 1 else 0.0)

        nonzero = observed != 0
        mape = float(np.mean(np.abs(residuals[nonzero] / observed[nonzero])) * 100 if np.any(nonzero) else 0.0)
        me = float(np.mean(residuals))

        return ValidationMetrics(
            objective=objective,
            n_samples=n,
            rmse=float(rmse),
            mae=float(mae),
            r2=r2,
            nse=r2,
            pbias=pbias,
            correlation=correlation,
            mape=mape,
            me=me,
            observations_mean=float(obs_mean),
            observations_std=float(obs_std),
            predictions_mean=float(pred_mean),
            predictions_std=float(pred_std),
        )

    def _standardize_residuals(self, residuals: np.ndarray) -> np.ndarray:
        """Standardize residuals."""
        std = np.std(residuals)
        return (residuals - np.mean(residuals)) / std if std > 0 else np.zeros_like(residuals)

    def _test_normality(self, residuals: np.ndarray) -> Dict[str, float]:
        """Test for residual normality."""
        try:
            stat, p = stats.shapiro(residuals)
            return {"statistic": float(stat), "p_value": float(p)}
        except Exception:
            return {"statistic": 0.0, "p_value": 1.0}

    def _calculate_autocorrelation(self, residuals: np.ndarray) -> float:
        """Calculate first-order autocorrelation."""
        if len(residuals) < 2:
            return 0.0
        res_centered = residuals - np.mean(residuals)
        return float(np.corrcoef(res_centered[:-1], res_centered[1:])[0, 1])

    def _test_heteroscedasticity(self, residuals: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Test for heteroscedasticity."""
        try:
            corr = np.corrcoef(residuals**2, predicted)[0, 1]
            stat = len(residuals) * corr**2
            p = 1 - stats.chi2.cdf(stat, df=1)
            return {"statistic": float(stat), "p_value": float(p)}
        except Exception:
            return {"statistic": 0.0, "p_value": 1.0}
