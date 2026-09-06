"""
Calibration Result Validation.

This module provides tools for assessing the quality of calibration results
by comparing simulated plant outputs with real measurement data using
various statistical metrics and residual analysis.
"""

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

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

    def to_dict(self) -> dict[str, Any]:
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
    normality_test: dict[str, float]
    autocorrelation: float
    heteroscedasticity_test: dict[str, float]
    outlier_indices: list[int] = field(default_factory=list)

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
    parameter_names: list[str]
    high_correlations: list[tuple[str, str, float]] = field(default_factory=list)
    vif: dict[str, float] | None = None

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

    def __init__(self, plant: Any, verbose: bool = True, time_varying_feed: bool = False):
        self.plant = plant
        self.verbose = verbose
        self.time_varying_feed = time_varying_feed

    def validate(
        self,
        parameters: dict[str, float],
        measurements: MeasurementData,
        objectives: list[str] | None = None,
        simulation_duration: float | None = None,
        warmup: MeasurementData | None = None,
    ) -> dict[str, ValidationMetrics]:
        """
        Validate parameters against measurement data.

        Args:
            parameters (Dict[str, float]): Calibrated parameters to test.
            measurements (MeasurementData): Reference measurement data.
            objectives (Optional[List[str]]): Variables to validate.
            simulation_duration (Optional[float]): Duration in days.
            warmup (Optional[MeasurementData]): Window to run before the scored
                one, so a later stretch of a record is not judged from the
                plant's initial state. Pass the training window when validating
                on the test window that follows it; without it the score
                contains the start-up transient rather than the model error.

        Returns:
            Dict[str, ValidationMetrics]: Metrics for each objective.
        """
        if objectives is None:
            objectives = ["Q_ch4", "pH", "VFA"]

        simulated_outputs = self._simulate_plant(measurements, parameters, warmup=warmup)

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
        simulated: dict[str, np.ndarray],
        objectives: list[str] | None = None,
    ) -> dict[str, ResidualAnalysis]:
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
        parameters: dict[str, float],
        measurements: MeasurementData,
        n_folds: int = 5,
        objectives: list[str] | None = None,
    ) -> dict[str, list[ValidationMetrics]]:
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
        cv_results: dict[str, list[ValidationMetrics]] = {obj: [] for obj in objectives}

        for fold in range(n_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_samples

            val_data = measurements.data.iloc[start_idx:end_idx].copy()
            val_measurements = type(measurements)(val_data)

            fold_metrics = self.validate(parameters, val_measurements, objectives)
            for obj, metrics in fold_metrics.items():
                cv_results[obj].append(metrics)

        return cv_results

    def _simulate_plant(
        self,
        measurements: MeasurementData,
        parameters: dict[str, float],
        warmup: MeasurementData | None = None,
    ) -> dict[str, np.ndarray]:
        """Run the plant over the measurement window with ``parameters`` applied.

        This delegates to :class:`PlantSimulator` on purpose. Validation used to
        carry its own copy of the setup, and it had drifted: parameters were
        written to ``_calibration_params``, which pyadm1 only reads for ``k_p``
        and ``k_L_a`` -- every kinetic rate silently had no effect, so a
        calibrated parameter set scored exactly like the default one. The
        simulator writes ``ADM1._kinetic`` where those rates are actually read,
        applies the substrate feeds from the measurements, and rewinds the plant
        afterwards.
        """
        from .core.simulator import PlantSimulator

        return PlantSimulator(
            self.plant, verbose=self.verbose, time_varying_feed=self.time_varying_feed
        ).simulate_with_parameters(parameters, measurements, warmup=warmup)

    def _extract_outputs_from_results(self, results: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        """Extract canonical observables from a multi-component simulation.

        Mirrors :meth:`PlantSimulator._extract_outputs_from_results`:
        gas flows aggregate over digesters, power aggregates over CHPs
        and heating systems, intensive quantities are averaged across
        digesters, and one gas-storage fill fraction is emitted per
        digester as ``stored_<digester_id>``.
        """
        type_by_id: dict[str, str] = {cid: comp.component_type.value for cid, comp in self.plant.components.items()}
        digester_ids = [cid for cid, t in type_by_id.items() if t == "digester"]

        scalar_keys = [
            "Q_ch4",
            "Q_gas",
            "Q_co2",
            "P_el",
            "P_th",
            "Q_gas_consumed",
            "P_aux_heat",
            "P_th_used",
            "pH",
            "VFA",
            "TAC",
        ]
        outputs: dict[str, list[float]] = {k: [] for k in scalar_keys}
        for cid in digester_ids:
            outputs[f"stored_{cid}"] = []

        for result in results:
            components = result.get("components", {})
            sums = {
                "Q_ch4": 0.0,
                "Q_gas": 0.0,
                "Q_co2": 0.0,
                "P_el": 0.0,
                "P_th": 0.0,
                "Q_gas_consumed": 0.0,
                "P_aux_heat": 0.0,
                "P_th_used": 0.0,
            }
            ph_list: list[float] = []
            vfa_list: list[float] = []
            tac_list: list[float] = []

            for cid, comp_result in components.items():
                ctype = type_by_id.get(cid, "")
                if ctype == "digester":
                    sums["Q_gas"] += comp_result.get("Q_gas", 0.0)
                    sums["Q_ch4"] += comp_result.get("Q_ch4", 0.0)
                    sums["Q_co2"] += comp_result.get("Q_co2", 0.0)
                    if "pH" in comp_result:
                        ph_list.append(comp_result["pH"])
                    if "VFA" in comp_result:
                        vfa_list.append(comp_result["VFA"])
                    if "TAC" in comp_result:
                        tac_list.append(comp_result["TAC"])
                elif ctype == "chp":
                    sums["P_el"] += comp_result.get("P_el", 0.0)
                    sums["P_th"] += comp_result.get("P_th", 0.0)
                    sums["Q_gas_consumed"] += comp_result.get("Q_gas_consumed", 0.0)
                elif ctype == "heating":
                    sums["P_aux_heat"] += comp_result.get("P_aux_heat", 0.0)
                    sums["P_th_used"] += comp_result.get("P_th_used", 0.0)

            for k, v in sums.items():
                outputs[k].append(v)
            outputs["pH"].append(float(np.mean(ph_list)) if ph_list else 7.0)
            outputs["VFA"].append(float(np.mean(vfa_list)) if vfa_list else 0.0)
            outputs["TAC"].append(float(np.mean(tac_list)) if tac_list else 0.0)

            for cid in digester_ids:
                gs = components.get(cid, {}).get("gas_storage", {})
                vol = gs.get("stored_volume_m3", float("nan"))
                cap = float(self.plant.components[cid].V_gas) if hasattr(self.plant.components[cid], "V_gas") else float("nan")
                frac = vol / cap if cap and cap > 0 else float("nan")
                outputs[f"stored_{cid}"].append(frac)

        return {k: np.array(v) for k, v in outputs.items()}

    def _extract_measurements(self, measurements: MeasurementData, objective: str) -> np.ndarray:
        """Extract measurement array for objective."""
        if objective not in measurements.data.columns:
            raise DataValidationError(f"Objective '{objective}' not found in measurements")

        series = measurements.get_measurement(objective)
        if series.empty:
            raise DataValidationError(f"No valid data for objective '{objective}'")

        return series.values

    def _align_arrays(self, observed: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

        # Correlation needs both series to vary. A flat channel, a mocked simulator or
        # a controlled quantity like the digester temperature makes numpy divide by a
        # zero standard deviation and answer NaN, which would then be reported as if
        # it were a measurement.
        can_correlate = n > 1 and obs_std > 0 and pred_std > 0
        correlation = float(np.corrcoef(observed, predicted)[0, 1]) if can_correlate else 0.0

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

    def _test_normality(self, residuals: np.ndarray) -> dict[str, float]:
        """Test for residual normality."""
        try:
            stat, p = stats.shapiro(residuals)
            return {"statistic": float(stat), "p_value": float(p)}
        except ValueError:
            return {"statistic": 0.0, "p_value": 1.0}

    def _calculate_autocorrelation(self, residuals: np.ndarray) -> float:
        """Calculate first-order autocorrelation."""
        if len(residuals) < 2:
            return 0.0
        res_centered = residuals - np.mean(residuals)
        # Residuals that never move have no autocorrelation to speak of.
        if np.std(res_centered[:-1]) == 0 or np.std(res_centered[1:]) == 0:
            return 0.0
        return float(np.corrcoef(res_centered[:-1], res_centered[1:])[0, 1])

    def _test_heteroscedasticity(self, residuals: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
        """Test for heteroscedasticity."""
        squared = residuals**2
        if len(residuals) < 2 or np.std(squared) == 0 or np.std(predicted) == 0:
            # Nothing to test: constant residuals or a constant prediction.
            return {"statistic": 0.0, "p_value": 1.0}
        try:
            corr = np.corrcoef(squared, predicted)[0, 1]
            stat = len(residuals) * corr**2
            p = 1 - stats.chi2.cdf(stat, df=1)
            return {"statistic": float(stat), "p_value": float(p)}
        except (ValueError, IndexError):
            return {"statistic": 0.0, "p_value": 1.0}
