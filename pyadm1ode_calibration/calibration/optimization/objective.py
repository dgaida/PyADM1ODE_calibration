"""Objective module."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class ErrorMetrics:
    """
    Container for different statistical error metrics.

    Calculates and stores various goodness-of-fit metrics for comparing
    observed measurement data with predicted simulation results.

    Attributes:
        mse (float): Mean Squared Error.
        rmse (float): Root Mean Squared Error.
        mae (float): Mean Absolute Error.
        mape (float): Mean Absolute Percentage Error.
        me (float): Mean Error (bias).
        r2 (float): Coefficient of Determination (R-squared).
        nse (float): Nash-Sutcliffe Efficiency.
    """

    mse: float
    rmse: float
    mae: float
    mape: float
    me: float
    r2: float
    nse: float

    @classmethod
    def compute(cls, observed: np.ndarray, predicted: np.ndarray) -> "ErrorMetrics":
        """
        Compute all error metrics for a pair of arrays.

        Args:
            observed (np.ndarray): Observed or measured values.
            predicted (np.ndarray): Predicted or simulated values.

        Returns:
            ErrorMetrics: Object containing calculated metrics.
        """
        observed = np.atleast_1d(observed)
        predicted = np.atleast_1d(predicted)

        valid = ~(np.isnan(observed) | np.isnan(predicted))
        if not np.any(valid):
            return cls(
                mse=float("inf"),
                rmse=float("inf"),
                mae=float("inf"),
                mape=float("inf"),
                me=float("inf"),
                r2=-float("inf"),
                nse=-float("inf"),
            )

        observed = observed[valid]
        predicted = predicted[valid]
        min_len = min(len(observed), len(predicted))
        observed, predicted = observed[:min_len], predicted[:min_len]

        residuals = observed - predicted
        abs_residuals = np.abs(residuals)

        mse = float(np.mean(residuals**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(abs_residuals))
        me = float(np.mean(residuals))

        nonzero = observed != 0
        if np.any(nonzero):
            mape = float(np.mean(abs_residuals[nonzero] / np.abs(observed[nonzero])) * 100)
        else:
            mape = float("inf")

        obs_mean = np.mean(observed)
        ss_tot = np.sum((observed - obs_mean) ** 2)
        ss_res = np.sum(residuals**2)

        if ss_tot > 0:
            r2 = float(1 - ss_res / ss_tot)
            nse = r2
        else:
            r2 = -float("inf")
            nse = -float("inf")

        return cls(mse=mse, rmse=rmse, mae=mae, mape=mape, me=me, r2=r2, nse=nse)


class ObjectiveFunction(ABC):
    """
    Abstract base class for calibration objective functions.

    An objective function defines the scalar value that the optimizer tries
    to minimize during calibration.

    Args:
        parameter_names (List[str]): Names of parameters in order.
        lower_is_better (bool): Whether to minimize (True) or maximize (False). Defaults to True.
    """

    def __init__(self, parameter_names: List[str], lower_is_better: bool = True):
        self.parameter_names = parameter_names
        self.lower_is_better = lower_is_better

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate the objective function.

        Args:
            x (np.ndarray): Array of parameter values.

        Returns:
            float: Calculated objective value.
        """
        pass

    def _params_to_dict(self, x: np.ndarray) -> Dict[str, float]:
        """
        Convert a parameter array to a dictionary.

        Args:
            x (np.ndarray): Parameter array.

        Returns:
            Dict[str, float]: Name-to-value mapping.
        """
        return {name: float(val) for name, val in zip(self.parameter_names, x)}


class SingleObjective(ObjectiveFunction):
    """
    Single-objective function for one output variable.

    Minimizes error between simulated and measured values for a single
    output (e.g., methane production).

    Args:
        simulator (Callable): Function that takes parameters dict and returns simulated outputs.
        measurements (np.ndarray): Measured values for the objective.
        objective_name (str): Name of output to match.
        parameter_names (List[str]): Names of parameters.
        error_metric (str): Error metric ("mse", "rmse", "mae", "mape"). Defaults to "rmse".
    """

    def __init__(
        self,
        simulator: Callable[[Dict[str, float]], Dict[str, np.ndarray]],
        measurements: np.ndarray,
        objective_name: str,
        parameter_names: List[str],
        error_metric: str = "rmse",
    ):
        super().__init__(parameter_names)
        self.simulator = simulator
        self.measurements = measurements
        self.objective_name = objective_name
        self.error_metric = error_metric.lower()

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate single objective.

        Args:
            x (np.ndarray): Parameter values.

        Returns:
            float: Error value.
        """
        params = self._params_to_dict(x)
        try:
            outputs = self.simulator(params)
            if self.objective_name not in outputs:
                return 1e10

            simulated = outputs[self.objective_name]
            metrics = ErrorMetrics.compute(self.measurements, simulated)

            error_map = {
                "mse": metrics.mse,
                "rmse": metrics.rmse,
                "mae": metrics.mae,
                "mape": metrics.mape,
                "nse": -metrics.nse,
                "r2": -metrics.r2,
            }
            return error_map.get(self.error_metric, metrics.rmse)
        except Exception:
            return 1e10


class MultiObjectiveFunction(ObjectiveFunction):
    """
    Multi-objective function with weighted combination.

    Combines errors from multiple plant outputs into a single scalar value.

    Args:
        simulator (Callable): Function that takes parameters and returns outputs.
        measurements_dict (Dict[str, np.ndarray]): Measured values for each objective.
        objectives (List[str]): Names of variables to include in the objective.
        weights (Dict[str, float]): Relative weights for each objective.
        parameter_names (List[str]): Names of optimized parameters.
        error_metric (str): Metric to minimize (e.g., 'rmse', 'mae'). Defaults to 'rmse'.
        normalize (bool): Whether to normalize errors by measurement mean. Defaults to True.
    """

    def __init__(
        self,
        simulator: Callable[[Dict[str, float]], Dict[str, np.ndarray]],
        measurements_dict: Dict[str, np.ndarray],
        objectives: List[str],
        weights: Dict[str, float],
        parameter_names: List[str],
        error_metric: str = "rmse",
        normalize: bool = True,
    ):
        super().__init__(parameter_names)
        self.simulator = simulator
        self.measurements_dict = measurements_dict
        self.objectives = objectives
        self.weights = weights
        self.error_metric = error_metric.lower()
        self.normalize = normalize

        total_weight = sum(weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in weights.items()}

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate the multi-objective weighted sum.

        Args:
            x (np.ndarray): Parameter array.

        Returns:
            float: Total weighted error.
        """
        params = self._params_to_dict(x)
        try:
            outputs = self.simulator(params)
            total_error = 0.0
            n_valid = 0

            for obj_name in self.objectives:
                if obj_name not in outputs or obj_name not in self.measurements_dict:
                    continue

                simulated = outputs[obj_name]
                measured = self.measurements_dict[obj_name]
                metrics = ErrorMetrics.compute(measured, simulated)

                error_map = {
                    "mse": metrics.mse,
                    "rmse": metrics.rmse,
                    "mae": metrics.mae,
                    "mape": metrics.mape,
                    "nse": -metrics.nse,
                    "r2": -metrics.r2,
                }
                error = error_map.get(self.error_metric, metrics.rmse)

                if self.normalize:
                    mean_measured = np.mean(np.abs(measured))
                    if mean_measured > 1e-10:
                        error /= mean_measured

                total_error += self.weights.get(obj_name, 0.0) * error
                n_valid += 1

            return total_error if n_valid > 0 else 1e10
        except Exception:
            return 1e10


class WeightedSumObjective(MultiObjectiveFunction):
    """
    Convenience class for MultiObjectiveFunction with equal weights by default.

    Args:
        simulator (Callable): Function that takes parameters and returns outputs.
        measurements_dict (Dict[str, np.ndarray]): Measured values for each objective.
        objectives (List[str]): Names of variables to include in the objective.
        parameter_names (List[str]): Names of optimized parameters.
        weights (Optional[Dict[str, float]]): Optional custom weights.
        **kwargs: Passed to MultiObjectiveFunction.
    """

    def __init__(
        self,
        simulator: Callable[[Dict[str, float]], Dict[str, np.ndarray]],
        measurements_dict: Dict[str, np.ndarray],
        objectives: List[str],
        parameter_names: List[str],
        weights: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ):
        if weights is None:
            weights = {obj: 1.0 / len(objectives) for obj in objectives}
        super().__init__(simulator, measurements_dict, objectives, weights, parameter_names, **kwargs)


class LikelihoodObjective(ObjectiveFunction):
    """
    Maximum likelihood objective function.

    Assumes Gaussian errors and maximizes likelihood (minimizes negative
    log-likelihood).

    Args:
        simulator (Callable): Simulator function.
        measurements_dict (Dict[str, np.ndarray]): Measurements for each objective.
        objectives (List[str]): List of objectives.
        parameter_names (List[str]): Parameter names.
        sigma (Optional[Dict[str, float]]): Standard deviations for each objective.
    """

    def __init__(
        self,
        simulator: Callable[[Dict[str, float]], Dict[str, np.ndarray]],
        measurements_dict: Dict[str, np.ndarray],
        objectives: List[str],
        parameter_names: List[str],
        sigma: Optional[Dict[str, float]] = None,
    ):
        super().__init__(parameter_names)
        self.simulator = simulator
        self.measurements_dict = measurements_dict
        self.objectives = objectives
        self.sigma = sigma or {}

        for obj_name in objectives:
            if obj_name not in self.sigma:
                measured = measurements_dict.get(obj_name, np.array([]))
                self.sigma[obj_name] = float(np.std(measured) + 1e-10) if measured.size > 0 else 1.0

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate negative log-likelihood.

        Args:
            x (np.ndarray): Parameter values.

        Returns:
            float: Negative log-likelihood.
        """
        params = self._params_to_dict(x)
        try:
            outputs = self.simulator(params)
            neg_log_likelihood = 0.0
            n_total = 0

            for obj_name in self.objectives:
                if obj_name not in outputs or obj_name not in self.measurements_dict:
                    continue

                simulated = np.atleast_1d(outputs[obj_name])
                measured = np.atleast_1d(self.measurements_dict[obj_name])
                min_len = min(len(measured), len(simulated))
                measured, simulated = measured[:min_len], simulated[:min_len]

                valid = ~(np.isnan(measured) | np.isnan(simulated))
                if not np.any(valid):
                    continue

                res = measured[valid] - simulated[valid]
                sigma = self.sigma[obj_name]
                n = len(res)
                nll = 0.5 * np.sum((res / sigma) ** 2) + n * np.log(sigma)
                neg_log_likelihood += nll
                n_total += n

            return neg_log_likelihood if n_total > 0 else 1e10
        except Exception:
            return 1e10


class CustomObjective(ObjectiveFunction):
    """
    Custom user-defined objective function.

    Args:
        simulator (Callable): Simulator function.
        measurements_dict (Dict[str, np.ndarray]): Measurements.
        objectives (List[str]): Objectives to evaluate.
        parameter_names (List[str]): Parameter names.
        custom_func (Callable): Custom function(simulated, measured) -> error.
    """

    def __init__(
        self,
        simulator: Callable[[Dict[str, float]], Dict[str, np.ndarray]],
        measurements_dict: Dict[str, np.ndarray],
        objectives: List[str],
        parameter_names: List[str],
        custom_func: Callable[[np.ndarray, np.ndarray], float],
    ):
        super().__init__(parameter_names)
        self.simulator = simulator
        self.measurements_dict = measurements_dict
        self.objectives = objectives
        self.custom_func = custom_func

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate custom objective.

        Args:
            x (np.ndarray): Parameter values.

        Returns:
            float: Custom error value.
        """
        params = self._params_to_dict(x)
        try:
            outputs = self.simulator(params)
            total_error = 0.0
            n_valid = 0

            for obj_name in self.objectives:
                if obj_name not in outputs or obj_name not in self.measurements_dict:
                    continue

                simulated = outputs[obj_name]
                measured = self.measurements_dict[obj_name]
                total_error += self.custom_func(simulated, measured)
                n_valid += 1

            return total_error / n_valid if n_valid > 0 else 1e10
        except Exception:
            return 1e10


def create_objective(
    objective_type: str,
    simulator: Callable[[Dict[str, float]], Dict[str, np.ndarray]],
    measurements_dict: Dict[str, np.ndarray],
    objectives: List[str],
    parameter_names: List[str],
    **kwargs: Any,
) -> ObjectiveFunction:
    """
    Factory function to create objective functions.

    Args:
        objective_type (str): Type of objective ("single", "multi", "weighted", "likelihood").
        simulator (Callable): Simulator function.
        measurements_dict (Dict[str, np.ndarray]): Measurements dictionary.
        objectives (List[str]): List of objectives.
        parameter_names (List[str]): Parameter names.
        **kwargs: Additional arguments.

    Returns:
        ObjectiveFunction: Created objective function.

    Raises:
        ValueError: If objective type is unknown.
    """
    objective_type = objective_type.lower()
    if objective_type == "single":
        return SingleObjective(
            simulator=simulator,
            measurements=measurements_dict[objectives[0]],
            objective_name=objectives[0],
            parameter_names=parameter_names,
            **kwargs,
        )
    elif objective_type in ["multi", "weighted"]:
        weights = kwargs.pop("weights", None)
        return MultiObjectiveFunction(
            simulator=simulator,
            measurements_dict=measurements_dict,
            objectives=objectives,
            weights=weights or {obj: 1.0 / len(objectives) for obj in objectives},
            parameter_names=parameter_names,
            **kwargs,
        )
    elif objective_type == "likelihood":
        return LikelihoodObjective(
            simulator=simulator,
            measurements_dict=measurements_dict,
            objectives=objectives,
            parameter_names=parameter_names,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")
