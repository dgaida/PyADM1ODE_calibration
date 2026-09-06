"""Initial calibration module."""

import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np

from pyadm1ode_calibration.io.loaders.measurement_data import MeasurementData

from ..analysis.identifiability import IdentifiabilityAnalyzer, IdentifiabilityResult, SubsetIdentifiability
from ..analysis.sensitivity import SensitivityAnalyzer, SensitivityResult
from ..core.base_calibrator import BaseCalibrator
from ..core.result import CalibrationResult
from ..optimization import MultiObjectiveFunction, ParameterConstraints, WeightedSumObjective, create_optimizer
from ..parameter_bounds import create_default_bounds
from ..validation import CalibrationValidator


class InitialCalibrator(BaseCalibrator):
    """
    Initial calibrator for ADM1 parameters from historical data.

    This calibrator is designed for batch optimization using a window of historical
    measurement data. It supports multi-objective optimization, sensitivity analysis,
    and cross-validation.

    Args:
        plant (Any): The PyADM1ODE plant model to calibrate.
        verbose (bool): Whether to enable verbose output. Defaults to True.
    """

    def __init__(self, plant: Any, verbose: bool = True, time_varying_feed: bool = False):
        super().__init__(plant, verbose, time_varying_feed=time_varying_feed)
        self.parameter_bounds = create_default_bounds()
        self.validator = CalibrationValidator(plant, verbose=False, time_varying_feed=time_varying_feed)
        self.sensitivity_analyzer = SensitivityAnalyzer(plant, self.simulator, verbose)
        self.identifiability_analyzer = IdentifiabilityAnalyzer(plant, self.sensitivity_analyzer, verbose)
        self._optimization_history: list[dict[str, Any]] = []
        self._best_objective_value: float = float("inf")
        self._original_parameters: dict[str, float] = self._get_current_parameters()

    def calibrate(
        self,
        measurements: MeasurementData,
        parameters: list[str],
        bounds: dict[str, tuple[float, float]] | None = None,
        method: str = "differential_evolution",
        objectives: list[str] | None = None,
        weights: dict[str, float] | None = None,
        validation_split: float = 0.2,
        max_iterations: int = 100,
        population_size: int = 15,
        tolerance: float = 1e-4,
        sensitivity_analysis: bool = True,
        use_constraints: bool = False,
        check_identifiability: bool = False,
        **kwargs: Any,
    ) -> CalibrationResult:
        """
        Run the initial calibration workflow.

        Args:
            measurements (MeasurementData): Historical measurement data for calibration.
            parameters (List[str]): Names of parameters to optimize.
            bounds (Optional[Dict[str, Tuple[float, float]]]): Custom search bounds for parameters.
            method (str): Optimization algorithm name. Defaults to 'differential_evolution'.
            objectives (Optional[List[str]]): List of objective variables (e.g., ['Q_ch4', 'pH']).
            weights (Optional[Dict[str, float]]): Weights for different objectives in the cost function.
            validation_split (float): Fraction of data to use for out-of-sample validation. Defaults to 0.2.
            max_iterations (int): Maximum number of optimizer iterations. Defaults to 100.
            population_size (int): Population size for evolutionary algorithms. Defaults to 15.
            tolerance (float): Convergence tolerance. Defaults to 1e-4.
            sensitivity_analysis (bool): Whether to perform sensitivity analysis after calibration.
            use_constraints (bool): Whether to apply parameter constraints. Defaults to False.
            check_identifiability (bool): Screen the parameter set for collinearity before
                optimizing and warn when it is unidentifiable as a set. Costs
                ``2 * len(parameters) + 1`` simulations against the several hundred a
                global search needs, so it is worth it for anything but a one-parameter
                fit. Defaults to False to leave existing behaviour untouched.
            **kwargs (Any): Additional keyword arguments passed to the optimizer.

        Returns:
            CalibrationResult: The calibration results including optimized parameters and metrics.
        """
        start_time = time.time()
        if objectives is None:
            objectives = ["Q_ch4"]

        # Split data
        train_data, val_data = self._split_data(measurements, validation_split)

        initial_params = self.parameter_bounds.get_default_values(parameters)
        param_bounds = self._setup_bounds(parameters, bounds)

        # Finding out that a set could never have been identified is worth far more
        # before the optimizer runs than after it.
        self.subset_identifiability: SubsetIdentifiability | None = None
        if check_identifiability and len(parameters) > 1:
            self.subset_identifiability = self.identifiability_analyzer.analyze_subset(initial_params, train_data, objectives)
            if not self.subset_identifiability.is_identifiable:
                warnings.warn(
                    f"Parameter set {parameters} is not identifiable together: " f"{self.subset_identifiability.reason}",
                    stacklevel=2,
                )
            elif self.verbose:
                print(f"  {self.subset_identifiability.reason}")

        # Create objective function
        def simulator_wrapper(params: dict[str, float]) -> dict[str, np.ndarray]:
            return self.simulator.simulate_with_parameters(params, train_data)

        measurements_dict: dict[str, np.ndarray] = {}
        for obj in objectives:
            if obj in train_data.data.columns:
                measurements_dict[obj] = train_data.get_measurement(obj).values

        objective_func: Callable[[np.ndarray], float]
        if weights is None:
            objective_func = WeightedSumObjective(
                simulator=simulator_wrapper,
                measurements_dict=measurements_dict,
                objectives=objectives,
                parameter_names=parameters,
                error_metric="rmse",
                normalize=True,
            )
        else:
            objective_func = MultiObjectiveFunction(
                simulator=simulator_wrapper,
                measurements_dict=measurements_dict,
                objectives=objectives,
                weights=weights,
                parameter_names=parameters,
                error_metric="rmse",
                normalize=True,
            )

        # Constraints
        obj_func_final: Callable[[np.ndarray], float]
        if use_constraints:
            constraints = ParameterConstraints()
            for param, (lb, ub) in param_bounds.items():
                constraints.add_box_constraint(param, lb, ub, hard=True)

            def penalized_objective(x: np.ndarray) -> float:
                params = {name: val for name, val in zip(parameters, x)}
                return objective_func(x) + constraints.calculate_penalty(params)

            obj_func_final = penalized_objective
        else:
            obj_func_final = objective_func

        optimizer_kwargs = {**kwargs}
        optimizer_kwargs["tolerance"] = tolerance
        if method in ["differential_evolution", "de"]:
            optimizer_kwargs["population_size"] = population_size

        optimizer = create_optimizer(
            method=method,
            bounds=param_bounds,
            max_iterations=max_iterations,
            verbose=self.verbose,
            **optimizer_kwargs,
        )

        initial_guess = (
            np.array([initial_params[p] for p in parameters]) if method in ["nelder_mead", "nm", "lbfgsb", "powell"] else None
        )
        opt_result = optimizer.optimize(obj_func_final, initial_guess=initial_guess)

        # Validation
        validation_metrics: dict[str, float] = {}
        if len(val_data) > 0:
            # Without the warmup the validation window would be simulated from the
            # plant's initial state, so a late stretch of a record could never line
            # up with its measurements no matter how good the parameters are.
            val_result = self.validator.validate(
                parameters=opt_result.parameter_dict,
                measurements=val_data,
                objectives=objectives,
                warmup=train_data,
            )
            for obj, metrics in val_result.items():
                validation_metrics.update(
                    {f"{obj}_rmse": float(metrics.rmse), f"{obj}_r2": float(metrics.r2), f"{obj}_nse": float(metrics.nse)}
                )

        # Sensitivity
        sensitivity_results: dict[str, float] = {}
        if sensitivity_analysis and opt_result.success:
            sens = self.sensitivity_analyzer.analyze(opt_result.parameter_dict, train_data, objectives)
            sensitivity_results = {p: float(max(abs(s) for s in r.sensitivity_indices.values())) for p, r in sens.items()}

        return CalibrationResult(
            success=opt_result.success,
            parameters=opt_result.parameter_dict,
            initial_parameters=initial_params,
            objective_value=float(opt_result.fun),
            n_iterations=int(opt_result.nit),
            execution_time=time.time() - start_time,
            method=method,
            message=str(opt_result.message) if hasattr(opt_result, "message") else "Optimization completed",
            validation_metrics=validation_metrics,
            sensitivity=sensitivity_results,
            history=opt_result.history,
        )

    def sensitivity_analysis(
        self, parameters: dict[str, float], measurements: MeasurementData, objectives: list[str] | None = None
    ) -> dict[str, SensitivityResult]:
        """
        Perform local sensitivity analysis for given parameters.

        Args:
            parameters (Dict[str, float]): Parameter set to analyze.
            measurements (MeasurementData): Data window for simulation.
            objectives (Optional[List[str]]): List of objective variables.

        Returns:
            Dict[str, SensitivityResult]: Mapping of parameter names to sensitivity indices.
        """
        return self.sensitivity_analyzer.analyze(parameters, measurements, objectives)

    def identifiability_analysis(
        self,
        parameters: dict[str, float],
        measurements: MeasurementData,
        optimization_history: list[dict[str, Any]] | None = None,
        confidence_level: float = 0.95,
        correlation_threshold: float = 0.8,
    ) -> dict[str, IdentifiabilityResult]:
        """
        Perform parameter identifiability analysis.

        Checks for parameter correlations and information content in the data.

        Args:
            parameters (Dict[str, float]): Parameter set to analyze.
            measurements (MeasurementData): Data window for simulation.
            optimization_history (Optional[List[Dict[str, Any]]]): History from a previous
                optimizer run, used to widen the confidence intervals.
            confidence_level (float): Confidence level for the intervals.
            correlation_threshold (float): Above this, two parameters count as correlated
                and neither is reported as identifiable on its own.

        Returns:
            IdentifiabilityResult: Analysis results including correlation matrix.
        """
        return self.identifiability_analyzer.analyze(
            parameters,
            measurements,
            optimization_history=optimization_history,
            confidence_level=confidence_level,
            correlation_threshold=correlation_threshold,
        )

    def _split_data(self, measurements: MeasurementData, split_ratio: float) -> tuple[MeasurementData, MeasurementData]:
        """
        Split measurement data into training and validation sets.

        Args:
            measurements (MeasurementData): Original measurement data.
            split_ratio (float): Fraction of data to use for validation.

        Returns:
            Tuple[MeasurementData, MeasurementData]: (train_data, val_data).
        """
        n_train = int(len(measurements) * (1 - split_ratio))
        return (
            MeasurementData(measurements.data.iloc[:n_train].copy(), metadata=measurements.metadata.copy()),
            MeasurementData(measurements.data.iloc[n_train:].copy(), metadata=measurements.metadata.copy()),
        )

    def _setup_bounds(
        self, parameters: list[str], custom_bounds: dict[str, tuple[float, float]] | None
    ) -> dict[str, tuple[float, float]]:
        """
        Configure parameter search bounds.

        Args:
            parameters (List[str]): List of parameters.
            custom_bounds (Optional[Dict[str, Tuple[float, float]]]): Override default bounds.

        Returns:
            Dict[str, Tuple[float, float]]: Mapping of parameter names to (min, max) tuples.
        """
        bounds: dict[str, tuple[float, float]] = {}
        for param in parameters:
            if custom_bounds and param in custom_bounds:
                bounds[param] = custom_bounds[param]
            else:
                b = self.parameter_bounds.get_bounds_tuple(param)
                if b:
                    bounds[param] = b
                else:
                    default = self.parameter_bounds.get_default_values([param])[param]
                    bounds[param] = (default * 0.5, default * 1.5)
        return bounds
