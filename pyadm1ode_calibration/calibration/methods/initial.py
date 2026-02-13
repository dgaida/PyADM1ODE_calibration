import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from ..core.base_calibrator import BaseCalibrator
from ..core.result import CalibrationResult
from ..parameter_bounds import create_default_bounds
from ..validation import CalibrationValidator
from ..optimization import create_optimizer, MultiObjectiveFunction, WeightedSumObjective, ParameterConstraints
from ..analysis.sensitivity import SensitivityAnalyzer, SensitivityResult
from ..analysis.identifiability import IdentifiabilityAnalyzer, IdentifiabilityResult
from pyadm1ode_calibration.io.loaders.measurement_data import MeasurementData

class InitialCalibrator(BaseCalibrator):
    """Initial calibrator for ADM1 parameters from historical data."""

    def __init__(self, plant: Any, verbose: bool = True):
        super().__init__(plant, verbose)
        self.parameter_bounds = create_default_bounds()
        self.validator = CalibrationValidator(plant, verbose=False)
        self.sensitivity_analyzer = SensitivityAnalyzer(plant, self.simulator, verbose)
        self.identifiability_analyzer = IdentifiabilityAnalyzer(plant, self.sensitivity_analyzer, verbose)
        self._optimization_history: List[Dict[str, Any]] = []
        self._best_objective_value: float = float("inf")
        self._original_parameters: Dict[str, float] = self._get_current_parameters()

    def calibrate(
        self,
        measurements: MeasurementData,
        parameters: List[str],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = "differential_evolution",
        objectives: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        validation_split: float = 0.2,
        max_iterations: int = 100,
        population_size: int = 15,
        tolerance: float = 1e-4,
        sensitivity_analysis: bool = True,
        use_constraints: bool = False,
        **kwargs: Any,
    ) -> CalibrationResult:
        start_time = time.time()
        if objectives is None:
            objectives = ["Q_ch4"]

        # Split data
        train_data, val_data = self._split_data(measurements, validation_split)

        initial_params = self.parameter_bounds.get_default_values(parameters)
        param_bounds = self._setup_bounds(parameters, bounds)

        # Create objective function
        def simulator_wrapper(params: Dict[str, float]) -> Dict[str, np.ndarray]:
            return self.simulator.simulate_with_parameters(params, train_data)

        measurements_dict: Dict[str, np.ndarray] = {}
        for obj in objectives:
            try:
                measurements_dict[obj] = train_data.get_measurement(obj).values
            except Exception:
                continue

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

        initial_guess = np.array([initial_params[p] for p in parameters]) if method in ["nelder_mead", "nm", "lbfgsb", "powell"] else None
        opt_result = optimizer.optimize(obj_func_final, initial_guess=initial_guess)

        # Validation
        validation_metrics: Dict[str, float] = {}
        if len(val_data) > 0:
            val_result = self.validator.validate(parameters=opt_result.parameter_dict, measurements=val_data, objectives=objectives)
            for obj, metrics in val_result.items():
                validation_metrics.update({
                    f"{obj}_rmse": float(metrics.rmse),
                    f"{obj}_r2": float(metrics.r2),
                    f"{obj}_nse": float(metrics.nse)
                })

        # Sensitivity
        sensitivity_results: Dict[str, float] = {}
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
        self,
        parameters: Dict[str, float],
        measurements: MeasurementData,
        objectives: Optional[List[str]] = None
    ) -> Dict[str, SensitivityResult]:
        return self.sensitivity_analyzer.analyze(parameters, measurements, objectives)

    def identifiability_analysis(
        self,
        parameters: Dict[str, float],
        measurements: MeasurementData
    ) -> Dict[str, IdentifiabilityResult]:
        return self.identifiability_analyzer.analyze(parameters, measurements)

    def _split_data(self, measurements: MeasurementData, split_ratio: float) -> Tuple[MeasurementData, MeasurementData]:
        n_train = int(len(measurements) * (1 - split_ratio))
        return (
            MeasurementData(measurements.data.iloc[:n_train].copy(), metadata=measurements.metadata.copy()),
            MeasurementData(measurements.data.iloc[n_train:].copy(), metadata=measurements.metadata.copy())
        )

    def _setup_bounds(self, parameters: List[str], custom_bounds: Optional[Dict[str, Tuple[float, float]]]) -> Dict[str, Tuple[float, float]]:
        bounds: Dict[str, Tuple[float, float]] = {}
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
