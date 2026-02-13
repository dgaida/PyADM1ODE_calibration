import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from ..core.base_calibrator import BaseCalibrator
from ..core.result import CalibrationResult
from ..parameter_bounds import create_default_bounds, ParameterBounds
from ..validation import CalibrationValidator
from ..optimization import create_optimizer, MultiObjectiveFunction, ParameterConstraints
from pyadm1ode_calibration.io.loaders.measurement_data import MeasurementData


@dataclass
class OnlineCalibrationTrigger:
    """Trigger conditions for online re-calibration.

    Attributes:
        variance_threshold: Trigger when prediction variance exceeds this value (0-1).
        time_threshold: Minimum time between calibrations [hours].
        residual_threshold: Trigger when residual exceeds this value.
        consecutive_violations: Number of consecutive threshold violations required.
        enabled: Whether automatic triggering is enabled.
    """

    variance_threshold: float = 0.15
    time_threshold: float = 24.0
    residual_threshold: Optional[float] = None
    consecutive_violations: int = 3
    enabled: bool = True


@dataclass
class ParameterChangeHistory:
    """History of parameter changes during online operation."""

    timestamp: datetime
    parameters: Dict[str, float]
    trigger_reason: str
    objective_value: float
    variance: float
    success: bool


@dataclass
class OnlineState:
    """Online calibrator state tracking."""

    last_calibration_time: Optional[datetime] = None
    consecutive_violations: int = 0
    current_variance: float = 0.0
    parameter_history: List[ParameterChangeHistory] = field(default_factory=list)
    total_calibrations: int = 0


class OnlineCalibrator(BaseCalibrator):
    """Online calibrator for real-time parameter adjustment.

    Performs fast, bounded re-calibration when model predictions deviate from
    measurements.
    """

    def __init__(self, plant: Any, verbose: bool = True, parameter_bounds: Optional[ParameterBounds] = None):
        """Initialize online calibrator.

        Args:
            plant: BiogasPlant instance
            verbose: Enable progress output
            parameter_bounds: Custom parameter bounds manager
        """
        super().__init__(plant, verbose)
        self.parameter_bounds: ParameterBounds = parameter_bounds or create_default_bounds()
        self.validator: CalibrationValidator = CalibrationValidator(plant, verbose=False)
        self.trigger: OnlineCalibrationTrigger = OnlineCalibrationTrigger()
        self.state: OnlineState = OnlineState()

    def calibrate(
        self,
        measurements: MeasurementData,
        parameters: Optional[List[str]] = None,
        current_parameters: Optional[Dict[str, float]] = None,
        variance_threshold: float = 0.15,
        max_parameter_change: float = 0.20,
        time_window: int = 7,
        method: str = "nelder_mead",
        max_iterations: int = 50,
        objectives: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        use_constraints: bool = True,
        **kwargs: Any,
    ) -> CalibrationResult:
        """Perform online re-calibration with bounded parameter adjustments.

        Args:
            measurements: Recent measurement data
            parameters: Parameters to adjust
            current_parameters: Current parameter values
            variance_threshold: Variance threshold for triggering (0-1)
            max_parameter_change: Maximum relative parameter change (0-1)
            time_window: Days of recent data to use
            method: Optimization method name
            max_iterations: Max optimization iterations
            objectives: List of outputs to match
            weights: Objective weights
            use_constraints: Whether to apply parameter constraints
            **kwargs: Extra optimizer settings

        Returns:
            CalibrationResult instance
        """
        start_time = time.time()
        if objectives is None:
            objectives = ["Q_ch4", "pH"]

        if parameters is None:
            if self.state.parameter_history:
                parameters = list(self.state.parameter_history[-1].parameters.keys())
            else:
                raise ValueError("No parameters specified and no calibration history available")

        if current_parameters is None:
            current_parameters = self._get_current_parameters()

        windowed_data = self._extract_time_window(measurements, time_window)
        current_variance = self._calculate_prediction_variance(windowed_data, current_parameters, objectives)
        self.state.current_variance = current_variance

        param_bounds = self._setup_online_bounds(parameters, current_parameters, max_parameter_change)

        def simulator_wrapper(params: Dict[str, float]) -> Dict[str, np.ndarray]:
            return self.simulator.simulate_with_parameters(params, windowed_data)

        measurements_dict: Dict[str, np.ndarray] = {
            obj: windowed_data.get_measurement(obj).values for obj in objectives if obj in windowed_data.data.columns
        }

        objective_func: Callable[[np.ndarray], float] = MultiObjectiveFunction(
            simulator=simulator_wrapper,
            measurements_dict=measurements_dict,
            objectives=objectives,
            weights=weights or {obj: 1.0 / len(objectives) for obj in objectives},
            parameter_names=parameters,
            error_metric="rmse",
            normalize=True,
        )

        obj_func_final: Callable[[np.ndarray], float]
        if use_constraints:
            constraints = ParameterConstraints()
            for p, (lb, ub) in param_bounds.items():
                constraints.add_box_constraint(p, lb, ub, hard=True)

            def penalized_objective(x: np.ndarray) -> float:
                params = {name: val for name, val in zip(parameters, x)}
                return objective_func(x) + constraints.calculate_penalty(params)

            obj_func_final = penalized_objective
        else:
            obj_func_final = objective_func

        optimizer = create_optimizer(
            method=method, bounds=param_bounds, max_iterations=max_iterations, verbose=self.verbose, **kwargs
        )

        initial_guess = np.array(
            [current_parameters.get(p, self.parameter_bounds.get_default_values([p])[p]) for p in parameters]
        )
        opt_result = optimizer.optimize(obj_func_final, initial_guess=initial_guess)

        validation_metrics: Dict[str, float] = {}
        if opt_result.success:
            val_res = self.validator.validate(
                parameters=opt_result.parameter_dict, measurements=windowed_data, objectives=objectives
            )
            validation_metrics = {f"{obj}_{k}": float(getattr(m, k)) for obj, m in val_res.items() for k in ["rmse", "r2"]}

        self.state.total_calibrations += 1
        self.state.last_calibration_time = datetime.now()

        history_entry = ParameterChangeHistory(
            timestamp=datetime.now(),
            parameters=opt_result.parameter_dict.copy(),
            trigger_reason="variance_threshold" if current_variance > variance_threshold else "manual",
            objective_value=float(opt_result.fun),
            variance=float(current_variance),
            success=bool(opt_result.success),
        )
        self.state.parameter_history.append(history_entry)

        return CalibrationResult(
            success=opt_result.success,
            parameters=opt_result.parameter_dict,
            initial_parameters=current_parameters,
            objective_value=float(opt_result.fun),
            n_iterations=int(opt_result.nit),
            execution_time=time.time() - start_time,
            method=method,
            message=str(getattr(opt_result, "message", "Online calibration completed")),
            validation_metrics=validation_metrics,
        )

    def should_recalibrate(
        self, recent_measurements: MeasurementData, objectives: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """Check if re-calibration should be triggered.

        Returns:
            Tuple of (should_recalibrate, reason)
        """
        if not self.trigger.enabled:
            return False, "Disabled"

        if objectives is None:
            objectives = ["Q_ch4", "pH"]

        if self.state.last_calibration_time:
            hours = (datetime.now() - self.state.last_calibration_time).total_seconds() / 3600
            if hours < self.trigger.time_threshold:
                return False, f"Too soon since last calibration ({hours:.1f}h)"

        variance = self._calculate_prediction_variance(recent_measurements, self._get_current_parameters(), objectives)
        self.state.current_variance = variance

        if variance > self.trigger.variance_threshold:
            self.state.consecutive_violations += 1
            if self.state.consecutive_violations >= self.trigger.consecutive_violations:
                return True, f"Variance {variance:.4f} > {self.trigger.variance_threshold}"
        else:
            self.state.consecutive_violations = 0

        return False, "Prediction within accuracy threshold"

    def apply_calibration(self, result: CalibrationResult) -> None:
        """Apply calibration parameters to the plant."""
        for component in self.plant.components.values():
            if component.component_type.value == "digester":
                component.apply_calibration_parameters(result.parameters)

    def _extract_time_window(self, measurements: MeasurementData, window_days: int) -> MeasurementData:
        """Extract recent time window from measurements."""
        last_time = measurements.data.index[-1]
        return measurements.get_time_window(last_time - timedelta(days=window_days), last_time)

    def _calculate_prediction_variance(
        self, measurements: MeasurementData, parameters: Dict[str, float], objectives: List[str]
    ) -> float:
        """Calculate prediction variance for current parameters."""
        try:
            outputs = self.simulator.simulate_with_parameters(parameters, measurements)
            variances: List[float] = []
            for obj in objectives:
                if obj not in outputs:
                    continue
                m = measurements.get_measurement(obj).values
                s = np.atleast_1d(outputs[obj])
                length = min(len(m), len(s))
                m, s = m[:length], s[:length]
                valid = ~(np.isnan(m) | np.isnan(s))
                if not np.any(valid):
                    continue
                res = m[valid] - s[valid]
                variances.append(float(np.std(res) / (np.mean(np.abs(m[valid])) + 1e-10)))
            return float(np.mean(variances)) if variances else 0.0
        except Exception:
            return 0.0

    def _setup_online_bounds(
        self, parameters: List[str], current_params: Dict[str, float], max_change: float
    ) -> Dict[str, Tuple[float, float]]:
        """Setup bounded parameter ranges for online calibration."""
        bounds: Dict[str, Tuple[float, float]] = {}
        for p in parameters:
            curr = current_params.get(p, 0.0)
            default = self.parameter_bounds.get_bounds_tuple(p) or (curr * 0.5, curr * 1.5)
            bounds[p] = (max(default[0], curr * (1 - max_change)), min(default[1], curr * (1 + max_change)))
        return bounds
