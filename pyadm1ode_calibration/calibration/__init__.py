"""
Calibration framework for ADM1 models.

Provides tools for initial batch calibration and online re-calibration
of biogas plant model parameters.
"""

from typing import List, Any, TYPE_CHECKING
from .core.base_calibrator import BaseCalibrator
from .core.result import CalibrationResult
from .methods.initial import InitialCalibrator
from .methods.online import OnlineCalibrator
from .parameter_bounds import ParameterBounds, ParameterBound, BoundType, create_default_bounds
from .validation import CalibrationValidator, ValidationMetrics

if TYPE_CHECKING:
    from pyadm1ode_calibration.io.loaders.measurement_data import MeasurementData


class Calibrator:
    """
    Orchestration layer for calibration workflows.

    Provides a simplified interface for running both initial and online
    calibrations on a plant model.

    Args:
        plant: The PyADM1ODE plant model instance.
        verbose (bool): Whether to enable verbose logging. Defaults to True.
    """

    def __init__(self, plant: Any, verbose: bool = True):
        self.plant = plant
        self.verbose = verbose
        self.initial_calibrator = InitialCalibrator(plant, verbose)
        self.online_calibrator = OnlineCalibrator(plant, verbose)

    def run_initial_calibration(
        self, measurements: "MeasurementData", parameters: List[str], **kwargs: Any
    ) -> CalibrationResult:
        """
        Run initial batch calibration from historical data.

        Args:
            measurements: Historical measurement data.
            parameters: List of parameter names to calibrate.
            **kwargs: Additional settings for InitialCalibrator.

        Returns:
            CalibrationResult: Results of the calibration.
        """
        return self.initial_calibrator.calibrate(measurements, parameters, **kwargs)

    def run_online_calibration(
        self, measurements: "MeasurementData", parameters: List[str], **kwargs: Any
    ) -> CalibrationResult:
        """
        Run online re-calibration for real-time adjustments.

        Args:
            measurements: Recent measurement data.
            parameters: List of parameter names to calibrate.
            **kwargs: Additional settings for OnlineCalibrator.

        Returns:
            CalibrationResult: Results of the calibration.
        """
        return self.online_calibrator.calibrate(measurements, parameters, **kwargs)

    def apply_calibration(self, result: CalibrationResult) -> None:
        """
        Apply calibration results to the plant model.

        Args:
            result (CalibrationResult): Result containing new parameters.
        """
        self.online_calibrator.apply_calibration(result)


__all__ = [
    "BaseCalibrator",
    "CalibrationResult",
    "InitialCalibrator",
    "OnlineCalibrator",
    "ParameterBounds",
    "ParameterBound",
    "BoundType",
    "create_default_bounds",
    "CalibrationValidator",
    "ValidationMetrics",
    "Calibrator",
]
