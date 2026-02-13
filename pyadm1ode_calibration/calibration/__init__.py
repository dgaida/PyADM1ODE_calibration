from .core.base_calibrator import BaseCalibrator
from .core.result import CalibrationResult
from .methods.initial import InitialCalibrator
from .methods.online import OnlineCalibrator
from .parameter_bounds import ParameterBounds, ParameterBound, BoundType, create_default_bounds
from .validation import CalibrationValidator, ValidationMetrics


class Calibrator:
    """Orchestration layer for calibration."""

    def __init__(self, plant, verbose: bool = True):
        self.plant = plant
        self.verbose = verbose
        self.initial_calibrator = InitialCalibrator(plant, verbose)
        self.online_calibrator = OnlineCalibrator(plant, verbose)

    def run_initial_calibration(self, measurements, parameters, **kwargs):
        return self.initial_calibrator.calibrate(measurements, parameters, **kwargs)

    def run_online_calibration(self, measurements, parameters, **kwargs):
        return self.online_calibrator.calibrate(measurements, parameters, **kwargs)

    def apply_calibration(self, result: CalibrationResult):
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
