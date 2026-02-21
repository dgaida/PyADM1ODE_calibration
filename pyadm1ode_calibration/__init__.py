"""
PyADM1ODE_calibration: Parameter Calibration Framework for Biogas Plant Models.

This package provides a comprehensive framework for the automated calibration
and re-calibration of Anaerobic Digestion Model No. 1 (ADM1) parameters
using measurement data from biogas plants.

Key modules:
- calibration: Optimization algorithms and calibration methods.
- io: Data loaders, persistence, and validation.
- exceptions: Custom error types.
"""

from .calibration import (
    Calibrator,
    InitialCalibrator,
    OnlineCalibrator,
    CalibrationResult,
    ParameterBounds,
    CalibrationValidator,
    ValidationMetrics,
)
from .io import (
    MeasurementData,
    CSVHandler,
    Database,
    DatabaseConfig,
    DataValidator,
    ValidationResult,
    OutlierDetector,
    Plant,
)
from .exceptions import (
    PyADM1CalibrationError,
    DataValidationError,
    CalibrationError,
    SimulationError,
    DatabaseError,
)

__version__ = "0.1.2"

__all__ = [
    "Calibrator",
    "InitialCalibrator",
    "OnlineCalibrator",
    "CalibrationResult",
    "ParameterBounds",
    "CalibrationValidator",
    "ValidationMetrics",
    "MeasurementData",
    "CSVHandler",
    "Database",
    "DatabaseConfig",
    "DataValidator",
    "ValidationResult",
    "OutlierDetector",
    "Plant",
    "PyADM1CalibrationError",
    "DataValidationError",
    "CalibrationError",
    "SimulationError",
    "DatabaseError",
]
