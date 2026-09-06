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
    CalibrationResult,
    CalibrationValidator,
    Calibrator,
    InitialCalibrator,
    OnlineCalibrator,
    ParameterBounds,
    ValidationMetrics,
    create_default_bounds,
)
from .exceptions import (
    CalibrationError,
    DatabaseError,
    DataValidationError,
    PyADM1CalibrationError,
    SimulationError,
)
from .io import (
    CSVHandler,
    Database,
    DatabaseConfig,
    DataValidator,
    MeasurementData,
    OutlierDetector,
    Plant,
    ValidationResult,
)

__version__ = "0.1.2"

__all__ = [
    "CSVHandler",
    "CalibrationError",
    "CalibrationResult",
    "CalibrationValidator",
    "Calibrator",
    "DataValidationError",
    "DataValidator",
    "Database",
    "DatabaseConfig",
    "DatabaseError",
    "InitialCalibrator",
    "MeasurementData",
    "OnlineCalibrator",
    "OutlierDetector",
    "ParameterBounds",
    "Plant",
    "PyADM1CalibrationError",
    "SimulationError",
    "ValidationMetrics",
    "ValidationResult",
    "create_default_bounds",
]
