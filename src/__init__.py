# ============================================================================
# src/__init__.py
# ============================================================================
"""
PyADM1ODE_calibration - Parameter Calibration Framework for Biogas Plant Models

This package provides comprehensive calibration tools for PyADM1ODE biogas plant models.
"""

from .io import CSVHandler, Database, DatabaseConfig, Plant, MeasurementData, DataValidator, OutlierDetector
from .calibration import (Calibrator, CalibrationResult, InitialCalibrator, OnlineCalibrator, ParameterBounds,
                          ParameterBound, BoundType, create_default_bounds, CalibrationValidator, ValidationMetrics)

__all__ = [
    "CSVHandler",
    "Database",
    "DatabaseConfig",
    "Plant",
    "MeasurementData",
    "DataValidator",
    "OutlierDetector",
    "Calibrator",
    "CalibrationResult",
    "InitialCalibrator",
    "OnlineCalibrator",
    "ParameterBounds",
    "ParameterBound",
    "BoundType",
    "create_default_bounds",
    "CalibrationValidator",
    "ValidationMetrics",
]

__version__ = "0.1.0"
__author__ = "Daniel Gaida"
__email__ = "daniel.gaida@th-koeln.de"
