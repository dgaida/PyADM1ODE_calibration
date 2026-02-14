"""
Custom exceptions for PyADM1ODE_calibration.

This module defines the exception hierarchy used throughout the
calibration package to handle various error states.
"""


class PyADM1CalibrationError(Exception):
    """
    Base exception for the pyadm1ode_calibration package.

    All custom exceptions in this package inherit from this class.
    """

    pass


class DataValidationError(PyADM1CalibrationError):
    """
    Raised when measurement data fails validation.

    This error indicates that the provided measurement data is missing
    required columns, has invalid ranges, or fails quality checks.
    """

    pass


class CalibrationError(PyADM1CalibrationError):
    """
    Raised when calibration process fails.

    This includes failures in optimization algorithms or invalid
    calibration configurations.
    """

    pass


class SimulationError(PyADM1CalibrationError):
    """
    Raised when plant simulation fails.

    This error is triggered when the underlying PyADM1ODE model
    fails to converge or encounters numerical errors during simulation.
    """

    pass


class DatabaseError(PyADM1CalibrationError):
    """
    Raised for database operation failures.

    Includes connection errors, query failures, or schema mismatches
    when interacting with the PostgreSQL persistence layer.
    """

    pass
