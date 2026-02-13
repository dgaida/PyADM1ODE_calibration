"""Custom exceptions for PyADM1ODE_calibration."""

class PyADM1CalibrationError(Exception):
    """Base exception for calibration package."""
    pass

class DataValidationError(PyADM1CalibrationError):
    """Raised when measurement data fails validation."""
    pass

class CalibrationError(PyADM1CalibrationError):
    """Raised when calibration fails."""
    pass

class SimulationError(PyADM1CalibrationError):
    """Raised when plant simulation fails."""
    pass

class DatabaseError(PyADM1CalibrationError):
    """Raised for database operation failures."""
    pass
