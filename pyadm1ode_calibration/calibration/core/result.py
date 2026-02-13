from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

@dataclass
class CalibrationResult:
    """
    Result from a calibration run.

    Attributes:
        success: Whether calibration converged successfully
        parameters: Calibrated parameter values
        initial_parameters: Initial parameter values before calibration
        objective_value: Final objective function value
        n_iterations: Number of optimization iterations
        execution_time: Wall clock time [seconds]
        method: Optimization method used
        message: Status message
        validation_metrics: Metrics on validation data
        sensitivity: Parameter sensitivity analysis results
        history: Optimization history (if available)
        timestamp: Calibration timestamp
    """

    success: bool
    parameters: Dict[str, float]
    initial_parameters: Dict[str, float]
    objective_value: float
    n_iterations: int
    execution_time: float
    method: str
    message: str
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    sensitivity: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.

        Returns:
            Dictionary containing all result data
        """
        return {
            "success": self.success,
            "parameters": self.parameters,
            "initial_parameters": self.initial_parameters,
            "objective_value": self.objective_value,
            "n_iterations": self.n_iterations,
            "execution_time": self.execution_time,
            "method": self.method,
            "message": self.message,
            "validation_metrics": self.validation_metrics,
            "sensitivity": self.sensitivity,
            "history": self.history,
            "timestamp": self.timestamp,
        }

    def to_json(self, filepath: str) -> None:
        """Save result to JSON file.

        Args:
            filepath: Destination file path
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationResult":
        """Create result from dictionary.

        Args:
            data: Dictionary with result attributes

        Returns:
            CalibrationResult instance
        """
        return cls(**data)
