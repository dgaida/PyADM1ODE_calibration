"""Result module."""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from datetime import datetime
import json


@dataclass
class CalibrationResult:
    """
    Result from a calibration run.

    Stores the output of an optimization process, including calibrated values,
    performance metrics, and diagnostic information.

    Attributes:
        success (bool): Whether calibration converged successfully.
        parameters (Dict[str, float]): Calibrated parameter values.
        initial_parameters (Dict[str, float]): Initial parameter values before calibration.
        objective_value (float): Final objective function value.
        n_iterations (int): Number of optimization iterations.
        execution_time (float): Wall clock time in seconds.
        method (str): Optimization method used (e.g., 'differential_evolution').
        message (str): Status message from the optimizer.
        validation_metrics (Dict[str, float]): Metrics on validation data (RMSE, R2, etc.).
        sensitivity (Dict[str, Any]): Parameter sensitivity analysis results.
        history (List[Dict[str, Any]]): Optimization history if tracking was enabled.
        timestamp (str): Calibration timestamp in ISO format.
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
        """
        Convert the result to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing all result data.
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
        """
        Save the result to a JSON file.

        Args:
            filepath (str): Destination file path.
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationResult":
        """
        Create a CalibrationResult instance from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary with result attributes.

        Returns:
            CalibrationResult: A new instance populated with the data.
        """
        return cls(**data)
