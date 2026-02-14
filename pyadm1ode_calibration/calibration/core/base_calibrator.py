"""Base calibrator module."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, TYPE_CHECKING
from .simulator import PlantSimulator

if TYPE_CHECKING:
    from pyadm1ode_calibration.io.loaders.measurement_data import MeasurementData
    from .result import CalibrationResult


class BaseCalibrator(ABC):
    """
    Base class for all calibrators.

    Provides common functionality for plant simulation and parameter management
    across different calibration methods (e.g., initial, online).

    Args:
        plant: The PyADM1ODE plant model to calibrate.
        verbose (bool): Whether to enable verbose logging. Defaults to True.
    """

    def __init__(self, plant, verbose: bool = True):
        self.plant = plant
        self.verbose = verbose
        self.simulator = PlantSimulator(plant, verbose)

    @abstractmethod
    def calibrate(self, measurements: "MeasurementData", parameters: List[str], **kwargs) -> "CalibrationResult":
        """
        Run the calibration workflow.

        Args:
            measurements (MeasurementData): The historical or real-time measurement data.
            parameters (List[str]): List of parameter names to calibrate.
            **kwargs: Additional method-specific configuration.

        Returns:
            CalibrationResult: The results of the calibration process.
        """
        pass

    def _simulate_with_parameters(
        self, parameters: Dict[str, float], measurements: "MeasurementData", restore_params: bool = False
    ) -> Dict[str, Any]:
        """
        Delegate simulation to the internal plant simulator.

        Args:
            parameters (Dict[str, float]): Parameters to use for the simulation.
            measurements (MeasurementData): Input data for the simulation.
            restore_params (bool): Whether to restore original parameters after simulation.

        Returns:
            Dict[str, Any]: Simulation results containing component-wise outputs.
        """
        return self.simulator.simulate_with_parameters(parameters, measurements, restore_params)

    def _get_current_parameters(self) -> Dict[str, float]:
        """
        Get current parameter values from the plant's digester component.

        Returns:
            Dict[str, float]: Current parameter names and their values.
        """
        for component in self.plant.components.values():
            if component.component_type.value == "digester":
                return getattr(component, "_calibration_params", {}).copy()
        return {}

    def _apply_parameters_to_plant(self, parameters: Dict[str, float]) -> None:
        """
        Apply a set of parameters to the plant model.

        Args:
            parameters (Dict[str, float]): Dictionary mapping parameter names to values.
        """
        self.simulator._apply_parameters(parameters)
