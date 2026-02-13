from abc import ABC, abstractmethod
from typing import Dict, List, Any, TYPE_CHECKING
from .simulator import PlantSimulator

if TYPE_CHECKING:
    from pyadm1ode_calibration.io.loaders.measurement_data import MeasurementData
    from .result import CalibrationResult


class BaseCalibrator(ABC):
    """Base class for all calibrators.

    Provides common functionality for plant simulation and parameter management.
    """

    def __init__(self, plant, verbose: bool = True):
        self.plant = plant
        self.verbose = verbose
        self.simulator = PlantSimulator(plant, verbose)

    @abstractmethod
    def calibrate(self, measurements: "MeasurementData", parameters: List[str], **kwargs) -> "CalibrationResult":
        """Run calibration workflow."""
        pass

    def _simulate_with_parameters(
        self, parameters: Dict[str, float], measurements: "MeasurementData", restore_params: bool = False
    ) -> Dict[str, Any]:
        """Delegate to simulator."""
        return self.simulator.simulate_with_parameters(parameters, measurements, restore_params)

    def _get_current_parameters(self) -> Dict[str, float]:
        """Get current parameter values from plant."""
        for component in self.plant.components.values():
            if component.component_type.value == "digester":
                return getattr(component, "_calibration_params", {}).copy()
        return {}

    def _apply_parameters_to_plant(self, parameters: Dict[str, float]) -> None:
        """Apply parameters to plant."""
        self.simulator._apply_parameters(parameters)
