from typing import Dict, List, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from pyadm1ode_calibration.io.loaders.measurement_data import MeasurementData


class PlantSimulator:
    """Handles plant simulation with parameter variations.

    Separates simulation logic from calibration algorithms.
    """

    def __init__(self, plant: Any, verbose: bool = True):
        """Initialize simulator.

        Args:
            plant: BiogasPlant instance
            verbose: Enable progress output
        """
        self.plant = plant
        self.verbose = verbose
        self._original_params: Dict[str, Dict[str, float]] = {}

    def simulate_with_parameters(
        self, parameters: Dict[str, float], measurements: "MeasurementData", restore_params: bool = True
    ) -> Dict[str, np.ndarray]:
        """Run simulation with given parameters.

        Args:
            parameters: Parameter values to apply {name: value}
            measurements: Measurement data for substrate feeds
            restore_params: Whether to restore original parameters after simulation

        Returns:
            Dictionary mapping output names to simulated arrays
        """
        if restore_params:
            self._backup_parameters()

        try:
            self._apply_parameters(parameters)

            # Simulation settings
            n_steps = len(measurements)
            dt = 1.0 / 24.0
            duration = n_steps * dt

            # Apply substrate feeds from measurements
            Q_substrates = self._extract_substrate_feeds(measurements)
            self._apply_substrate_feeds(Q_substrates)

            # Run simulation
            results = self.plant.simulate(duration=duration, dt=dt, save_interval=dt)

            return self._extract_outputs_from_results(results)
        finally:
            if restore_params:
                self._restore_parameters()

    def _backup_parameters(self) -> None:
        """Store current parameter values from all digesters."""
        self._original_params = {}
        for component_id, component in self.plant.components.items():
            if component.component_type.value == "digester":
                self._original_params[component_id] = getattr(component, "_calibration_params", {}).copy()

    def _restore_parameters(self) -> None:
        """Restore previous parameter values to all digesters."""
        for component_id, params in self._original_params.items():
            component = self.plant.components[component_id]
            component._calibration_params = params.copy()

    def _apply_parameters(self, parameters: Dict[str, float]) -> None:
        """Apply parameters to all digesters in the plant.

        Args:
            parameters: Dictionary of parameter names and values
        """
        for component in self.plant.components.values():
            if component.component_type.value == "digester":
                if not hasattr(component, "_calibration_params"):
                    component._calibration_params = {}
                for name, val in parameters.items():
                    component._calibration_params[name] = val

    def _extract_substrate_feeds(self, measurements: "MeasurementData") -> List[float]:
        """Extract mean substrate feed rates from measurements.

        Args:
            measurements: MeasurementData instance

        Returns:
            List of average feed rates for each substrate
        """
        try:
            Q = measurements.get_substrate_feeds()
            return list(np.mean(Q, axis=0))
        except Exception:
            # Default substrate mix if not found
            return [15.0, 10.0] + [0.0] * 8

    def _apply_substrate_feeds(self, Q_substrates: List[float]) -> None:
        """Apply substrate feeds to all digester components.

        Args:
            Q_substrates: List of substrate feed rates
        """
        for component in self.plant.components.values():
            if component.component_type.value == "digester":
                component.Q_substrates = Q_substrates
                component.adm1.create_influent(Q_substrates, 0)

    def _extract_outputs_from_results(self, results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract relevant outputs from simulation results.

        Args:
            results: List of result dictionaries from plant.simulate()

        Returns:
            Dictionary mapping output names to numpy arrays
        """
        outputs: Dict[str, List[float]] = {
            "Q_ch4": [],
            "Q_gas": [],
            "Q_co2": [],
            "pH": [],
            "VFA": [],
            "TAC": [],
        }

        for result in results:
            components = result.get("components", {})
            q_ch4_total = 0.0
            q_gas_total = 0.0
            q_co2_total = 0.0
            pH_list, vfa_list, tac_list = [], [], []

            for component_result in components.values():
                q_ch4_total += component_result.get("Q_ch4", 0.0)
                q_gas_total += component_result.get("Q_gas", 0.0)
                q_co2_total += component_result.get("Q_co2", 0.0)
                if "pH" in component_result:
                    pH_list.append(component_result["pH"])
                if "VFA" in component_result:
                    vfa_list.append(component_result["VFA"])
                if "TAC" in component_result:
                    tac_list.append(component_result["TAC"])

            outputs["Q_ch4"].append(q_ch4_total)
            outputs["Q_gas"].append(q_gas_total)
            outputs["Q_co2"].append(q_co2_total)
            outputs["pH"].append(np.mean(pH_list) if pH_list else 7.0)
            outputs["VFA"].append(np.mean(vfa_list) if vfa_list else 0.0)
            outputs["TAC"].append(np.mean(tac_list) if tac_list else 0.0)

        return {k: np.array(v) for k, v in outputs.items()}
