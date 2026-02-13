# calibration/base_calibrator.py
from abc import ABC
from typing import Dict, List, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    # Nur für Type Checker, nicht zur Laufzeit
    from pyadm1ode_calibration.io.measurement_data import MeasurementData


class BaseCalibrator(ABC):
    """Gemeinsame Basis für Initial- und Online-Kalibrierer."""

    def __init__(self, plant, verbose: bool = True):
        self.plant = plant
        self.verbose = verbose

    def _simulate_with_parameters(
        self, parameters: Dict[str, float], measurements: "MeasurementData", restore_params: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Simuliere Plant mit gegebenen Parametern.

        Args:
            parameters: Parameter-Werte
            measurements: Messdaten
            restore_params: Ob ursprüngliche Parameter wiederhergestellt werden sollen

        Returns:
            Dictionary mit simulierten Outputs
        """
        # Backup bei Bedarf
        original_params = {}
        if restore_params:
            original_params = self._backup_parameters()

        try:
            # Parameter anwenden
            self._apply_parameters_to_plant(parameters)

            # Simulation vorbereiten
            n_steps = len(measurements)
            dt = 1.0 / 24.0
            duration = n_steps * dt

            # Substrat-Feeds extrahieren und anwenden
            Q_substrates = self._extract_substrate_feeds(measurements)
            self._apply_substrate_feeds(Q_substrates)

            # Simulation durchführen
            results = self.plant.simulate(duration=duration, dt=dt, save_interval=dt)

            # Outputs extrahieren
            return self._extract_outputs_from_results(results)

        finally:
            if restore_params:
                self._restore_parameters(original_params)

    def _extract_outputs_from_results(self, results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Extract relevant outputs from simulation results.

        Args:
            results: List of simulation results from plant.simulate().

        Returns:
            Dict[str, np.ndarray]: Extracted outputs as {output_name: array}.
        """
        outputs = {
            "Q_ch4": [],
            "Q_gas": [],
            "Q_co2": [],
            "pH": [],
            "VFA": [],
            "TAC": [],
        }

        for result in results:
            components = result.get("components", {})

            # Summiere über alle Komponenten
            q_ch4_total = 0.0
            q_gas_total = 0.0
            q_co2_total = 0.0
            pH_list = []
            vfa_list = []
            tac_list = []

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

    def _apply_parameters_to_plant(self, parameters: Dict[str, float]) -> None:
        """Wende Parameter auf alle Fermenter an."""
        digester_count = 0

        for component in self.plant.components.values():
            if component.component_type.value == "digester":
                digester_count += 1

                if not hasattr(component, "_calibration_params"):
                    component._calibration_params = {}

                for param_name, param_value in parameters.items():
                    component._calibration_params[param_name] = param_value

        if digester_count == 0:
            raise ValueError("Keine Fermenter in der Anlage gefunden")

    def _backup_parameters(self) -> Dict[str, Dict[str, float]]:
        """Sichere aktuelle Parameter."""
        backup = {}
        for component_id, component in self.plant.components.items():
            if component.component_type.value == "digester":
                backup[component_id] = getattr(component, "_calibration_params", {}).copy()
        return backup

    def _restore_parameters(self, backup: Dict[str, Dict[str, float]]) -> None:
        """Stelle gesicherte Parameter wieder her."""
        for component_id, params in backup.items():
            component = self.plant.components[component_id]
            component._calibration_params = params.copy()

    def _extract_substrate_feeds(self, measurements: "MeasurementData") -> List[float]:
        """
        Extract substrate feed rates from measurements.

        Looks for columns with names like 'Q_sub1', 'Q_sub2', etc., or
        uses a default substrate mix if not found.

        Args:
            measurements: Measurement data potentially containing substrate feeds.

        Returns:
            List[float]: Substrate feed rates in m³/d for each substrate.
        """
        try:
            Q = measurements.get_substrate_feeds()
            return list(np.mean(Q, axis=0))
        except Exception:
            # Default-Mix wenn nicht in Messdaten
            return [15.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def _apply_substrate_feeds(self, Q_substrates: List[float]) -> None:
        """Wende Substrat-Feeds auf Fermenter an."""
        for component in self.plant.components.values():
            if component.component_type.value == "digester":
                component.Q_substrates = Q_substrates
                component.adm1.create_influent(Q_substrates, 0)

    def _get_current_parameters(self) -> Dict[str, float]:
        """Hole aktuelle Parameter vom ersten Fermenter."""
        for component in self.plant.components.values():
            if component.component_type.value == "digester":
                return getattr(component, "_calibration_params", {}).copy()
        return {}
