import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from ..core.simulator import PlantSimulator


@dataclass
class SensitivityResult:
    """Result from sensitivity analysis."""

    parameter: str
    base_value: float
    sensitivity_indices: Dict[str, float]
    local_gradient: Dict[str, float]
    normalized_sensitivity: Dict[str, float]
    variance_contribution: float


class SensitivityAnalyzer:
    """Performs local sensitivity analysis on plant parameters."""

    def __init__(self, plant, simulator: Optional[PlantSimulator] = None, verbose: bool = True):
        self.plant = plant
        self.simulator = simulator or PlantSimulator(plant, verbose)
        self.verbose = verbose

    def analyze(
        self,
        parameters: Dict[str, float],
        measurements: Any,
        objectives: Optional[List[str]] = None,
        perturbation: float = 0.01,
    ) -> Dict[str, SensitivityResult]:
        """Perform local sensitivity analysis."""
        if objectives is None:
            objectives = ["Q_ch4"]

        if self.verbose:
            print(f"\nSensitivity analysis for {len(parameters)} parameters")
            print(f"Objectives: {objectives}")
            print(f"Perturbation: {perturbation * 100:.1f}%")

        results = {}

        for param_name, base_value in parameters.items():
            delta = base_value * perturbation if base_value != 0 else perturbation

            param_plus = parameters.copy()
            param_plus[param_name] = base_value + delta

            param_minus = parameters.copy()
            param_minus[param_name] = base_value - delta

            outputs_base = self.simulator.simulate_with_parameters(parameters, measurements)
            outputs_plus = self.simulator.simulate_with_parameters(param_plus, measurements)
            outputs_minus = self.simulator.simulate_with_parameters(param_minus, measurements)

            local_gradient, sensitivity_indices, normalized_sensitivity = {}, {}, {}

            for obj in objectives:
                if all(obj in out for out in [outputs_base, outputs_plus, outputs_minus]):
                    base_val = np.mean(outputs_base[obj])
                    plus_val = np.mean(outputs_plus[obj])
                    minus_val = np.mean(outputs_minus[obj])

                    gradient = (plus_val - minus_val) / (2 * delta)
                    local_gradient[obj] = gradient
                    sensitivity_indices[obj] = gradient * (base_value / base_val) if base_val != 0 else 0.0

                    base_std = np.std(outputs_base[obj])
                    normalized_sensitivity[obj] = abs(gradient * delta / base_std) if base_std > 0 else 0.0

            variance_contrib = sum(s**2 for s in sensitivity_indices.values())

            results[param_name] = SensitivityResult(
                parameter=param_name,
                base_value=base_value,
                sensitivity_indices=sensitivity_indices,
                local_gradient=local_gradient,
                normalized_sensitivity=normalized_sensitivity,
                variance_contribution=variance_contrib,
            )

        return results
