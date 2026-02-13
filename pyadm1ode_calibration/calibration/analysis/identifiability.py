from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from .sensitivity import SensitivityAnalyzer


@dataclass
class IdentifiabilityResult:
    """Result from identifiability analysis."""

    parameter: str
    is_identifiable: bool
    confidence_interval: Tuple[float, float]
    correlation_with: Dict[str, float]
    objective_sensitivity: float
    reason: str


class IdentifiabilityAnalyzer:
    """Assesses parameter identifiability."""

    def __init__(self, plant, sensitivity_analyzer: Optional[SensitivityAnalyzer] = None, verbose: bool = True):
        self.plant = plant
        self.sensitivity_analyzer = sensitivity_analyzer or SensitivityAnalyzer(plant, verbose=verbose)
        self.verbose = verbose

    def analyze(
        self,
        parameters: Dict[str, float],
        measurements: Any,
        optimization_history: Optional[List[Dict[str, Any]]] = None,
        confidence_level: float = 0.95,
        correlation_threshold: float = 0.8,
    ) -> Dict[str, IdentifiabilityResult]:
        """Assess parameter identifiability."""
        sensitivity = self.sensitivity_analyzer.analyze(parameters, measurements, objectives=["Q_ch4", "pH", "VFA"])

        # Simple correlation analysis could be added here if history is provided

        results = {}
        for param_name, param_value in parameters.items():
            max_sensitivity = 0.0
            if param_name in sensitivity:
                max_sensitivity = max(abs(s) for s in sensitivity[param_name].sensitivity_indices.values())

            is_identifiable = max_sensitivity >= 1e-6
            reason = "Parameter is identifiable" if is_identifiable else f"Very low sensitivity (max: {max_sensitivity:.2e})"

            uncertainty = 0.1 * param_value / max(max_sensitivity, 1e-6)
            ci = (max(0, param_value - uncertainty), param_value + uncertainty) if is_identifiable else (0, param_value * 10)

            results[param_name] = IdentifiabilityResult(
                parameter=param_name,
                is_identifiable=is_identifiable,
                confidence_interval=ci,
                correlation_with={},  # Placeholder
                objective_sensitivity=max_sensitivity,
                reason=reason,
            )
        return results
