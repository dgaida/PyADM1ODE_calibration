"""Sensitivity analysis module."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.simulator import PlantSimulator


@dataclass
class SensitivityResult:
    """
    Result from sensitivity analysis for a single parameter.

    Attributes:
        parameter (str): Name of the analyzed parameter.
        base_value (float): Nominal value used during analysis.
        sensitivity_indices (Dict[str, float]): Dimensionless sensitivity indices
            for each objective.
        local_gradient (Dict[str, float]): Partial derivatives of objectives
            with respect to this parameter.
        normalized_sensitivity (Dict[str, float]): Sensitivity normalized by
            output standard deviation.
        variance_contribution (float): Total variance contribution (sum of squares).
        sensitivity_column (np.ndarray): One scaled sensitivity per observation, that
            is per objective and time step, concatenated in the order of the
            ``objectives`` argument. This is the column this parameter contributes to
            the sensitivity matrix that :func:`..identifiability.collinearity_index`
            works on. The scalars above are means of it and cannot replace it: two
            parameters can have the same mean effect while deforming the curve in
            completely different ways, and only the full column shows that.
    """

    parameter: str
    base_value: float
    sensitivity_indices: dict[str, float]
    local_gradient: dict[str, float]
    normalized_sensitivity: dict[str, float]
    variance_contribution: float
    sensitivity_column: np.ndarray = field(default_factory=lambda: np.empty(0))


class SensitivityAnalyzer:
    """
    Performs local sensitivity analysis on ADM1 plant parameters.

    Calculates how changes in input parameters affect specific model outputs,
    helping to identify which parameters are most influential.

    Args:
        plant: The PyADM1ODE plant model instance.
        simulator (Optional[PlantSimulator]): Simulator instance to use.
        verbose (bool): Whether to enable progress output. Defaults to True.
    """

    def __init__(self, plant, simulator: PlantSimulator | None = None, verbose: bool = True):
        self.plant = plant
        self.simulator = simulator or PlantSimulator(plant, verbose)
        self.verbose = verbose

    def analyze(
        self,
        parameters: dict[str, float],
        measurements: Any,
        objectives: list[str] | None = None,
        perturbation: float = 0.01,
    ) -> dict[str, SensitivityResult]:
        """
        Perform local sensitivity analysis using finite differences.

        Args:
            parameters (Dict[str, float]): Baseline parameter set.
            measurements (Any): Input data for simulation.
            objectives (Optional[List[str]]): List of objective variables to analyze.
            perturbation (float): Relative step size for finite differences. Defaults to 0.01.

        Returns:
            Dict[str, SensitivityResult]: Mapping of parameter names to their
                respective sensitivity analysis results.
        """
        if objectives is None:
            objectives = ["Q_ch4"]

        if self.verbose:
            print(f"\nSensitivity analysis for {len(parameters)} parameters")
            print(f"Objectives: {objectives}")
            print(f"Perturbation: {perturbation * 100:.1f}%")

        results = {}

        # The base case does not depend on which parameter is being perturbed, so run
        # it once instead of once per parameter. 2N+1 simulations rather than 3N.
        outputs_base = self.simulator.simulate_with_parameters(parameters, measurements)

        for param_name, base_value in parameters.items():
            delta = base_value * perturbation if base_value != 0 else perturbation

            param_plus = parameters.copy()
            param_plus[param_name] = base_value + delta

            param_minus = parameters.copy()
            param_minus[param_name] = base_value - delta

            outputs_plus = self.simulator.simulate_with_parameters(param_plus, measurements)
            outputs_minus = self.simulator.simulate_with_parameters(param_minus, measurements)

            local_gradient, sensitivity_indices, normalized_sensitivity = {}, {}, {}
            column_parts: list[np.ndarray] = []

            for obj in objectives:
                if all(obj in out for out in [outputs_base, outputs_plus, outputs_minus]):
                    series_base = np.asarray(outputs_base[obj], dtype=float)
                    series_plus = np.asarray(outputs_plus[obj], dtype=float)
                    series_minus = np.asarray(outputs_minus[obj], dtype=float)

                    # A channel the simulation did not produce carries no information.
                    # Averaging it would yield NaN and poison every index built on it.
                    if min(series_base.size, series_plus.size, series_minus.size) == 0:
                        continue

                    base_val = np.mean(series_base)
                    plus_val = np.mean(series_plus)
                    minus_val = np.mean(series_minus)

                    gradient = (plus_val - minus_val) / (2 * delta)
                    local_gradient[obj] = gradient
                    sensitivity_indices[obj] = gradient * (base_value / base_val) if base_val != 0 else 0.0

                    base_std = np.std(series_base)
                    normalized_sensitivity[obj] = abs(gradient * delta / base_std) if base_std > 0 else 0.0

                    # Per-observation sensitivity, scaled the way Brun et al. 2001 do:
                    # by the parameter value so the columns are comparable across
                    # parameters, and by the spread of the channel so they are
                    # comparable across channels of different units. A flat channel
                    # carries no information and contributes zeros rather than noise
                    # blown up by a near-zero divisor.
                    scale = base_std if base_std > 0 else 0.0
                    if scale > 0:
                        column_parts.append((series_plus - series_minus) / (2 * delta) * base_value / scale)
                    else:
                        column_parts.append(np.zeros_like(series_base))

            variance_contrib = sum(s**2 for s in sensitivity_indices.values())
            column = np.concatenate(column_parts) if column_parts else np.empty(0)

            results[param_name] = SensitivityResult(
                parameter=param_name,
                base_value=base_value,
                sensitivity_indices=sensitivity_indices,
                local_gradient=local_gradient,
                normalized_sensitivity=normalized_sensitivity,
                variance_contribution=variance_contrib,
                sensitivity_column=column,
            )

        return results
