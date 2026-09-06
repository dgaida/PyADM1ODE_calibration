"""Identifiability analysis module.

A parameter is only worth calibrating if the data pin it down. Two ways it can fail,
following the taxonomy of Raue et al. 2009 (Bioinformatics 25:1923):

* *structurally* non-identifiable: the parameter does not move the measured outputs at
  all, so the likelihood is flat along it and its confidence interval is unbounded no
  matter how much data is collected.
* *practically* non-identifiable: it does move them, but too weakly for the data at
  hand, so the confidence interval stays wider than the estimate itself.

Both measures of Brun et al. 2001 (Water Resour. Res. 37:1015) are computed: the
sensitivity of each parameter on its own, and the near-linear dependence between them,
their collinearity index. The second matters because a set of individually identifiable
parameters can still be unidentifiable together, when one parameter's effect on the
outputs can be undone by another's.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .sensitivity import SensitivityAnalyzer

#: Below this a parameter moves no output beyond numerical noise: structurally
#: non-identifiable, and no amount of data changes that.
STRUCTURAL_SENSITIVITY_FLOOR = 1e-6

#: The confidence half-width may not exceed the estimate itself. Raue et al. 2009 call
#: a parameter practically non-identifiable once its interval is unbounded; an interval
#: reaching from zero to twice the estimate is that case in all but name, and it is
#: what a near-zero sensitivity produces here.
MAX_RELATIVE_UNCERTAINTY = 1.0

#: Above this collinearity index a parameter subset counts as unidentifiable as a set,
#: however well each member scores alone. Brun et al. 2001 put the practical limit
#: between 10 and 20; 20 is the customary one and corresponds to two sensitivity
#: directions enclosing an angle whose cosine is 0.9975, so the parameters do very
#: nearly the same thing to the model. The R package FME implements the same measure
#: as ``collin``.
MAX_COLLINEARITY_INDEX = 20.0

#: Guard against normalising a column that is numerically zero. This is not the test
#: for whether a parameter is worth including, which is the per-parameter criterion
#: above: a solver-noise column can be far larger than this and still mean nothing.
DEAD_COLUMN_RATIO = 1e-12


def collinearity_index(sensitivity_matrix: np.ndarray) -> float:
    """Brun's collinearity index for the given sensitivity matrix.

    Args:
        sensitivity_matrix: One column per parameter, one row per observation.

    Returns:
        float: ``1 / smallest singular value`` of the column-normalised matrix.
        1.0 means the parameters act on the outputs in completely independent
        directions, and the value grows without bound as they become linearly
        dependent. ``inf`` when a column is all zeros, that is when a parameter does
        nothing at all.
    """
    matrix = np.atleast_2d(np.asarray(sensitivity_matrix, dtype=float))
    if matrix.size == 0 or matrix.shape[1] == 0:
        return float("inf")

    norms = np.linalg.norm(matrix, axis=0)
    # Normalising to unit length is what makes the columns comparable, but it also
    # stretches a column of pure rounding noise into a full-blown direction, which
    # would make a parameter that does nothing look perfectly independent. Anything
    # negligible next to the strongest column is treated as the zero it is.
    if np.all(norms == 0) or np.any(norms <= norms.max() * DEAD_COLUMN_RATIO):
        return float("inf")

    smallest = np.linalg.svd(matrix / norms, compute_uv=False)[-1]
    return float("inf") if smallest <= 0 else float(1.0 / smallest)


@dataclass
class SubsetIdentifiability:
    """Whether a set of parameters can be estimated together.

    Attributes:
        parameters (List[str]): The subset this verdict is about.
        collinearity_index (float): Brun's index, see :func:`collinearity_index`.
        is_identifiable (bool): Whether the index stays below
            :data:`MAX_COLLINEARITY_INDEX`.
        reason (str): Explanation of the verdict.
    """

    parameters: list[str]
    collinearity_index: float
    is_identifiable: bool
    reason: str


@dataclass
class IdentifiabilityResult:
    """
    Result from parameter identifiability analysis.

    Determines if a parameter can be reliably estimated from the available data.

    Attributes:
        parameter (str): Name of the parameter.
        is_identifiable (bool): Whether the parameter is considered identifiable.
        confidence_interval (Tuple[float, float]): Estimated 95% confidence interval.
        correlation_with (Dict[str, float]): Correlation coefficients with other parameters.
        objective_sensitivity (float): Maximum sensitivity across all objectives.
        reason (str): Explanation for the identifiability status.
    """

    parameter: str
    is_identifiable: bool
    confidence_interval: tuple[float, float]
    correlation_with: dict[str, float]
    objective_sensitivity: float
    reason: str


class IdentifiabilityAnalyzer:
    """
    Assesses parameter identifiability for ADM1 calibration.

    Analyzes whether parameters have enough influence on model outputs
    and whether they are correlated with each other, which can hinder
    accurate estimation.

    Args:
        plant: The PyADM1ODE plant model instance.
        sensitivity_analyzer (Optional[SensitivityAnalyzer]): Analyzer for calculating gradients.
        verbose (bool): Whether to enable verbose output. Defaults to True.
    """

    def __init__(self, plant, sensitivity_analyzer: SensitivityAnalyzer | None = None, verbose: bool = True):
        self.plant = plant
        self.sensitivity_analyzer = sensitivity_analyzer or SensitivityAnalyzer(plant, verbose=verbose)
        self.verbose = verbose

    def analyze(
        self,
        parameters: dict[str, float],
        measurements: Any,
        optimization_history: list[dict[str, Any]] | None = None,
        confidence_level: float = 0.95,
        correlation_threshold: float = 0.8,
    ) -> dict[str, IdentifiabilityResult]:
        """
        Assess parameter identifiability based on sensitivity and correlation.

        Args:
            parameters (Dict[str, float]): Parameter set to analyze.
            measurements (Any): Measurement data for simulation.
            optimization_history (Optional[List[Dict[str, Any]]]): History from optimizer.
            confidence_level (float): Level for confidence intervals. Defaults to 0.95.
            correlation_threshold (float): Threshold for identifying high correlations.

        Returns:
            Dict[str, IdentifiabilityResult]: Results for each parameter. For whether
            the parameters can be estimated *together*, see :meth:`analyze_subset`.
        """
        sensitivity = self.sensitivity_analyzer.analyze(parameters, measurements, objectives=["Q_ch4", "pH", "VFA"])
        return self._verdicts(parameters, sensitivity, correlation_threshold)

    def _verdicts(
        self,
        parameters: dict[str, float],
        sensitivity: dict[str, Any],
        correlation_threshold: float = 0.8,
    ) -> dict[str, IdentifiabilityResult]:
        """Judge each parameter on its own, from an existing sensitivity analysis."""
        correlations = self._pairwise_correlations(sensitivity)

        results = {}
        for param_name, param_value in parameters.items():
            max_sensitivity = 0.0
            if param_name in sensitivity:
                # `default` matters: when not one objective could be evaluated the dict
                # is empty, and a bare max() would raise instead of reporting the
                # parameter as having no measurable effect, which is what it has.
                max_sensitivity = max((abs(s) for s in sensitivity[param_name].sensitivity_indices.values()), default=0.0)

            # A 10 % change in the parameter shifts the outputs by max_sensitivity
            # times that, so the change the data can still resolve scales inversely.
            uncertainty = 0.1 * param_value / max(max_sensitivity, STRUCTURAL_SENSITIVITY_FLOOR)
            relative_uncertainty = uncertainty / abs(param_value) if param_value else float("inf")

            if max_sensitivity < STRUCTURAL_SENSITIVITY_FLOOR:
                is_identifiable = False
                reason = f"No effect on any objective (max sensitivity {max_sensitivity:.2e}), structurally non-identifiable"
            elif relative_uncertainty > MAX_RELATIVE_UNCERTAINTY:
                is_identifiable = False
                reason = (
                    f"Confidence interval wider than the estimate "
                    f"(+-{relative_uncertainty * 100:.0f} %, sensitivity {max_sensitivity:.2e}), "
                    f"practically non-identifiable"
                )
            else:
                is_identifiable = True
                reason = f"Identifiable to +-{relative_uncertainty * 100:.0f} %"

            # Report the interval that was actually computed, wide or not, rather than
            # a placeholder. A structurally dead parameter shows up as a huge range,
            # which is the honest picture.
            ci = (max(0.0, param_value - uncertainty), param_value + uncertainty)

            results[param_name] = IdentifiabilityResult(
                parameter=param_name,
                is_identifiable=is_identifiable,
                confidence_interval=ci,
                correlation_with={
                    other: value
                    for other, value in correlations.get(param_name, {}).items()
                    if abs(value) >= correlation_threshold
                },
                objective_sensitivity=max_sensitivity,
                reason=reason,
            )
        return results

    def analyze_subset(
        self,
        parameters: dict[str, float],
        measurements: Any,
        objectives: list[str] | None = None,
    ) -> SubsetIdentifiability:
        """Can this set of parameters be estimated together?

        Individual sensitivity is necessary but not sufficient. Two parameters that
        each move the outputs strongly are still unidentifiable as a pair when one can
        undo what the other did, which is what the collinearity index measures.

        Costs ``2 * len(parameters) + 1`` simulations, so it is cheap enough to run
        before a calibration rather than after it.

        Args:
            parameters: The candidate set, name to nominal value.
            measurements: Data window the sensitivities are evaluated over.
            objectives: Channels to score against. Defaults to gas, pH and VFA.

        Returns:
            SubsetIdentifiability: The verdict for the set as a whole.
        """
        sensitivity = self.sensitivity_analyzer.analyze(
            parameters, measurements, objectives=objectives or ["Q_ch4", "pH", "VFA"]
        )
        return self.subset_verdict(sensitivity, self._verdicts(parameters, sensitivity))

    @staticmethod
    def subset_verdict(
        sensitivity: dict[str, Any],
        per_parameter: dict[str, IdentifiabilityResult],
    ) -> SubsetIdentifiability:
        """Judge a parameter set from an existing sensitivity analysis.

        Two stages, in the order Brun et al. 2001 prescribe. First each parameter is
        screened on its own, because one that fails alone cannot be rescued by the
        company it keeps. The collinearity index is then computed over the survivors
        only. Doing it the other way round lets a parameter made of solver noise pass
        as an independent direction, since normalising its column to unit length hides
        how small it was.
        """
        names = list(sensitivity)
        weak = [n for n in names if n in per_parameter and not per_parameter[n].is_identifiable]
        survivors = [n for n in names if n not in weak]

        columns = {n: np.asarray(sensitivity[n].sensitivity_column, dtype=float) for n in survivors}
        sizes = {c.size for c in columns.values()}
        if len(survivors) < 2 or not sizes or 0 in sizes or len(sizes) != 1:
            gamma = 1.0 if len(survivors) == 1 else float("inf")
        else:
            gamma = collinearity_index(np.column_stack([columns[n] for n in survivors]))

        collinear = gamma > MAX_COLLINEARITY_INDEX
        parts = []
        if weak:
            parts.append(f"{weak} not identifiable on their own")
        if collinear and len(survivors) >= 2:
            parts.append(
                f"collinearity index {gamma:.1f} above {MAX_COLLINEARITY_INDEX:.0f} for {survivors}, "
                f"they compensate each other, drop one"
            )
        elif len(survivors) >= 2:
            parts.append(f"collinearity index {gamma:.1f} for {survivors}")

        identifiable = not weak and not collinear
        reason = "; ".join(parts) if parts else "Nothing to judge"
        return SubsetIdentifiability(names, gamma, identifiable, reason)

    @staticmethod
    def _pairwise_correlations(sensitivity: dict[str, Any]) -> dict[str, dict[str, float]]:
        """Cosine of the angle between each pair of sensitivity directions.

        1.0 means two parameters deform the model output in exactly the same way, so
        the data cannot tell them apart. This is the pairwise view of the collinearity
        index, which considers the whole set at once.
        """
        names = list(sensitivity)
        unit: dict[str, np.ndarray] = {}
        for name in names:
            column = np.asarray(sensitivity[name].sensitivity_column, dtype=float)
            norm = np.linalg.norm(column) if column.size else 0.0
            if norm > 0:
                unit[name] = column / norm

        out: dict[str, dict[str, float]] = {n: {} for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if a in unit and b in unit and unit[a].size == unit[b].size:
                    value = float(np.dot(unit[a], unit[b]))
                    out[a][b] = value
                    out[b][a] = value
        return out
