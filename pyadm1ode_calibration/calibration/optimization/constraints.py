"""Constraints module."""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class BoxConstraint:
    """
    Box constraint (bounds) for a single parameter.

    Attributes:
        parameter_name (str): Name of the parameter.
        lower (float): Lower bound.
        upper (float): Upper bound.
        hard (bool): If True, violations are strictly prohibited. Defaults to True.
    """

    parameter_name: str
    lower: float
    upper: float
    hard: bool = True

    def is_feasible(self, value: float) -> bool:
        """Check if value is within bounds."""
        return self.lower <= value <= self.upper

    def project(self, value: float) -> float:
        """Project value to bounds."""
        return np.clip(value, self.lower, self.upper)

    def violation(self, value: float) -> float:
        """Calculate violation magnitude."""
        if value < self.lower:
            return self.lower - value
        elif value > self.upper:
            return value - self.upper
        return 0.0


@dataclass
class LinearConstraint:
    """
    Linear constraint on multiple parameters.

    Attributes:
        coefficients (Dict[str, float]): Parameter coefficients.
        lower_bound (Optional[float]): Min sum.
        upper_bound (Optional[float]): Max sum.
        constraint_type (str): 'inequality' or 'equality'.
    """

    coefficients: Dict[str, float]
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    constraint_type: str = "inequality"

    def evaluate(self, parameters: Dict[str, float]) -> float:
        """Evaluate LHS."""
        return sum(coef * parameters.get(name, 0.0) for name, coef in self.coefficients.items())

    def is_feasible(self, parameters: Dict[str, float]) -> bool:
        """Check feasibility."""
        val = self.evaluate(parameters)
        if self.constraint_type == "equality":
            return abs(val - (self.upper_bound or 0.0)) < 1e-6
        if self.lower_bound is not None and val < self.lower_bound:
            return False
        if self.upper_bound is not None and val > self.upper_bound:
            return False
        return True

    def violation(self, parameters: Dict[str, float]) -> float:
        """Calculate violation."""
        val = self.evaluate(parameters)
        if self.constraint_type == "equality":
            return abs(val - (self.upper_bound or 0.0))
        violation = 0.0
        if self.lower_bound is not None and val < self.lower_bound:
            violation = max(violation, self.lower_bound - val)
        if self.upper_bound is not None and val > self.upper_bound:
            violation = max(violation, val - self.upper_bound)
        return violation


@dataclass
class NonlinearConstraint:
    """Nonlinear constraint."""

    name: str
    function: Callable[[Dict[str, float]], float]
    constraint_type: str = "inequality"
    tolerance: float = 1e-6

    def evaluate(self, parameters: Dict[str, float]) -> float:
        """Evaluate function."""
        return self.function(parameters)

    def is_feasible(self, parameters: Dict[str, float]) -> bool:
        """Check feasibility."""
        val = self.evaluate(parameters)
        if self.constraint_type == "equality":
            return abs(val) <= self.tolerance
        return val <= self.tolerance

    def violation(self, parameters: Dict[str, float]) -> float:
        """Calculate violation."""
        val = self.evaluate(parameters)
        if self.constraint_type == "equality":
            return abs(val)
        return max(0.0, val)


class PenaltyFunction(ABC):
    """Base penalty function."""

    @abstractmethod
    def __call__(self, violation: float, weight: float = 1.0) -> float:
        """Calculate penalty."""
        pass


class QuadraticPenalty(PenaltyFunction):
    """Quadratic penalty."""

    def __call__(self, violation: float, weight: float = 1.0) -> float:
        return weight * violation**2


class LinearPenalty(PenaltyFunction):
    """Linear penalty."""

    def __call__(self, violation: float, weight: float = 1.0) -> float:
        return weight * abs(violation)


class LogarithmicPenalty(PenaltyFunction):
    """Logarithmic penalty."""

    def __call__(self, violation: float, weight: float = 1.0) -> float:
        if violation <= 0:
            return 0.0
        return -weight * np.log(max(1e-10, violation))


class ExponentialPenalty(PenaltyFunction):
    """Exponential penalty."""

    def __call__(self, violation: float, weight: float = 1.0) -> float:
        if violation <= 0:
            return 0.0
        return weight * (np.exp(violation) - 1.0)


class BarrierPenalty(PenaltyFunction):
    """Barrier penalty."""

    def __call__(self, violation: float, weight: float = 1.0) -> float:
        if violation <= 0:
            return 0.0
        return weight / max(1e-10, violation)


class ParameterConstraints:
    """Manager for parameter constraints."""

    def __init__(self, penalty_function: Optional[PenaltyFunction] = None):
        self.box_constraints: Dict[str, BoxConstraint] = {}
        self.linear_constraints: List[LinearConstraint] = []
        self.nonlinear_constraints: List[NonlinearConstraint] = []
        self.penalty_function = penalty_function or QuadraticPenalty()
        self.penalty_weights: Dict[str, float] = {}

    def add_box_constraint(self, parameter_name: str, lower: float, upper: float, hard: bool = True, weight: float = 1.0):
        """Add box constraint."""
        self.box_constraints[parameter_name] = BoxConstraint(parameter_name, lower, upper, hard)
        if not hard:
            self.penalty_weights[f"box_{parameter_name}"] = weight

    def add_linear_inequality(self, coefficients, lower_bound=None, upper_bound=None, weight=1.0):
        """Add linear inequality."""
        c = LinearConstraint(coefficients, lower_bound, upper_bound, "inequality")
        self.linear_constraints.append(c)
        self.penalty_weights[f"linear_{len(self.linear_constraints)}"] = weight

    def add_linear_equality(self, coefficients, target, weight=1.0):
        """Add linear equality."""
        c = LinearConstraint(coefficients, None, target, "equality")
        self.linear_constraints.append(c)
        self.penalty_weights[f"linear_eq_{len(self.linear_constraints)}"] = weight

    def add_nonlinear_constraint(self, name, function, constraint_type="inequality", weight=1.0):
        """Add nonlinear constraint."""
        c = NonlinearConstraint(name, function, constraint_type)
        self.nonlinear_constraints.append(c)
        self.penalty_weights[f"nonlinear_{name}"] = weight

    def is_feasible(self, parameters: Dict[str, float]) -> bool:
        """Check if parameters satisfy all hard constraints."""
        for c in self.box_constraints.values():
            if c.hard and not c.is_feasible(parameters.get(c.parameter_name, 0.0)):
                return False
        for c in self.linear_constraints:
            if not c.is_feasible(parameters):
                return False
        for c in self.nonlinear_constraints:
            if not c.is_feasible(parameters):
                return False
        return True

    def calculate_penalty(self, parameters: Dict[str, float]) -> float:
        """Calculate total penalty for all violated constraints."""
        p = 0.0
        for name, c in self.box_constraints.items():
            v = c.violation(parameters.get(name, 0.0))
            if v > 0:
                if c.hard:
                    return float("inf")
                p += self.penalty_function(v, self.penalty_weights.get(f"box_{name}", 1.0))
        for i, c in enumerate(self.linear_constraints, 1):
            v = c.violation(parameters)
            if v > 0:
                p += self.penalty_function(v, self.penalty_weights.get(f"linear_{i}", 1.0))
        for c in self.nonlinear_constraints:
            v = c.violation(parameters)
            if v > 0:
                p += self.penalty_function(v, self.penalty_weights.get(f"nonlinear_{c.name}", 1.0))
        return p

    def project_to_feasible(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """Project parameters to feasible region (box constraints only)."""
        projected = parameters.copy()
        for name, constraint in self.box_constraints.items():
            if name in projected:
                projected[name] = constraint.project(projected[name])
        return projected

    def get_scipy_constraints(self, parameter_names: List[str]) -> List[Dict]:
        """Convert constraints to scipy format."""
        scipy_constraints = []
        for constraint in self.linear_constraints:
            coef_array = np.array([constraint.coefficients.get(name, 0.0) for name in parameter_names])
            if constraint.constraint_type == "equality":
                scipy_constraints.append(
                    {"type": "eq", "fun": lambda x, c=coef_array, b=constraint.upper_bound: np.dot(c, x) - b}
                )
            else:
                if constraint.lower_bound is not None:
                    scipy_constraints.append(
                        {"type": "ineq", "fun": lambda x, c=coef_array, b=constraint.lower_bound: np.dot(c, x) - b}
                    )
                if constraint.upper_bound is not None:
                    scipy_constraints.append(
                        {"type": "ineq", "fun": lambda x, c=coef_array, b=constraint.upper_bound: b - np.dot(c, x)}
                    )
        for constraint in self.nonlinear_constraints:

            def constraint_func(x, names=parameter_names, func=constraint.function):
                params = {name: val for name, val in zip(names, x)}
                return func(params)

            if constraint.constraint_type == "equality":
                scipy_constraints.append({"type": "eq", "fun": constraint_func})
            else:
                scipy_constraints.append({"type": "ineq", "fun": lambda x, f=constraint_func: -f(x)})
        return scipy_constraints

    def validate_parameters(self, parameters: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Validate parameters and return detailed error messages."""
        errors = []
        for name, constraint in self.box_constraints.items():
            if name in parameters:
                value = parameters[name]
                if not constraint.is_feasible(value):
                    errors.append(
                        f"Parameter '{name}' = {value:.4f} violates bounds [{constraint.lower:.4f}, {constraint.upper:.4f}]"
                    )
        for i, constraint in enumerate(self.linear_constraints, 1):
            if not constraint.is_feasible(parameters):
                value = constraint.evaluate(parameters)
                if constraint.constraint_type == "equality":
                    errors.append(f"Linear constraint {i}: {value:.4f} != {constraint.upper_bound:.4f}")
                else:
                    if constraint.lower_bound and value < constraint.lower_bound:
                        errors.append(f"Linear constraint {i}: {value:.4f} < {constraint.lower_bound:.4f}")
                    if constraint.upper_bound and value > constraint.upper_bound:
                        errors.append(f"Linear constraint {i}: {value:.4f} > {constraint.upper_bound:.4f}")
        for constraint in self.nonlinear_constraints:
            if not constraint.is_feasible(parameters):
                value = constraint.evaluate(parameters)
                errors.append(
                    f"Nonlinear constraint '{constraint.name}': g(x) = {value:.4f} violates "
                    f"{constraint.constraint_type} constraint"
                )
        return len(errors) == 0, errors


def create_penalty_function(penalty_type: str) -> PenaltyFunction:
    """Penalty factory."""
    m = {
        "quadratic": QuadraticPenalty,
        "linear": LinearPenalty,
        "logarithmic": LogarithmicPenalty,
        "exponential": ExponentialPenalty,
        "barrier": BarrierPenalty,
    }
    if penalty_type.lower() not in m:
        raise ValueError(f"Unknown penalty: {penalty_type}")
    return m[penalty_type.lower()]()
