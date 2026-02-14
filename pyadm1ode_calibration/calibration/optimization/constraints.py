"""Constraints module."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable
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
        """Check feasibility."""
        return self.lower <= value <= self.upper

    def project(self, value: float) -> float:
        """Project to bounds."""
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
        """Check feasibility."""
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
        """Calculate total penalty."""
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
