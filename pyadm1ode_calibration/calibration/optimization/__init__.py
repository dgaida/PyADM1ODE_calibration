"""
Optimization Algorithms and Objective Functions

Optimization methods for parameter calibration with various algorithms and
customizable objective functions.

Modules:
    optimizer: Abstract Optimizer base class and concrete implementations including
              gradient-free methods (Nelder-Mead, Powell, differential evolution,
              particle swarm), gradient-based methods (L-BFGS-B, SLSQP), and
              multi-objective optimization (NSGA-II).

    objective: Objective function classes for single and multi-objective optimization
              including weighted sum of errors, likelihood-based objectives, and
              custom cost functions with support for different error metrics (MSE,
              MAE, log-likelihood).

    constraints: Constraint handling for parameter optimization including box constraints,
                linear constraints, nonlinear constraints, and penalty methods with
                different penalty functions (quadratic, logarithmic, barrier).

Example:
    >>> from pyadm1ode_calibration.calibration.optimization import (
    ...     DifferentialEvolutionOptimizer,
    ...     MultiObjectiveFunction,
    ...     ParameterConstraints
    ... )
    >>>
    >>> # Define objective function
    >>> objective = MultiObjectiveFunction(
    ...     targets=["Q_ch4", "pH", "VFA"],
    ...     weights=[0.6, 0.2, 0.2],
    ...     error_metric="rmse"
    ... )
    >>>
    >>> # Set up optimizer with constraints
    >>> optimizer = DifferentialEvolutionOptimizer(
    ...     objective=objective,
    ...     bounds=parameter_bounds,
    ...     population_size=50,
    ...     max_iterations=100
    ... )
    >>>
    >>> # Run optimization
    >>> result = optimizer.optimize(
    ...     plant=plant,
    ...     measurements=measurements
    ... )
"""

from .constraints import (
    BoxConstraint,
    LinearConstraint,
    NonlinearConstraint,
    ParameterConstraints,
    PenaltyFunction,
)
from .objective import (
    LikelihoodObjective,
    MultiObjectiveFunction,
    ObjectiveFunction,
    SingleObjective,
    WeightedSumObjective,
)
from .optimizer import (
    DifferentialEvolutionOptimizer,
    GradientBasedOptimizer,
    GradientFreeOptimizer,
    LBFGSBOptimizer,
    NelderMeadOptimizer,
    Optimizer,
    ParticleSwarmOptimizer,
    create_optimizer,
)

__all__ = [
    "BoxConstraint",
    "DifferentialEvolutionOptimizer",
    "GradientBasedOptimizer",
    "GradientFreeOptimizer",
    "LBFGSBOptimizer",
    "LikelihoodObjective",
    "LinearConstraint",
    "MultiObjectiveFunction",
    "NelderMeadOptimizer",
    "NonlinearConstraint",
    "ObjectiveFunction",
    "Optimizer",
    "ParameterConstraints",
    "ParticleSwarmOptimizer",
    "PenaltyFunction",
    "SingleObjective",
    "WeightedSumObjective",
    "create_optimizer",
]
