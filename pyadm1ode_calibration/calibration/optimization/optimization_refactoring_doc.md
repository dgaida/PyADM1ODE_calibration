# PyADM1 Optimization Package - Refactoring Guide

## Overview

The optimization package has been created to provide a clean, modular, and extensible architecture for parameter calibration. This refactoring separates concerns and makes the codebase more maintainable.

## Package Structure

```
pyadm1/calibration/optimization/
├── __init__.py          # Package initialization with imports
├── optimizer.py         # Optimization algorithm implementations
├── objective.py         # Objective function classes
└── constraints.py       # Constraint handling
```

## Key Components

### 1. Optimizer Classes (`optimizer.py`)

**Abstract Base Classes:**
- `Optimizer`: Base class for all optimizers
- `GradientFreeOptimizer`: Base for gradient-free methods
- `GradientBasedOptimizer`: Base for gradient-based methods

**Concrete Implementations:**
- `DifferentialEvolutionOptimizer`: Global optimization (recommended for initial calibration)
- `ParticleSwarmOptimizer`: Swarm intelligence
- `NelderMeadOptimizer`: Local simplex method (good for online calibration)
- `PowellOptimizer`: Conjugate direction method
- `LBFGSBOptimizer`: Gradient-based with bounds
- `SLSQPOptimizer`: Sequential least squares with constraints

**Key Features:**
- Automatic history tracking
- Progress monitoring
- Consistent interface across all methods
- Easy extension for custom optimizers

### 2. Objective Functions (`objective.py`)

**Classes:**
- `SingleObjective`: Single output optimization
- `MultiObjectiveFunction`: Weighted multi-objective
- `WeightedSumObjective`: Convenience wrapper
- `LikelihoodObjective`: Maximum likelihood estimation
- `CustomObjective`: User-defined functions

**Features:**
- Multiple error metrics (RMSE, MAE, MAPE, NSE, R²)
- Automatic normalization
- Flexible weighting schemes
- Statistical metrics computation

### 3. Constraint Handling (`constraints.py`)

**Constraint Types:**
- `BoxConstraint`: Simple bounds
- `LinearConstraint`: Linear equality/inequality
- `NonlinearConstraint`: Custom nonlinear constraints

**Penalty Functions:**
- `QuadraticPenalty`: Smooth quadratic penalty
- `LinearPenalty`: Linear penalty
- `LogarithmicPenalty`: Barrier function
- `ExponentialPenalty`: Exponential growth
- `BarrierPenalty`: Inverse barrier

**Features:**
- Hard and soft constraints
- Penalty-based constraint handling
- Projection to feasible region
- Integration with scipy optimizers

## Refactoring Changes

### What Was Moved

**From `initial.py` → `optimizer.py`:**
- `_optimize_differential_evolution()` → `DifferentialEvolutionOptimizer`
- `_optimize_nelder_mead()` → `NelderMeadOptimizer`
- `_optimize_lbfgsb()` → `LBFGSBOptimizer`

**From `parameter_bounds.py` → `constraints.py`:**
- Penalty function logic → `PenaltyFunction` classes
- Bound checking → `BoxConstraint` class

**New Functionality:**
- Particle Swarm Optimization
- Powell's method
- SLSQP with constraints
- Multi-objective functions
- Likelihood objectives
- Linear and nonlinear constraints

### What Stayed

**In `calibrator.py`:**
- High-level calibration interface
- Simulation orchestration
- Result management
- History tracking

**In `initial.py` and `online.py`:**
- Calibration-specific logic
- ADM1 integration
- Sensitivity analysis
- Identifiability analysis

## Usage Examples

### Example 1: Basic Optimization

```python
from pyadm1.calibration.optimization import (
    DifferentialEvolutionOptimizer,
    SingleObjective
)

# Define objective
objective = SingleObjective(
    simulator=my_simulator,
    measurements=measured_ch4,
    objective_name="Q_ch4",
    parameter_names=["k_dis", "Y_su"],
    error_metric="rmse"
)

# Create optimizer
optimizer = DifferentialEvolutionOptimizer(
    bounds={"k_dis": (0.3, 0.8), "Y_su": (0.05, 0.15)},
    max_iterations=100,
    population_size=15
)

# Run optimization
result = optimizer.optimize(objective)

print(f"Optimal parameters: {result.parameter_dict}")
print(f"Objective value: {result.fun:.4f}")
```

### Example 2: Multi-Objective with Constraints

```python
from pyadm1.calibration.optimization import (
    LBFGSBOptimizer,
    MultiObjectiveFunction,
    ParameterConstraints
)

# Multi-objective
objective = MultiObjectiveFunction(
    simulator=simulator,
    measurements_dict={
        "Q_ch4": measured_ch4,
        "pH": measured_ph,
        "VFA": measured_vfa
    },
    objectives=["Q_ch4", "pH", "VFA"],
    weights={"Q_ch4": 0.6, "pH": 0.2, "VFA": 0.2},
    parameter_names=["k_dis", "Y_su", "k_hyd_ch"],
    error_metric="rmse"
)

# Add constraints
constraints = ParameterConstraints()
constraints.add_box_constraint("k_dis", 0.3, 0.8)
constraints.add_box_constraint("Y_su", 0.05, 0.15)
constraints.add_box_constraint("k_hyd_ch", 5.0, 15.0)

# Linear constraint: Y_su + 0.1*k_dis <= 0.2
constraints.add_linear_inequality(
    coefficients={"Y_su": 1.0, "k_dis": 0.1},
    upper_bound=0.2
)

# Wrap objective with penalty
def penalized_objective(x):
    params = {name: val for name, val in zip(["k_dis", "Y_su", "k_hyd_ch"], x)}
    penalty = constraints.calculate_penalty(params)
    return objective(x) + penalty

# Optimize
optimizer = LBFGSBOptimizer(
    bounds={"k_dis": (0.3, 0.8), "Y_su": (0.05, 0.15), "k_hyd_ch": (5.0, 15.0)},
    max_iterations=100
)

result = optimizer.optimize(penalized_objective)
```

### Example 3: Custom Objective Function

```python
from pyadm1.calibration.optimization import (
    NelderMeadOptimizer,
    CustomObjective
)

# Define custom error function
def my_custom_error(simulated, measured):
    """Custom error with emphasis on peaks."""
    residuals = simulated - measured
    # Weight larger values more heavily
    weights = 1.0 + np.abs(measured) / np.max(np.abs(measured))
    return np.sum(weights * residuals**2)

# Create custom objective
objective = CustomObjective(
    simulator=simulator,
    measurements_dict={"Q_ch4": measured_ch4},
    objectives=["Q_ch4"],
    parameter_names=["k_dis"],
    custom_func=my_custom_error
)

# Optimize
optimizer = NelderMeadOptimizer(
    bounds={"k_dis": (0.3, 0.8)},
    max_iterations=50
)

result = optimizer.optimize(objective, initial_guess=np.array([0.5]))
```

## Integration with Existing Code

### Updating `initial.py`

The `InitialCalibrator` class should now use the optimization package:

```python
from pyadm1.calibration.optimization import (
    create_optimizer,
    MultiObjectiveFunction,
    ParameterConstraints
)

class InitialCalibrator:
    def calibrate(self, ...):
        # Create objective
        objective = MultiObjectiveFunction(
            simulator=self._simulate_wrapper,
            measurements_dict=self._prepare_measurements(measurements),
            objectives=objectives,
            weights=weights,
            parameter_names=parameters
        )

        # Create optimizer
        optimizer = create_optimizer(
            method=method,
            bounds=param_bounds,
            max_iterations=max_iterations,
            population_size=population_size
        )

        # Run optimization
        result = optimizer.optimize(objective)

        return self._convert_to_calibration_result(result)
```

### Updating `online.py`

Similar refactoring for online calibration:

```python
from pyadm1.calibration.optimization import NelderMeadOptimizer

class OnlineCalibrator:
    def calibrate(self, ...):
        # Use fast local optimizer
        optimizer = NelderMeadOptimizer(
            bounds=bounded_ranges,
            max_iterations=max_iterations
        )

        # Use current parameters as initial guess
        initial_guess = np.array([current_parameters[p] for p in parameters])

        result = optimizer.optimize(objective, initial_guess)
```

## Benefits of Refactoring

1. **Separation of Concerns**: Optimization algorithms, objectives, and constraints are cleanly separated

2. **Extensibility**: Easy to add new optimizers or objective functions without modifying existing code

3. **Reusability**: Optimization components can be used in other parts of the codebase

4. **Testability**: Each component can be tested independently

5. **Maintainability**: Clear structure makes code easier to understand and modify

6. **Flexibility**: Users can easily customize optimization behavior

## Migration Guide

### For Users

**Old way:**
```python
calibrator.calibrate(method="differential_evolution", ...)
```

**New way (same interface, just refactored internally):**
```python
calibrator.calibrate(method="differential_evolution", ...)
```

The high-level interface remains the same!

### For Developers

When adding new optimization methods:

1. Create new class inheriting from `Optimizer`
2. Implement `optimize()` method
3. Add to factory function
4. Write tests

When adding new objective types:

1. Create new class inheriting from `ObjectiveFunction`
2. Implement `__call__()` method
3. Add documentation and examples

## Testing

Each module should have comprehensive tests:

```
tests/unit/test_calibration/test_optimization/
├── test_optimizers.py
├── test_objectives.py
└── test_constraints.py
```

## Future Enhancements

Potential additions:
- Trust region methods
- Simulated annealing
- Genetic algorithms
- Multi-objective Pareto optimization (NSGA-II)
- Adaptive penalty methods
- Surrogate-based optimization
- Parallel optimization strategies

## Summary

This refactoring creates a clean, modular optimization framework that:
- Maintains backward compatibility
- Improves code organization
- Enables easy extension
- Provides more flexibility
- Better aligns with software engineering best practices

The separation into optimizer, objective, and constraints modules follows the Single Responsibility Principle and makes the codebase more maintainable and testable.
