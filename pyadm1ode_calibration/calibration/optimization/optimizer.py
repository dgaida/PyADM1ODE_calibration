"""Optimization module."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import differential_evolution, minimize, OptimizeResult
import time


@dataclass
class OptimizationResult:
    """
    Result from an optimization run.

    Stores the output of a mathematical optimization process.

    Attributes:
        success (bool): Whether optimization converged successfully.
        x (np.ndarray): Optimal parameter values found by the optimizer.
        fun (float): Objective function value at the optimum point.
        nit (int): Number of iterations performed.
        nfev (int): Number of function evaluations performed.
        message (str): Status message from the optimization algorithm.
        parameter_names (List[str]): Names of the optimized parameters.
        parameter_dict (Dict[str, float]): Parameters as a name-to-value mapping.
        history (List[Dict[str, Any]]): Optimization history log if tracked.
        execution_time (float): Wall clock time in seconds.
    """

    success: bool
    x: np.ndarray
    fun: float
    nit: int
    nfev: int
    message: str
    parameter_names: List[str]
    parameter_dict: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0

    @classmethod
    def from_scipy_result(
        cls, result: OptimizeResult, parameter_names: List[str], execution_time: float, history: Optional[List] = None
    ) -> "OptimizationResult":
        """
        Create an OptimizationResult from a scipy OptimizeResult.

        Args:
            result (OptimizeResult): The raw result from scipy.
            parameter_names (List[str]): Names of parameters in the 'x' array.
            execution_time (float): Time taken for optimization.
            history (Optional[List]): Optimization progress history.

        Returns:
            OptimizationResult: A standardized result object.
        """
        param_dict = {name: float(val) for name, val in zip(parameter_names, result.x)}

        return cls(
            success=result.success,
            x=result.x,
            fun=result.fun,
            nit=result.nit,
            nfev=result.nfev,
            message=result.message,
            parameter_names=parameter_names,
            parameter_dict=param_dict,
            history=history or [],
            execution_time=execution_time,
        )


class Optimizer(ABC):
    """
    Abstract base class for optimization algorithms.

    All optimizers must implement the optimize() method and provide
    a consistent interface for parameter calibration.

    Args:
        bounds (Dict[str, Tuple[float, float]]): Parameter bounds as {name: (min, max)}.
        max_iterations (int): Maximum number of iterations. Defaults to 100.
        tolerance (float): Convergence tolerance. Defaults to 1e-6.
        verbose (bool): Whether to enable progress output. Defaults to True.
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
    ):
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose

        self.parameter_names = list(bounds.keys())
        self.bounds_array = np.array([bounds[name] for name in self.parameter_names])

        # History tracking
        self.history: List[Dict[str, Any]] = []
        self._best_value = float("inf")
        self._n_evaluations = 0

    @abstractmethod
    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Run the optimization process.

        Args:
            objective_func (Callable): Function to minimize f(x) -> float.
            initial_guess (Optional[np.ndarray]): Optional starting point for the optimizer.

        Returns:
            OptimizationResult: The result of the optimization.
        """
        pass

    def _wrap_objective(self, objective_func: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
        """
        Wrap the objective function to track evaluations and history.

        Args:
            objective_func (Callable): The original objective function.

        Returns:
            Callable: The wrapped function with side-effects for tracking.
        """

        def wrapped(x: np.ndarray) -> float:
            # Evaluate objective
            value = objective_func(x)

            # Track evaluation
            self._n_evaluations += 1

            # Update history
            param_dict = {name: float(val) for name, val in zip(self.parameter_names, x)}
            self.history.append({"parameters": param_dict, "objective": float(value), "iteration": self._n_evaluations})

            # Track best
            if value < self._best_value:
                self._best_value = value
                if self.verbose:
                    param_str = ", ".join([f"{name}={val:.4f}" for name, val in param_dict.items()])
                    print(f"  Iteration {self._n_evaluations}: f={value:.6f} | {param_str}")

            return value

        return wrapped

    def _reset_tracking(self):
        """
        Reset history and counters.
        """
        self.history = []
        self._best_value = float("inf")
        self._n_evaluations = 0

    def _check_bounds(self, x: np.ndarray) -> bool:
        """
        Check if parameters are within defined bounds.

        Args:
            x (np.ndarray): Parameter array to check.

        Returns:
            bool: True if all parameters are within bounds.
        """
        return np.all(x >= self.bounds_array[:, 0]) and np.all(x <= self.bounds_array[:, 1])

    def _project_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """
        Project parameters onto the feasible space defined by the bounds.

        Args:
            x (np.ndarray): Parameter array to project.

        Returns:
            np.ndarray: Projected parameter array.
        """
        return np.clip(x, self.bounds_array[:, 0], self.bounds_array[:, 1])


class GradientFreeOptimizer(Optimizer):
    """Base class for gradient-free optimization methods."""

    pass


class GradientBasedOptimizer(Optimizer):
    """Base class for gradient-based optimization methods."""

    pass


class DifferentialEvolutionOptimizer(GradientFreeOptimizer):
    """
    Differential Evolution optimizer.

    Global optimization using evolutionary algorithm. Good for multimodal
    problems with many local minima.

    Args:
        bounds (Dict[str, Tuple[float, float]]): Parameter bounds.
        max_iterations (int): Max generations. Defaults to 100.
        tolerance (float): Convergence tolerance.
        verbose (bool): Enable output.
        population_size (int): Population size per parameter. Defaults to 15.
        strategy (str): DE strategy. Defaults to 'best1bin'.
        mutation (Tuple[float, float]): Mutation scale. Defaults to (0.5, 1.0).
        recombination (float): Crossover probability. Defaults to 0.7.
        seed (Optional[int]): Random seed.
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        population_size: int = 15,
        strategy: str = "best1bin",
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        seed: Optional[int] = None,
    ):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.population_size = population_size
        self.strategy = strategy
        self.mutation = mutation
        self.recombination = recombination
        self.seed = seed

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Run differential evolution optimization.

        Args:
            objective_func (Callable): Objective function to minimize.
            initial_guess (Optional[np.ndarray]): Not used by DE.

        Returns:
            OptimizationResult: Results of the optimization.
        """
        if self.verbose:
            print("Starting Differential Evolution optimization")
            print(f"  Population size: {self.population_size}")
            print(f"  Max iterations: {self.max_iterations}")

        self._reset_tracking()
        wrapped_objective = self._wrap_objective(objective_func)
        start_time = time.time()

        result = differential_evolution(
            func=wrapped_objective,
            bounds=self.bounds_array,
            strategy=self.strategy,
            maxiter=self.max_iterations,
            popsize=self.population_size,
            tol=self.tolerance,
            mutation=self.mutation,
            recombination=self.recombination,
            seed=self.seed,
            disp=False,
            polish=True,
        )

        execution_time = time.time() - start_time

        if self.verbose:
            print(f"\nOptimization complete in {execution_time:.1f}s")

        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class ParticleSwarmOptimizer(GradientFreeOptimizer):
    """
    Particle Swarm Optimization.

    Requires 'pyswarm' package.
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        swarm_size: int = 20,
        omega: float = 0.5,
        phip: float = 0.5,
        phig: float = 0.5,
    ):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.swarm_size = swarm_size
        self.omega = omega
        self.phip = phip
        self.phig = phig

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """Run particle swarm optimization."""
        try:
            from pyswarm import pso
        except ImportError:
            raise ImportError("Particle Swarm requires 'pyswarm' package: pip install pyswarm")

        if self.verbose:
            print("Starting Particle Swarm optimization")

        self._reset_tracking()
        wrapped_objective = self._wrap_objective(objective_func)
        start_time = time.time()

        lb = self.bounds_array[:, 0]
        ub = self.bounds_array[:, 1]

        xopt, fopt = pso(
            wrapped_objective,
            lb,
            ub,
            swarmsize=self.swarm_size,
            omega=self.omega,
            phip=self.phip,
            phig=self.phig,
            maxiter=self.max_iterations,
            debug=self.verbose,
        )

        execution_time = time.time() - start_time
        result = OptimizeResult(
            x=xopt,
            fun=fopt,
            success=True,
            nit=self.max_iterations,
            nfev=len(self.history),
            message="Optimization terminated successfully",
        )
        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class NelderMeadOptimizer(GradientFreeOptimizer):
    """
    Nelder-Mead simplex optimizer.

    Local optimization method. Fast but may not find global optimum.
    Good for online calibration.

    Args:
        bounds (Dict[str, Tuple[float, float]]): Parameter bounds.
        max_iterations (int): Max iterations.
        tolerance (float): Convergence tolerance.
        verbose (bool): Enable output.
        adaptive (bool): Use adaptive simplex. Defaults to True.
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        adaptive: bool = True,
    ):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.adaptive = adaptive

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Run Nelder-Mead optimization.

        Args:
            objective_func (Callable): Objective function to minimize.
            initial_guess (Optional[np.ndarray]): Starting point. Midpoint of bounds if None.

        Returns:
            OptimizationResult: Results of the optimization.
        """
        if initial_guess is None:
            initial_guess = np.mean(self.bounds_array, axis=1)

        if self.verbose:
            print("Starting Nelder-Mead optimization")

        self._reset_tracking()

        def penalized_objective(x):
            if not self._check_bounds(x):
                return 1e10
            return objective_func(x)

        wrapped_objective = self._wrap_objective(penalized_objective)
        start_time = time.time()

        result = minimize(
            fun=wrapped_objective,
            x0=initial_guess,
            method="Nelder-Mead",
            options={
                "maxiter": self.max_iterations,
                "xatol": self.tolerance,
                "fatol": self.tolerance,
                "adaptive": self.adaptive,
                "disp": False,
            },
        )

        execution_time = time.time() - start_time

        if self.verbose:
            print(f"\nOptimization complete in {execution_time:.1f}s")

        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class PowellOptimizer(GradientFreeOptimizer):
    """
    Powell's conjugate direction method.
    """

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """Run Powell optimization."""
        if initial_guess is None:
            initial_guess = np.mean(self.bounds_array, axis=1)

        if self.verbose:
            print("Starting Powell optimization")

        self._reset_tracking()

        def penalized_objective(x):
            if not self._check_bounds(x):
                return 1e10
            return objective_func(x)

        wrapped_objective = self._wrap_objective(penalized_objective)
        start_time = time.time()

        result = minimize(
            fun=wrapped_objective,
            x0=initial_guess,
            method="Powell",
            options={"maxiter": self.max_iterations, "ftol": self.tolerance, "disp": False},
        )

        execution_time = time.time() - start_time

        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class LBFGSBOptimizer(GradientBasedOptimizer):
    """
    L-BFGS-B optimizer.

    Fast gradient-based method with box constraints.

    Args:
        bounds (Dict[str, Tuple[float, float]]): Parameter bounds.
        max_iterations (int): Max iterations.
        tolerance (float): Convergence tolerance.
        verbose (bool): Enable output.
        gtol (float): Gradient tolerance. Defaults to 1e-5.
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        gtol: float = 1e-5,
    ):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.gtol = gtol

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Run L-BFGS-B optimization.

        Args:
            objective_func (Callable): Objective function to minimize.
            initial_guess (Optional[np.ndarray]): Starting point. Midpoint of bounds if None.

        Returns:
            OptimizationResult: Results of the optimization.
        """
        if initial_guess is None:
            initial_guess = np.mean(self.bounds_array, axis=1)

        if self.verbose:
            print("Starting L-BFGS-B optimization")

        self._reset_tracking()
        wrapped_objective = self._wrap_objective(objective_func)
        start_time = time.time()

        result = minimize(
            fun=wrapped_objective,
            x0=initial_guess,
            method="L-BFGS-B",
            bounds=self.bounds_array,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance, "gtol": self.gtol, "disp": False},
        )

        execution_time = time.time() - start_time

        if self.verbose:
            print(f"\nOptimization complete in {execution_time:.1f}s")

        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class SLSQPOptimizer(GradientBasedOptimizer):
    """
    Sequential Least Squares Programming.
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        constraints: Optional[List] = None,
    ):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.constraints = constraints or []

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """Run SLSQP optimization."""
        if initial_guess is None:
            initial_guess = np.mean(self.bounds_array, axis=1)

        if self.verbose:
            print("Starting SLSQP optimization")

        self._reset_tracking()
        wrapped_objective = self._wrap_objective(objective_func)
        start_time = time.time()

        result = minimize(
            fun=wrapped_objective,
            x0=initial_guess,
            method="SLSQP",
            bounds=self.bounds_array,
            constraints=self.constraints,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance, "disp": False},
        )

        execution_time = time.time() - start_time

        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


def create_optimizer(
    method: str, bounds: Dict[str, Tuple[float, float]], max_iterations: int = 100, verbose: bool = True, **kwargs
) -> Optimizer:
    """
    Factory function to create optimizer instances.

    Args:
        method (str): Optimization method name (e.g., 'differential_evolution', 'nelder_mead').
        bounds (Dict[str, Tuple[float, float]]): Parameter bounds.
        max_iterations (int): Maximum iterations.
        verbose (bool): Whether to enable output.
        **kwargs: Additional method-specific arguments.

    Returns:
        Optimizer: An instance of a concrete Optimizer class.

    Raises:
        ValueError: If the requested optimization method is unknown.
    """
    method = method.lower().replace("-", "_").replace(" ", "_")

    optimizer_map = {
        "differential_evolution": DifferentialEvolutionOptimizer,
        "de": DifferentialEvolutionOptimizer,
        "particle_swarm": ParticleSwarmOptimizer,
        "pso": ParticleSwarmOptimizer,
        "nelder_mead": NelderMeadOptimizer,
        "nm": NelderMeadOptimizer,
        "powell": PowellOptimizer,
        "lbfgsb": LBFGSBOptimizer,
        "l_bfgs_b": LBFGSBOptimizer,
        "slsqp": SLSQPOptimizer,
    }

    if method not in optimizer_map:
        available = ", ".join(optimizer_map.keys())
        raise ValueError(f"Unknown optimization method: {method}. Available: {available}")

    optimizer_class = optimizer_map[method]
    return optimizer_class(bounds=bounds, max_iterations=max_iterations, verbose=verbose, **kwargs)
