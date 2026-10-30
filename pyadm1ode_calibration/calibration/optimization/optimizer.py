"""Optimization module."""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution, minimize


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
    parameter_names: list[str]
    parameter_dict: dict[str, float] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0

    @classmethod
    def from_scipy_result(
        cls, result: OptimizeResult, parameter_names: list[str], execution_time: float, history: list | None = None
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
        bounds: dict[str, tuple[float, float]],
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
        self.history: list[dict[str, Any]] = []
        self._best_value = float("inf")
        self._n_evaluations = 0

    @abstractmethod
    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: np.ndarray | None = None
    ) -> OptimizationResult:
        """
        Run the optimization process.

        Args:
            objective_func (Callable): Function to minimize f(x) -> float.
            initial_guess (Optional[np.ndarray]): Optional starting point for the optimizer.

        Returns:
            OptimizationResult: The result of the optimization.
        """

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
                    # Note: _n_evaluations counts function evaluations,
                    # not optimizer iterations. For DE these differ by a
                    # factor of popsize * n_params per generation.
                    print(f"  Eval {self._n_evaluations}: f={value:.6f} | {param_str}")

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


class GradientBasedOptimizer(Optimizer):
    """Base class for gradient-based optimization methods."""


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
        init_sigma (Optional[float]): If given together with an initial guess,
            the initial population is drawn from a normal distribution with this
            standard deviation (in parameter units) around the guess instead of
            Latin hypercube sampling over the whole box. Use the prior spread of
            the parameters (e.g. 0.25 for log-factors) so a good literature
            start is not thrown away.
        polish (bool): Polish the best member with L-BFGS-B at the end.
            Defaults to True; set False when the budget is counted in
            simulations, because the polish step is not budgeted.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        population_size: int = 15,
        strategy: str = "best1bin",
        mutation: tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        seed: int | None = None,
        init_sigma: float | None = None,
        polish: bool = True,
    ):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.population_size = population_size
        self.strategy = strategy
        self.mutation = mutation
        self.recombination = recombination
        self.seed = seed
        self.init_sigma = init_sigma
        self.polish = polish

    def _initial_population(self, x0: np.ndarray) -> np.ndarray:
        """Draw the initial population around ``x0`` and clip it to the bounds.

        The first member is ``x0`` itself so the start point is always evaluated.
        """
        rng = np.random.default_rng(self.seed)
        n_members = self.population_size * len(self.parameter_names)
        pop = rng.normal(loc=x0, scale=self.init_sigma, size=(n_members, len(x0)))
        pop[0] = x0
        return np.clip(pop, self.bounds_array[:, 0], self.bounds_array[:, 1])

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: np.ndarray | None = None
    ) -> OptimizationResult:
        """
        Run differential evolution optimization.

        Args:
            objective_func (Callable): Objective function to minimize.
            initial_guess (Optional[np.ndarray]): Start point. Without
                ``init_sigma`` it only replaces the best member of the random
                initial population; with ``init_sigma`` the whole population is
                drawn around it.

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

        init: str | np.ndarray = "latinhypercube"
        x0 = None
        if initial_guess is not None:
            x0 = self._project_to_bounds(np.asarray(initial_guess, dtype=float))
            if self.init_sigma is not None:
                init = self._initial_population(x0)

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
            polish=self.polish,
            init=init,
            x0=x0,
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
        bounds: dict[str, tuple[float, float]],
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
        self, objective_func: Callable[[np.ndarray], float], initial_guess: np.ndarray | None = None
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

        raw = pso(
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

        # pyswarm changed its return contract: 0.6 handed back ``(xopt, fopt)``,
        # 1.0 returns a scipy-style OptimizeResult. Accept both so the method
        # does not depend on which release happens to be installed.
        if isinstance(raw, tuple):
            xopt, fopt = raw
            result = OptimizeResult(
                x=np.asarray(xopt, dtype=float),
                fun=float(fopt),
                success=True,
                nit=self.max_iterations,
                nfev=len(self.history),
                message="Optimization terminated successfully",
            )
        else:
            result = OptimizeResult(
                x=np.asarray(raw.x, dtype=float),
                fun=float(raw.fun),
                success=bool(getattr(raw, "success", True)),
                nit=int(getattr(raw, "nit", self.max_iterations)),
                nfev=len(self.history) or int(getattr(raw, "nfev", 0)),
                message=str(getattr(raw, "message", "Optimization terminated successfully")),
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
        initial_simplex_scale (Optional[float]): Edge length of the initial
            simplex in parameter units. SciPy's default perturbs each
            coordinate by 5 % and uses 0.00025 for coordinates that are exactly
            zero, which is far too small for log-factor parameters starting at
            0. Set it to the expected parameter spread (e.g. 0.25).
        max_evaluations (Optional[int]): Cap on objective evaluations
            (``maxfev``), for sharing one budget between several runs.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        adaptive: bool = True,
        initial_simplex_scale: float | None = None,
        max_evaluations: int | None = None,
    ):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.adaptive = adaptive
        self.initial_simplex_scale = initial_simplex_scale
        self.max_evaluations = max_evaluations

    def _initial_simplex(self, x0: np.ndarray) -> np.ndarray:
        """Build a right-angled simplex of edge ``initial_simplex_scale`` at ``x0``.

        A vertex that would leave the box is reflected to the other side of
        ``x0`` so every vertex stays feasible.
        """
        n = len(x0)
        simplex = np.tile(x0, (n + 1, 1))
        lo, hi = self.bounds_array[:, 0], self.bounds_array[:, 1]
        for i in range(n):
            step = self.initial_simplex_scale
            if x0[i] + step > hi[i]:
                step = -step
            simplex[i + 1, i] = np.clip(x0[i] + step, lo[i], hi[i])
        return simplex

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: np.ndarray | None = None
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

        options: dict[str, Any] = {
            "maxiter": self.max_iterations,
            "maxfev": self.max_evaluations,
            "xatol": self.tolerance,
            "fatol": self.tolerance,
            "adaptive": self.adaptive,
            "disp": False,
        }
        if self.initial_simplex_scale is not None:
            options["initial_simplex"] = self._initial_simplex(np.asarray(initial_guess, dtype=float))

        result = minimize(
            fun=wrapped_objective,
            x0=initial_guess,
            method="Nelder-Mead",
            options=options,
        )

        execution_time = time.time() - start_time

        if self.verbose:
            print(f"\nOptimization complete in {execution_time:.1f}s")

        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class PowellOptimizer(GradientFreeOptimizer):
    """
    Powell's conjugate direction method.

    Args:
        bounds (Dict[str, Tuple[float, float]]): Parameter bounds.
        max_iterations (int): Max iterations (direction sweeps).
        tolerance (float): Convergence tolerance (``ftol``).
        verbose (bool): Enable output.
        max_evaluations (Optional[int]): Cap on objective evaluations
            (``maxfev``). Needed when several Powell runs share one budget,
            e.g. inside :class:`MultiStartOptimizer`; a single iteration can
            otherwise consume hundreds of evaluations.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        max_evaluations: int | None = None,
    ):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.max_evaluations = max_evaluations

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: np.ndarray | None = None
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
            options={
                "maxiter": self.max_iterations,
                "maxfev": self.max_evaluations,
                "ftol": self.tolerance,
                "disp": False,
            },
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
        eps (Optional[float]): Finite-difference step for the numerical
            gradient. SciPy's default of 1e-8 is far below the noise floor of
            an adaptive ODE solver (relative tolerance 1e-6), so the gradient is
            dominated by solver noise. Use roughly the square root of the
            solver tolerance, e.g. 1e-3, or supply ``jac``. Ignored when
            ``jac`` is given.
        jac (Optional[Callable]): Exact gradient ``g(x) -> np.ndarray``, e.g.
            from the differentiable PyTorch ADM1. Replaces finite differences.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        gtol: float = 1e-5,
        eps: float | None = None,
        jac: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.gtol = gtol
        self.eps = eps
        self.jac = jac

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: np.ndarray | None = None
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

        options: dict[str, Any] = {"maxiter": self.max_iterations, "ftol": self.tolerance, "gtol": self.gtol}
        if self.jac is None and self.eps is not None:
            options["eps"] = self.eps

        result = minimize(
            fun=wrapped_objective,
            x0=initial_guess,
            method="L-BFGS-B",
            jac=self.jac,
            bounds=self.bounds_array,
            # `disp` is deprecated for L-BFGS-B and removed in SciPy 1.18. Silence is
            # the default, so dropping it changes nothing but the warning.
            options=options,
        )

        execution_time = time.time() - start_time

        if self.verbose:
            print(f"\nOptimization complete in {execution_time:.1f}s")

        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class SLSQPOptimizer(GradientBasedOptimizer):
    """
    Sequential Least Squares Programming.

    Args:
        bounds (Dict[str, Tuple[float, float]]): Parameter bounds.
        max_iterations (int): Max iterations.
        tolerance (float): Convergence tolerance.
        verbose (bool): Enable output.
        constraints (Optional[list]): SciPy constraint dictionaries.
        eps (Optional[float]): Finite-difference step, see
            :class:`LBFGSBOptimizer`. Ignored when ``jac`` is given.
        jac (Optional[Callable]): Exact gradient ``g(x) -> np.ndarray``.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        constraints: list | None = None,
        eps: float | None = None,
        jac: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.constraints = constraints or []
        self.eps = eps
        self.jac = jac

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: np.ndarray | None = None
    ) -> OptimizationResult:
        """Run SLSQP optimization."""
        if initial_guess is None:
            initial_guess = np.mean(self.bounds_array, axis=1)

        if self.verbose:
            print("Starting SLSQP optimization")

        self._reset_tracking()
        wrapped_objective = self._wrap_objective(objective_func)
        start_time = time.time()

        options: dict[str, Any] = {"maxiter": self.max_iterations, "ftol": self.tolerance, "disp": False}
        if self.jac is None and self.eps is not None:
            options["eps"] = self.eps

        result = minimize(
            fun=wrapped_objective,
            x0=initial_guess,
            method="SLSQP",
            jac=self.jac,
            bounds=self.bounds_array,
            constraints=self.constraints,
            options=options,
        )

        execution_time = time.time() - start_time

        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class CMAESOptimizer(GradientFreeOptimizer):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES).

    The reference evolutionary method for continuous black-box problems. The
    search distribution starts at the initial guess with width ``sigma0`` and
    adapts its covariance to the shape of the objective, so parameter
    correlations (k_m with K_S, uptake with decay) are followed automatically.
    Requires the 'cma' package.

    Args:
        bounds (Dict[str, Tuple[float, float]]): Parameter bounds.
        max_iterations (int): Maximum number of generations.
        tolerance (float): Tolerance on the objective (``tolfun``).
        verbose (bool): Enable output.
        sigma0 (Optional[float]): Initial step size in parameter units.
            Defaults to a quarter of the smallest bound width; set it to the
            expected parameter spread (e.g. 0.25 for log-factors).
        population_size (Optional[int]): Offspring per generation. Defaults to
            the CMA-ES rule ``4 + 3 ln(n)``.
        seed (Optional[int]): Random seed.
        max_evaluations (Optional[int]): Cap on objective evaluations.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        sigma0: float | None = None,
        population_size: int | None = None,
        seed: int | None = None,
        max_evaluations: int | None = None,
    ):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.sigma0 = sigma0
        self.population_size = population_size
        self.seed = seed
        self.max_evaluations = max_evaluations

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: np.ndarray | None = None
    ) -> OptimizationResult:
        """Run CMA-ES.

        Args:
            objective_func (Callable): Objective function to minimize.
            initial_guess (Optional[np.ndarray]): Centre of the initial search
                distribution. Midpoint of bounds if None.

        Returns:
            OptimizationResult: Results of the optimization.
        """
        try:
            import cma
        except ImportError:
            raise ImportError("CMA-ES requires the 'cma' package: pip install cma")

        if initial_guess is None:
            initial_guess = np.mean(self.bounds_array, axis=1)
        x0 = self._project_to_bounds(np.asarray(initial_guess, dtype=float))
        widths = self.bounds_array[:, 1] - self.bounds_array[:, 0]
        sigma0 = self.sigma0 if self.sigma0 is not None else 0.25 * float(np.min(widths))

        if self.verbose:
            print(f"Starting CMA-ES optimization (sigma0={sigma0:.4g})")

        self._reset_tracking()
        wrapped_objective = self._wrap_objective(objective_func)
        start_time = time.time()

        options: dict[str, Any] = {
            "bounds": [self.bounds_array[:, 0].tolist(), self.bounds_array[:, 1].tolist()],
            "maxiter": self.max_iterations,
            "tolfun": self.tolerance,
            "verbose": -9,
        }
        if self.population_size is not None:
            options["popsize"] = self.population_size
        if self.seed is not None:
            options["seed"] = self.seed
        if self.max_evaluations is not None:
            options["maxfevals"] = self.max_evaluations

        es = cma.CMAEvolutionStrategy(x0, sigma0, options)
        while not es.stop():
            candidates = es.ask()
            es.tell(candidates, [wrapped_objective(np.asarray(c)) for c in candidates])

        execution_time = time.time() - start_time
        best = es.result
        result = OptimizeResult(
            x=np.asarray(best.xbest, dtype=float),
            fun=float(best.fbest),
            success=True,
            nit=int(best.iterations),
            nfev=int(best.evaluations),
            message="; ".join(f"{k}={v}" for k, v in es.stop().items()) or "stopped",
        )
        if self.verbose:
            print(f"\nOptimization complete in {execution_time:.1f}s")
        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class BOBYQAOptimizer(GradientFreeOptimizer):
    """
    Powell's BOBYQA (Bound Optimization BY Quadratic Approximation).

    Powell's own successor to his 1964 conjugate-direction method: a trust-region
    method that interpolates a quadratic model from past evaluations, so every
    simulation keeps informing the search. Handles the bounds natively.
    Requires the 'Py-BOBYQA' package (falls back to 'pdfo' if installed).

    Args:
        bounds (Dict[str, Tuple[float, float]]): Parameter bounds.
        max_iterations (int): Used as the evaluation cap when
            ``max_evaluations`` is not given.
        tolerance (float): Final trust-region radius (``rhoend``).
        verbose (bool): Enable output.
        rhobeg (Optional[float]): Initial trust-region radius in parameter
            units. Powell recommends about a tenth of the expected change;
            defaults to a tenth of the smallest bound width.
        max_evaluations (Optional[int]): Cap on objective evaluations.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        rhobeg: float | None = None,
        max_evaluations: int | None = None,
    ):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.rhobeg = rhobeg
        self.max_evaluations = max_evaluations

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: np.ndarray | None = None
    ) -> OptimizationResult:
        """Run BOBYQA.

        Args:
            objective_func (Callable): Objective function to minimize.
            initial_guess (Optional[np.ndarray]): Start point. Midpoint of bounds if None.

        Returns:
            OptimizationResult: Results of the optimization.
        """
        if initial_guess is None:
            initial_guess = np.mean(self.bounds_array, axis=1)
        x0 = self._project_to_bounds(np.asarray(initial_guess, dtype=float))
        widths = self.bounds_array[:, 1] - self.bounds_array[:, 0]
        rhobeg = self.rhobeg if self.rhobeg is not None else 0.1 * float(np.min(widths))
        # BOBYQA needs the initial radius to fit inside the box.
        rhobeg = min(rhobeg, 0.5 * float(np.min(widths)))
        rhoend = min(self.tolerance, rhobeg)
        maxfev = self.max_evaluations or self.max_iterations

        if self.verbose:
            print(f"Starting BOBYQA optimization (rhobeg={rhobeg:.4g})")

        self._reset_tracking()
        wrapped_objective = self._wrap_objective(objective_func)
        start_time = time.time()

        lo, hi = self.bounds_array[:, 0], self.bounds_array[:, 1]
        try:
            # Pure-Python implementation by the DFO-LS authors; preferred because the
            # compiled pdfo wheels lag behind NumPy releases.
            import pybobyqa

            soln = pybobyqa.solve(wrapped_objective, x0, bounds=(lo, hi), maxfun=maxfev, rhobeg=rhobeg, rhoend=rhoend)
            result = OptimizeResult(
                x=np.asarray(soln.x, dtype=float),
                fun=float(soln.f),
                success=soln.flag == soln.EXIT_SUCCESS,
                nit=int(soln.nf),
                nfev=int(soln.nf),
                message=str(soln.msg),
            )
        except ImportError:
            try:
                from pdfo import pdfo
                from scipy.optimize import Bounds
            except ImportError:
                raise ImportError("BOBYQA requires 'Py-BOBYQA' (pip install Py-BOBYQA) or 'pdfo' (pip install pdfo)")
            raw = pdfo(
                wrapped_objective,
                x0,
                method="bobyqa",
                bounds=Bounds(lo, hi),
                options={"maxfev": maxfev, "rhobeg": rhobeg, "rhoend": rhoend, "quiet": True},
            )
            result = OptimizeResult(
                x=np.asarray(raw.x, dtype=float),
                fun=float(raw.fun),
                success=bool(raw.success),
                nit=int(getattr(raw, "nfev", len(self.history))),
                nfev=int(getattr(raw, "nfev", len(self.history))),
                message=str(raw.message),
            )

        execution_time = time.time() - start_time
        if self.verbose:
            print(f"\nOptimization complete in {execution_time:.1f}s")
        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class ResidualOptimizer(Optimizer):
    """Base class for least-squares methods that need the residual vector.

    Unlike the scalar optimizers, ``optimize`` expects ``objective_func`` to
    return the **vector of weighted residuals** ``r(x)`` (e.g. one entry per
    measurement, each divided by its sensor sigma). The scalar recorded in the
    history is ``sum(r**2)``, so it is directly comparable with a chi-square
    objective up to its normalisation.
    """

    def _wrap_residuals(self, residual_func: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
        """Wrap a residual function so evaluations and the best sum of squares are tracked.

        Args:
            residual_func (Callable): Function returning a 1-D residual array.

        Returns:
            Callable: The wrapped function with tracking side-effects.
        """

        def wrapped(x: np.ndarray) -> np.ndarray:
            residuals = np.asarray(residual_func(x), dtype=float).ravel()
            value = float(np.sum(residuals**2))
            self._n_evaluations += 1
            param_dict = {name: float(val) for name, val in zip(self.parameter_names, x)}
            self.history.append({"parameters": param_dict, "objective": value, "iteration": self._n_evaluations})
            if value < self._best_value:
                self._best_value = value
                if self.verbose:
                    param_str = ", ".join([f"{name}={val:.4f}" for name, val in param_dict.items()])
                    print(f"  Eval {self._n_evaluations}: sum(r^2)={value:.6f} | {param_str}")
            return residuals

        return wrapped


class DFOLSOptimizer(ResidualOptimizer):
    """
    DFO-LS: derivative-free Gauss-Newton for nonlinear least squares.

    Builds linear models of the residuals from past evaluations, which makes it
    far cheaper per step than a general derivative-free method on a sum of
    squares. Designed for noisy, expensive residuals and therefore a natural
    fit for an adaptive ODE solver. Requires the 'DFO-LS' package.

    Args:
        bounds (Dict[str, Tuple[float, float]]): Parameter bounds.
        max_iterations (int): Used as the evaluation cap when
            ``max_evaluations`` is not given.
        tolerance (float): Final trust-region radius (``rhoend``).
        verbose (bool): Enable output.
        rhobeg (Optional[float]): Initial trust-region radius. Defaults to a
            tenth of the smallest bound width.
        max_evaluations (Optional[int]): Cap on residual evaluations.
        noisy (bool): Tell DFO-LS the residuals are noisy, which switches on
            its restart and regression safeguards. Defaults to True.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        rhobeg: float | None = None,
        max_evaluations: int | None = None,
        noisy: bool = True,
    ):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.rhobeg = rhobeg
        self.max_evaluations = max_evaluations
        self.noisy = noisy

    def optimize(
        self, objective_func: Callable[[np.ndarray], np.ndarray], initial_guess: np.ndarray | None = None
    ) -> OptimizationResult:
        """Run DFO-LS.

        Args:
            objective_func (Callable): Function returning the residual vector.
            initial_guess (Optional[np.ndarray]): Start point. Midpoint of bounds if None.

        Returns:
            OptimizationResult: Results with ``fun`` = sum of squared residuals.
        """
        try:
            import dfols
        except ImportError:
            raise ImportError("DFO-LS requires the 'DFO-LS' package: pip install DFO-LS")

        if initial_guess is None:
            initial_guess = np.mean(self.bounds_array, axis=1)
        x0 = self._project_to_bounds(np.asarray(initial_guess, dtype=float))
        widths = self.bounds_array[:, 1] - self.bounds_array[:, 0]
        rhobeg = self.rhobeg if self.rhobeg is not None else 0.1 * float(np.min(widths))
        rhobeg = min(rhobeg, 0.5 * float(np.min(widths)))

        if self.verbose:
            print(f"Starting DFO-LS optimization (rhobeg={rhobeg:.4g})")

        self._reset_tracking()
        wrapped = self._wrap_residuals(objective_func)
        start_time = time.time()

        soln = dfols.solve(
            wrapped,
            x0,
            bounds=(self.bounds_array[:, 0], self.bounds_array[:, 1]),
            maxfun=self.max_evaluations or self.max_iterations,
            rhobeg=rhobeg,
            rhoend=min(self.tolerance, rhobeg),
            objfun_has_noise=self.noisy,
        )

        execution_time = time.time() - start_time
        result = OptimizeResult(
            x=np.asarray(soln.x, dtype=float),
            fun=float(soln.obj),
            success=soln.flag == soln.EXIT_SUCCESS,
            nit=int(soln.nf),
            nfev=int(soln.nf),
            message=str(soln.msg),
        )
        if self.verbose:
            print(f"\nOptimization complete in {execution_time:.1f}s")
        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class LeastSquaresOptimizer(ResidualOptimizer):
    """
    Trust-region least squares (SciPy ``least_squares``, Gauss-Newton family).

    The standard choice in systems biology when exact sensitivities are
    available: pass ``jac`` computed from the differentiable PyTorch ADM1.
    Without ``jac`` it falls back to finite differences with step
    ``diff_step``, which suffers from solver noise just like L-BFGS-B.

    Args:
        bounds (Dict[str, Tuple[float, float]]): Parameter bounds.
        max_iterations (int): Maximum number of residual evaluations
            (``max_nfev``).
        tolerance (float): Used for ``ftol``, ``xtol`` and ``gtol``.
        verbose (bool): Enable output.
        jac (Optional[Callable]): Jacobian ``J(x) -> (n_residuals, n_params)``.
        diff_step (Optional[float]): Relative finite-difference step when
            ``jac`` is None.
        method (str): 'trf' (default, respects bounds), 'dogbox' or 'lm'
            (no bounds).
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        jac: Callable[[np.ndarray], np.ndarray] | None = None,
        diff_step: float | None = None,
        method: str = "trf",
    ):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.jac = jac
        self.diff_step = diff_step
        self.method = method

    def optimize(
        self, objective_func: Callable[[np.ndarray], np.ndarray], initial_guess: np.ndarray | None = None
    ) -> OptimizationResult:
        """Run trust-region least squares.

        Args:
            objective_func (Callable): Function returning the residual vector.
            initial_guess (Optional[np.ndarray]): Start point. Midpoint of bounds if None.

        Returns:
            OptimizationResult: Results with ``fun`` = sum of squared residuals.
        """
        from scipy.optimize import least_squares

        if initial_guess is None:
            initial_guess = np.mean(self.bounds_array, axis=1)
        x0 = self._project_to_bounds(np.asarray(initial_guess, dtype=float))

        if self.verbose:
            print(f"Starting least-squares optimization ({self.method})")

        self._reset_tracking()
        wrapped = self._wrap_residuals(objective_func)
        start_time = time.time()

        kwargs: dict[str, Any] = {
            "jac": self.jac if self.jac is not None else "2-point",
            "method": self.method,
            "max_nfev": self.max_iterations,
            "ftol": self.tolerance,
            "xtol": self.tolerance,
            "gtol": self.tolerance,
        }
        if self.method != "lm":
            kwargs["bounds"] = (self.bounds_array[:, 0], self.bounds_array[:, 1])
        if self.jac is None and self.diff_step is not None:
            kwargs["diff_step"] = self.diff_step

        raw = least_squares(wrapped, x0, **kwargs)

        execution_time = time.time() - start_time
        result = OptimizeResult(
            x=np.asarray(raw.x, dtype=float),
            fun=float(2.0 * raw.cost),
            success=bool(raw.success),
            nit=int(raw.nfev),
            nfev=int(raw.nfev),
            message=str(raw.message),
        )
        if self.verbose:
            print(f"\nOptimization complete in {execution_time:.1f}s")
        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


def _merge_histories(parts: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Concatenate per-stage histories with a running evaluation counter."""
    merged: list[dict[str, Any]] = []
    for part in parts:
        for entry in part:
            merged.append({**entry, "iteration": len(merged) + 1})
    return merged


class HybridOptimizer(Optimizer):
    """
    Global search followed by local refinement.

    Runs a global method (default: differential evolution with its initial
    population drawn around the start point) and hands its best point to a
    local method (default: Powell). This is the scheme Carecci et al. (2025)
    use for an agricultural ADM1 and the natural fix for differential
    evolution ignoring a good literature start.

    Args:
        bounds (Dict[str, Tuple[float, float]]): Parameter bounds.
        max_iterations (int): Default iteration cap for both stages.
        tolerance (float): Convergence tolerance for both stages.
        verbose (bool): Enable output.
        global_method (str): Name accepted by :func:`create_optimizer`.
        local_method (str): Name accepted by :func:`create_optimizer`.
        global_kwargs (Optional[dict]): Extra arguments for the global stage.
        local_kwargs (Optional[dict]): Extra arguments for the local stage.
        global_max_iterations (Optional[int]): Iteration cap of the global
            stage; defaults to ``max_iterations``.
        local_max_iterations (Optional[int]): Iteration cap of the local
            stage; defaults to ``max_iterations``.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        global_method: str = "differential_evolution",
        local_method: str = "powell",
        global_kwargs: dict[str, Any] | None = None,
        local_kwargs: dict[str, Any] | None = None,
        global_max_iterations: int | None = None,
        local_max_iterations: int | None = None,
    ):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.global_method = global_method
        self.local_method = local_method
        self.global_kwargs = dict(global_kwargs or {})
        self.local_kwargs = dict(local_kwargs or {})
        self.global_max_iterations = global_max_iterations
        self.local_max_iterations = local_max_iterations

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: np.ndarray | None = None
    ) -> OptimizationResult:
        """Run the global stage, then refine its best point locally.

        Args:
            objective_func (Callable): Objective function to minimize.
            initial_guess (Optional[np.ndarray]): Start point for the global stage.

        Returns:
            OptimizationResult: The better of the two stages, with the
            histories of both stages concatenated.
        """
        self._reset_tracking()
        start_time = time.time()

        global_opt = create_optimizer(
            self.global_method,
            self.bounds,
            max_iterations=self.global_max_iterations or self.max_iterations,
            verbose=self.verbose,
            tolerance=self.tolerance,
            **self.global_kwargs,
        )
        global_result = global_opt.optimize(objective_func, initial_guess)

        local_opt = create_optimizer(
            self.local_method,
            self.bounds,
            max_iterations=self.local_max_iterations or self.max_iterations,
            verbose=self.verbose,
            tolerance=self.tolerance,
            **self.local_kwargs,
        )
        local_result = local_opt.optimize(objective_func, global_result.x)

        best = local_result if local_result.fun <= global_result.fun else global_result
        self.history = _merge_histories([global_result.history, local_result.history])
        self._n_evaluations = len(self.history)
        execution_time = time.time() - start_time
        result = OptimizeResult(
            x=np.asarray(best.x, dtype=float),
            fun=float(best.fun),
            success=bool(global_result.success or local_result.success),
            nit=int(global_result.nit + local_result.nit),
            nfev=int(global_result.nfev + local_result.nfev),
            message=f"global: {global_result.message}; local: {local_result.message}",
        )
        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class MultiStartOptimizer(Optimizer):
    """
    Multi-start local optimization.

    Runs a local method from several start points and keeps the best run. The
    standard remedy for local minima when exact gradients are available (Raue
    et al. 2013). Start points are either Latin-hypercube samples over the box
    or normal draws around the initial guess; the initial guess itself is
    always the first start.

    Args:
        bounds (Dict[str, Tuple[float, float]]): Parameter bounds.
        max_iterations (int): Iteration cap per local run.
        tolerance (float): Convergence tolerance per local run.
        verbose (bool): Enable output.
        local_method (str): Name accepted by :func:`create_optimizer`.
        local_kwargs (Optional[dict]): Extra arguments for the local method.
        n_starts (int): Number of local runs.
        spread (Optional[float]): If given, starts are drawn from a normal
            distribution with this standard deviation around the initial
            guess; otherwise from a Latin hypercube over the whole box.
        seed (Optional[int]): Random seed for the start points.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        local_method: str = "powell",
        local_kwargs: dict[str, Any] | None = None,
        n_starts: int = 5,
        spread: float | None = None,
        seed: int | None = None,
    ):
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.local_method = local_method
        self.local_kwargs = dict(local_kwargs or {})
        self.n_starts = n_starts
        self.spread = spread
        self.seed = seed

    def _start_points(self, x0: np.ndarray) -> np.ndarray:
        """Return ``n_starts`` feasible start points, the first being ``x0``."""
        rng = np.random.default_rng(self.seed)
        lo, hi = self.bounds_array[:, 0], self.bounds_array[:, 1]
        n_extra = max(self.n_starts - 1, 0)
        if n_extra == 0:
            return x0[None, :]
        if self.spread is not None:
            extra = rng.normal(loc=x0, scale=self.spread, size=(n_extra, len(x0)))
        else:
            from scipy.stats import qmc

            unit = qmc.LatinHypercube(d=len(x0), seed=rng).random(n_extra)
            extra = lo + unit * (hi - lo)
        return np.vstack([x0[None, :], np.clip(extra, lo, hi)])

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: np.ndarray | None = None
    ) -> OptimizationResult:
        """Run the local method from every start point and keep the best run.

        Args:
            objective_func (Callable): Objective function to minimize.
            initial_guess (Optional[np.ndarray]): First start point. Midpoint of bounds if None.

        Returns:
            OptimizationResult: The best local run, with all histories concatenated.
        """
        if initial_guess is None:
            initial_guess = np.mean(self.bounds_array, axis=1)
        x0 = self._project_to_bounds(np.asarray(initial_guess, dtype=float))

        self._reset_tracking()
        start_time = time.time()
        results: list[OptimizationResult] = []
        for k, start in enumerate(self._start_points(x0)):
            if self.verbose:
                print(f"Multi-start {k + 1}/{self.n_starts}")
            local_opt = create_optimizer(
                self.local_method,
                self.bounds,
                max_iterations=self.max_iterations,
                verbose=self.verbose,
                tolerance=self.tolerance,
                **self.local_kwargs,
            )
            results.append(local_opt.optimize(objective_func, start))

        best = min(results, key=lambda r: r.fun)
        self.history = _merge_histories([r.history for r in results])
        self._n_evaluations = len(self.history)
        execution_time = time.time() - start_time
        result = OptimizeResult(
            x=np.asarray(best.x, dtype=float),
            fun=float(best.fun),
            success=bool(best.success),
            nit=int(sum(r.nit for r in results)),
            nfev=int(sum(r.nfev for r in results)),
            message=f"best of {len(results)} starts: {best.message}",
        )
        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


def create_optimizer(
    method: str, bounds: dict[str, tuple[float, float]], max_iterations: int = 100, verbose: bool = True, **kwargs
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
        "cma_es": CMAESOptimizer,
        "cmaes": CMAESOptimizer,
        "cma": CMAESOptimizer,
        "bobyqa": BOBYQAOptimizer,
        "dfols": DFOLSOptimizer,
        "dfo_ls": DFOLSOptimizer,
        "least_squares": LeastSquaresOptimizer,
        "trf": LeastSquaresOptimizer,
        "hybrid": HybridOptimizer,
        "de_powell": HybridOptimizer,
        "multistart": MultiStartOptimizer,
        "multi_start": MultiStartOptimizer,
    }

    if method not in optimizer_map:
        available = ", ".join(optimizer_map.keys())
        raise ValueError(f"Unknown optimization method: {method}. Available: {available}")

    optimizer_class = optimizer_map[method]
    return optimizer_class(bounds=bounds, max_iterations=max_iterations, verbose=verbose, **kwargs)
