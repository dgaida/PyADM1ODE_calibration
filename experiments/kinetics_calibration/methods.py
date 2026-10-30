"""The calibration methods under test.

Wraps the optimisers implemented in ``pyadm1ode_calibration`` behind one
signature so they can be compared at an equal budget of forward simulations.

Equal budget, not equal iterations
----------------------------------
These optimisers differ by an order of magnitude in how many model evaluations
one "iteration" costs (a Nelder-Mead iteration is ~1 simulation, a differential-
evolution generation is ``popsize * n_params``). Comparing them at equal
iterations would compare nothing. Every method therefore gets the same number of
forward simulations, ``max_iterations`` is set high enough that the budget binds,
and the run keeps the best point seen when the budget runs out.

Differential evolution is **start-point agnostic** — it samples the whole box and
scipy's implementation ignores ``x0``. It is flagged as such and run once per
configuration rather than pretending the near/far start means anything to it.
Its population size is scaled to the budget so it gets ~20 generations instead of
one and a half, which is the difference between a search and a random sample.
The ``de_prior`` and ``de_powell`` variants instead draw the initial population
around the start point with the prior spread, so they *do* use it.

Two families of entries
-----------------------
The first five entries are the methods of the original study at SciPy defaults.
The ``*_sigma``/``*_eps``/``de_prior`` entries are the same algorithms with their
scale parameters set from the problem (prior spread 0.25, finite-difference step
1e-3), and the remaining entries are additional algorithms. Comparing a default
entry with its scaled twin separates the algorithm from its settings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyadm1ode_calibration.calibration.optimization import create_optimizer

#: Generations differential evolution should be able to complete inside the budget.
DE_TARGET_GENERATIONS = 20


@dataclass(frozen=True)
class Method:
    """A calibration method.

    Attributes:
        name: Registry key.
        optimizer: ``create_optimizer`` method string.
        uses_start: Whether the start point influences the result.
        kwargs: Extra optimiser arguments.
    """

    name: str
    optimizer: str
    uses_start: bool
    kwargs: dict[str, Any]


#: Expected spread of the log-factors around the literature values (the prior
#: sigma of the benchmark). Every method that has a scale parameter gets it, so
#: the comparison is between algorithms, not between SciPy defaults.
PRIOR_SIGMA = 0.25

#: Finite-difference step for the gradient methods in the "_eps" variants: about
#: the square root of the solver's relative tolerance (1e-6), far above SciPy's
#: default of 1e-8 that only measures solver noise.
FD_STEP = 1e-3

METHODS: dict[str, Method] = {
    # --- the five methods of the original study, SciPy defaults ---------------
    "nelder_mead": Method("nelder_mead", "nelder_mead", True, {"adaptive": True}),
    "powell": Method("powell", "powell", True, {}),
    "lbfgsb": Method("lbfgsb", "lbfgsb", True, {}),
    "slsqp": Method("slsqp", "slsqp", True, {}),
    "differential_evolution": Method("differential_evolution", "differential_evolution", False, {"seed": 0}),
    # --- the same methods with problem-scaled settings -------------------------
    "nelder_mead_sigma": Method(
        "nelder_mead_sigma", "nelder_mead", True, {"adaptive": True, "initial_simplex_scale": PRIOR_SIGMA}
    ),
    "lbfgsb_eps": Method("lbfgsb_eps", "lbfgsb", True, {"eps": FD_STEP}),
    "slsqp_eps": Method("slsqp_eps", "slsqp", True, {"eps": FD_STEP}),
    "de_prior": Method("de_prior", "differential_evolution", True, {"seed": 0, "init_sigma": PRIOR_SIGMA, "polish": False}),
    # --- additional algorithms ------------------------------------------------
    "cmaes": Method("cmaes", "cma_es", True, {"sigma0": PRIOR_SIGMA, "seed": 0}),
    "bobyqa": Method("bobyqa", "bobyqa", True, {"rhobeg": PRIOR_SIGMA}),
    "de_powell": Method(
        "de_powell",
        "hybrid",
        True,
        {
            "global_method": "differential_evolution",
            "local_method": "powell",
            "global_kwargs": {"seed": 0, "init_sigma": PRIOR_SIGMA, "polish": False},
        },
    ),
    "multistart_powell": Method(
        "multistart_powell", "multistart", True, {"local_method": "powell", "spread": PRIOR_SIGMA, "seed": 0}
    ),
}

#: Share of the budget the global stage of ``de_powell`` may spend before Powell
#: takes over from its best point.
HYBRID_GLOBAL_SHARE = 0.5

#: Local runs of ``multistart_powell``; each gets an equal share of the budget.
MULTISTART_RUNS = 3


def run_method(
    method_name: str,
    objective: Callable[[np.ndarray], float],
    bounds: dict[str, tuple[float, float]],
    x0: np.ndarray,
    budget: int,
) -> dict[str, Any]:
    """Run one optimiser until it converges or exhausts the budget.

    Args:
        method_name: Key into :data:`METHODS`.
        objective: The problem's objective over the active parameters. Raises
            ``BudgetExhausted`` once the allowance is gone.
        bounds: Search box keyed by parameter name.
        x0: Start point for the active parameters.
        budget: Forward simulations allowed.

    Returns:
        ``{"success", "message", "n_iterations"}``. The estimate itself is read
        from the problem's ``best_theta``, not from here, so a run stopped
        mid-search still reports its best point.

    Raises:
        ValueError: If ``x0`` lies outside ``bounds`` for a method that uses the
            start point. SciPy's Powell answers that case with ``success=True``
            after *zero* evaluations, which used to be recorded as a finished run
            whose "estimate" was the start point itself. Refusing up front makes
            the misconfiguration visible instead. Start-agnostic methods
            (differential evolution) are exempt — they never read ``x0``.
    """
    method = METHODS[method_name]
    n_p = len(bounds)
    kwargs = dict(method.kwargs)

    x0 = np.asarray(x0, dtype=float)
    if method.uses_start:
        outside = [
            f"{name}: {v:+.3f} not in [{lo:+.3f}, {hi:+.3f}]"
            for (name, (lo, hi)), v in zip(bounds.items(), x0)
            if v < lo or v > hi
        ]
        if outside:
            raise ValueError(
                "start point outside the search bounds — the optimiser would " "return it unchanged: " + "; ".join(outside)
            )

    def de_sizing(de_budget: int) -> tuple[int, int]:
        """Population size per parameter and generation cap for a DE budget."""
        # popsize is *per parameter* in scipy, so the generation cost is
        # popsize * n_p. Size it so the budget buys DE_TARGET_GENERATIONS.
        popsize = max(4, de_budget // max(1, DE_TARGET_GENERATIONS * n_p))
        return int(popsize), DE_TARGET_GENERATIONS * 4

    if method.optimizer == "differential_evolution":
        kwargs["population_size"], max_iterations = de_sizing(budget)
    elif method.optimizer == "hybrid":
        # The global stage gets its share of the budget; the local stage runs
        # until the budget exception stops it, so it inherits the remainder.
        global_budget = int(budget * HYBRID_GLOBAL_SHARE)
        popsize, generations = de_sizing(global_budget)
        kwargs["global_kwargs"] = {**kwargs.get("global_kwargs", {}), "population_size": popsize}
        kwargs["global_max_iterations"] = max(1, global_budget // max(1, popsize * n_p))
        kwargs["local_max_iterations"] = budget * 4
        max_iterations = generations
    elif method.optimizer in ("cma_es", "bobyqa"):
        # Both take an explicit evaluation cap; keep it one above the budget so
        # the budget exception, not the solver, ends the run.
        kwargs["max_evaluations"] = budget + 1
        max_iterations = budget * 4
    elif method.optimizer == "multistart":
        # Each start gets an equal share of the budget through the local
        # method's evaluation cap; without it the first run would spend the
        # whole budget and the "multi-start" would just be a single Powell.
        kwargs["n_starts"] = MULTISTART_RUNS
        kwargs["local_kwargs"] = {**kwargs.get("local_kwargs", {}), "max_evaluations": budget // MULTISTART_RUNS}
        max_iterations = budget * 4
    else:
        # Generous, so the budget is what actually stops the search.
        max_iterations = budget * 4

    optimizer = create_optimizer(
        method=method.optimizer,
        bounds=bounds,
        max_iterations=max_iterations,
        verbose=False,
        tolerance=1e-8,
        **kwargs,
    )

    from problem import BudgetExhausted  # local import: avoids a cycle

    try:
        result = optimizer.optimize(objective, initial_guess=np.asarray(x0, float))
        return {
            "success": bool(result.success),
            "message": str(result.message),
            "n_iterations": int(result.nit),
        }
    except BudgetExhausted as exc:
        return {"success": False, "message": f"budget exhausted: {exc}", "n_iterations": -1}
    except Exception as exc:  # noqa: BLE001 - a blown-up solver must not kill the sweep
        return {"success": False, "message": f"{type(exc).__name__}: {exc}", "n_iterations": -1}
