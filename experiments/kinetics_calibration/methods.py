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


METHODS: dict[str, Method] = {
    "nelder_mead": Method("nelder_mead", "nelder_mead", True, {"adaptive": True}),
    "powell": Method("powell", "powell", True, {}),
    "lbfgsb": Method("lbfgsb", "lbfgsb", True, {}),
    "slsqp": Method("slsqp", "slsqp", True, {}),
    "differential_evolution": Method("differential_evolution", "differential_evolution", False, {"seed": 0}),
}


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

    if method.optimizer == "differential_evolution":
        # popsize is *per parameter* in scipy, so the generation cost is
        # popsize * n_p. Size it so the budget buys DE_TARGET_GENERATIONS.
        popsize = max(4, budget // max(1, DE_TARGET_GENERATIONS * n_p))
        kwargs["population_size"] = int(popsize)
        max_iterations = DE_TARGET_GENERATIONS * 4
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
