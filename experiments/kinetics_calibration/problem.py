"""The calibration problem: measurements in, kinetic log-factors out.

Defines the objective every method minimises, the two start points (``near`` /
``far``), the shared evaluation budget, and the recovery metrics.

Objective
---------
A chi-square in the dataset's own noise model::

    J(theta) = mean_{t,c} ( (y_meas[t,c] - y_sim(theta)[t,c]) / sigma[t,c] )^2

With the true kinetics and the true initial state this is ~1.0 by construction
(verified in ``test_problem.py``), so ``J`` is directly interpretable: 1.0 means
"fits to within sensor noise", and the value the optimiser reaches can be compared
against ``J(theta_true)`` to tell a *search* failure from an *identifiability*
failure — the distinction the whole study turns on.

Budget
------
Every method gets the same number of forward simulations (``budget``). The wrapper
raises :class:`BudgetExhausted` on overrun and the runner keeps the best point seen
so far, so "accuracy at equal cost" is a fair comparison across optimisers whose
natural iteration counts differ by an order of magnitude.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastsim import (
    KINETIC_KEYS,
    ForwardModel,
    adm1_torch_params,
    fostac_from_states,
    fostac_sigma,
    sensor_sigma,
)

#: Search box on the log-factors: a factor between 1/4 and 4 of nominal. The truth
#: is lognormal(0, 0.25), so this is ~5.5 sigma wide — generous enough that the box
#: never binds on the truth, tight enough to stay physically sane.
LOG_FACTOR_BOUND = float(np.log(4.0))

#: Spread of the ``far`` start (vs 0.25 for the truth), so a far start sits a
#: typical factor ~1.8 off the true value instead of ~1.2.
FAR_START_SIGMA = 0.6

#: A failed simulation (solver blow-up on an absurd kinetic set) scores this.
FAILURE_PENALTY = 1e8


class BudgetExhausted(RuntimeError):
    """Raised once the objective has used its allowance of forward simulations."""


# --------------------------------------------------------------------------
# Start points
# --------------------------------------------------------------------------
def start_point(kind: str, seed: int, n: int = len(KINETIC_KEYS)) -> np.ndarray:
    """A reproducible start in log-factor space.

    Args:
        kind: ``"near"`` — all factors 1.0, i.e. the literature default values a
            practitioner would actually start from (~20 % off the truth on
            average). ``"far"`` — a seeded lognormal(0, :data:`FAR_START_SIGMA`)
            draw, ~80 % off on average. The far draw is *independent of the
            truth*, so it leaks nothing; how far it landed is recorded per run as
            ``start_dist``.
        seed: Seed for the ``far`` draw (ignored for ``near``).
        n: Number of parameters.

    Returns:
        ``(n,)`` log-factors.
    """
    if kind == "near":
        return np.zeros(n)
    if kind == "far":
        rng = np.random.default_rng(seed)
        return np.clip(
            rng.normal(0.0, FAR_START_SIGMA, size=n),
            -LOG_FACTOR_BOUND,
            LOG_FACTOR_BOUND,
        )
    raise ValueError(f"Unknown start kind: {kind!r} (expected 'near' or 'far')")


# --------------------------------------------------------------------------
# Problem
# --------------------------------------------------------------------------
@dataclass
class CalibrationProblem:
    """One calibration task: one series, one window, one parameter subset.

    Args:
        series: A dataset series dict (from ``loader.load_test``).
        n_days: Length of the calibration window in days, taken from ``t=0``.
        active: Indices into :data:`KINETIC_KEYS` that are optimised. Inactive
            parameters are held at their start value, *not* at the truth — a
            reduced parameter set therefore carries the bias of whatever it left
            behind, which is the honest cost of screening.
        known_x0: If ``True`` the calibrator is handed the dataset's true initial
            state — the identifiability upper bound. If ``False`` it must settle
            the plant itself under the kinetics it is trying (``warmup_days``),
            which is the realistic case.
        warmup_days: Settling time for the ``known_x0=False`` variant.
        model: Pre-built forward model (one per worker process).
        budget: Max forward simulations; ``None`` for unlimited.
        use_fostac: Add the weekly FOS/TAC titration to the objective. Only 9
            values per series against 7205 sensor terms, but they observe the
            VFA pool directly, which is the direction the gas sensors cannot
            see. Each term is weighted by its own sigma, so the handful of
            points carry their true information content rather than being
            drowned out by sheer count.
    """

    series: dict[str, Any]
    n_days: int
    active: np.ndarray
    known_x0: bool = True
    warmup_days: float = 30.0
    model: ForwardModel | None = None
    budget: int | None = None
    use_fostac: bool = False
    fostac_every_days: float = 7.0
    #: Weight of the lognormal prior penalty (Bayesian MAP). 0 disables it.
    #: The diagnosed failure mode is that calibrated parameters get *bent* to
    #: absorb the error of the ones held fixed, and that happens well inside any
    #: sane box — so a penalty that grows with the distance catches it where a
    #: hard bound does not.
    prior_weight: float = 0.0
    #: Per-parameter search box in log space; None uses the literature bounds.
    log_bounds: np.ndarray | None = None

    n_evals: int = field(default=0, init=False)
    best_value: float = field(default=np.inf, init=False)
    best_theta: np.ndarray | None = field(default=None, init=False)
    trace: list[tuple[int, float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = ForwardModel.build(prune=True)
        self.active = np.asarray(self.active, dtype=int)
        # Inactive parameters sit at nominal until set_base() says otherwise.
        self._base_theta = np.zeros(len(KINETIC_KEYS))
        from bounds import prior_log_sigma

        self._prior_sigma = prior_log_sigma()
        self.n_steps = int(self.n_days * 24)
        # The calibrator only ever sees the *reported* (noisy) feed and the noisy
        # sensors — feed_true / states are held back for scoring.
        self.feed = np.asarray(self.series["feed_noisy"], dtype=float)
        self.meas = np.asarray(self.series["measurements"], dtype=float)[: self.n_steps + 1]
        self.x0_true = np.asarray(self.series["states"], dtype=float)[0]
        self.f_true = np.asarray(self.series["kinetic_factors"], dtype=float)
        self.theta_true = np.log(self.f_true)
        # Sigma is fixed on the *measured* magnitudes, not recomputed per
        # candidate — otherwise the optimiser could lower the chi-square by
        # inflating the simulated gas flow (and hence its own error bars).
        self._sigma = sensor_sigma(self.meas)

        # FOS/TAC. The dataset stores an independent titration every hour, so the
        # sampling frequency is chosen HERE and never inferred from the array —
        # inferring it would silently turn 9 weekly lab values into 1441.
        self._fostac_idx = np.zeros(0, dtype=int)
        if self.use_fostac:
            if "fostac" not in self.series:
                raise KeyError(
                    "use_fostac=True but the series has no 'fostac' array — "
                    "regenerate the benchmark, or run its add_fostac.py."
                )
            import paths

            paths.add_dataset_to_path()
            from fostac import sample_indices

            lab = np.asarray(self.series["fostac"], dtype=float)[: self.n_steps + 1]
            rows = sample_indices(len(lab), every_days=self.fostac_every_days)
            # A clipped titration (true FOS below the detection limit) carries no
            # usable information and would divide by a zero sigma.
            rows = rows[np.isfinite(lab[rows, 0]) & (lab[rows, 0] > 0.0)]
            self._fostac_idx = rows
            self._fostac_meas = lab[rows]
            self._fostac_sigma = fostac_sigma(self._fostac_meas)
            assert self.model is not None
            self._params = adm1_torch_params(self.model.plant)

    # -- forward evaluation ----------------------------------------------
    def simulate(self, theta_full: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """``(states, observables)`` for a full 26-vector, or ``None`` on failure.

        The states come back alongside the observables because the FOS/TAC
        channels are read out of them — the lab measurement costs no extra
        simulation, only the state trajectory the run already produced.
        """
        assert self.model is not None
        try:
            self.model.set_log_factors(theta_full)
            if self.known_x0:
                x0 = self.x0_true
            else:
                x0 = self.model.warmup(self.model.initial_state(), self.warmup_days, self.feed[0])
            states, obs = self.model.simulate(x0, self.feed, self.n_steps)
            if not np.all(np.isfinite(obs)):
                return None
            return states, obs
        except Exception:  # noqa: BLE001 - an unsolvable parameter set scores as invalid
            return None

    def expand(self, theta_active: Sequence[float]) -> np.ndarray:
        """Scatter the optimised subset back into the full 26-vector."""
        full = np.array(self._base_theta, dtype=float, copy=True)
        full[self.active] = np.asarray(theta_active, dtype=float)
        return full

    def set_base(self, theta_start_full: np.ndarray) -> None:
        """Fix the values the inactive parameters are held at (the start point)."""
        self._base_theta = np.asarray(theta_start_full, dtype=float).copy()

    # -- objective --------------------------------------------------------
    def objective(self, theta_active: Sequence[float]) -> float:
        """Chi-square of the active parameters; tracks budget and best-so-far."""
        if self.budget is not None and self.n_evals >= self.budget:
            raise BudgetExhausted(f"budget of {self.budget} simulations used")
        self.n_evals += 1

        theta_full = self.expand(theta_active)
        result = self.simulate(theta_full)
        if result is None:
            return FAILURE_PENALTY
        states, obs = result

        residuals = ((self.meas - obs) / self._sigma) ** 2
        if self.use_fostac and len(self._fostac_idx):
            sim_lab = fostac_from_states(states[self._fostac_idx], self._params)
            lab_res = ((self._fostac_meas - sim_lab) / self._fostac_sigma) ** 2
            # Mean over ALL terms, so each measurement counts once regardless of
            # channel. The 9 lab rows are a small share of the total, which is
            # exactly their real weight — the point is what they observe, not
            # how many they are.
            value = float((residuals.sum() + lab_res.sum()) / (residuals.size + lab_res.size))
        else:
            value = float(np.mean(residuals))

        if self.prior_weight:
            # Lognormal prior centred on the literature value (theta = 0).
            act = self.active
            value += self.prior_weight * float(np.mean((theta_full[act] / self._prior_sigma[act]) ** 2))
        if not np.isfinite(value):
            return FAILURE_PENALTY
        if value < self.best_value:
            self.best_value = value
            self.best_theta = theta_full.copy()
            self.trace.append((self.n_evals, value))
        return value

    def make_objective(self) -> Callable[[np.ndarray], float]:
        """The objective as a plain callable, for the optimiser factories."""
        return lambda x: self.objective(x)

    # -- scoring ----------------------------------------------------------
    def bounds(self) -> dict[str, tuple[float, float]]:
        """Search box for the active parameters, keyed by kinetic name.

        Defaults to the literature-derived per-parameter bounds in
        :mod:`bounds`; the old flat box is available by passing ``log_bounds``.
        """
        if self.log_bounds is None:
            from bounds import log_bounds as _lit

            lb = _lit()
        else:
            lb = np.asarray(self.log_bounds, dtype=float)
        return {KINETIC_KEYS[i]: (float(lb[i, 0]), float(lb[i, 1])) for i in self.active}

    def recovery(self, theta_hat_full: np.ndarray) -> dict[str, Any]:
        """Parameter-recovery metrics for an estimate.

        The headline number is the **log-error** ``|ln(f_hat / f_true)|``: 0 means
        exact, 0.22 means "off by 25 %", 0.69 means "off by a factor of 2". It is
        reported only over the *active* parameters — a method cannot be blamed for
        the ones it was not allowed to move.
        """
        theta_hat_full = np.asarray(theta_hat_full, dtype=float)
        err = np.abs(theta_hat_full - self.theta_true)
        act = self.active

        # Fair scoring when K_S is held fixed. In the first-order regime only the
        # ratio k_m/K_S acts, so a k_m that is free while its K_S is frozen at
        # nominal must absorb the true K_S perturbation to fit at all. Comparing
        # such a k_m against the true k_m alone would report an error the data
        # could never have avoided; the identifiable quantity is the ratio.
        from bounds import FIRST_ORDER_PAIRS

        active_set = {int(i) for i in act}
        for km_name, ks_name in FIRST_ORDER_PAIRS:
            i_km = KINETIC_KEYS.index(km_name)
            i_ks = KINETIC_KEYS.index(ks_name)
            if i_km in active_set and i_ks not in active_set:
                ratio_hat = theta_hat_full[i_km] - self._base_theta[i_ks]
                ratio_true = self.theta_true[i_km] - self.theta_true[i_ks]
                err[i_km] = abs(ratio_hat - ratio_true)
        return {
            "log_err_median": float(np.median(err[act])),
            "log_err_mean": float(np.mean(err[act])),
            "log_err_max": float(np.max(err[act])),
            "frac_within_10pct": float(np.mean(err[act] < np.log(1.10))),
            "frac_within_25pct": float(np.mean(err[act] < np.log(1.25))),
            "log_err_per_param": {KINETIC_KEYS[i]: float(err[i]) for i in range(len(KINETIC_KEYS))},
        }

    def predictive_score(self, theta_hat_full: np.ndarray) -> dict[str, float]:
        """Fit and forecast quality over the **full 60 days** of the series.

        Simulating the whole series from ``t=0`` under the estimate and scoring the
        part beyond the calibration window measures what calibration is actually
        for — predicting operation it was not fitted to. Reported as a chi-square
        so it is on the same scale as the training objective.
        """
        assert self.model is not None
        total = len(self.series["measurements"]) - 1
        saved_steps, self.n_steps = self.n_steps, total
        try:
            result = self.simulate(np.asarray(theta_hat_full, dtype=float))
        finally:
            self.n_steps = saved_steps
        if result is None:
            return {"chi2_full": float("nan"), "chi2_holdout": float("nan")}
        _, obs = result

        meas_full = np.asarray(self.series["measurements"], dtype=float)
        sigma_full = sensor_sigma(meas_full)
        chi2 = ((meas_full - obs) / sigma_full) ** 2
        cut = self.n_steps + 1
        holdout = float(np.mean(chi2[cut:])) if cut < len(chi2) else float("nan")
        return {"chi2_full": float(np.mean(chi2)), "chi2_holdout": holdout}
