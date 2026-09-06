"""Sensitivity and identifiability screening (Brun et al. 2001).

Answers "which kinetics can be found at all, and how much data does it take"
(Q2 and Q3) *before* any optimiser is run — and far more cheaply, because the
answer comes from one Jacobian rather than hundreds of searches.

Uses nothing from ``pyadm1ode_calibration``. Its ``analysis/sensitivity.py``
averages each output series to a single mean before differencing (erasing the
post-switch transients that carry the information) and reports unscaled relative
indices, and its ``analysis/identifiability.py`` leaves the correlation analysis
as an empty placeholder with a ``1e-6`` identifiability threshold. Both are
documented in the README; the Brun et al. (2001) measures are implemented here
instead.

The Jacobian
------------
Central differences of the noise-scaled observables with respect to the log-factors::

    S[t, c, j] = (1 / sigma[t,c]) * d y_sim[t,c] / d theta_j

Scaling by the sensor sigma is what makes the columns comparable: a column's norm
then reads directly as "how many noise standard deviations does the signal move if
this kinetic changes by one log unit". Below ~1 the parameter is buried in the
noise no matter which optimiser you point at it.

Because the model is propagated forward in time, the rows of ``S`` for the first
``N`` days are a prefix of the 60-day Jacobian — so **every window length is scored
from a single 60-day computation**, which is why the data-length question is
answered here at 1/4 of the cost.

Two measures per Brun et al. (2001), *Practical identifiability of ASM2d
parameters*, Wat. Res. 35(16):
* ``delta_msqr[j]`` — importance: RMS of column ``j``. High = the data respond to it.
* ``gamma[K]`` — collinearity index of a subset: ``1 / sqrt(lambda_min)`` of the
  correlation matrix of the *normalised* columns. Large = the subset's effects
  cancel, so the members cannot be told apart even when each is individually
  strong. The literature threshold for "practically identifiable together" is
  ``gamma < 10`` (loose) to ``gamma < 5`` (strict).
"""

from __future__ import annotations

from collections.abc import Sequence
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

#: Step for the central difference in log-factor space (~5 % parameter change).
DEFAULT_STEP = 0.05


@dataclass
class Jacobian:
    """Noise-scaled output Jacobian of one series over the full 60 days.

    Attributes:
        S: ``(T, C, P)`` sensitivity of each channel/time to each log-factor.
        keys: Parameter names matching the last axis.
        n_sims: Forward simulations spent building it.
    """

    S: np.ndarray
    keys: tuple[str, ...]
    n_sims: int
    #: Sensitivity of the weekly FOS/TAC rows, ``(n_samples, 2, P)``. Empty when
    #: the lab channel was not requested.
    S_lab: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 0)))
    #: Sample day of each row of :attr:`S_lab`, needed to truncate it per window.
    lab_days: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def window(self, n_days: int | None = None) -> np.ndarray:
        """Flatten to ``(T*C, P)``, optionally truncated to the first ``n_days``.

        When the lab channel is present its rows are appended, truncated to the
        same window, so a caller cannot accidentally score sensors over 10 days
        against titrations over 60.
        """
        S = self.S if n_days is None else self.S[: int(n_days * 24) + 1]
        flat = S.reshape(-1, S.shape[-1])
        if self.S_lab.size:
            keep = np.ones(len(self.S_lab), dtype=bool) if n_days is None else self.lab_days <= n_days
            lab = self.S_lab[keep].reshape(-1, self.S_lab.shape[-1])
            flat = np.concatenate([flat, lab], axis=0)
        return flat


def compute_jacobian(
    series: dict[str, Any],
    model: ForwardModel | None = None,
    step: float = DEFAULT_STEP,
    around: str = "truth",
    n_days: float = 60.0,
    use_fostac: bool = False,
    fostac_every_days: float | None = None,
) -> Jacobian:
    """Central-difference Jacobian of the scaled observables.

    Args:
        series: A dataset series dict.
        model: Pre-built forward model; built here if omitted.
        step: Central-difference step in log-factor space.
        around: ``"truth"`` linearises at the series' true kinetics (the honest
            local identifiability of the real system), ``"nominal"`` at the
            literature defaults (what a practitioner could compute without
            knowing the answer). They agree closely, which is itself worth
            reporting: the screening is usable in practice.
        n_days: Horizon.
        fostac_every_days: Override the lab sampling interval to answer
            "would measuring more often help?". ``None`` uses the interval
            actually stored in the dataset (weekly).

    Returns:
        The :class:`Jacobian`.
    """
    model = model or ForwardModel.build(prune=True)
    n_steps = int(n_days * 24)
    feed = np.asarray(series["feed_noisy"], dtype=float)
    x0 = np.asarray(series["states"], dtype=float)[0]
    meas = np.asarray(series["measurements"], dtype=float)[: n_steps + 1]
    sigma = sensor_sigma(meas)

    base = np.log(np.asarray(series["kinetic_factors"], dtype=float)) if around == "truth" else np.zeros(len(KINETIC_KEYS))

    # The weekly lab rows are appended as two extra "channels" that are only
    # populated on sample days. Scaling them by their own sigma puts them on the
    # same footing as the online sensors, so importance and collinearity stay
    # comparable with and without them.
    lab_rows = np.zeros(0, dtype=int)
    if use_fostac:
        import paths

        paths.add_dataset_to_path()
        from fostac import SAMPLE_EVERY_DAYS, sample_indices

        params = adm1_torch_params(model.plant)
        every = SAMPLE_EVERY_DAYS if fostac_every_days is None else fostac_every_days
        lab_rows = sample_indices(n_steps + 1, every_days=every)
        # Sigma from the *true* values at those instants: the Fisher information
        # depends on where and how precisely one measures, not on the particular
        # numbers drawn, so this works for any hypothetical frequency.
        truth = fostac_from_states(np.asarray(series["states"], dtype=float)[lab_rows], params)
        lab_rows = lab_rows[truth[:, 0] > 0.0]
        lab_sigma = fostac_sigma(truth[truth[:, 0] > 0.0])

    n_p = len(KINETIC_KEYS)
    n_chan = sigma.shape[1]
    S = np.zeros((n_steps + 1, n_chan, n_p))
    S_lab = np.zeros((len(lab_rows), 2, n_p))
    n_sims = 0
    for j in range(n_p):
        cols, labs = [], []
        for sign in (+1.0, -1.0):
            theta = base.copy()
            theta[j] += sign * step
            model.set_log_factors(theta)
            states, obs = model.simulate(x0, feed, n_steps)
            cols.append(obs)
            if use_fostac:
                labs.append(fostac_from_states(states[lab_rows], params))
            n_sims += 1
        S[:, :, j] = (cols[0] - cols[1]) / (2.0 * step) / sigma
        if use_fostac:
            S_lab[:, :, j] = (labs[0] - labs[1]) / (2.0 * step) / lab_sigma
    model.reset_kinetics()
    return Jacobian(
        S=S,
        keys=tuple(KINETIC_KEYS),
        n_sims=n_sims,
        S_lab=S_lab,
        lab_days=lab_rows / 24.0,
    )


def delta_msqr(S_flat: np.ndarray) -> np.ndarray:
    """Per-parameter importance: RMS of each Jacobian column.

    Units are "sensor sigmas of output movement per log unit of parameter".
    """
    return np.sqrt(np.mean(S_flat**2, axis=0))


def collinearity(S_flat: np.ndarray, subset: Sequence[int]) -> float:
    """Brun collinearity index of a parameter subset (``inf`` if degenerate).

    Columns are normalised to unit length first, so the index measures *only*
    whether the parameters' effects are linearly dependent — not how strong they
    are. That separation is the point: a subset can be highly sensitive and still
    unidentifiable.
    """
    sub = S_flat[:, list(subset)]
    norms = np.linalg.norm(sub, axis=0)
    if np.any(norms <= 0):
        return float("inf")
    normed = sub / norms
    eig = np.linalg.eigvalsh(normed.T @ normed)
    lam_min = float(np.min(eig))
    if lam_min <= 1e-14:
        return float("inf")
    return float(1.0 / np.sqrt(lam_min))


def rank_identifiable_subsets(
    S_flat: np.ndarray,
    max_size: int = 6,
    gamma_max: float = 10.0,
    candidates: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Greedily grow the largest well-conditioned parameter subset.

    Starts from the most important parameter and repeatedly adds whichever
    remaining candidate keeps the collinearity index lowest, stopping when no
    addition stays under ``gamma_max``. This is the standard practical recipe:
    exhaustive search over all subsets of 26 parameters is not tractable, and the
    greedy path is what a practitioner would follow anyway.

    Returns:
        One record per accepted subset size, each with ``size``, ``indices``,
        ``names`` and ``gamma``.
    """
    imp = delta_msqr(S_flat)
    pool = list(range(S_flat.shape[1])) if candidates is None else list(candidates)
    pool = [j for j in pool if imp[j] > 0]
    if not pool:
        return []

    chosen = [max(pool, key=lambda j: imp[j])]
    out = [
        {
            "size": 1,
            "indices": list(chosen),
            "names": [KINETIC_KEYS[j] for j in chosen],
            "gamma": 1.0,
        }
    ]
    while len(chosen) < max_size:
        rest = [j for j in pool if j not in chosen]
        if not rest:
            break
        scored = [(collinearity(S_flat, chosen + [j]), j) for j in rest]
        gamma, best = min(scored, key=lambda t: t[0])
        if gamma > gamma_max:
            break
        chosen.append(best)
        out.append(
            {
                "size": len(chosen),
                "indices": list(chosen),
                "names": [KINETIC_KEYS[j] for j in chosen],
                "gamma": float(gamma),
            }
        )
    return out


def marginal_std_err(S_flat: np.ndarray) -> np.ndarray:
    """Standard error of each log-factor **with every other parameter known**.

    ``1 / sqrt(sum_t S[t,j]^2)`` — the Cramer-Rao bound of the one-parameter
    problem, in log units (0.01 ~ 1 %). Optimistic by construction: it ignores
    the variance inflation that comes from estimating the others at the same
    time. It is the *best case*, so a parameter whose marginal error is already
    large is hopeless, full stop.
    """
    with np.errstate(divide="ignore"):
        return 1.0 / np.sqrt(np.sum(S_flat**2, axis=0))


def joint_std_err(S_flat: np.ndarray, subset: Sequence[int]) -> np.ndarray:
    """Standard errors when the whole ``subset`` is estimated together.

    The square roots of the diagonal of ``(S_K^T S_K)^-1`` — the honest bound,
    because it charges each parameter for the directions it shares with the
    others. The ratio to :func:`marginal_std_err` is the variance-inflation
    factor that collinearity buys you.
    """
    sub = S_flat[:, list(subset)]
    fim = sub.T @ sub
    try:
        cov = np.linalg.inv(fim)
    except np.linalg.LinAlgError:
        return np.full(len(subset), np.inf)
    diag = np.diag(cov)
    return np.sqrt(np.where(diag > 0, diag, np.inf))


def screen(
    jac: Jacobian,
    windows: Sequence[int] = (10, 20, 30, 60),
    gamma_max: float = 10.0,
    max_size: int = 8,
    joint_subset: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Full screening report for one series across several window lengths.

    Args:
        jac: The 60-day Jacobian.
        windows: Window lengths in days to score.
        gamma_max: Collinearity ceiling for the greedy subset search.
        max_size: Largest subset the greedy search will build.
        joint_subset: Parameter indices to also report *joint* standard errors
            for. Defaults to the greedy subset found at that window.

    Returns:
        A JSON-serialisable report. The Gram matrix ``S^T S`` is stored per
        window (26x26, ~5 kB) so joint uncertainties for any subset can be
        recomputed later without redoing the 18-minute Jacobian.
    """
    out: dict[str, Any] = {"keys": list(jac.keys), "n_sims": jac.n_sims, "windows": {}}
    for w in windows:
        S = jac.window(w)
        imp = delta_msqr(S)
        order = np.argsort(imp)[::-1]
        subsets = rank_identifiable_subsets(S, max_size=max_size, gamma_max=gamma_max)
        best = subsets[-1]["indices"] if subsets else []
        use = list(joint_subset) if joint_subset is not None else best

        se_marg = marginal_std_err(S)
        se_joint = joint_std_err(S, use) if use else np.array([])
        out["windows"][str(w)] = {
            "delta_msqr": {KINETIC_KEYS[j]: float(imp[j]) for j in range(len(imp))},
            "ranking": [KINETIC_KEYS[j] for j in order],
            "identifiable_subsets": subsets,
            "best_subset": subsets[-1]["names"] if subsets else [],
            "best_subset_gamma": subsets[-1]["gamma"] if subsets else float("nan"),
            "marginal_std_err": {KINETIC_KEYS[j]: float(se_marg[j]) for j in range(len(se_marg))},
            "joint_std_err": {KINETIC_KEYS[j]: float(se_joint[i]) for i, j in enumerate(use)},
            # S^T S: everything above can be rederived from this.
            "gram": (S.T @ S).tolist(),
        }
    return out
