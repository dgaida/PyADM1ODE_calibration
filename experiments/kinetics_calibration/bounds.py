"""Literature-based search bounds and priors for the ADM1 kinetics.

Replaces the flat "factor 1/4 to 4 for everything" box with per-parameter limits,
and offers a soft prior as the better-behaved alternative to hard limits.

Why per-parameter
-----------------
The families differ by orders of magnitude in how much they genuinely vary:

* **Disintegration / hydrolysis** are substrate properties. Published calibrations
  span 0.16 d^-1 (food waste) to 10 d^-1 (maize silage + cattle manure), and split
  slow/fast fractions differ by a factor ~40. Wide bounds are honest here.
* **Uptake rates** ``k_m`` vary across plants but within roughly one order of
  magnitude (e.g. ``k_m_ac`` reported 15-75 kg COD/kg COD/d).
* **Decay rates** ``k_dec`` are biologically conserved and are the most tightly
  constrained family (0.001-0.04 d^-1 across the literature, with 0.02 standard).
* **Half-saturation** ``K_S`` is reported over a wide range (``K_S_ac`` 0.42-1.59
  kg COD/m^3) — but see the warning below: at this plant's operating point it is
  not separately identifiable anyway.

Bounds are expressed as **factors on the model's own nominal value**, not as
absolute numbers. The nominal values here are ADM1da, temperature-corrected to
the plant's 42 °C, so they do not equal the ADM1 STR defaults and absolute
literature limits cannot be transplanted directly.

Hard bounds or soft prior?
--------------------------
Hard bounds only stop a search at the wall. This study's diagnosis was that the
calibrated parameters get *bent* to compensate for the ones held fixed — and that
bending mostly happens well inside any sane box, so bounds alone will not catch
it. :data:`PRIOR_LOG_SIGMA` therefore defines a lognormal prior for an optional
penalty term (Bayesian MAP), which resists the bending in proportion to how far
it goes rather than only at the boundary.

Honesty note
------------
The benchmark's true factors are drawn lognormal(0, 0.25). Tightening bounds
towards that distribution improves the score *by construction*. The widths below
are therefore taken from the published spread of each family, NOT from the
generating distribution, and every one of them stays well outside +-3 sigma of it
(factor 2.1). On a real plant the truth may still sit outside — that is what the
soft prior handles more gracefully than a hard wall.

Sources
-------
* Batstone et al. (2002), ADM1 STR; Rosen & Jeppsson (2006), BSM2 defaults.
* ``k_m_ac`` 15-75, ``K_S_ac`` 0.42-1.59: reported ranges in ADM1 VFA-uptake
  studies (Renewable Energy 2018, S0960148118303495).
* ``k_hyd`` 10 d^-1 for maize silage + cattle manure; 3 d^-1 food waste.
* ``k_dis`` 0.16 (food waste), 0.26 (grass silage), 1.4-1.7 / 60 slow vs rapid.
* ``k_dec`` 0.001-0.04, 0.02 standard.
"""

from __future__ import annotations

import numpy as np
from fastsim import KINETIC_KEYS

#: Multiplicative factor bounds per parameter family, as (low, high) on nominal.
#: Wider where the literature genuinely disagrees, tighter where biology pins it.
_FAMILY_BOUNDS: dict[str, tuple[float, float]] = {
    # Substrate properties: published calibrations differ by ~40x between the
    # slow and fast fractions alone, so this family gets the widest box.
    "k_dis": (0.1, 10.0),
    "k_hyd": (0.1, 10.0),
    # Uptake rates: roughly one order of magnitude across plants.
    "k_m_": (0.25, 4.0),
    # Half-saturation: wide in the literature, but not separately identifiable at
    # this operating point (see IDENTIFIABLE_ALONE) — the box is a formality.
    "K_S_": (0.25, 4.0),
    # Decay: the most conserved family; 0.001-0.04 around a 0.02 standard.
    "k_dec": (0.3, 3.0),
}

#: 1-sigma of the lognormal prior per family, in log units, for the optional MAP
#: penalty. Narrower than the bounds: a bound says "impossible beyond here", a
#: prior says "increasingly implausible from here on".
_FAMILY_PRIOR_SIGMA: dict[str, float] = {
    "k_dis": 0.7,  # substrate-dependent, weakly informative
    "k_hyd": 0.7,
    "k_m_": 0.4,
    "K_S_": 0.4,
    "k_dec": 0.25,  # biologically conserved
}

#: Parameters that are NOT separately identifiable at this plant's operating
#: point, with the reason. Measured, not assumed: the normalised Fisher matrix
#: gives corr(k_m_X, K_S_X) = -1.000 for su/aa/c4/h2 and -0.998 for fa, and the
#: substrate levels explain why.
#:
#: * su, aa, fa, c4, h2: S/K_S = 0.00-0.06, i.e. deep in the first-order regime
#:   where the Monod term collapses to (k_m/K_S)*S — only the RATIO acts.
#: * pro (S/K_S = 26) and ac (S/K_S = 3.6): saturated, so K_S barely enters.
#:
#: Either way K_S carries no independent information here and calibrating it only
#: adds a direction the data cannot constrain.
NOT_IDENTIFIABLE_ALONE: dict[str, str] = {
    "K_S_su": "S/K_S=0.01, first-order: only k_m_su/K_S_su identifiable",
    "K_S_aa": "S/K_S=0.00, first-order: only k_m_aa/K_S_aa identifiable",
    "K_S_fa": "S/K_S=0.06, first-order: only k_m_fa/K_S_fa identifiable",
    "K_S_c4": "S/K_S=0.03, first-order: only k_m_c4/K_S_c4 identifiable",
    "K_S_h2": "S/K_S=0.01, first-order: only k_m_h2/K_S_h2 identifiable",
    "K_S_pro": "S/K_S=26, saturated: K_S_pro has almost no effect",
    "K_S_ac": "S/K_S=3.6, saturated: K_S_ac has little effect",
}

#: Monod pairs whose ratio is the identifiable quantity in the first-order
#: regime. Used to score a run fairly when K_S is held fixed: the recovered k_m
#: then absorbs the true K_S perturbation, so comparing it against the true k_m
#: alone would report an error the data could never have avoided.
FIRST_ORDER_PAIRS: tuple[tuple[str, str], ...] = (
    ("k_m_su", "K_S_su"),
    ("k_m_aa", "K_S_aa"),
    ("k_m_fa", "K_S_fa"),
    ("k_m_c4", "K_S_c4"),
    ("k_m_h2", "K_S_h2"),
)


def _family(name: str) -> str:
    for prefix in _FAMILY_BOUNDS:
        if name.startswith(prefix):
            return prefix
    raise KeyError(f"No bound family for {name!r}")


def log_bounds() -> np.ndarray:
    """``(26, 2)`` array of (low, high) bounds in log-factor space."""
    out = np.zeros((len(KINETIC_KEYS), 2))
    for i, name in enumerate(KINETIC_KEYS):
        lo, hi = _FAMILY_BOUNDS[_family(name)]
        out[i] = (np.log(lo), np.log(hi))
    return out


def prior_log_sigma() -> np.ndarray:
    """``(26,)`` prior standard deviations in log-factor space."""
    return np.array([_FAMILY_PRIOR_SIGMA[_family(n)] for n in KINETIC_KEYS], dtype=float)


def parameters_without_ks() -> np.ndarray:
    """Indices of every kinetic except the seven ``K_S`` (19 parameters).

    The recommended active set: dropping ``K_S`` removes exactly the directions
    the normalised Fisher matrix shows as degenerate, without giving up anything
    the measurements could have resolved.
    """
    return np.array(
        [i for i, n in enumerate(KINETIC_KEYS) if n not in NOT_IDENTIFIABLE_ALONE],
        dtype=int,
    )


def describe() -> str:
    """Human-readable table of the bounds, for the report and for review."""
    lb = log_bounds()
    ps = prior_log_sigma()
    lines = [f"{'Kinetik':<12}{'Faktor unten':>14}{'oben':>8}{'Prior sigma':>13}  Hinweis"]
    for i, n in enumerate(KINETIC_KEYS):
        note = NOT_IDENTIFIABLE_ALONE.get(n, "")
        lines.append(f"{n:<12}{np.exp(lb[i, 0]):>14.2f}{np.exp(lb[i, 1]):>8.2f}{ps[i]:>13.2f}  {note}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
