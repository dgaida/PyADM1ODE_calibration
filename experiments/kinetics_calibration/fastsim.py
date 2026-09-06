"""Fast forward model for the kinetics-calibration study.

Rebuilds exactly the simulator that produced ``datasets/benchmark`` — same plant
(``build_multi_stage_plant``, digester ``primary``), same ``k_dec_ac`` operating
point, same 5 substrates and 5 sensors, same hourly propagation — but **prunes the
plant down to the primary digester**.

Why not ``pyadm1ode_calibration.calibration.core.PlantSimulator``
----------------------------------------------------------------
That class reduces the substrate feed to ``np.mean(Q, axis=0)`` and holds it
constant for the whole simulation. This benchmark's signal *is* the load changes
(6 phases, 5 switches per series), so a constant mean feed would erase exactly
what makes the kinetics identifiable. :meth:`ForwardModel.simulate` imposes the
schedule step by step instead. Every calibrator in that package routes through
``PlantSimulator``, which is why the study uses its *optimisers* only.

Why pruning is safe *and* worth it
----------------------------------
The benchmark's five sensors all sit on ``primary``, and the plant cascade is
one-directional (``primary -> secondary -> storage``), so the downstream stages
cannot influence anything that is observed. Stepping them anyway costs ~12x the
runtime because their stiffer trajectories dominate the BDF solver. Pruning gives
**bit-identical** primary observables (verified in ``test_fastsim.py``) at ~1/12
the cost — the difference between a 3-minute and a 15-second 60-day simulation,
which is what makes a calibration study with hundreds of runs feasible at all.

Parameterisation
----------------
The study optimises in **log-factor space**: ``theta_j = ln(f_j)`` where ``f_j``
multiplies the nominal (literature) value of kinetic ``j``. This matches how the
dataset was generated (``f ~ lognormal(0, 0.25)``), makes all 26 parameters
dimensionless and identically scaled — which the raw physical bounds emphatically
are not (``K_S_h2 ~ 7e-6`` next to ``k_m_su ~ 30``) — and makes the recovery error
``|ln(f_hat / f_true)|`` a symmetric, interpretable "off by a factor of x".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from pyadm1ode_estimation.estimation.quickstart import build_filter_components
from pyadm1ode_estimation.estimation.specs import InputSpec
from pyadm1ode_estimation.example_plants import build_multi_stage_plant

# --------------------------------------------------------------------------
# Config — mirrors old/benchmark/generate_benchmark.py (the data generator)
# --------------------------------------------------------------------------
DIGESTER_ID = "primary"
#: Components kept when pruning. The storage is the digester's own gas storage.
KEEP_COMPONENTS = frozenset({"primary", "primary_storage"})

SUBSTRATES = [
    InputSpec("maize_silage", substrate_index=0, initial_flow=4.74),
    InputSpec("solid_manure", substrate_index=1, initial_flow=13.70),
    InputSpec("chicken_litter", substrate_index=2, initial_flow=1.09),
    InputSpec("slurry", substrate_index=3, initial_flow=3.68),
    InputSpec("cereal_grain", substrate_index=4, initial_flow=0.20),
]
SENSORS = ["q_gas", "q_ch4", "q_co2", "ph", "ts"]
CHANNEL_NAMES = ["Q_gas", "Q_ch4", "Q_co2", "pH", "TS"]
DT_HOURS = 1.0

#: Acetate-methanogen decay reverted to the ADM1 standard, as in the generator —
#: without it the example plant has no healthy operating point.
K_DEC_AC = 0.02

#: Sensor 1-sigma. Relative for the gas channels, absolute for the installed
#: pH/TS probes — identical to the noise actually added to the dataset.
SENSOR_SIGMA_REL = {"Q_gas": 0.03, "Q_ch4": 0.04, "Q_co2": 0.04}
SENSOR_SIGMA_ABS = {"pH": 0.02, "TS": 0.2}

#: The 26 perturbed kinetics, in the dataset's ``kinetic_keys`` order.
KINETIC_KEYS: tuple[str, ...] = (
    "k_dis_PS",
    "k_dis_PF",
    "k_hyd_ch",
    "k_hyd_pr",
    "k_hyd_li",
    "k_m_su",
    "k_m_aa",
    "k_m_fa",
    "k_m_c4",
    "k_m_pro",
    "k_m_ac",
    "k_m_h2",
    "K_S_su",
    "K_S_aa",
    "K_S_fa",
    "K_S_c4",
    "K_S_pro",
    "K_S_ac",
    "K_S_h2",
    "k_dec_su",
    "k_dec_aa",
    "k_dec_fa",
    "k_dec_c4",
    "k_dec_pro",
    "k_dec_ac",
    "k_dec_h2",
)


def build_plant(prune: bool = True) -> Any:
    """The benchmark plant, optionally pruned to the primary digester.

    Args:
        prune: Drop every component that cannot influence the primary
            digester's sensors (~12x faster, identical observables).

    Returns:
        The :class:`pyadm1.BiogasPlant`.
    """
    plant = build_multi_stage_plant()
    plant.components[DIGESTER_ID].adm1._kinetic["k_dec_ac"] = K_DEC_AC
    if not prune:
        return plant

    for cid in list(plant.components):
        if cid not in KEEP_COMPONENTS:
            del plant.components[cid]
    plant.connections = [
        c for c in plant.connections if c.from_component in KEEP_COMPONENTS and c.to_component in KEEP_COMPONENTS
    ]
    for comp in plant.components.values():
        comp.inputs = [i for i in getattr(comp, "inputs", []) if i in KEEP_COMPONENTS]
        comp.outputs = [o for o in getattr(comp, "outputs", []) if o in KEEP_COMPONENTS]
    return plant


@dataclass
class ForwardModel:
    """A reusable, pre-built forward simulator for one plant instance.

    Building the plant and the filter components costs ~1 s, so a worker builds
    one :class:`ForwardModel` and then reuses it for every objective evaluation,
    only rewriting the kinetic dict in between.
    """

    plant: Any
    process: Any
    obs: Any
    spec: Any
    nominal_kinetics: dict[str, float]

    @classmethod
    def build(cls, prune: bool = True) -> ForwardModel:
        plant = build_plant(prune=prune)
        process, obs, spec = build_filter_components(plant, digester_id=DIGESTER_ID, substrates=SUBSTRATES, sensors=SENSORS)
        kinetic = plant.components[DIGESTER_ID].adm1._kinetic
        missing = [k for k in KINETIC_KEYS if k not in kinetic]
        if missing:
            raise KeyError(f"Kinetics missing from the plant: {missing}")
        nominal = {k: float(kinetic[k]) for k in KINETIC_KEYS}
        return cls(plant, process, obs, spec, nominal)

    # -- kinetics ---------------------------------------------------------
    def set_log_factors(self, theta: Sequence[float]) -> None:
        """Apply ``k_j = nominal_j * exp(theta_j)`` for the 26 study kinetics."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (len(KINETIC_KEYS),):
            raise ValueError(f"theta must have shape ({len(KINETIC_KEYS)},), got {theta.shape}")
        kinetic = self.plant.components[DIGESTER_ID].adm1._kinetic
        for key, t in zip(KINETIC_KEYS, theta):
            kinetic[key] = self.nominal_kinetics[key] * float(np.exp(t))

    def reset_kinetics(self) -> None:
        """Restore the nominal (literature) kinetic values."""
        kinetic = self.plant.components[DIGESTER_ID].adm1._kinetic
        kinetic.update(self.nominal_kinetics)

    # -- simulation -------------------------------------------------------
    def initial_state(self) -> np.ndarray:
        """The plant's default 41-state ADM1 vector, in state-vector layout."""
        return self.spec.read_adm1_state(self.plant)

    def simulate(
        self,
        x0_adm1: np.ndarray,
        feed: np.ndarray,
        n_steps: int,
        dt_hours: float = DT_HOURS,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate ``n_steps`` hourly steps under an imposed feed schedule.

        Mirrors the generator's ``_propagate`` (with ``n_burnin=0``): the feed for
        the upcoming step is written into the state vector's ``input_flow``
        channels, the observables are read *before* stepping, so row ``k`` of both
        returns corresponds to time ``k * dt``.

        Args:
            x0_adm1: The 41 ADM1 states at ``t=0`` (dataset ``states[0]`` layout).
            feed: ``(>= n_steps + 1, 5)`` substrate flows [m3/d].
            n_steps: Number of hourly steps.
            dt_hours: Step size in hours.

        Returns:
            ``(states, observations)`` with shapes ``(n_steps + 1, 41)`` and
            ``(n_steps + 1, 5)``; observation columns follow
            :data:`CHANNEL_NAMES`.
        """
        dt = dt_hours / 24.0
        adm1_pos = self.spec.kind_indices("adm1")
        aug = self.spec.kind_indices("input_flow")

        x = np.zeros(len(self.spec))
        x[adm1_pos] = np.asarray(x0_adm1, dtype=float)

        states = np.zeros((n_steps + 1, len(adm1_pos)))
        obs_rows = np.zeros((n_steps + 1, len(self.obs.channels)))
        for k in range(n_steps + 1):
            for j, i_aug in enumerate(aug):
                x[i_aug] = feed[min(k, len(feed) - 1), j]
            states[k] = x[adm1_pos]
            self.process.refresh_outputs(x, equilibration_dt=dt)
            obs_rows[k] = [float(c.extractor(self.process.plant, x)) for c in self.obs.channels]
            if k < n_steps:
                x = self.process.step(x, dt)
        return states, obs_rows

    def warmup(
        self,
        x0_adm1: np.ndarray,
        warmup_days: float,
        feed_flows: Sequence[float],
        dt_hours: float = DT_HOURS,
    ) -> np.ndarray:
        """Settle the digester at a constant feed and return the 41 ADM1 states.

        Used by the *unknown-x0* variant, where the calibrator does not get the
        dataset's true initial state and must settle the plant itself under the
        kinetics it is currently trying.
        """
        dt = dt_hours / 24.0
        adm1_pos = self.spec.kind_indices("adm1")
        x = np.zeros(len(self.spec))
        x[adm1_pos] = np.asarray(x0_adm1, dtype=float)
        for j, i_aug in enumerate(self.spec.kind_indices("input_flow")):
            x[i_aug] = float(feed_flows[j])
        for _ in range(round(warmup_days * 24.0 / dt_hours)):
            x = self.process.step(x, dt)
        return x[adm1_pos]


def fostac_from_states(states: np.ndarray, params: Any) -> np.ndarray:
    """FOS and TAC [mg/L] for a simulated state trajectory, as ``(T, 2)``.

    Reads out of the states the simulator already returns, so adding the FOS/TAC
    channels to an experiment costs **no extra simulation** — the same run that
    produced the gas observables also produces these.
    """
    import torch
    from pyadm1.core.adm1_torch import tac_torch, vfa_torch

    x = torch.as_tensor(np.asarray(states, dtype=float))
    return np.stack([vfa_torch(x).numpy() * 1000.0, tac_torch(x, params).numpy() * 1000.0], axis=-1)


def adm1_torch_params(plant: Any) -> Any:
    """``Adm1TorchParams`` of the digester, for :func:`fostac_from_states`."""
    from pyadm1.core.adm1_torch import Adm1TorchParams

    return Adm1TorchParams.from_adm1(plant.components[DIGESTER_ID].adm1)


def fostac_sigma(fostac_meas: np.ndarray) -> np.ndarray:
    """1-sigma of a Nordmann titration, propagated from the titrant volumes.

    Mirrors ``datasets/benchmark/fostac.py``: the reported value is converted
    back to a titrant volume, the volume uncertainty is applied there, and the
    result is converted forward again. That reproduces the strong concentration
    dependence of the FOS error (a short second leg means a large relative
    error) instead of assuming one fixed percentage.

    Args:
        fostac_meas: ``(T, 2)`` measured FOS/TAC [mg/L]; ``NaN`` rows allowed.

    Returns:
        ``(T, 2)`` standard deviations in the same units.
    """
    import paths

    paths.add_dataset_to_path()
    from fostac import (
        BURETTE_ML_SIGMA,
        ENDPOINT_ML_SIGMA,
        FOS_OFFSET,
        FOS_PER_ML,
        FOS_SLOPE,
        SAMPLE_REL_SIGMA,
        TAC_PER_ML,
    )

    meas = np.asarray(fostac_meas, dtype=float)
    v_endpoint = ENDPOINT_ML_SIGMA**2 + BURETTE_ML_SIGMA**2

    v1 = meas[:, 1] / TAC_PER_ML
    v2 = (meas[:, 0] / FOS_PER_ML + FOS_OFFSET) / FOS_SLOPE
    sd_v1 = np.sqrt((SAMPLE_REL_SIGMA * v1) ** 2 + v_endpoint)
    sd_v2 = np.sqrt((SAMPLE_REL_SIGMA * v2) ** 2 + v_endpoint)

    sigma = np.empty_like(meas)
    sigma[:, 0] = sd_v2 * FOS_SLOPE * FOS_PER_ML
    sigma[:, 1] = sd_v1 * TAC_PER_ML
    return sigma


def sensor_sigma(obs_clean: np.ndarray) -> np.ndarray:
    """Per-sample 1-sigma for the 5 channels, matching the dataset's noise model.

    Relative channels scale with the reading (floored so a near-zero reading does
    not get infinite weight); pH/TS are absolute.

    Args:
        obs_clean: ``(T, 5)`` observations in :data:`CHANNEL_NAMES` order.

    Returns:
        ``(T, 5)`` standard deviations.
    """
    obs_clean = np.asarray(obs_clean, dtype=float)
    sigma = np.zeros_like(obs_clean)
    for j, name in enumerate(CHANNEL_NAMES):
        if name in SENSOR_SIGMA_ABS:
            sigma[:, j] = SENSOR_SIGMA_ABS[name]
        else:
            rel = SENSOR_SIGMA_REL[name]
            scale = np.maximum(np.abs(obs_clean[:, j]), 1e-6)
            # Floor at 1 % of the channel median so a transient near-zero gas
            # reading cannot dominate the chi-square.
            floor = 0.01 * max(float(np.median(np.abs(obs_clean[:, j]))), 1e-6)
            sigma[:, j] = rel * np.maximum(scale, floor)
    return sigma
