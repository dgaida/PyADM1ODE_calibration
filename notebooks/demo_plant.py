"""
Shared setup for the example notebooks: a small plant and a twin dataset.

Why a *twin* dataset instead of the shipped ``data/plant_measurements.csv``?
Because a tutorial should have a known answer. The measurements here are
produced by simulating this very plant with parameter values we choose, plus
measurement noise. A calibration can therefore be judged against the truth:
either it recovers the value we put in, or it does not.

The shipped CSV stays the subject of notebook 01, where the point is handling
real-world data problems (outliers, gaps, resampling) rather than fitting.

The plant is deliberately tiny - one digester and one CHP - so a five-day
simulation takes about half a second and every notebook runs in well under a
minute.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

from pyadm1ode_calibration import MeasurementData
from pyadm1ode_calibration.calibration.core.simulator import PlantSimulator

#: The substrate columns of the measurement frame map onto these, in order.
#: The plant must declare exactly as many substrates as the data has feed
#: columns, otherwise the feed vector cannot be applied.
#:
#: Given as paths into this repository rather than as bare IDs. pyadm1 resolves a bare
#: ID against its own package data directory, which is populated for an editable
#: install but empty for a wheel - as on Colab, where these notebooks are meant to run.
_SUBSTRATE_DIR = Path(__file__).resolve().parent.parent / "data" / "substrates"
SUBSTRATES = [str(_SUBSTRATE_DIR / f"{name}.yaml") for name in ("maize_silage_milk_ripeness", "cattle_manure", "grass_silage")]

#: Daily feed of the demo plant [m3/d], one entry per substrate.
FEED = [15.0, 10.0, 0.0]

#: Parameter value the twin data is generated with. The model's own default for
#: ``k_hyd_ch`` is 4.0, so an uncalibrated simulation visibly overshoots - which
#: is what notebook 02 shows.
TRUE_PARAMETERS = {"k_hyd_ch": 2.0}

#: The value an uncalibrated model starts from (PyADM1ODE's ADM1 default).
DEFAULT_K_HYD_CH = 4.0


def build_demo_plant(days: float = 5.0) -> BiogasPlant:
    """One digester (2000 m3) feeding one 500 kW CHP.

    Args:
        days: Simulation horizon the feedstock is prepared for.
    """
    feedstock = Feedstock(SUBSTRATES, feeding_freq=24, total_simtime=days)
    plant = BiogasPlant("demo")
    cfg = PlantConfigurator(plant, feedstock)
    cfg.add_digester("F1", V_liq=2000, V_gas=400, T_ad=313.15, Q_substrates=FEED, name="Digester F1")
    cfg.add_chp("chp", P_el_nom=500.0, eta_el=0.40, eta_th=0.45, name="CHP 500 kW")
    cfg.auto_connect_digester_to_chp("F1", "chp")
    plant.initialize()
    return plant


#: Feed columns of the measurement frame, in the order of :data:`SUBSTRATES`.
FEED_COLUMNS = ("Q_sub_maize", "Q_sub_manure", "Q_sub_grass")

#: Channels :func:`add_measurement_noise` turns into "measurements".
MEASURED_CHANNELS = ("Q_gas", "Q_ch4", "P_el", "pH", "VFA")


def feed_frame(
    days: float = 5.0,
    start: str | pd.Timestamp = "2024-01-01",
    changes: Sequence[tuple[float, Sequence[float]]] = (),
) -> pd.DataFrame:
    """Hourly feed table - the input side of a measurement frame.

    Args:
        days: Length of the table.
        start: First timestamp.
        changes: Feed changes as ``(day, [maize, manure, grass])`` pairs. From that
            day on, counted from ``start``, the plant gets the given mix [m3/d].
            Without changes the feed is :data:`FEED` throughout.
    """
    index = pd.date_range(start, periods=int(days * 24), freq="h")
    frame = pd.DataFrame({"timestamp": index})
    day = np.arange(len(index)) / 24.0
    for i, column in enumerate(FEED_COLUMNS):
        frame[column] = FEED[i]
    for change_day, mix in changes:
        for column, value in zip(FEED_COLUMNS, mix):
            frame.loc[day >= change_day, column] = float(value)
    return frame


def add_measurement_noise(frame: pd.DataFrame, simulated: dict, noise: float, rng: np.random.Generator) -> MeasurementData:
    """Turn a simulation into noisy "measurements" on top of its feed table.

    Args:
        frame: Feed table the simulation was run with.
        simulated: Simulator output for that table.
        noise: Relative Gaussian noise added to every measured channel.
        rng: Random generator, so a sequence of windows draws fresh noise.
    """
    out = frame.copy()
    for channel in MEASURED_CHANNELS:
        values = np.asarray(simulated[channel], dtype=float)
        out[channel] = values * (1.0 + rng.normal(0.0, noise, values.size))
    return MeasurementData(out)


def make_twin_measurements(
    days: float = 5.0,
    noise: float = 0.02,
    seed: int = 0,
    parameters: dict[str, float] | None = None,
    start: str = "2024-01-01",
) -> MeasurementData:
    """Simulate the demo plant and return the result as noisy "measurements".

    Args:
        days: Length of the record.
        noise: Relative Gaussian noise added to every measured channel.
        seed: Seed for reproducibility.
        parameters: Parameters to generate with; defaults to
            :data:`TRUE_PARAMETERS`.
        start: First timestamp.
    """
    frame = feed_frame(days, start)
    plant = build_demo_plant(days)
    simulated = PlantSimulator(plant, verbose=False).simulate_with_parameters(
        parameters or TRUE_PARAMETERS, MeasurementData(frame)
    )

    return add_measurement_noise(frame, simulated, noise, np.random.default_rng(seed))


def simulate(
    plant: BiogasPlant,
    measurements: MeasurementData,
    parameters: dict[str, float],
    time_varying_feed: bool = False,
) -> dict:
    """Run one simulation over the measurement window - a one-line shorthand.

    The plant is rewound afterwards, so it can be called again from the same state.

    Args:
        plant: Plant to simulate.
        measurements: Window that supplies the feed and the time axis.
        parameters: Parameter values for this run.
        time_varying_feed: Apply the feed hour by hour instead of its mean.
    """
    simulator = PlantSimulator(plant, verbose=False, time_varying_feed=time_varying_feed)
    return simulator.simulate_with_parameters(parameters, measurements)


def advance(plant: BiogasPlant, window: MeasurementData | pd.DataFrame, parameters: dict[str, float]) -> dict:
    """Run the plant through a window and keep the state it ends in.

    This is the opposite of :func:`simulate`: the plant is not rewound, so the next
    window continues where this one stopped, as a real plant does.

    Args:
        plant: Plant to advance.
        window: Window or feed table that supplies the hourly feed.
        parameters: Parameter values the plant runs with in this window.

    Returns:
        The simulator output of the window.
    """
    if isinstance(window, pd.DataFrame):
        window = MeasurementData(window)
    simulator = PlantSimulator(plant, verbose=False, time_varying_feed=True)
    # pyadm1 prints a progress line every 100 steps of a constant-feed run.
    with contextlib.redirect_stdout(io.StringIO()):
        return simulator.simulate_with_parameters(parameters, window, restore_state=False)
