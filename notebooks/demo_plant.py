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

import numpy as np
import pandas as pd
from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

from pyadm1ode_calibration import MeasurementData
from pyadm1ode_calibration.calibration.core.simulator import PlantSimulator

#: The substrate columns of the measurement frame map onto these, in order.
#: The plant must declare exactly as many substrates as the data has feed
#: columns, otherwise the feed vector cannot be applied.
SUBSTRATES = ["maize_silage_milk_ripeness", "cattle_manure", "grass_silage"]

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


def feed_frame(days: float = 5.0, start: str = "2024-01-01") -> pd.DataFrame:
    """Hourly feed table - the input side of a measurement frame."""
    index = pd.date_range(start, periods=int(days * 24), freq="h")
    return pd.DataFrame(
        {
            "timestamp": index,
            "Q_sub_maize": FEED[0],
            "Q_sub_manure": FEED[1],
            "Q_sub_grass": FEED[2],
        }
    )


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

    rng = np.random.default_rng(seed)
    out = frame.copy()
    for channel in ("Q_gas", "Q_ch4", "P_el", "pH", "VFA"):
        values = np.asarray(simulated[channel], dtype=float)
        out[channel] = values * (1.0 + rng.normal(0.0, noise, values.size))
    return MeasurementData(out)


def simulate(plant: BiogasPlant, measurements: MeasurementData, parameters: dict[str, float]) -> dict:
    """Run one simulation over the measurement window - a one-line shorthand."""
    return PlantSimulator(plant, verbose=False).simulate_with_parameters(parameters, measurements)
