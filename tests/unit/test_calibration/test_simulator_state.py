"""
The simulator must hand every evaluation the same starting point.

A plant keeps integrating where the previous simulation stopped. Without a
rewind, the second evaluation of a parameter set starts from a different
digester state than the first and returns a different error - the optimizer
then searches a landscape that changes under its feet, and a calibration walks
away from the value that generated the data.

These tests pin the rewind down at both levels: the component state round-trip
in pyadm1, and the rewind that ``PlantSimulator`` performs around a run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pyadm1 = pytest.importorskip("pyadm1", reason="needs the PyADM1ODE base package")

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

from pyadm1ode_calibration import MeasurementData
from pyadm1ode_calibration.calibration.core.simulator import PlantSimulator

DAYS = 2


def _set_state_writes_attributes() -> bool:
    """Does the installed pyadm1 mirror a restored state back onto the attributes?

    ``Component.set_state`` assigns the report dict. Whether it also writes the values
    onto the attributes the model integrates from is a property of pyadm1, and older
    releases do not. ``PlantSimulator`` no longer depends on it either way, so only the
    test that pins the upstream contract needs to know.
    """
    from pyadm1.components.energy import CHP

    probe = CHP(component_id="probe", P_el_nom=1.0)
    probe._state_probe = 1
    probe.set_state({"_state_probe": 2})
    return getattr(probe, "_state_probe", None) == 2


SET_STATE_WRITES_ATTRIBUTES = _set_state_writes_attributes()

#: Substrate files shipped with this repository. Passing paths rather than bare IDs
#: keeps the test working whether pyadm1 is an editable install next to a checkout
#: that carries the substrate data or a wheel whose data directory is empty.
SUBSTRATE_DIR = Path(__file__).resolve().parents[3] / "data" / "substrates"
SUBSTRATES = [str(SUBSTRATE_DIR / f"{name}.yaml") for name in ("maize_silage_milk_ripeness", "cattle_manure", "grass_silage")]


def _plant() -> BiogasPlant:
    feedstock = Feedstock(
        SUBSTRATES,
        feeding_freq=24,
        total_simtime=DAYS,
    )
    plant = BiogasPlant("test")
    cfg = PlantConfigurator(plant, feedstock)
    cfg.add_digester("F1", V_liq=2000, V_gas=400, T_ad=313.15, Q_substrates=[15.0, 10.0, 0.0])
    cfg.add_chp("chp", P_el_nom=500.0, eta_el=0.40, eta_th=0.45)
    cfg.auto_connect_digester_to_chp("F1", "chp")
    plant.initialize()
    return plant


def _measurements() -> MeasurementData:
    return MeasurementData(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=DAYS * 24, freq="h"),
                "Q_sub_maize": 15.0,
                "Q_sub_manure": 10.0,
                "Q_sub_grass": 0.0,
            }
        )
    )


class TestComponentStateRoundTrip:
    def test_snapshot_is_independent_of_the_live_state(self) -> None:
        """``get_state`` must not hand out references the simulation mutates."""
        plant = _plant()
        digester = plant.components["F1"]
        snapshot = digester.get_state()
        before = list(snapshot["adm1_state"])

        PlantSimulator(plant, verbose=False).simulate_with_parameters({"k_hyd_ch": 2.0}, _measurements(), restore_state=False)

        assert list(snapshot["adm1_state"]) == before, "snapshot followed the live state"

    @pytest.mark.skipif(
        not SET_STATE_WRITES_ATTRIBUTES,
        reason="installed pyadm1 does not mirror set_state onto attributes; "
        "PlantSimulator compensates, see test_repeated_evaluation_is_reproducible",
    )
    def test_set_state_reaches_the_attribute_the_model_reads(self) -> None:
        """Restoring the dict alone would leave ``adm1_state`` untouched.

        This one pins pyadm1's own contract rather than ours. The rewind that
        calibration depends on is covered by :class:`TestSimulatorRewind`, which passes
        either way because ``PlantSimulator`` writes the attributes back itself.
        """
        plant = _plant()
        digester = plant.components["F1"]
        snapshot = digester.get_state()
        before = np.asarray(digester.adm1_state, dtype=float)

        PlantSimulator(plant, verbose=False).simulate_with_parameters({"k_hyd_ch": 2.0}, _measurements(), restore_state=False)
        assert not np.allclose(np.asarray(digester.adm1_state, dtype=float), before)

        digester.set_state(snapshot)
        assert np.allclose(np.asarray(digester.adm1_state, dtype=float), before)


class TestSimulatorRewind:
    def test_repeated_evaluation_is_reproducible(self) -> None:
        """Same parameters, same plant, same answer - what an optimizer needs."""
        simulator = PlantSimulator(_plant(), verbose=False)
        measurements = _measurements()

        runs = [
            np.asarray(
                simulator.simulate_with_parameters({"k_hyd_ch": 2.0}, measurements)["Q_gas"],
                dtype=float,
            )
            for _ in range(3)
        ]

        assert np.allclose(runs[0], runs[1])
        assert np.allclose(runs[0], runs[2])

    def test_without_the_rewind_the_runs_drift(self) -> None:
        """The opposite case, so the test above cannot pass for the wrong reason."""
        simulator = PlantSimulator(_plant(), verbose=False)
        measurements = _measurements()

        first = np.asarray(
            simulator.simulate_with_parameters({"k_hyd_ch": 2.0}, measurements, restore_state=False)["Q_gas"],
            dtype=float,
        )
        second = np.asarray(
            simulator.simulate_with_parameters({"k_hyd_ch": 2.0}, measurements, restore_state=False)["Q_gas"],
            dtype=float,
        )

        assert not np.allclose(first, second)


class TestWarmUp:
    """A later window must be judged from the state the plant is actually in."""

    def test_warmup_changes_the_starting_state(self) -> None:
        simulator = PlantSimulator(_plant(), verbose=False)
        window = _measurements()

        cold = np.asarray(
            simulator.simulate_with_parameters({"k_hyd_ch": 2.0}, window)["Q_gas"],
            dtype=float,
        )
        warmed = np.asarray(
            simulator.simulate_with_parameters({"k_hyd_ch": 2.0}, window, warmup=window)["Q_gas"],
            dtype=float,
        )

        # the warm-up leaves the digester further along, so the first hours differ most
        assert not np.allclose(cold, warmed)
        assert abs(warmed[0] - cold[0]) > abs(warmed[-1] - cold[-1])

    def test_warmup_still_rewinds_the_plant(self) -> None:
        """``restore_state`` must cover the warm-up too, or runs stop being comparable."""
        simulator = PlantSimulator(_plant(), verbose=False)
        window = _measurements()

        first = np.asarray(
            simulator.simulate_with_parameters({"k_hyd_ch": 2.0}, window, warmup=window)["Q_gas"],
            dtype=float,
        )
        second = np.asarray(
            simulator.simulate_with_parameters({"k_hyd_ch": 2.0}, window, warmup=window)["Q_gas"],
            dtype=float,
        )

        assert np.allclose(first, second)
