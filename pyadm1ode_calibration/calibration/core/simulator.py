"""Simulator module."""

import types
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from pyadm1ode_calibration.io.loaders.measurement_data import MeasurementData


class PlantSimulator:
    """
    Handles plant simulation with parameter variations.

    This class separates the simulation logic from calibration algorithms,
    providing a consistent interface for running the ADM1 model with
    modified parameter sets.

    Args:
        plant (Any): The PyADM1ODE plant model instance.
        verbose (bool): Whether to enable progress output and logging. Defaults to True.
    """

    def __init__(self, plant: Any, verbose: bool = True, time_varying_feed: bool = False):
        self.plant = plant
        self.verbose = verbose
        self.time_varying_feed = time_varying_feed
        self._original_params: dict[str, dict[str, float]] = {}
        self._original_state: dict[str, dict[str, Any]] = {}

    def simulate_with_parameters(
        self,
        parameters: dict[str, float],
        measurements: "MeasurementData",
        restore_params: bool = True,
        chp_load_setpoints: dict[str, np.ndarray] | None = None,
        restore_state: bool = True,
        warmup: Optional["MeasurementData"] = None,
    ) -> dict[str, np.ndarray]:
        """
        Run a plant simulation using a specific set of parameters.

        Args:
            parameters: Parameter values to apply {name: value}.
            measurements: Input data for substrate feeds and timing.
            restore_params: Whether to restore the original plant parameters
                after the simulation finishes. Defaults to True.
            restore_state: Whether to rewind the plant to the state it had
                before the run. Defaults to True, and calibration depends on
                it: a plant keeps integrating where the last simulation
                stopped, so without the rewind the second evaluation of a
                parameter set starts from a different digester state than the
                first and returns a different error. The optimizer then sees a
                landscape that changes under its feet.
            warmup: Optional window to run *before* the scored one, keeping its
                end state. Use it when ``measurements`` is a later stretch of a
                record: a digester that has been running for weeks is not in the
                state the model starts from, so scoring such a window cold
                measures the start-up transient instead of the parameters. The
                warm-up itself is not returned, and the plant is still rewound
                afterwards when ``restore_state`` is set.
            chp_load_setpoints: Optional mapping from CHP component ID to an
                array of load setpoints (one value per simulation step, in
                [0, 1]). When provided, the simulator drives each CHP from
                this time series instead of letting the plant engine default
                to full load. Useful for replaying measured CHP operation
                so the comparison isolates the gas-production model.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping output names (e.g., 'Q_ch4')
                to simulated numpy arrays.
        """
        if restore_params:
            self._backup_parameters()
        if restore_state:
            self._backup_state()

        try:
            self._apply_parameters(parameters)

            if warmup is not None and len(warmup) > 0:
                self._run_window(warmup, chp_load_setpoints=None)

            results = self._run_window(measurements, chp_load_setpoints)
            return self._extract_outputs_from_results(results)
        finally:
            if restore_params:
                self._restore_parameters()
            if restore_state:
                self._restore_state()

    def _run_window(
        self,
        window: "MeasurementData",
        chp_load_setpoints: dict[str, np.ndarray] | None = None,
    ) -> list[dict[str, Any]]:
        """Feed the plant with one window and step it through, leaving the state."""
        n_steps = len(window)
        dt = 1.0 / 24.0

        feed_series = self._extract_substrate_feed_series(window) if self.time_varying_feed else None
        if feed_series is None:
            self._apply_substrate_feeds(self._extract_substrate_feeds(window))

        if chp_load_setpoints or feed_series is not None:
            return self._step_through(n_steps, dt, chp_load_setpoints, feed_series)
        return self.plant.simulate(duration=n_steps * dt, dt=dt, save_interval=dt)

    def _backup_state(self) -> None:
        """Snapshot the dynamic state of every component that exposes one."""
        self._original_state = {
            cid: comp.get_state() for cid, comp in self.plant.components.items() if hasattr(comp, "get_state")
        }

    def _restore_state(self) -> None:
        """Rewind every component to the snapshot taken by :meth:`_backup_state`."""
        for cid, state in self._original_state.items():
            comp = self.plant.components.get(cid)
            if comp is not None and hasattr(comp, "set_state"):
                comp.set_state(state)

    def _step_through(
        self,
        n_steps: int,
        dt: float,
        setpoints: dict[str, np.ndarray] | None = None,
        feed_series: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Step the plant n_steps times, forcing CHP load setpoints and/or feeds.

        The plant engine hardcodes ``load_setpoint=1.0`` when it calls each
        CHP's ``step``. To inject a time-varying setpoint without modifying
        pyadm1, we monkey-patch the ``step`` method of every targeted CHP
        for the duration of this run so it overwrites ``inputs["load_setpoint"]``
        with the externally supplied value just before delegating to the
        original implementation. Patches are reverted in a ``finally`` block
        so the plant object is left exactly as we found it.

        Setpoints shorter than ``n_steps`` are extended by holding their last
        value; ``NaN`` entries fall back to ``1.0`` (default full load).

        ``feed_series`` is an ``(n_steps, n_substrates)`` array applied one row per
        step, which is what makes a load or substrate change visible to the model.
        Without it the caller has already set a constant feed.
        """
        setpoints = setpoints or {}
        unknown = [cid for cid in setpoints if cid not in self.plant.components]
        if unknown:
            raise KeyError(f"chp_load_setpoints references unknown CHP IDs: {unknown}")

        clean_series: dict[str, np.ndarray] = {}
        for cid, series in setpoints.items():
            arr = np.asarray(series, dtype=float)
            if len(arr) < n_steps:
                pad = np.full(n_steps - len(arr), arr[-1] if len(arr) else 1.0)
                arr = np.concatenate([arr, pad])
            arr = np.clip(np.nan_to_num(arr, nan=1.0), 0.0, 1.0)
            clean_series[cid] = arr

        patched_chps: list[Any] = []
        try:
            for cid in clean_series:
                chp = self.plant.components[cid]
                self._patch_chp_step(chp)
                patched_chps.append(chp)

            results: list[dict[str, Any]] = []
            for i in range(n_steps):
                for cid, arr in clean_series.items():
                    self.plant.components[cid]._forced_load_setpoint = float(arr[i])
                if feed_series is not None:
                    self._apply_substrate_feeds([float(q) for q in feed_series[i]])
                step_result = self.plant.step(dt)
                results.append({"time": self.plant.simulation_time, "components": step_result})
                if self.verbose and (i + 1) % 100 == 0:
                    print(f"Simulated {i + 1}/{n_steps} steps")
            return results
        finally:
            for chp in patched_chps:
                self._unpatch_chp_step(chp)

    @staticmethod
    def _patch_chp_step(chp: Any) -> None:
        """Override chp.step so it honours ``chp._forced_load_setpoint``."""
        if getattr(chp, "_original_step", None) is not None:
            return  # already patched
        original = chp.step
        chp._original_step = original
        chp._forced_load_setpoint = None

        def _patched(self_, t, dt, inputs):
            sp = getattr(self_, "_forced_load_setpoint", None)
            if sp is not None:
                inputs = {**inputs, "load_setpoint": sp}
            return original(t, dt, inputs)

        chp.step = types.MethodType(_patched, chp)

    @staticmethod
    def _unpatch_chp_step(chp: Any) -> None:
        original = getattr(chp, "_original_step", None)
        if original is None:
            return
        chp.step = original
        del chp._original_step
        if hasattr(chp, "_forced_load_setpoint"):
            del chp._forced_load_setpoint

    def _backup_parameters(self) -> None:
        """Snapshot the current per-digester calibration parameters and the
        effective kinetic dict so :meth:`_restore_parameters` can revert
        every channel ``_apply_parameters`` touches.

        pyadm1's ``ADM1._calibration_params`` only takes effect for
        ``k_p`` and ``k_L_a`` (those are read at evaluation time). All
        other kinetic rates (``k_dis``, ``k_hyd_ch``, ``k_hyd_pr``,
        ``k_hyd_li``, …) live in ``ADM1._kinetic`` and are read every
        ODE step from there. Calibrating those parameters therefore
        requires overwriting ``_kinetic`` directly — and backing it up
        so the next sim sees the original temperature-corrected base.
        """
        self._original_params = {}
        self._original_kinetics: dict[str, dict[str, float]] = {}
        for component_id, component in self.plant.components.items():
            if component.component_type.value != "digester":
                continue
            self._original_params[component_id] = getattr(component, "_calibration_params", {}).copy()
            adm1 = getattr(component, "adm1", None)
            if adm1 is not None and hasattr(adm1, "_kinetic"):
                self._original_kinetics[component_id] = dict(adm1._kinetic)

    def _restore_parameters(self) -> None:
        """Revert ``_calibration_params`` and ``_kinetic`` to the snapshots."""
        for component_id, params in self._original_params.items():
            component = self.plant.components[component_id]
            component._calibration_params = params.copy()
        for component_id, kinetics in self._original_kinetics.items():
            component = self.plant.components[component_id]
            adm1 = getattr(component, "adm1", None)
            if adm1 is not None and hasattr(adm1, "_kinetic"):
                adm1._kinetic = dict(kinetics)

    def _apply_parameters(self, parameters: dict[str, float]) -> None:
        """Apply parameter overrides to every digester.

        Each name lands wherever pyadm1 actually consumes it:

        - kinetic rates (``k_dis``, ``k_hyd_*``, ``k_m_*``, ``K_S_*``,
          ``k_dec_*``, …) are written into ``ADM1._kinetic`` — that is
          the dict the ODE right-hand-side reads at every step.
        - ``k_p`` / ``k_L_a`` additionally go through
          :meth:`ADM1.set_calibration_parameters`, which is the channel
          pyadm1 looks at for those two parameters.

        Names that are neither in the kinetic dict nor a known
        calibration-channel name are accepted silently — they end up
        in ``_calibration_params`` and may be read by future hooks.
        """
        for component in self.plant.components.values():
            if component.component_type.value != "digester":
                continue
            if not hasattr(component, "_calibration_params"):
                component._calibration_params = {}
            adm1 = getattr(component, "adm1", None)
            kinetic = getattr(adm1, "_kinetic", None) if adm1 is not None else None
            for name, val in parameters.items():
                component._calibration_params[name] = val
                fval = float(val)
                if kinetic is not None and name in kinetic:
                    kinetic[name] = fval
                if adm1 is not None and hasattr(adm1, "set_calibration_parameters"):
                    adm1.set_calibration_parameters({name: fval})

    def _extract_substrate_feeds(self, measurements: "MeasurementData") -> list[float]:
        """
        Extract mean substrate feed rates from measurement data.

        Args:
            measurements (MeasurementData): Measurement data containing substrate columns.

        Returns:
            List[float]: List of average feed rates for each substrate.
        """
        try:
            Q = measurements.get_substrate_feeds()
            return list(np.mean(Q, axis=0))
        except (ValueError, KeyError):
            # No substrate columns in the measurements: fall back to a default mix.
            return [15.0, 10.0] + [0.0] * 8

    def _extract_substrate_feed_series(self, measurements: "MeasurementData") -> np.ndarray | None:
        """The per-step substrate feeds, or ``None`` when the window has none.

        Returns ``None`` for a constant feed as well, so a record without any load
        change takes the cheap single-simulate path.
        """
        try:
            Q = np.asarray(measurements.get_substrate_feeds(), dtype=float)
        except (ValueError, KeyError):
            return None
        if Q.ndim != 2 or len(Q) < 2 or np.allclose(Q, Q[0], equal_nan=True):
            return None
        return Q

    def _apply_substrate_feeds(self, Q_substrates: list[float]) -> None:
        """
        Apply substrate feed rates respecting the plant's stage cascade.

        In a multi-stage plant (primary -> secondary -> storage), only the
        first digester component receives the fresh substrate; subsequent
        stages receive a zero pass-through and accumulate substrate via
        the liquid connections established with ``cfg.connect(...)``.
        Setting the same fresh feed on every stage would triple-count the
        substrate and produce ~Nx too much biogas.

        For single-digester plants this behaves identically to the
        previous version.

        Args:
            Q_substrates (List[float]): List of substrate feed rates in
                m³/d. Applied to the first digester component; later
                stages receive an all-zero vector of the same length.
        """
        passthrough = [0.0] * len(Q_substrates)
        primary_assigned = False
        for component in self.plant.components.values():
            if component.component_type.value != "digester":
                continue
            feed = Q_substrates if not primary_assigned else passthrough
            primary_assigned = True
            component.Q_substrates = feed
            component.adm1.create_influent(feed, 0)

    def _extract_outputs_from_results(self, results: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        """Extract and aggregate observables from a multi-component sim.

        Aggregations:

        * Gas flows (``Q_gas``, ``Q_ch4``, ``Q_co2``) — summed over
          digesters.
        * Power (``P_el``, ``P_th``, ``Q_gas_consumed``) — summed over CHPs.
        * Heating (``P_aux_heat``, ``P_th_used``) — summed over
          HeatingSystems.
        * Intensive quantities (``pH``, ``VFA``, ``TAC``) — averaged over
          digesters that report them.
        * Gas-storage fill — one fraction (0..1) per digester, exposed as
          ``stored_<digester_id>``. Convert to a plant-specific channel
          (e.g. ``stored_primary`` ↔ measured ``gas_storage_F1`` %) in
          the runner.

        Component types are read from the live ``plant.components``
        dictionary keyed by component_id, so the same extractor works
        for any plant topology.
        """
        type_by_id: dict[str, str] = {cid: comp.component_type.value for cid, comp in self.plant.components.items()}
        digester_ids = [cid for cid, t in type_by_id.items() if t == "digester"]

        scalar_keys = [
            "Q_ch4",
            "Q_gas",
            "Q_co2",
            "P_el",
            "P_th",
            "Q_gas_consumed",
            "P_aux_heat",
            "P_th_used",
            "pH",
            "VFA",
            "TAC",
        ]
        outputs: dict[str, list[float]] = {k: [] for k in scalar_keys}
        for cid in digester_ids:
            outputs[f"stored_{cid}"] = []

        for result in results:
            components = result.get("components", {})
            sums = {
                "Q_ch4": 0.0,
                "Q_gas": 0.0,
                "Q_co2": 0.0,
                "P_el": 0.0,
                "P_th": 0.0,
                "Q_gas_consumed": 0.0,
                "P_aux_heat": 0.0,
                "P_th_used": 0.0,
            }
            ph_list: list[float] = []
            vfa_list: list[float] = []
            tac_list: list[float] = []

            for cid, comp_result in components.items():
                ctype = type_by_id.get(cid, "")
                if ctype == "digester":
                    sums["Q_gas"] += comp_result.get("Q_gas", 0.0)
                    sums["Q_ch4"] += comp_result.get("Q_ch4", 0.0)
                    sums["Q_co2"] += comp_result.get("Q_co2", 0.0)
                    if "pH" in comp_result:
                        ph_list.append(comp_result["pH"])
                    if "VFA" in comp_result:
                        vfa_list.append(comp_result["VFA"])
                    if "TAC" in comp_result:
                        tac_list.append(comp_result["TAC"])
                elif ctype == "chp":
                    sums["P_el"] += comp_result.get("P_el", 0.0)
                    sums["P_th"] += comp_result.get("P_th", 0.0)
                    sums["Q_gas_consumed"] += comp_result.get("Q_gas_consumed", 0.0)
                elif ctype == "heating":
                    sums["P_aux_heat"] += comp_result.get("P_aux_heat", 0.0)
                    sums["P_th_used"] += comp_result.get("P_th_used", 0.0)

            for k, v in sums.items():
                outputs[k].append(v)
            outputs["pH"].append(float(np.mean(ph_list)) if ph_list else 7.0)
            outputs["VFA"].append(float(np.mean(vfa_list)) if vfa_list else 0.0)
            outputs["TAC"].append(float(np.mean(tac_list)) if tac_list else 0.0)

            # Per-digester gas-storage fill (0..1). The digester writes its
            # storage state into the nested ``gas_storage`` dict; if a
            # different topology omits it, the channel reports NaN so
            # downstream code does not silently compare against zero.
            for cid in digester_ids:
                gs = components.get(cid, {}).get("gas_storage", {})
                vol = gs.get("stored_volume_m3", float("nan"))
                cap = float(self.plant.components[cid].V_gas) if hasattr(self.plant.components[cid], "V_gas") else float("nan")
                frac = vol / cap if cap and cap > 0 else float("nan")
                outputs[f"stored_{cid}"].append(frac)

        return {k: np.array(v) for k, v in outputs.items()}
