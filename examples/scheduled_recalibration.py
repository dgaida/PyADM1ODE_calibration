#!/usr/bin/env python3
"""Unattended recalibration job: one window in, one stored result out.

What it does, once per invocation:

1. read the newest measurement window (CSV here, a plant historian in reality),
2. look up the parameters currently in use, from the database if a previous run stored
   any, otherwise from a default,
3. recalibrate with the guardrails of :class:`OnlineCalibrator`,
4. refuse the result if the data or the fit look untrustworthy,
5. store the accepted result so the next run starts from it.

Usage
-----
::
    # first run, creates the SQLite file and stores the accepted result
    python examples/scheduled_recalibration.py --db sqlite:///calibration.db

    # dry run: calibrate and report, store nothing
    python examples/scheduled_recalibration.py --dry-run

    # against a real database, credentials from DB_HOST / DB_USER / DB_PASSWORD / ...
    python examples/scheduled_recalibration.py --db env

A scheduler entry then only has to check the exit code::

    0  accepted and stored
    1  rejected by a guardrail, previous parameters kept
    2  input unusable (no data, too short, too many gaps)
    3  unexpected error, see the log
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from pathlib import Path

import numpy as np
import pandas as pd

from pyadm1ode_calibration import Database, MeasurementData, OnlineCalibrator

# The demo plant of the notebook series, so this script and the notebooks describe the
# same plant. Replace both with your own builder.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
from demo_plant import build_demo_plant, make_twin_measurements

LOG = logging.getLogger("recalibration")

#: Identifier under which results are stored. One row per plant per run.
PLANT_ID = "demo-f1"

#: The parameter that is tracked, and where to start when no history exists.
PARAMETER = "k_hyd_ch"
FALLBACK_VALUE = 2.0

#: Guardrails. ``max_parameter_change`` is the important one: it caps how far a single
#: run may move, so one bad window cannot swing the model.
MAX_PARAMETER_CHANGE = 0.20
VARIANCE_THRESHOLD = 0.15

#: Acceptance thresholds applied *after* the fit. A calibration that converges to a
#: worse objective than the current parameters is a regression, not an improvement.
MIN_ROWS = 48
MAX_MISSING_FRACTION = 0.20
REQUIRE_IMPROVEMENT = True


class Exit(IntEnum):
    """Exit codes. The scheduler reads these, not the log."""

    ACCEPTED = 0
    REJECTED = 1
    BAD_INPUT = 2
    ERROR = 3


@dataclass
class Outcome:
    """What one run decided, and why."""

    code: Exit
    reason: str
    previous: float | None = None
    fitted: float | None = None
    objective_before: float | None = None
    objective_after: float | None = None


# --------------------------------------------------------------------------- input ---
def load_window(csv: Path | None, days: float) -> MeasurementData:
    """The newest ``days`` of measurements.

    In production this reads the historian. Here it falls back to the twin generator so
    the script is runnable without any data, which is what makes it testable as a
    *schedule* before it is testable as a *calibration*.
    """
    if csv is None:
        LOG.info("no --csv given, generating a twin window of %.0f d", days)
        return make_twin_measurements(days=days, noise=0.02, seed=int(datetime.now(timezone.utc).timestamp()) % 1000)

    frame = pd.read_csv(csv, parse_dates=["timestamp"])
    cutoff = frame["timestamp"].max() - timedelta(days=days)
    window = frame[frame["timestamp"] >= cutoff].reset_index(drop=True)
    LOG.info("read %d rows from %s (last %.0f d)", len(window), csv, days)
    return MeasurementData(window)


def check_input(measurements: MeasurementData, channel: str) -> str | None:
    """Reasons not to calibrate at all. Returns ``None`` when the window is usable.

    Checked before the optimiser runs, because a fit on unusable data still converges to
    *something* and that something then looks like a result.
    """
    frame = measurements.data
    if len(frame) < MIN_ROWS:
        return f"only {len(frame)} rows, need {MIN_ROWS}"
    if channel not in frame.columns:
        return f"objective channel {channel!r} missing from the window"
    missing = float(frame[channel].isna().mean())
    if missing > MAX_MISSING_FRACTION:
        return f"{missing:.0%} of {channel} missing, limit {MAX_MISSING_FRACTION:.0%}"
    if float(np.nanstd(frame[channel])) == 0.0:
        return f"{channel} is constant, nothing to fit"
    return None


# ------------------------------------------------------------------------ database ---
def open_database(spec: str | None) -> Database | None:
    """``None``, a connection string, or ``"env"`` for DB_* environment variables."""
    if spec is None:
        return None
    db = Database.from_env() if spec == "env" else Database(connection_string=spec)
    db.create_all_tables()
    return db


def current_parameters(db: Database | None) -> tuple[float, str]:
    """The parameters in use: the last accepted run, else the fallback."""
    if db is None:
        return FALLBACK_VALUE, "no database, using the fallback"
    latest = db.get_latest_calibration(PLANT_ID)
    if not latest or PARAMETER not in (latest.get("parameters") or {}):
        return FALLBACK_VALUE, "no stored calibration yet, using the fallback"
    return float(latest["parameters"][PARAMETER]), f"stored {latest.get('created_at')}"


# ----------------------------------------------------------------------- the run ---
def objective_of(plant, measurements: MeasurementData, value: float, channel: str) -> float:
    """Normalised RMSE of one parameter value, so before and after are comparable."""
    from pyadm1ode_calibration.calibration.core.simulator import PlantSimulator

    predicted = PlantSimulator(plant, verbose=False).simulate_with_parameters({PARAMETER: value}, measurements)
    observed = np.asarray(measurements.data[channel], dtype=float)
    modelled = np.asarray(predicted[channel], dtype=float)
    return float(np.sqrt(np.mean((modelled - observed) ** 2)) / np.mean(np.abs(observed)))


def run_once(args: argparse.Namespace) -> Outcome:
    """One scheduled invocation."""
    measurements = load_window(args.csv, args.days)

    bad = check_input(measurements, args.channel)
    if bad:
        return Outcome(Exit.BAD_INPUT, bad)

    db = open_database(args.db)
    try:
        previous, origin = current_parameters(db)
        LOG.info("current %s = %.4f (%s)", PARAMETER, previous, origin)

        plant = build_demo_plant(days=args.days)
        before = objective_of(plant, measurements, previous, args.channel)

        result = OnlineCalibrator(build_demo_plant(days=args.days), verbose=False).calibrate(
            measurements,
            parameters=[PARAMETER],
            current_parameters={PARAMETER: previous},
            objectives=[args.channel],
            max_parameter_change=MAX_PARAMETER_CHANGE,
            variance_threshold=VARIANCE_THRESHOLD,
            max_iterations=args.max_iterations,
            use_constraints=False,
        )
        fitted = float(result.parameters[PARAMETER])
        after = objective_of(plant, measurements, fitted, args.channel)
        LOG.info("fit: %.4f -> %.4f  (objective %.4f -> %.4f)  %s", previous, fitted, before, after, result.message)

        common = {"previous": previous, "fitted": fitted, "objective_before": before, "objective_after": after}
        if not result.success:
            return Outcome(Exit.REJECTED, f"optimiser did not converge: {result.message}", **common)
        if REQUIRE_IMPROVEMENT and after >= before:
            return Outcome(Exit.REJECTED, f"no improvement ({before:.4f} -> {after:.4f}), keeping {previous:.4f}", **common)

        if args.dry_run:
            return Outcome(Exit.ACCEPTED, "accepted, not stored (--dry-run)", **common)
        if db is None:
            return Outcome(Exit.ACCEPTED, "accepted, not stored (no --db)", **common)

        stamps = pd.to_datetime(measurements.data.index)
        db.store_calibration(
            plant_id=PLANT_ID,
            calibration_type="online",
            method="scheduled",
            parameters={PARAMETER: fitted},
            objective_value=after,
            objectives=[args.channel],
            validation_metrics={"objective_before": before, "objective_after": after},
            data_start=stamps.min().to_pydatetime(),
            data_end=stamps.max().to_pydatetime(),
            success=True,
            message=result.message,
        )
        return Outcome(Exit.ACCEPTED, "accepted and stored", **common)
    finally:
        if db is not None:
            db.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, default=None, help="measurement file; omitted generates a twin window")
    ap.add_argument("--days", type=float, default=3.0, help="length of the window [d]")
    ap.add_argument("--channel", default="Q_gas", help="objective channel")
    ap.add_argument("--db", default=None, help="SQLAlchemy URL, or 'env' for the DB_* variables; omitted = stateless")
    ap.add_argument("--max-iterations", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true", help="calibrate but store nothing")
    ap.add_argument("--log", default="INFO", help="DEBUG, INFO, WARNING")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        outcome = run_once(args)
    except Exception:
        # A scheduled job must never die silently: log the traceback, return a code the
        # scheduler can alert on.
        LOG.exception("recalibration failed")
        return int(Exit.ERROR)

    level = logging.INFO if outcome.code is Exit.ACCEPTED else logging.WARNING
    LOG.log(level, "%s: %s", outcome.code.name, outcome.reason)
    if outcome.fitted is not None:
        LOG.log(
            level,
            "  %s %.4f -> %.4f | objective %.4f -> %.4f",
            PARAMETER,
            outcome.previous,
            outcome.fitted,
            outcome.objective_before,
            outcome.objective_after,
        )
    return int(outcome.code)


if __name__ == "__main__":
    raise SystemExit(main())
