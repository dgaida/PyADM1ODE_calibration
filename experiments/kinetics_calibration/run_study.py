"""Runner for the kinetics-calibration study.

Three stages, each writing JSONL into ``results/`` so a sweep can be interrupted
and resumed:

* ``screen``    — Jacobian + Brun identifiability per series (no optimisation).
* ``calibrate`` — the method x start x window x parameter-set sweep.
* ``all``       — both.

Everything runs in a process pool: one forward model per worker, reused across
tasks. Set ``OMP_NUM_THREADS=1`` — the ADM1 right-hand side is small, so BLAS
threads only fight each other and serialise the pool.

The benchmark series live in ``PyADM1ODE_estimate`` and are located
automatically; see :mod:`paths` and the ``--dataset`` / ``--results`` flags.

Examples::

    python run_study.py screen --series 0 1 2 3
    python run_study.py calibrate --preset smoke
    python run_study.py all --preset main --workers 14
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import paths

# --------------------------------------------------------------------------
# Presets — the compute knob. Costs assume ~0.23 s per simulated day per core.
# --------------------------------------------------------------------------
#: The test set is stored grouped by operating mode (0-4 low_high, 5-9 stable_low,
#: 10-14 stable_high, 15-19 oscillating), so a contiguous slice would silently
#: study one or two modes. These picks take two series per mode — identifiability
#: depends on how hard the feed excites the plant, so the mode is a first-class
#: factor of the study, not a nuisance to average over.
BALANCED_8 = [0, 1, 5, 6, 10, 11, 15, 16]
BALANCED_4 = [0, 5, 10, 15]

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "series": [0, 15],
        "methods": ["nelder_mead", "differential_evolution"],
        "windows": [10],
        "starts": ["near", "far"],
        "param_sets": ["top5"],
        "budget": 120,
        "known_x0": [True],
    },
    # Q4 — all five optimisers head to head at two window lengths (~2 h on 14
    # workers). Deliberately not run at 60 days for every method: a 60-day task
    # costs 6x a 10-day one, and the length question is answered separately by
    # the two best methods rather than by all five.
    "shootout": {
        "series": BALANCED_4,
        "methods": [
            "nelder_mead",
            "powell",
            "lbfgsb",
            "slsqp",
            "differential_evolution",
        ],
        "windows": [10, 30],
        "starts": ["near", "far"],
        "param_sets": ["top5"],
        "budget": 300,
        "known_x0": [True],
    },
    # Q3 — the days-of-data answer on the recovery side (~2.5 h). Methods list is
    # overridden from the shootout result.
    "length": {
        "series": BALANCED_4,
        "methods": ["nelder_mead", "differential_evolution"],
        "windows": [10, 20, 30, 60],
        "starts": ["near", "far"],
        "param_sets": ["top5"],
        "budget": 300,
        "known_x0": [True],
    },
    # Closes a gap left by the original split: `shootout` ran all five optimisers
    # but only at 10 and 30 days, while `length` covered all four window lengths
    # with only Nelder-Mead and DE. That split was decided *before* Powell turned
    # out to be the best method, so the window-length table had to fall back on
    # the runner-up methods. Same config as `length` in every other respect, so
    # the results drop straight into the same comparison.
    "powell_length": {
        "series": BALANCED_4,
        "methods": ["powell"],
        "windows": [20, 60],
        "starts": ["near", "far"],
        "param_sets": ["top5"],
        "budget": 300,
        "known_x0": [True],
        # Same flat box the runs it extends were searched with.
        "legacy_bounds": True,
    },
    # F5 under genuinely equal conditions. `shootout` ran L-BFGS-B and SLSQP at
    # 10 and 30 days only, while Nelder-Mead, Powell and DE were later extended
    # to 20 and 60 days — so a table pooled over windows averaged the gradient
    # methods over a *different* window mix than the rest, which flatters or
    # penalises them depending on how the error moves with window length.
    #
    # This preset states the intended full design (5 methods x 4 windows) rather
    # than only the missing cells; the resume logic in `run_calibration` skips
    # what is already stored, so pointing it at the existing JSONL computes
    # exactly the gaps. `legacy_bounds` because all 112 stored runs used the flat
    # box — mixing two search spaces in one table would confound method with
    # search space, which is the mistake this preset exists to avoid.
    "shootout_length": {
        "series": BALANCED_4,
        "methods": [
            "nelder_mead",
            "powell",
            "lbfgsb",
            "slsqp",
            "differential_evolution",
        ],
        "windows": [10, 20, 30, 60],
        "starts": ["near", "far"],
        "param_sets": ["top5"],
        "budget": 300,
        "known_x0": [True],
        "legacy_bounds": True,
    },
    # `shootout_length` doubled to eight series (two per operating mode instead
    # of one). Reason: with four series a table cell rests on four runs, which
    # makes the interquartile ranges shaky and leaves the window trends
    # underpowered once near/far are reported separately — the k_dis_PF trend
    # lands at p = 0.18 in the near half purely for lack of runs.
    #
    # Everything else is identical to `shootout_length` (same windows, budget,
    # legacy box, both starts for every method including DE), so the new runs
    # merge into the same tables. Resume skips the 160 runs already stored, so
    # only the four added series are computed: 4 x 40 = 160 runs, ~140 CPU-hours.
    #
    # The `top5` set is deliberately NOT rescreened. It is pooled over series, so
    # rescreening could reorder it and make the new runs incomparable with the
    # stored ones.
    "shootout_series8": {
        "series": BALANCED_8,
        "methods": [
            "nelder_mead",
            "powell",
            "lbfgsb",
            "slsqp",
            "differential_evolution",
        ],
        "windows": [10, 20, 30, 60],
        "starts": ["near", "far"],
        "param_sets": ["top5"],
        "budget": 300,
        "known_x0": [True],
        "legacy_bounds": True,
        "force_all_starts": True,
    },
    # The parameter-count question rerun on the study's strongest footing:
    # eight series instead of four, and the full 60-day window instead of 30.
    # Powell only — it beats every other method with p < 0.03 on the eight-series
    # comparison, so a second method would cost 8x the compute to re-answer a
    # settled question.
    #
    # `legacy_bounds` for two reasons. It matches the method comparison, so the
    # two are on the same search space for the first time; and the flat box is
    # exactly the interval the `far` start is clipped to, so no start point can
    # land outside it. With the literature bounds a `far` start *can* fall
    # outside (k_dec is the tightest family), which SciPy's Powell answers by
    # returning x0 unchanged after zero evaluations — that silently corrupted
    # three cells of the original 30-day sweep.
    #
    # Writes to its own file: mixing 30-day/literature-bounds runs with
    # 60-day/flat-box runs in one JSONL would let `analyze_count.py` pool two
    # different experiments.
    "count60": {
        "series": BALANCED_8,
        "methods": ["powell"],
        "windows": [60],
        "starts": ["near", "far"],
        "param_sets": ["top3", "top5", "top10", "top18", "nokS", "all26"],
        "budget": 0,  # replaced per task by BUDGET_PER_PARAM * n_active
        "known_x0": [True],
        "legacy_bounds": True,
    },
    # Tests the assumption that the start point is irrelevant for Differential
    # Evolution. It is irrelevant for DE's *search* — scipy ignores x0 and samples
    # the whole box — but NOT for the problem: `set_base` pins the 21 inactive
    # kinetics to the start too, so a `far` start leaves DE with frozen parameters
    # further from the truth, which the five free ones must absorb. Identical to
    # `shootout_length` in every other respect (same series, windows, budget,
    # legacy box), so the new runs drop straight into the same table.
    "de_far": {
        "series": BALANCED_4,
        "methods": ["differential_evolution"],
        "windows": [10, 20, 30, 60],
        "starts": ["near", "far"],
        "param_sets": ["top5"],
        "budget": 300,
        "known_x0": [True],
        "legacy_bounds": True,
        "force_all_starts": True,
    },
    # The cost of asking for all 26 at once, and the realistic unknown-x0 case.
    "wide": {
        "series": BALANCED_4,
        "methods": ["nelder_mead", "differential_evolution"],
        "windows": [30],
        "starts": ["near", "far"],
        "param_sets": ["all26", "top5"],
        "budget": 600,
        "known_x0": [True, False],
    },
    # How many parameters SHOULD one calibrate? Too few and the ones held fixed
    # bias the free ones (measured: chi2 at the true top-5 is 1.05 near / 1.59
    # far, against a 1.00 noise floor); too many and the search spends its budget
    # on directions the data cannot constrain. The budget is scaled with the
    # dimension (see BUDGET_PER_PARAM) because a fixed budget would simply
    # starve the larger sets and confound "too many parameters" with "too few
    # evaluations". Powell only — it won the optimiser comparison outright.
    "count": {
        "series": BALANCED_4,
        "methods": ["powell"],
        "windows": [30],
        "starts": ["near", "far"],
        "param_sets": ["top3", "top5", "top10", "top18", "nokS", "all26"],
        "budget": 0,  # replaced per task by BUDGET_PER_PARAM * n_active
        "known_x0": [True],
    },
}

#: Forward simulations granted per free parameter when ``budget`` is 0.
BUDGET_PER_PARAM = 60

#: Screening only needs a series list, and is shared by every preset.
PRESETS["main"] = PRESETS["shootout"] | {"series": BALANCED_8}


@dataclass(frozen=True)
class Task:
    """One calibration run."""

    series_idx: int
    method: str
    start: str
    n_days: int
    param_set: str
    known_x0: bool
    use_fostac: bool
    fostac_every_days: float
    #: Use the original flat box (factor 1/4 to 4 for every parameter) instead of
    #: the literature bounds. Needed to extend an existing comparison: mixing
    #: two bound settings in one table would confound method with search space.
    legacy_bounds: bool
    budget: int


# --------------------------------------------------------------------------
# Worker-side state (one forward model per process)
# --------------------------------------------------------------------------
_MODEL = None
_TEST = None


def _worker_init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _get_model():
    global _MODEL
    if _MODEL is None:
        from fastsim import ForwardModel

        _MODEL = ForwardModel.build(prune=True)
    return _MODEL


def _get_series(idx: int) -> dict[str, Any]:
    global _TEST
    if _TEST is None:
        paths.add_dataset_to_path()
        from loader import load_test

        _TEST = load_test()
    return _TEST[idx]


# --------------------------------------------------------------------------
# Parameter sets
# --------------------------------------------------------------------------
def resolve_param_set(name: str, screening_path: Path | None = None) -> np.ndarray:
    """Indices of the kinetics a run is allowed to move.

    ``all26`` is every perturbed kinetic. ``top<k>`` takes the k most important
    parameters from the screening stage (pooled across series), which is the set a
    practitioner would actually calibrate — screening first is part of the method,
    not a shortcut around it.
    """
    from fastsim import KINETIC_KEYS

    if name == "all26":
        return np.arange(len(KINETIC_KEYS))
    if name == "nokS":
        # The 19 kinetics minus every K_S. The normalised Fisher matrix gives
        # corr(k_m_X, K_S_X) = -1.00 for five pairs (first-order regime) and the
        # remaining two are saturated, so K_S adds only directions the data
        # cannot constrain. See bounds.NOT_IDENTIFIABLE_ALONE.
        from bounds import parameters_without_ks

        return parameters_without_ks()
    if name.startswith("top"):
        k = int(name[3:])
        path = screening_path or (paths.results_dir() / "screening.json")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — run the 'screen' stage before using '{name}'.")
        report = json.loads(path.read_text(encoding="utf-8"))
        ranking = report["pooled"]["ranking"]
        return np.array([KINETIC_KEYS.index(n) for n in ranking[:k]], dtype=int)
    raise ValueError(f"Unknown parameter set: {name!r}")


# --------------------------------------------------------------------------
# Stage: screening
# --------------------------------------------------------------------------
def _screen_one(series_idx: int, n_days: float, around: str, use_fostac: bool = False) -> dict[str, Any]:
    from screening import compute_jacobian, screen

    series = _get_series(series_idx)
    jac = compute_jacobian(series, _get_model(), around=around, n_days=n_days, use_fostac=use_fostac)
    rep = screen(jac)
    rep["series_idx"] = series_idx
    rep["regime"] = str(series["regime"])
    # Keep the raw importance per window; the Jacobian itself is too big to store.
    return rep


def run_screening(
    series_ids: list[int],
    workers: int,
    n_days: float = 60.0,
    around: str = "truth",
    use_fostac: bool = False,
    out_name: str = "screening.json",
) -> dict[str, Any]:
    """Screen every series and pool the ranking."""
    from fastsim import KINETIC_KEYS

    print(f"[screen] {len(series_ids)} series x 52 sims x {n_days:.0f} d, {workers} workers")
    per_series: list[dict[str, Any]] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        futs = {ex.submit(_screen_one, i, n_days, around, use_fostac): i for i in series_ids}
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                rep = fut.result()
                per_series.append(rep)
                print(
                    f"[screen] series {idx} ({rep['regime']}) done "
                    f"({len(per_series)}/{len(series_ids)}, {time.time()-t0:.0f}s)"
                )
            except Exception:  # noqa: BLE001 - one dead worker must not abort the sweep
                print(f"[screen] series {idx} FAILED\n{traceback.format_exc()}")

    per_series.sort(key=lambda r: r["series_idx"])
    # Pool by the geometric mean of importance across series, so a parameter has to
    # be visible in *most* operating modes to rank high — not just in one.
    pooled: dict[str, float] = {}
    for key in KINETIC_KEYS:
        vals = [max(r["windows"]["60"]["delta_msqr"][key], 1e-12) for r in per_series]
        pooled[key] = float(np.exp(np.mean(np.log(vals))))
    ranking = sorted(pooled, key=lambda k: -pooled[k])

    report = {
        "n_days": n_days,
        "around": around,
        "use_fostac": use_fostac,
        "per_series": per_series,
        "pooled": {"delta_msqr_geomean": pooled, "ranking": ranking},
    }
    results = paths.results_dir()
    results.mkdir(parents=True, exist_ok=True)
    (results / out_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[screen] wrote {results/out_name} ({time.time()-t0:.0f}s)")
    print("[screen] pooled ranking:", ", ".join(ranking[:8]))
    return report


# --------------------------------------------------------------------------
# Stage: calibration
# --------------------------------------------------------------------------
def _legacy_log_bounds() -> np.ndarray:
    """The original flat search box, one factor range for all 26 parameters."""
    from fastsim import KINETIC_KEYS
    from problem import LOG_FACTOR_BOUND

    return np.tile([-LOG_FACTOR_BOUND, LOG_FACTOR_BOUND], (len(KINETIC_KEYS), 1))


def _calibrate_one(task: Task, active: list[int]) -> dict[str, Any]:
    from methods import METHODS, run_method
    from problem import CalibrationProblem, start_point

    series = _get_series(task.series_idx)
    problem = CalibrationProblem(
        series=series,
        n_days=task.n_days,
        active=np.asarray(active, dtype=int),
        known_x0=task.known_x0,
        model=_get_model(),
        budget=task.budget,
        use_fostac=task.use_fostac,
        fostac_every_days=task.fostac_every_days,
        log_bounds=_legacy_log_bounds() if task.legacy_bounds else None,
    )
    # The far start is seeded per series so every method attacks the *same* far
    # point — otherwise the methods would be compared on different problems.
    theta_start = start_point(task.start, seed=1000 + task.series_idx)
    problem.set_base(theta_start)

    t0 = time.time()
    info = run_method(
        task.method,
        problem.make_objective(),
        problem.bounds(),
        theta_start[problem.active],
        task.budget,
    )
    wall = time.time() - t0

    # A run that never evaluated the objective has no estimate. Falling back to
    # the start point here would record it as a finished calibration whose
    # "result" happens to equal its input — indistinguishable from a real run in
    # the JSONL, and it silently drags the medians towards the start value.
    if problem.best_theta is None:
        raise RuntimeError(
            f"optimiser returned without a single objective evaluation "
            f"(n_evals={problem.n_evals}, optimiser said: "
            f"{info.get('message', '')!r})"
        )
    theta_hat = problem.best_theta
    rec: dict[str, Any] = {
        **asdict(task),
        "regime": str(series["regime"]),
        "uses_start": METHODS[task.method].uses_start,
        "n_active": len(active),
        "active": [int(i) for i in active],
        "n_evals": problem.n_evals,
        "wall_s": wall,
        "chi2_final": problem.best_value,
        "optimizer": info,
        "theta_hat": [float(v) for v in theta_hat],
    }
    rec.update(problem.recovery(theta_hat))
    rec["start_dist"] = float(np.median(np.abs(theta_start[problem.active] - problem.theta_true[problem.active])))

    # Reference chi-squares, charged outside the budget. These are what turn a
    # number into a verdict:
    #   chi2_start    — what the fit was before calibrating.
    #   chi2_truth    — the true values for the *active* parameters, the rest left
    #                   at the start. The best a run with this subset could do.
    #   chi2_truth_all— every one of the 26 kinetics at its true value: the noise
    #                   floor of the problem (~1.0 by construction).
    # If chi2_final <= chi2_truth while the parameter error stays large, the data
    # simply do not distinguish the parameter sets — an identifiability verdict,
    # not an optimiser verdict.
    problem.budget = None
    rec["chi2_start"] = float(problem.objective(theta_start[problem.active]))
    rec["chi2_truth"] = float(problem.objective(problem.theta_true[problem.active]))
    saved_base = problem._base_theta
    problem.set_base(problem.theta_true)
    rec["chi2_truth_all"] = float(problem.objective(problem.theta_true[problem.active]))
    problem.set_base(saved_base)

    rec.update(problem.predictive_score(theta_hat))
    return rec


def iter_tasks(cfg: dict[str, Any]) -> Iterator[Task]:
    """Expand a preset into tasks, skipping the start dimension for DE.

    ``force_all_starts`` overrides that skip. It exists because "DE ignores the
    start point" is only half true: DE ignores it for its own search, but
    :meth:`CalibrationProblem.set_base` pins the *inactive* parameters to the
    start as well, so a ``far`` start still hands DE a harder problem (21 frozen
    kinetics further from the truth). Setting the flag runs DE on both starts so
    that claim can be measured instead of assumed.
    """
    from methods import METHODS

    force_starts = bool(cfg.get("force_all_starts", False))
    for s in cfg["series"]:
        for m in cfg["methods"]:
            starts = cfg["starts"] if force_starts or METHODS[m].uses_start else cfg["starts"][:1]
            for start in starts:
                for w in cfg["windows"]:
                    for ps in cfg["param_sets"]:
                        for kx0 in cfg["known_x0"]:
                            for lab in cfg.get("use_fostac", [False]):
                                yield Task(
                                    s,
                                    m,
                                    start,
                                    w,
                                    ps,
                                    kx0,
                                    lab,
                                    cfg.get("fostac_every_days", 7.0),
                                    bool(cfg.get("legacy_bounds", False)),
                                    cfg["budget"],
                                )


def run_calibration(cfg: dict[str, Any], workers: int, out_name: str) -> None:
    """Run the sweep, appending each finished task to a JSONL file."""
    tasks = list(iter_tasks(cfg))
    active_by_set = {ps: resolve_param_set(ps) for ps in cfg["param_sets"]}
    if not cfg["budget"]:
        # Scale the allowance with the number of free parameters, so the
        # comparison is "accuracy per parameter searched" rather than a handicap
        # for the larger sets.
        tasks = [replace(t, budget=BUDGET_PER_PARAM * len(active_by_set[t.param_set])) for t in tasks]

    results = paths.results_dir()
    results.mkdir(parents=True, exist_ok=True)
    out = results / out_name
    done: set[tuple] = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add(
                (
                    r["series_idx"],
                    r["method"],
                    r["start"],
                    r["n_days"],
                    r["param_set"],
                    r["known_x0"],
                    r.get("use_fostac", False),
                    r.get("fostac_every_days", 7.0),
                )
            )
        tasks = [
            t
            for t in tasks
            if (t.series_idx, t.method, t.start, t.n_days, t.param_set, t.known_x0, t.use_fostac, t.fostac_every_days)
            not in done
        ]
        print(f"[calibrate] resuming — {len(done)} done, {len(tasks)} to go")

    #: Measured wall-clock per simulated day per forward simulation, on one core.
    #: (0.325 s/day: a 10-day, 300-simulation task takes ~16 min.)
    est_s = sum(t.budget * t.n_days * 0.325 for t in tasks) / max(workers, 1)
    print(f"[calibrate] {len(tasks)} tasks on {workers} workers " f"(~{est_s/3600:.1f} h estimated)")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:
        futs = {ex.submit(_calibrate_one, t, [int(i) for i in active_by_set[t.param_set]]): t for t in tasks}
        for n, fut in enumerate(as_completed(futs), 1):
            t = futs[fut]
            try:
                rec = fut.result()
            except Exception:  # noqa: BLE001 - one dead worker must not abort the sweep
                print(f"[calibrate] FAILED {t}\n{traceback.format_exc()}")
                continue
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(
                f"[calibrate] {n}/{len(tasks)} s{t.series_idx} {t.method[:12]:12s} "
                f"{t.start:4s} {t.n_days:2d}d {t.param_set:6s} | "
                f"chi2 {rec['chi2_start']:.2f}->{rec['chi2_final']:.2f} "
                f"(truth {rec['chi2_truth']:.2f}) | "
                f"log-err {rec['start_dist']:.3f}->{rec['log_err_median']:.3f} | "
                f"{rec['wall_s']:.0f}s | {time.time()-t0:.0f}s elapsed"
            )
    print(f"[calibrate] wrote {out} ({time.time()-t0:.0f}s)")


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["screen", "calibrate", "all"])
    p.add_argument("--preset", default="main", choices=sorted(PRESETS))
    p.add_argument("--series", type=int, nargs="*", default=None, help="Override the preset's series list.")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--budget", type=int, default=None, help="Override the budget.")
    p.add_argument("--out", default=None, help="JSONL filename for the sweep.")
    p.add_argument("--screen-days", type=float, default=60.0)
    p.add_argument("--around", default="truth", choices=["truth", "nominal"])
    p.add_argument("--fostac", action="store_true", help="Include the FOS/TAC titration in the objective.")
    p.add_argument(
        "--fostac-every-days",
        type=float,
        default=7.0,
        help="Lab sampling interval in days (7 = realistic weekly, " "0.0417 = hourly upper bound).",
    )
    p.add_argument("--screen-out", default="screening.json")
    paths.add_results_argument(p)
    args = p.parse_args()
    # Before anything resolves a location — and before the pool is created, so
    # the workers inherit the choice through the environment.
    paths.apply_location_args(args)

    cfg = dict(PRESETS[args.preset])
    if args.series is not None:
        cfg["series"] = args.series
    if args.budget is not None:
        cfg["budget"] = args.budget
    if args.fostac:
        cfg["use_fostac"] = [True]
        cfg["fostac_every_days"] = args.fostac_every_days
    out_name = args.out or f"calibration_{args.preset}.jsonl"

    if args.stage in ("screen", "all"):
        run_screening(
            cfg["series"], args.workers, args.screen_days, args.around, use_fostac=args.fostac, out_name=args.screen_out
        )
    if args.stage in ("calibrate", "all"):
        run_calibration(cfg, args.workers, out_name)


if __name__ == "__main__":
    main()
