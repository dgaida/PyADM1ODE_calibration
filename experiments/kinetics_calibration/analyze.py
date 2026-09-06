"""Aggregate the study into the four tables the questions actually ask for.

Usage::

    python analyze.py                      # all sections, markdown to stdout
    python analyze.py --section length     # just the days-of-data answer
    python analyze.py --out report.md

Reads ``results/screening.json`` and ``results/calibration_*.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import paths

#: Log-error below which a parameter counts as "recovered" (within 25 %).
RECOVERED = float(np.log(1.25))


#: The sweep these tables describe. Named explicitly rather than globbed: the
#: results directory also holds ``calibration_count*.jsonl`` (a different budget
#: and different parameter sets) and ``calibration_fostac.jsonl``, so a glob
#: would mix experiments and quietly corrupt every median in this file.
MAIN_SWEEP = "calibration_main.jsonl"


def load_runs(pattern: str = MAIN_SWEEP) -> list[dict[str, Any]]:
    """Completed calibration runs of one sweep.

    Args:
        pattern: File name or glob under ``results/``. Defaults to the main
            sweep; pass a glob only when you are sure the matched files belong
            to the same experiment.
    """
    runs: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for path in sorted(paths.results_dir().glob(pattern)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            # Guard against the same configuration arriving from two files.
            key = (
                r["series_idx"],
                r["method"],
                r["start"],
                r["n_days"],
                r["param_set"],
                r["known_x0"],
                r.get("use_fostac", False),
            )
            if key in seen:
                continue
            seen.add(key)
            runs.append(r)
    return runs


def _fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def _median(values: Iterable[float]) -> float:
    """Median over the finite entries only.

    The 60-day runs have no holdout window left (the series is 60 days long), so
    their ``chi2_holdout`` is NaN by design. A plain median would propagate that
    NaN and blank out the whole group, hiding the shorter windows' results.
    """
    arr = np.asarray([v for v in values], dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def _table(header: list[str], rows: Iterable[list[str]]) -> str:
    rows = list(rows)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------------
def section_identifiability() -> str:
    """Which kinetics are visible, and which can be told apart."""
    path = paths.results_dir() / "screening.json"
    if not path.exists():
        return "_(screening not run)_\n"
    rep = json.loads(path.read_text(encoding="utf-8"))
    pooled = rep["pooled"]["delta_msqr_geomean"]
    ranking = rep["pooled"]["ranking"]

    per_series = rep["per_series"]
    by_regime: dict[str, list[dict]] = defaultdict(list)
    for r in per_series:
        by_regime[r["regime"]].append(r)
    regimes = sorted(by_regime)

    rows = []
    for name in ranking[:12]:
        cells = [name, _fmt(pooled[name], 2)]
        for reg in regimes:
            vals = [s["windows"]["60"]["delta_msqr"][name] for s in by_regime[reg]]
            cells.append(_fmt(float(np.exp(np.mean(np.log(np.maximum(vals, 1e-12))))), 2))
        rows.append(cells)

    txt = [
        "### Q2 — Which kinetics can be found?",
        "",
        "Importance `delta_msqr` = how many **sensor standard deviations** the five",
        "measurements move when a kinetic changes by one log unit (a factor e).",
        "Below ~1 the parameter's effect is smaller than the noise, so no optimiser",
        "can recover it. Geometric mean over series, 60-day window.",
        "",
        _table(["kinetic", "pooled"] + regimes, rows),
        "",
    ]

    # Being individually strong is not enough — the effects also have to be
    # separable. A subset with a high collinearity index cannot be resolved even
    # when every member moves the sensors a lot.
    sub_rows = []
    for s in per_series:
        w60 = s["windows"]["60"]
        sub_rows.append(
            [
                f"{s['series_idx']}",
                s["regime"],
                str(len(w60["best_subset"])),
                ", ".join(f"`{n}`" for n in w60["best_subset"]),
                _fmt(w60["best_subset_gamma"], 1),
            ]
        )
    txt += [
        "Largest **jointly** identifiable subset per series (Brun greedy, stopping",
        "when the collinearity index would exceed 10):",
        "",
        _table(["series", "mode", "size", "subset", "gamma"], sub_rows),
        "",
    ]
    return "\n".join(txt)


def section_length() -> str:
    """How the answer changes with the length of the measurement record."""
    path = paths.results_dir() / "screening.json"
    if not path.exists():
        return "_(screening not run)_\n"
    rep = json.loads(path.read_text(encoding="utf-8"))
    per_series = rep["per_series"]
    windows = sorted(per_series[0]["windows"], key=int)
    ranking = rep["pooled"]["ranking"][:6]

    rows = []
    for name in ranking:
        cells = [name]
        for w in windows:
            vals = [s["windows"][w]["delta_msqr"][name] for s in per_series]
            cells.append(_fmt(float(np.exp(np.mean(np.log(np.maximum(vals, 1e-12))))), 2))
        rows.append(cells)

    txt = [
        "### Q3 — How many days of measurements?",
        "",
        "**Information side** — importance vs. window length (geometric mean over series):",
        "",
        _table(["kinetic"] + [f"{w} d" for w in windows], rows),
        "",
    ]

    runs = load_runs()
    if runs:
        txt += ["**Recovery side** — median log-error of the calibrated parameters:", ""]
        methods = sorted({r["method"] for r in runs})
        lengths = sorted({r["n_days"] for r in runs})
        rows = []
        for m in methods:
            for start in sorted({r["start"] for r in runs if r["method"] == m}):
                cells = [m, start]
                for L in lengths:
                    sel = [r for r in runs if r["method"] == m and r["start"] == start and r["n_days"] == L]
                    cells.append(_fmt(_median([r["log_err_median"] for r in sel])) if sel else "-")
                rows.append(cells)
        txt += [_table(["method", "start"] + [f"{L} d" for L in lengths], rows), ""]
        txt += [
            "Lower is better; the start column of the same row is the reference —",
            "a value that does not fall below the start distance means calibration",
            "did not learn the kinetics, whatever it did to the fit.",
            "",
        ]
    return "\n".join(txt)


def section_methods() -> str:
    """Accuracy and cost per optimiser."""
    runs = load_runs()
    if not runs:
        return "_(no calibration runs)_\n"

    rows = []
    for m in sorted({r["method"] for r in runs}):
        sel = [r for r in runs if r["method"] == m]
        improved = [r for r in sel if r["log_err_median"] < r["start_dist"]]
        rows.append(
            [
                m,
                str(len(sel)),
                _fmt(_median([r["start_dist"] for r in sel])),
                _fmt(_median([r["log_err_median"] for r in sel])),
                _fmt(_median([r["frac_within_25pct"] for r in sel]), 2),
                _fmt(_median([r["chi2_final"] for r in sel]), 2),
                _fmt(_median([r["chi2_holdout"] for r in sel]), 2),
                f"{100.0*len(improved)/len(sel):.0f} %",
                _fmt(_median([r["n_evals"] for r in sel]), 0),
                _fmt(_median([r["wall_s"] for r in sel]), 0),
            ]
        )

    txt = [
        "### Q4 — Which algorithm is most accurate and most efficient?",
        "",
        "All methods get the **same budget of forward simulations**, so this is",
        "accuracy at equal cost. `chi2` values are per-measurement chi-squares:",
        "1.0 means the fit is inside sensor noise.",
        "",
        _table(
            ["method", "n", "start err", "final err", "frac<25%", "chi2 fit", "chi2 holdout", "improved", "sims", "wall s"],
            rows,
        ),
        "",
        "`improved` is the share of runs whose parameters ended **closer** to the",
        "truth than they started — the share where calibration helped rather than",
        "just fitting noise.",
        "",
    ]

    # Pooling over windows is only fair while every method covers the same ones.
    # It silently was not: the first sweep ran the gradient methods at 10 and 30
    # days and the others at 10/20/30/60, so the pooled row mixed a different
    # window mix per method. This breakdown makes the coverage visible and lets
    # each window be compared on its own.
    windows = sorted({r["n_days"] for r in runs})
    methods = sorted({r["method"] for r in runs})
    rows = []
    gaps = []
    for m in methods:
        cells = [m]
        for w in windows:
            sel = [r for r in runs if r["method"] == m and r["n_days"] == w]
            if sel:
                err = _median([r["log_err_median"] for r in sel])
                cells.append(f"{_fmt(err)} ({len(sel)})")
            else:
                cells.append("—")
                gaps.append((m, w))
        rows.append(cells)

    txt += [
        "**Per window** — median log-error, run count in brackets. A dash is a",
        "cell that was never computed, and any dash makes the pooled table above",
        "an average over a different window mix for that method.",
        "",
        _table(["method"] + [f"{w} d" for w in windows], rows),
        "",
    ]
    if gaps:
        txt += [
            "> **Incomplete matrix.** Missing: "
            + ", ".join(f"`{m}` at {w} d" for m, w in gaps)
            + ". Run `run_study.py calibrate --preset shootout_length` to fill it.",
            "",
        ]
    return "\n".join(txt)


def section_possible() -> str:
    """The headline question: can the kinetics be recovered at all?"""
    runs = load_runs()
    if not runs:
        return "_(no calibration runs)_\n"

    txt = ["### Q1 — Is it possible to recover the kinetics from the measurements?", ""]
    rows = []
    for start in sorted({r["start"] for r in runs}):
        sel = [r for r in runs if r["start"] == start]
        rows.append(
            [
                start,
                str(len(sel)),
                _fmt(_median([r["start_dist"] for r in sel])),
                _fmt(_median([r["log_err_median"] for r in sel])),
                _fmt(_median([r["chi2_start"] for r in sel]), 2),
                _fmt(_median([r["chi2_final"] for r in sel]), 2),
                _fmt(_median([r["chi2_truth"] for r in sel]), 2),
            ]
        )
    txt += [
        _table(
            ["start", "n", "start err", "final err", "chi2 start", "chi2 final", "chi2 at truth"],
            rows,
        ),
        "",
        "The decisive comparison is **chi2 final vs. chi2 at truth**. If the search",
        "reaches a fit as good as (or better than) the true parameters while the",
        "parameter error stays high, the failure is *identifiability*, not the",
        "optimiser: many kinetic sets explain these five sensors equally well.",
        "",
    ]

    # Per-parameter recovery, counted only where the parameter was actually
    # optimised — a kinetic held fixed at its start value says nothing about
    # whether the data could have identified it.
    from fastsim import KINETIC_KEYS

    active_names: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        for idx in r["active"]:
            name = KINETIC_KEYS[idx]
            active_names[name].append(r["log_err_per_param"][name])
    if active_names:
        rows = [
            [name, str(len(v)), _fmt(float(np.median(v))), f"{100.0*float(np.mean(np.asarray(v) < RECOVERED)):.0f} %"]
            for name, v in sorted(active_names.items(), key=lambda kv: np.median(kv[1]))
        ]
        txt += [
            "Per calibrated parameter (median over all runs):",
            "",
            _table(["kinetic", "n", "median log-err", "within 25 %"], rows),
            "",
        ]
    return "\n".join(txt)


def section_regime() -> str:
    """Does the operating mode change the answer?"""
    runs = load_runs()
    if not runs:
        return ""
    rows = []
    for reg in sorted({r["regime"] for r in runs}):
        sel = [r for r in runs if r["regime"] == reg]
        rows.append(
            [
                reg,
                str(len(sel)),
                _fmt(_median([r["start_dist"] for r in sel])),
                _fmt(_median([r["log_err_median"] for r in sel])),
                _fmt(_median([r["chi2_holdout"] for r in sel]), 2),
            ]
        )
    return "\n".join(
        [
            "### Operating mode",
            "",
            "How hard the feed excites the plant decides how much the measurements say",
            "about the kinetics.",
            "",
            _table(["mode", "n", "start err", "final err", "chi2 holdout"], rows),
            "",
        ]
    )


def section_confounding() -> str:
    """Which parameters are traded off against each other by the search.

    The screening's collinearity index predicts this from the Jacobian; this
    measures it directly from the estimates. Correlating the *signed* log-errors
    across runs shows which parameters the data cannot separate: a strong
    correlation means the objective constrains a combination of the two, so the
    search slides along that direction and lands anywhere on it.
    """
    paths.add_dataset_to_path()
    from fastsim import KINETIC_KEYS
    from loader import load_test

    runs = [r for r in load_runs() if r["method"] in ("powell", "nelder_mead", "differential_evolution")]
    if not runs:
        return ""
    test = load_test()
    names = [KINETIC_KEYS[i] for i in runs[0]["active"]]
    idx = [KINETIC_KEYS.index(n) for n in names]

    err = {n: [] for n in names}
    for r in runs:
        truth = np.log(np.asarray(test[r["series_idx"]]["kinetic_factors"], float))
        hat = np.asarray(r["theta_hat"], float)
        for n, i in zip(names, idx):
            err[n].append(hat[i] - truth[i])
    E = np.array([err[n] for n in names])

    rows = []
    for a, n in enumerate(names):
        rows.append([n] + [f"{np.corrcoef(E[a], E[b])[0, 1]:.2f}" for b in range(len(names))])

    bias = [[n, f"{np.mean(err[n]):+.3f}", f"{np.std(err[n]):.3f}", f"{np.exp(np.mean(err[n])):.2f}x"] for n in names]
    return "\n".join(
        [
            "### Which parameters get confounded",
            "",
            "Correlation of the **signed** log-errors across runs. A strong value means",
            "the measurements constrain a *combination* of the two parameters rather than",
            "each one, so the search slides along that direction.",
            "",
            _table([""] + [n[:9] for n in names], rows),
            "",
            "Bias and spread of each estimate:",
            "",
            _table(["kinetic", "mean signed err", "sd", "bias factor"], bias),
            "",
            "Near-zero means with large spreads say the estimates are essentially",
            "**unbiased** — the problem is variance along the confounded directions,",
            "not a systematic offset.",
            "",
        ]
    )


SECTIONS = {
    "possible": section_possible,
    "identifiability": section_identifiability,
    "length": section_length,
    "methods": section_methods,
    "confounding": section_confounding,
    "regime": section_regime,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--section", choices=sorted(SECTIONS), default=None)
    p.add_argument("--out", default=None)
    paths.add_results_argument(p)
    args = p.parse_args()
    paths.apply_location_args(args)

    names = [args.section] if args.section else ["possible", "identifiability", "length", "methods", "confounding", "regime"]
    text = "# Kinetics-calibration study — results\n\n" + "\n".join(SECTIONS[n]() for n in names)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
