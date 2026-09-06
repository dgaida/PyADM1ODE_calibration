"""How many kinetics should one calibrate?

Compares parameter sets of growing size and separates the two effects that pull
in opposite directions:

* **Bias from freezing.** Parameters left at their (wrong) start value force the
  free ones to bend and absorb their error. ``chi2_truth`` measures this: it is
  the best fit reachable with the active set at its true values and the rest
  frozen. The further above the 1.0 noise floor it sits, the more the free
  parameters must lie to compensate.
* **Variance from searching.** Every extra parameter adds a direction the data
  may not constrain, and the budget has to cover it.

Comparing medians over the *active* set would be misleading
---------------------------------------------------------
A run calibrating only the top 3 is scored on the three easiest parameters; one
calibrating all 26 is scored on 26 including hopeless ones. The headline metric
here is therefore the error over **all 26 kinetics**, with parameters a run did
not calibrate counted at the value they kept — their start. That is what the
resulting model actually carries, and it is the same yardstick for every
configuration.

Usage::

    python analyze_count.py                                  # baseline sweep
    python analyze_count.py --file calibration_count_fostac7d.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # The FOS/TAC repeats of this sweep write to their own files, so the
    # "does the lab measurement change the answer?" comparison is a matter of
    # pointing this at each of them in turn.
    p.add_argument("--file", default="calibration_count.jsonl", help="JSONL des Anzahl-Sweeps unter dem Ergebnisverzeichnis.")
    paths.add_results_argument(p)
    args = p.parse_args()
    paths.apply_location_args(args)

    paths.add_dataset_to_path()
    from fastsim import KINETIC_KEYS
    from loader import load_test
    from problem import start_point

    path = paths.results_dir() / args.file
    if not path.exists():
        raise SystemExit(f"{path} fehlt — Sweep zuerst laufen lassen.")
    runs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    test = load_test()
    print(f"{path.name}: n = {len(runs)} Laeufe\n")

    # Full-vector error: what the resulting 26-parameter model is actually worth.
    for r in runs:
        truth = np.log(np.asarray(test[r["series_idx"]]["kinetic_factors"], float))
        hat = np.asarray(r["theta_hat"], float)
        start = start_point(r["start"], seed=1000 + r["series_idx"])
        r["_err_all"] = float(np.median(np.abs(hat - truth)))
        r["_err_start_all"] = float(np.median(np.abs(start - truth)))
        # Common yardstick: the five parameters every configuration calibrates.
        common = [KINETIC_KEYS.index(n) for n in ("k_dis_PS", "k_m_pro", "k_m_ac", "k_dec_pro", "k_dis_PF")]
        r["_err_top5"] = float(np.median(np.abs(hat[common] - truth[common])))

    order = ["top3", "top5", "top10", "top18", "nokS", "all26"]
    sets = [p for p in order if any(r["param_set"] == p for r in runs)]

    # The start distance differs by start point, so it must be reported per
    # column — a pooled "Start" would silently compare near results against a
    # near/far average and invert the verdict.
    start_ref = {st: np.median([r["_err_start_all"] for r in runs if r["start"] == st]) for st in ("near", "far")}
    print("Fehler ueber ALLE 26 Kinetiken (nicht kalibrierte zaehlen mit ihrem Startwert)")
    print(f"{'Satz':<8}{'n_par':>6}{'near':>9}{'far':>9}{'chi2_T near':>13}{'chi2_T far':>12}{'Sims':>8}")
    print(f"{'(Start)':<8}{'':>6}{start_ref['near']:>9.3f}{start_ref['far']:>9.3f}")
    for ps in sets:
        sel = [r for r in runs if r["param_set"] == ps]
        if not sel:
            continue
        npar = sel[0]["n_active"]
        row = f"{ps:<8}{npar:>6}"
        for st in ("near", "far"):
            s2 = [r for r in sel if r["start"] == st]
            row += f"{(np.median([r['_err_all'] for r in s2]) if s2 else np.nan):>9.3f}"
        for st in ("near", "far"):
            s2 = [r for r in sel if r["start"] == st]
            row += f"{(np.median([r['chi2_truth'] for r in s2]) if s2 else np.nan):>13.3f}"
        row += f"{np.median([r['n_evals'] for r in sel]):>8.0f}"
        print(row)

    print("\nFehler auf den 5 gemeinsamen Kinetiken (gleicher Massstab fuer alle Saetze)")
    print(f"{'Satz':<8}{'near':>9}{'far':>9}{'chi2 fit near':>15}{'chi2 fit far':>14}")
    for ps in sets:
        sel = [r for r in runs if r["param_set"] == ps]
        row = f"{ps:<8}"
        for st in ("near", "far"):
            s2 = [r for r in sel if r["start"] == st]
            row += f"{(np.median([r['_err_top5'] for r in s2]) if s2 else np.nan):>9.3f}"
        for st in ("near", "far"):
            s2 = [r for r in sel if r["start"] == st]
            row += f"{(np.median([r['chi2_final'] for r in s2]) if s2 else np.nan):>15.3f}"
        print(row)

    print("\nVorhersage auf den nicht kalibrierten Tagen (chi2_holdout)")
    print(f"{'Satz':<8}{'near':>9}{'far':>9}")
    for ps in sets:
        sel = [r for r in runs if r["param_set"] == ps]
        row = f"{ps:<8}"
        for st in ("near", "far"):
            v = [r["chi2_holdout"] for r in sel if r["start"] == st and np.isfinite(r["chi2_holdout"])]
            row += f"{(np.median(v) if v else np.nan):>9.3f}"
        print(row)


if __name__ == "__main__":
    main()
