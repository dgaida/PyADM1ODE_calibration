"""What does the weekly FOS/TAC titration actually add?

Compares two screening runs (with and without the lab channel) on the measure
that can answer the question, and explains why the obvious one cannot.

Why not ``delta_msqr``
----------------------
Brun's importance is an RMS *average* over the rows of the Jacobian. The lab
channel contributes 9 rows against 7205 sensor rows, so averaging dilutes it to
0.12 % of the total no matter how informative those 9 rows are. Comparing two
measurement sets of different size with an average is meaningless.

What works instead
------------------
The Fisher information ``S^T S`` is a **sum**, so it is additive over
measurements::

    FIM_total = FIM_sensors + FIM_lab

The stored Gram matrix per window is exactly that, so the lab contribution can
be isolated by subtraction, and the quantity that actually matters — the joint
standard error of a parameter when the whole subset is estimated together —
follows from the inverse.

Usage::

    python compare_fostac.py                       # summary tables
    python compare_fostac.py --window 30
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


def _load(name: str) -> dict:
    path = paths.results_dir() / name
    if not path.exists():
        raise SystemExit(f"{path} fehlt — Screening zuerst laufen lassen.")
    return json.loads(path.read_text(encoding="utf-8"))


def _gram(rep: dict, series_pos: int, window: str) -> np.ndarray:
    return np.asarray(rep["per_series"][series_pos]["windows"][window]["gram"], float)


def joint_se(gram: np.ndarray, subset: list[int]) -> np.ndarray:
    """Standard errors of ``subset`` when estimated jointly, from the Gram matrix."""
    sub = gram[np.ix_(subset, subset)]
    try:
        cov = np.linalg.inv(sub)
    except np.linalg.LinAlgError:
        return np.full(len(subset), np.inf)
    d = np.diag(cov)
    return np.sqrt(np.where(d > 0, d, np.inf))


def compare_runs() -> None:
    """With/without comparison of the actual calibration runs.

    The screening answers what the lab channel adds *locally, at the truth*. This
    answers what it adds to a real search starting far away — a non-local effect
    the Fisher information cannot see, because a measurement can help by keeping
    the optimiser out of wrong regions even when it barely sharpens the optimum.
    """
    from fastsim import KINETIC_KEYS

    def load(name: str) -> list[dict]:
        path = paths.results_dir() / name
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    base = [r for r in load("calibration_main.jsonl") if r["method"] in ("nelder_mead", "differential_evolution")]
    lab = load("calibration_fostac.jsonl")
    if not lab:
        print("\n(noch keine FOS/TAC-Kalibrierlaeufe)")
        return

    # Pair the runs by their configuration so like is compared with like.
    def key(r: dict) -> tuple:
        return (r["series_idx"], r["method"], r["start"], r["n_days"])

    bmap = {key(r): r for r in base}
    pairs = [(bmap[key(r)], r) for r in lab if key(r) in bmap]
    print(f"\n4. Kalibrierlaeufe: {len(pairs)} paarweise vergleichbare Konfigurationen")
    if not pairs:
        return

    print(f"{'Fenster':>8}{'n':>4}{'Fehler ohne':>13}{'Fehler mit':>12}{'besser?':>10}")
    for w in sorted({b["n_days"] for b, _ in pairs}):
        sel = [(b, l) for b, l in pairs if b["n_days"] == w]
        eb = np.median([b["log_err_median"] for b, _ in sel])
        el = np.median([l["log_err_median"] for _, l in sel])
        wins = 100.0 * np.mean([l["log_err_median"] < b["log_err_median"] for b, l in sel])
        print(f"{w:>8}{len(sel):>4}{eb:13.3f}{el:12.3f}{wins:9.0f}%")

    print(f"\n{'Kinetik':<12}{'Fehler ohne':>13}{'Fehler mit':>12}{'Verbesserung':>14}")
    for name in [KINETIC_KEYS[i] for i in pairs[0][0]["active"]]:
        eb = np.median([b["log_err_per_param"][name] for b, _ in pairs])
        el = np.median([l["log_err_per_param"][name] for _, l in pairs])
        print(f"{name:<12}{eb:13.3f}{el:12.3f}{eb/max(el,1e-9):13.2f}x")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--window", default="60")
    p.add_argument("--base", default="screening_base.json")
    p.add_argument("--lab", default="screening_fostac.json")
    p.add_argument("--runs-only", action="store_true", help="Nur den Kalibrierlauf-Vergleich, ohne Screening.")
    paths.add_results_argument(p)
    args = p.parse_args()
    paths.apply_location_args(args)

    if args.runs_only:
        compare_runs()
        return

    from fastsim import KINETIC_KEYS

    base, lab = _load(args.base), _load(args.lab)
    top5 = [KINETIC_KEYS.index(n) for n in base["pooled"]["ranking"][:5]]
    names = [KINETIC_KEYS[i] for i in top5]

    print(f"Fenster: {args.window} d | Parametersatz: {', '.join(names)}\n")

    # --- 1. Information content per measurement -------------------------
    print("1. Fisher-Information der Top-5 (Diagonale von S^T S, Summe ueber Reihen)")
    print(f"{'Kinetik':<12}{'Sensoren':>12}{'Labor':>12}{'Labor-Anteil':>14}")
    inf_sensor = np.zeros(len(top5))
    inf_lab = np.zeros(len(top5))
    for pos in range(len(base["per_series"])):
        gb = _gram(base, pos, args.window)
        gl = _gram(lab, pos, args.window)
        inf_sensor += np.diag(gb)[top5]
        inf_lab += (np.diag(gl) - np.diag(gb))[top5]
    for k, n in enumerate(names):
        share = inf_lab[k] / max(inf_sensor[k] + inf_lab[k], 1e-30)
        print(f"{n:<12}{inf_sensor[k]:12.3g}{inf_lab[k]:12.3g}{100*share:13.2f}%")

    # --- 2. Joint standard errors ---------------------------------------
    print("\n2. Gemeinsamer Standardfehler der Top-5 [log-Einheiten, kleiner=besser]")
    print(f"{'Kinetik':<12}{'ohne Labor':>12}{'mit Labor':>12}{'Verbesserung':>14}")
    se_b, se_l = [], []
    for pos in range(len(base["per_series"])):
        se_b.append(joint_se(_gram(base, pos, args.window), top5))
        se_l.append(joint_se(_gram(lab, pos, args.window), top5))
    se_b, se_l = np.median(se_b, axis=0), np.median(se_l, axis=0)
    for k, n in enumerate(names):
        print(f"{n:<12}{se_b[k]:12.4f}{se_l[k]:12.4f}{se_b[k]/max(se_l[k],1e-30):13.2f}x")

    # --- 3. The confounded propionate pair ------------------------------
    pair = [KINETIC_KEYS.index("k_m_pro"), KINETIC_KEYS.index("k_dec_pro")]
    print("\n3. Das gekoppelte Propionat-Paar (k_m_pro / k_dec_pro)")
    print(f"{'Serie':>6} {'Regime':<13}{'gamma ohne':>12}{'gamma mit':>12}{'Faktor':>9}")
    for pos, s in enumerate(base["per_series"]):
        rows = []
        for rep in (base, lab):
            g = _gram(rep, pos, args.window)[np.ix_(pair, pair)]
            d = np.sqrt(np.diag(g))
            corr = g / np.outer(d, d)
            rows.append(float(1.0 / np.sqrt(max(np.linalg.eigvalsh(corr).min(), 1e-30))))
        print(f"{s['series_idx']:>6} {s['regime']:<13}{rows[0]:12.2f}{rows[1]:12.2f}" f"{rows[0]/max(rows[1],1e-30):8.2f}x")
    print(
        "\ngamma = Kollinearitaetsindex des Paares. Sinkt er deutlich, trennt die\n"
        "Labormessung die beiden Parameter; bleibt er gleich, tut sie es nicht."
    )

    compare_runs()


if __name__ == "__main__":
    main()
