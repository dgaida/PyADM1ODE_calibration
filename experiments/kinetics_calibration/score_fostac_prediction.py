"""Does calibrating *with* FOS/TAC predict FOS/TAC better?

Recovering the kinetics and predicting the acidification indicator are different
goals, and a measurement can serve the second without helping the first. This
re-simulates each stored estimate over the full 60 days and scores the predicted
FOS/TAC against the truth, so the two questions are answered separately.

Scored on the **whole trajectory**, not only the 9 sample days: what an operator
wants from a model is the indicator on the days it was *not* measured.

Usage::

    python score_fostac_prediction.py --workers 24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import paths

_MODEL = None
_TEST = None
_PARAMS = None


def _init() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _score(rec: dict[str, Any]) -> dict[str, Any]:
    global _MODEL, _TEST, _PARAMS
    if _MODEL is None:
        from fastsim import ForwardModel

        _MODEL = ForwardModel.build(prune=True)
    if _TEST is None:
        paths.add_dataset_to_path()
        from loader import load_test

        _TEST = load_test()
    if _PARAMS is None:
        from fastsim import adm1_torch_params

        _PARAMS = adm1_torch_params(_MODEL.plant)

    from fastsim import fostac_from_states

    s = _TEST[rec["series_idx"]]
    n_steps = len(s["measurements"]) - 1
    _MODEL.set_log_factors(np.asarray(rec["theta_hat"], float))
    states, _ = _MODEL.simulate(s["states"][0], np.asarray(s["feed_noisy"], float), n_steps)
    _MODEL.reset_kinetics()

    pred = fostac_from_states(states, _PARAMS)
    truth = fostac_from_states(np.asarray(s["states"], float), _PARAMS)
    ratio_p = pred[:, 0] / np.maximum(pred[:, 1], 1e-9)
    ratio_t = truth[:, 0] / np.maximum(truth[:, 1], 1e-9)

    return {
        "series_idx": rec["series_idx"],
        "method": rec["method"],
        "start": rec["start"],
        "n_days": rec["n_days"],
        "use_fostac": rec.get("use_fostac", False),
        # Relative error on FOS, absolute on the ratio (which is already a ratio).
        "fos_rel_rmse": float(np.sqrt(np.mean(((pred[:, 0] - truth[:, 0]) / np.maximum(truth[:, 0], 1e-9)) ** 2))),
        "tac_rel_rmse": float(np.sqrt(np.mean(((pred[:, 1] - truth[:, 1]) / np.maximum(truth[:, 1], 1e-9)) ** 2))),
        "ratio_rmse": float(np.sqrt(np.mean((ratio_p - ratio_t) ** 2))),
        "ratio_max_err": float(np.max(np.abs(ratio_p - ratio_t))),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    paths.add_results_argument(p)
    args = p.parse_args()
    paths.apply_location_args(args)

    def load(name: str) -> list[dict]:
        path = paths.results_dir() / name
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []

    runs = [r for r in load("calibration_main.jsonl") if r["method"] in ("nelder_mead", "differential_evolution")]
    runs += load("calibration_fostac.jsonl")
    print(f"[score] {len(runs)} Laeufe auf {args.workers} Workern")

    out = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        for f in as_completed([ex.submit(_score, r) for r in runs]):
            out.append(f.result())
    results = paths.results_dir()
    results.mkdir(parents=True, exist_ok=True)
    (results / "fostac_prediction.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    key = lambda r: (r["series_idx"], r["method"], r["start"], r["n_days"])
    base = {key(r): r for r in out if not r["use_fostac"]}
    pairs = [(base[key(r)], r) for r in out if r["use_fostac"] and key(r) in base]

    print(f"\nVorhersage von FOS/TAC ueber die vollen 60 Tage ({len(pairs)} Paare)")
    print(f"{'Groesse':<18}{'ohne Labor':>12}{'mit Labor':>12}{'Verbesserung':>14}")
    for label, field in [
        ("FOS rel. RMSE", "fos_rel_rmse"),
        ("TAC rel. RMSE", "tac_rel_rmse"),
        ("FOS/TAC RMSE", "ratio_rmse"),
        ("FOS/TAC max. Fehler", "ratio_max_err"),
    ]:
        b = np.median([x[field] for x, _ in pairs])
        l = np.median([y[field] for _, y in pairs])
        print(f"{label:<18}{b:12.4f}{l:12.4f}{b/max(l,1e-12):13.2f}x")

    wins = 100.0 * np.mean([y["ratio_rmse"] < x["ratio_rmse"] for x, y in pairs])
    print(f"\nFOS/TAC-Vorhersage besser in {wins:.0f}% der Paare (Zufall waere 50%)")


if __name__ == "__main__":
    main()
