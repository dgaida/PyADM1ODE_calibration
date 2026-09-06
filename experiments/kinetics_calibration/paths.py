"""Where the benchmark dataset and the result files live.

The study code lives in this repository, but the data it runs on does not: the
benchmark series are generated and maintained in ``PyADM1ODE_estimate``. Rather
than hard-coding a relative path — which only worked while the study sat inside
that repository — the dataset is looked up in three steps:

1. ``$PYADM1ODE_BENCHMARK``, if set. Wins over everything.
2. The installed ``pyadm1ode_estimation`` package. It is required by
   :mod:`fastsim` anyway (the plant definition comes from it), and in an editable
   install its ``__file__`` points straight into the source checkout, so the
   dataset sits at ``<repo>/datasets/benchmark``.
3. A sibling checkout next to this repository (``../PyADM1ODE_estimate``), for
   the case where the package is installed non-editable.

Step 2 is what makes the default case work without configuration.

Environment variables are used rather than a module-level global because the
sweeps run in a :class:`~concurrent.futures.ProcessPoolExecutor`. On Windows the
workers are *spawned*, so they re-import this module from scratch and would not
see an in-process override — an environment variable is inherited.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

#: Overrides the dataset lookup. Must point at the directory holding
#: ``loader.py``, ``test.npz`` and ``meta.json``.
ENV_DATASET = "PYADM1ODE_BENCHMARK"

#: Overrides where sweeps write and analyses read. Defaults to ``./results``.
ENV_RESULTS = "KINETICS_RESULTS"

#: Files that must be present for a directory to count as the benchmark.
_REQUIRED = ("loader.py", "test.npz", "meta.json")


def _is_benchmark(path: Path) -> bool:
    return all((path / f).exists() for f in _REQUIRED)


def _candidates() -> list[Path]:
    """Every place the dataset might be, in priority order."""
    out: list[Path] = []
    env = os.environ.get(ENV_DATASET)
    if env:
        out.append(Path(env).expanduser())

    # The estimate package is a hard dependency of fastsim, so if the study can
    # run at all this import succeeds and points at its checkout.
    try:
        import pyadm1ode_estimation

        pkg = Path(pyadm1ode_estimation.__file__).resolve().parent
        out.append(pkg.parent / "datasets" / "benchmark")
    except Exception:  # noqa: BLE001, S110 - a missing package is handled by the caller
        pass

    # Sibling checkout: <parent>/PyADM1ODE_calibration and
    # <parent>/PyADM1ODE_estimate next to each other.
    repo_root = _HERE.parent.parent
    out.append(repo_root.parent / "PyADM1ODE_estimate" / "datasets" / "benchmark")
    return out


def benchmark_dir() -> Path:
    """The benchmark dataset directory.

    Raises:
        FileNotFoundError: If no candidate holds the dataset, listing every path
            that was tried so the fix is obvious.
    """
    tried = _candidates()
    for path in tried:
        if _is_benchmark(path):
            return path.resolve()
    raise FileNotFoundError(
        "Benchmark-Datensatz nicht gefunden. Gesucht in:\n  "
        + "\n  ".join(str(p) for p in tried)
        + f"\n\nSetze {ENV_DATASET} auf das Verzeichnis mit "
        + ", ".join(_REQUIRED)
        + " (in PyADM1ODE_estimate: datasets/benchmark)."
    )


def add_dataset_to_path() -> Path:
    """Make ``loader`` and ``fostac`` importable, and return the dataset dir.

    Both modules are plain scripts inside the dataset directory rather than an
    installable package, so they are reached by path rather than by import name.
    """
    path = benchmark_dir()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def use_dataset(path: str | os.PathLike[str] | None) -> None:
    """Pin the dataset directory for this process *and its workers*.

    Called from the CLI flags. Writing to the environment rather than to a
    module global is deliberate: spawned pool workers inherit the environment
    but not the global.
    """
    if path is not None:
        os.environ[ENV_DATASET] = str(Path(path).expanduser().resolve())


def results_dir() -> Path:
    """Where sweeps write their JSONL and the analyses read it from.

    Defaults to ``results/`` beside this file. The published results of the
    study are *not* stored here — they live with the report in
    ``PyADM1ODE_estimate``; point :data:`ENV_RESULTS` there to reproduce its
    tables without recomputing anything.
    """
    env = os.environ.get(ENV_RESULTS)
    if env:
        return Path(env).expanduser().resolve()
    return _HERE / "results"


def use_results(path: str | os.PathLike[str] | None) -> None:
    """Pin the results directory for this process and its workers."""
    if path is not None:
        os.environ[ENV_RESULTS] = str(Path(path).expanduser().resolve())


def add_results_argument(parser) -> None:
    """Add the two location flags every entry point shares."""
    parser.add_argument(
        "--results",
        default=None,
        help="Verzeichnis fuer die JSONL-/JSON-Ergebnisse " f"(Standard: ./results, oder ${ENV_RESULTS}).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Verzeichnis des Benchmark-Datensatzes " f"(Standard: automatisch, oder ${ENV_DATASET}).",
    )


def apply_location_args(args) -> None:
    """Honour ``--results`` / ``--dataset`` before anything reads them."""
    use_results(getattr(args, "results", None))
    use_dataset(getattr(args, "dataset", None))


if __name__ == "__main__":
    print(f"Datensatz : {benchmark_dir()}")
    print(f"Ergebnisse: {results_dir()}")
