# PyADM1ODE Calibration

[![PyPI version](https://img.shields.io/pypi/v/pyadm1ode-calibration.svg)](https://pypi.org/project/pyadm1ode-calibration/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyadm1ode-calibration.svg)](https://pypi.org/project/pyadm1ode-calibration/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://dgaida.github.io/PyADM1ODE_calibration/)
[![Interrogate](assets/interrogate.svg)](development/metrics.md)
[![Open In Colab](assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/PyADM1ODE_calibration/blob/main/notebooks/00_quickstart.ipynb)

**Parameter calibration for [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) biogas plant models.**

Fits ADM1 parameters to measured plant data, either once from a historical window or continuously during operation.

## What it does

- **Initial calibration**: global search over a historical window, with a train/test split.
- **Online recalibration**: bounded parameter updates, triggered by prediction variance.
- **Analysis**: sensitivity and identifiability, so you can tell which parameters the data actually resolves.
- **Data**: CSV and database sources behind one schema file, plus validation, outlier removal and gap filling.

## Contents

| Page | Content |
|------|---------|
| [Getting Started](getting-started.md) | Prerequisites and the core concepts |
| [Installation](installation.md) | pip, conda, Docker |
| [Configuration](configuration.md) | Parameters, bounds, optimizers, database |
| [Usage](usage/index.md) | The two workflows in code |
| [Tutorials](tutorials/index.md) | Notebook series 00 to 06 |
| [Examples](examples/index.md) | Runnable scripts |
| [API Reference](api/index.md) | Classes and functions |
| [Architecture](architecture/index.md) | Modules and data flow |

## Quickstart

```python
from pyadm1ode_calibration import Calibrator, MeasurementData

measurements = MeasurementData.from_csv("plant_data.csv")
calibrator = Calibrator(plant)          # plant: a pyadm1 BiogasPlant

result = calibrator.run_initial_calibration(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch"],
    objectives=["Q_ch4", "pH"],
)

if result.success:
    calibrator.apply_calibration(result)
```

## Citation

If you use PyADM1ODE_calibration in your research, please cite:

```bibtex
@software{pyadm1_calibration,
  author = {Gaida, Daniel},
  title = {PyADM1ODE\_calibration: Parameter Calibration Framework for Biogas Plant Models},
  year = {2026},
  url = {https://github.com/dgaida/PyADM1ODE_calibration}
}
```
