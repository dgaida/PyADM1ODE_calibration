# PyADM1ODE Calibration

[![PyPI version](https://img.shields.io/pypi/v/pyadm1ode-calibration.svg)](https://pypi.org/project/pyadm1ode-calibration/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyadm1ode-calibration.svg)](https://pypi.org/project/pyadm1ode-calibration/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://dgaida.github.io/PyADM1ODE_calibration/)
[![Interrogate](assets/interrogate.svg)](development/metrics.md)

**Advanced parameter calibration framework for PyADM1ODE biogas plant models.**

PyADM1ODE_calibration provides a complete solution for the calibration of [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) models. It enables precise tuning of complex ADM1 parameters to real plant data using state-of-the-most optimization techniques.

## Key Features

- 🎯 **Precision**: Highly accurate matching of ADM1 parameters to real plant data.
- ⚡ **Efficiency**: Fast local optimizers for online use and robust global optimizers for initial calibration.
- 📊 **Analysis**: Integrated sensitivity and identifiability analysis to identify critical parameters.
- 💾 **Integration**: Seamless connection to PostgreSQL databases and CSV workflows.
- 🌍 **Multilingual**: Documentation available in German and English.

## Table of Contents

- [Getting Started](getting-started.md) — Quick start with the project.
- [Installation](installation.md) — Installation guides for various environments.
- [Configuration](configuration.md) — Overview of configuration options and parameters.
- [Tutorials](tutorials/index.md) — Step-by-step guides (also for Google Colab).
- [API Reference](api/index.md) — Detailed documentation of classes and functions.

## Quickstart

```python
from pyadm1ode_calibration.calibration import InitialCalibrator
from pyadm1ode_calibration.io.loaders import MeasurementData

# 1. Load data
measurements = MeasurementData.from_csv("plant_data.csv")

# 2. Create calibrator
calibrator = InitialCalibrator(plant_model)

# 3. Run calibration
result = calibrator.calibrate(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch"],
    objectives=["Q_ch4", "pH"]
)

# 4. Apply results
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
