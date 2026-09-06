# Getting Started

## Prerequisites

- **Python** 3.10 or newer.  
- **[PyADM1ODE](https://github.com/dgaida/PyADM1ODE)**, which provides the plant model this package calibrates.  
- **Measurement data** as a time series, from CSV or a database. Useful channels are gas flow, methane content, pH and VFA.  

## Install

```bash
pip install pyadm1ode-calibration
```

For development, see [Installation](installation.md).

## Core concepts

**`MeasurementData`** wraps one time-indexed table. Every calibration reads from it, and it also does the pre-processing: `remove_outliers`, `fill_gaps`, `get_time_window`.

**`InitialCalibrator`** fits a fresh model to a historical window. It splits the data into training and validation, runs a global optimizer and reports both fits.

**`OnlineCalibrator`** keeps a running model on track. `should_recalibrate` returns a decision plus its reason, and `calibrate` caps how far a single update may move a parameter.

**`Calibrator`** is a thin facade over both. It and `OnlineCalibrator` can write a result back into the plant with `apply_calibration`, `InitialCalibrator` cannot.

**Objectives** are the measured channels the fit is scored against. Several can be combined with weights, for example 80 % on gas flow and 20 % on pH.

## Next steps

- Work through the notebooks, starting at [Tutorials](tutorials/index.md).  
- Look up parameters and optimizers in [Configuration](configuration.md).  
- See both workflows in code under [Usage](usage/index.md).  
