# PyADM1ODE Calibration

**Advanced parameter calibration framework for PyADM1ODE biogas plant models.**

PyADM1ODE_calibration provides a complete solution for the calibration of [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) models:

- **Initial Calibration**: Batch optimization based on historical measurement data.
- **Online Re-Calibration**: Real-time parameter adjustment during plant operation.
- **Multiple Optimization Algorithms**: Differential Evolution, Nelder-Mead, L-BFGS-B, Particle Swarm.
- **Multi-Objective Optimization**: Balance multiple outputs (CH₄, pH, VFA) with weighted objectives.
- **Comprehensive Validation**: Goodness-of-fit metrics, residual analysis, cross-validation.
- **Data Management**: CSV/database import, validation, outlier detection, gap filling.

## Key Features

- 🎯 **Precision**: Highly accurate matching of ADM1 parameters to real plant data.
- ⚡ **Efficiency**: Fast local optimizers for online use.
- 📊 **Analysis**: Integrated sensitivity and identifiability analysis.
- 💾 **Integration**: Seamless connection to PostgreSQL databases and CSV workflows.

## Getting Started

Start with the [Installation](getting-started.md) and follow our [Quickstart Guide](usage/index.md).
