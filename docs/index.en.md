# PyADM1ODE_calibration

> **Note**: This is a calibration-focused package that only works together with [PyADM1ODE](https://github.com/dgaida/PyADM1ODE).

**Advanced parameter calibration framework for PyADM1ODE biogas plant models**

Automated calibration and re-calibration of Anaerobic Digestion Model No. 1 (ADM1) parameters using real plant measurement data with multiple optimization algorithms, comprehensive validation, and online adaptation capabilities.

## Overview

PyADM1ODE_calibration provides a complete calibration framework for [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) biogas plant models:

- **Initial Calibration**: Batch optimization from historical measurement data
- **Online Re-Calibration**: Real-time parameter adjustment during plant operation
- **Multiple Optimization Algorithms**: Differential Evolution, Nelder-Mead, L-BFGS-B, Particle Swarm
- **Multi-Objective Optimization**: Balance multiple outputs (CH₄, pH, VFA) with weighted objectives
- **Comprehensive Validation**: Goodness-of-fit metrics, residual analysis, cross-validation
- **Parameter Identifiability**: Sensitivity analysis and correlation detection
- **Data Management**: CSV/database import, validation, outlier detection, gap filling
