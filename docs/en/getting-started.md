# Getting Started

This guide will help you get started with PyADM1ODE_calibration.

## Prerequisites

- **Python 3.10** or higher.
- **PyADM1ODE**: The base package must be installed.
- **Data**: Historical biogas plant measurement data in CSV format (recommended).

## Installation

The easiest way is to install via pip directly from the repository:

```bash
pip install git+https://github.com/dgaida/PyADM1ODE_calibration.git
```

For developers, we recommend installing in editable mode:

```bash
git clone https://github.com/dgaida/PyADM1ODE_calibration.git
cd PyADM1ODE_calibration
pip install -e .
```

## Quick Overview

A typical calibration workflow consists of four steps:

1. **Load Data**: Import and validate your measurement data.
2. **Prepare Model**: Configuration of your PyADM1ODE plant model.
3. **Run Calibration**: Choose an algorithm and start the optimization.
4. **Verify Results**: Validate parameters using goodness-of-fit metrics.

Check the [Calibration Example](usage/calibration.md) for a detailed code walkthrough.
