# Getting Started

This guide helps you with the first steps using **PyADM1ODE_calibration**.

## Prerequisites

Before you begin, ensure your environment meets the following requirements:

- **Python**: 3.10 or higher.
- **PyADM1ODE**: The base package for biogas plant simulation.
- **Data**: Historical plant measurement data (e.g., CH₄ production, pH value) in CSV format or in a PostgreSQL database.

## Installation

Install the package directly via pip:

```bash
pip install pyadm1ode-calibration
```

Or for development:

```bash
git clone https://github.com/dgaida/PyADM1ODE_calibration.git
cd PyADM1ODE_calibration
pip install -e ".[dev]"
```

## Core Concepts

### 1. Data Loading (`MeasurementData`)
All calibrations are based on the `MeasurementData` object. It manages time series of measurements and provides functions for validation and pre-processing.

### 2. Calibrators
- **InitialCalibrator**: Used for the initial tuning of the model to a historical dataset (batch optimization).
- **OnlineCalibrator**: Enables continuous adjustment of parameters during ongoing operation to respond to changes in substrate quality or biology.

### 3. Objective Functions
You can define and weight multiple target variables (objectives), e.g., 80% weight on methane production and 20% on the pH value.

## Next Steps

- Follow the [Tutorial for Initial Calibration](tutorials/calibration.md).
- Learn more about [Parameter Configuration](configuration.md).
- Check the [API Reference](api/index.md) for detailed information.
