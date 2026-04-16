# Usage

Welcome to the usage guide for PyADM1ODE_calibration. Here you will learn how to integrate the core functionalities of the package into your workflow.

## Typical Workflows

### 1. Initial Calibration
Used to fit a new plant model to historical data.
[Learn more](calibration.md#initial-calibration)

### 2. Online Monitoring & Re-Calibration
Continuous monitoring of model quality and automatic parameter adjustment.
[Learn more](calibration.md#online-re-calibration)

### 3. Data Management
Efficient loading, validation, and storage of measurements.
[API Reference for IO](../api/io.md)

## Code Structure

- **`calibration`**: Contains the optimization logic and calibrators.
- **`io`**: Handles data import (CSV, database) and validation.
- **`optimization`**: Implements various optimization algorithms.

## Examples

Practical examples can be found in the [Examples section](../examples/index.md) or in the interactive [Tutorials](../tutorials/index.md).
