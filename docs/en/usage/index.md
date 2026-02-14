# Usage

This section contains detailed guides on how to use PyADM1ODE_calibration.

## Core Workflows

- [Batch Calibration](calibration.md): Initial calibration based on historical data.
- [Online Re-Calibration](calibration.md#online-re-calibration): Dynamic adjustment during operation.

## Examples

- [Complete Workflow](../examples/calibration_workflow.md): A step-by-step example from data preparation to results analysis.

## Data Formats

The framework expects measurement data as time series. Ensure your CSV files contain a `timestamp` column and corresponding columns for the objectives (e.g., `Q_ch4`, `pH`).
