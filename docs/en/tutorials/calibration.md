# Calibration Tutorial

This tutorial describes the basic steps for calibrating an ADM1 model.

## 1. Data Preparation

Measurement data must be available as time series. The `MeasurementData` object expects column names corresponding to ADM1 state variables.

```python
from pyadm1ode_calibration.io.loaders import MeasurementData
measurements = MeasurementData.from_csv("data.csv")
```

## 2. Initializing the Calibrator

The `InitialCalibrator` requires an instance of the plant model.

```python
from pyadm1ode_calibration.calibration import InitialCalibrator
calibrator = InitialCalibrator(plant)
```

## 3. Running the Calibration

Select the parameters to be optimized. Typically, these are hydrolysis constants or yield coefficients.

```python
result = calibrator.calibrate(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch"],
    method="differential_evolution"
)
```

## 4. Evaluation

Check `result.success` and the optimized values in `result.parameters`.
