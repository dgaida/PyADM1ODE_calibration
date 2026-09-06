# Calibration Tutorial

The short version of notebooks 01 to 03.

## 1. Load and clean the data

```python
from pyadm1ode_calibration import MeasurementData

measurements = MeasurementData.from_csv("data.csv")
measurements.remove_outliers(method="zscore", threshold=3.0)
measurements.fill_gaps(method="interpolate", limit=3)
```

The index must be a timestamp, and the columns carry the channel names the objectives refer to
(`Q_gas`, `Q_ch4`, `pH`, ...).

## 2. Pick the parameters

Fit few of them. Start with the hydrolysis rate of the dominant substrate fraction, since it is what
most directly moves gas production. [Configuration](../configuration.md) lists the usual candidates
with their bounds.

## 3. Fit

```python
from pyadm1ode_calibration import InitialCalibrator

calibrator = InitialCalibrator(plant)
result = calibrator.calibrate(
    measurements=measurements,
    parameters=["k_hyd_ch"],
    objectives=["Q_gas"],
    validation_split=0.2,
)
```

## 4. Check the result

`result.success` only says the optimizer terminated. What matters is whether the error on the
held-out part dropped as well: compare `result.validation_metrics` against the training metrics. A
large gap between the two means the fit followed noise, not the plant.
