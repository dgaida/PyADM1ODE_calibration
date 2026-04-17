# Calibration

Calibration is the core of this framework. It is divided into initial calibration and online re-calibration.

## Initial Calibration

Initial calibration is usually performed once when historical data is available for a period (e.g., 30 days).

```python
from pyadm1ode_calibration.calibration import InitialCalibrator
from pyadm1ode_calibration.io.loaders import MeasurementData

# 1. Load measurement data
measurements = MeasurementData.from_csv("hist_data.csv")

# 2. Initialize calibrator
# plant is an instance of pyadm1ode.Plant
calibrator = InitialCalibrator(plant)

# 3. Start calibration
result = calibrator.calibrate(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch", "Y_su"],
    method="differential_evolution",
    validation_split=0.2
)

# 4. Evaluate results
print(f"Best fit: {result.objective_value}")
print(f"Calibrated parameters: {result.parameters}")
```

## Online Re-Calibration

Online re-calibration is used to adjust the model during ongoing operation if the prediction quality decreases.

```python
from pyadm1ode_calibration.calibration import OnlineCalibrator

calibrator = OnlineCalibrator(plant)

# Define trigger conditions
calibrator.set_trigger(
    variance_threshold=0.15,  # 15% deviation
    time_threshold=24.0       # Minimum 24h interval
)

# Check if re-calibration is necessary
if calibrator.should_recalibrate(recent_data):
    result = calibrator.calibrate(
        measurements=recent_data,
        method="nelder-mead"
    )
    if result.success:
        calibrator.apply_calibration(result)
```

## Validation

After each calibration, a validation is automatically performed (if a `validation_split` was specified). You can find the results in `result.validation_metrics`:

- **RMSE**: Root Mean Square Error  
- **R²**: Coefficient of Determination  
- **NSE**: Nash-Sutcliffe Efficiency  
