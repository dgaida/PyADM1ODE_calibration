# Calibration

## Initial calibration

Run once, on a historical window of stable operation. Seven to thirty days is a usual span.

```python
from pyadm1ode_calibration import InitialCalibrator, MeasurementData

measurements = MeasurementData.from_csv("hist_data.csv")
measurements.remove_outliers(method="zscore", threshold=3.0)
measurements.fill_gaps(method="interpolate", limit=3)

calibrator = InitialCalibrator(plant)
result = calibrator.calibrate(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch", "Y_su"],
    objectives=["Q_ch4", "pH"],
    weights={"Q_ch4": 0.8, "pH": 0.2},
    method="differential_evolution",
    validation_split=0.2,
    max_iterations=50,
    population_size=10,
)

print(result.success, result.parameters, result.validation_metrics)
```

`InitialCalibrator` does not write to the plant. Use the `Calibrator` facade for that, or apply the
values yourself.

### Which parameters are worth fitting

```python
sensitivity = calibrator.sensitivity_analysis(result.parameters, measurements)
identifiability = calibrator.identifiability_analysis(
    result.parameters, measurements, correlation_threshold=0.8
)
```

`identifiability_analysis` reports two distinct failures, after Raue et al. 2009:

- **structurally non-identifiable**: the parameter moves no measured output at all, so no amount of  
  data will pin it down. On the example plant `k_dis` is this case, because the substrates enter as
  hydrolysable fractions and disintegration never has a substrate.  
- **practically non-identifiable**: it does move the outputs, but so weakly that the confidence  
  interval stays wider than the estimate itself.

A set can also fail as a set. `analyze_subset` applies the two measures of Brun et al. 2001 in their
order: first each parameter alone, then the **collinearity index** over the survivors. It is 1 when
they act on the outputs in independent directions and grows without bound as they become
interchangeable, and the customary limit is 20.

```python
verdict = calibrator.identifiability_analyzer.analyze_subset(parameters, measurements)
print(verdict.collinearity_index, verdict.reason)
```

Cheaper than finding out afterwards: it costs `2 * len(parameters) + 1` simulations against the
several hundred a global search needs, so `calibrate(..., check_identifiability=True)` runs it first
and warns before the optimizer starts.

## Online recalibration

Trigger thresholds live on `calibrator.trigger`, the step limit is an argument of `calibrate`.

```python
from pyadm1ode_calibration import OnlineCalibrator

calibrator = OnlineCalibrator(plant)
calibrator.trigger.variance_threshold = 0.15     # 15 % relative deviation
calibrator.trigger.time_threshold = 24.0         # hours since the last run
calibrator.trigger.consecutive_violations = 2

needed, reason = calibrator.should_recalibrate(recent_data)
if needed:
    result = calibrator.calibrate(
        measurements=recent_data,
        parameters=["k_hyd_ch"],
        current_parameters={"k_hyd_ch": 2.0},
        max_parameter_change=0.20,               # at most 20 % per run
        method="nelder_mead",
    )
    if result.success:
        calibrator.apply_calibration(result)
```

`should_recalibrate` returns a pair, the decision and the reason for it. Testing the pair itself is
always true.

## Validation

With a `validation_split`, metrics for the held-out part land in `result.validation_metrics`:

- **RMSE**, **MAE**: absolute error  
- **R2**: coefficient of determination  
- **NSE**: Nash-Sutcliffe efficiency  
- **PBIAS**: systematic over- or underprediction  

`CalibrationValidator` adds residual diagnostics: normality, autocorrelation and heteroscedasticity.
