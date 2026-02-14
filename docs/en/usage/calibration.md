# Model Calibration Guide

This guide covers parameter calibration in PyADM1ODE, including initial calibration from historical data and online re-calibration during plant operation.

## Overview

Model calibration is essential for accurate biogas plant simulation. PyADM1ODE provides:

- **Initial Calibration**: Batch optimization from historical measurement data
- **Online Re-calibration**: Adaptive parameter adjustment during operation
- **Sensitivity Analysis**: Identification of influential parameters
- **Identifiability Assessment**: Detection of over-parameterization
- **Validation Tools**: Goodness-of-fit metrics and residual analysis

## Quick Start

```python
from pyadm1ode_calibration.calibration import Calibrator
from pyadm1ode_calibration.io import MeasurementData
# Assuming plant and feedstock are available from pyadm1ode
from pyadm1ode import BiogasPlant

# Load plant and measurements
plant = BiogasPlant.from_json("plant.json", feedstock)
measurements = MeasurementData.from_csv("plant_data.csv")

# Create calibrator
calibrator = Calibrator(plant)

# Calibrate parameters
result = calibrator.run_initial_calibration(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch", "Y_su"],
    objectives=["Q_ch4", "pH"],
    weights={"Q_ch4": 0.8, "pH": 0.2}
)

# Apply calibrated parameters
if result.success:
    calibrator.apply_calibration(result)
```

## Initial Calibration

The initial calibration is typically performed using 7-30 days of stable plant operation data.

### When to Use Initial Calibration

Use initial calibration when you have:
- Historical measurement data (≥2 weeks recommended)
- Stable plant operation during measurement period
- Reliable measurements of key outputs (gas production, pH, VFA)
- Known substrate feed rates and composition

### Parameter Selection

Choose parameters based on:

**High Priority** (most sensitive to calibration):
- `k_dis`: Disintegration rate [1/d]
- `k_hyd_ch`, `k_hyd_pr`, `k_hyd_li`: Hydrolysis rates [1/d]
- `Y_su`, `Y_aa`: Yield coefficients [kg COD/kg COD]

**Medium Priority**:
- `k_m_c4`, `k_m_pro`, `k_m_ac`, `k_m_h2`: Maximum uptake rates [1/d]
- `K_S_su`, `K_S_aa`: Half-saturation constants [kg COD/m³]

**Low Priority** (usually well-defined):
- Stoichiometric coefficients (C, N content)
- Physical-chemical constants (K_a, K_H)

## Online Re-Calibration

Online re-calibration allows for continuous adjustment of parameters during operation to account for gradual changes in substrate properties or equipment drift.

### Using the Online Calibrator

```python
# Re-calibrate with bounded changes
result = calibrator.run_online_calibration(
    measurements=recent_data,
    parameters=["k_dis", "Y_su"],
    max_parameter_change=0.10,  # Max 10% change
    time_window=7,               # Use last 7 days
    method="nelder_mead"         # Fast local optimization
)

if result.success:
    calibrator.apply_calibration(result)
```

### When to Use Online Re-Calibration

Appropriate for:
- Long-term operation (months)
- Gradual substrate property changes
- Seasonal variations
- Equipment drift

## Validation and Quality Checks

### Goodness-of-Fit Metrics

The `CalibrationValidator` provides comprehensive metrics to assess the quality of the calibration.

| Metric                          | Excellent | Good | Fair | Poor |
|---------------------------------|-----------|------|------|------|
| R²                              | > 0.90 | 0.75-0.90 | 0.50-0.75 | < 0.50 |
| NSE (Nash-Sutcliffe Efficiency) | > 0.90 | 0.70-0.90 | 0.50-0.70 | < 0.50 |
| PBIAS (Percent Bias)            | < ±5% | ±5-±10% | ±10-±25% | > ±25% |

## References

1. **Batstone et al. (2002)**: Anaerobic Digestion Model No. 1 (ADM1). IWA Publishing.
2. **Gaida (2014)**: Dynamic real-time substrate feed optimization of anaerobic co-digestion plants. PhD thesis.
3. **Dochain & Vanrolleghem (2001)**: Dynamical Modelling & Estimation in Wastewater Treatment Processes.
