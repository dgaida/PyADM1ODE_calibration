# PyADM1ODE_calibration

> **Note**: This is a calibration-focused package that only works together with [PyADM1ODE](https://github.com/dgaida/PyADM1ODE).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/github/v/tag/dgaida/PyADM1ODE_calibration?label=version)](https://github.com/dgaida/PyADM1ODE_calibration/tags)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/dgaida/PyADM1ODE_calibration/branch/main/graph/badge.svg)](https://codecov.io/gh/dgaida/PyADM1ODE_calibration)
[![Code Quality](https://github.com/dgaida/PyADM1ODE_calibration/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE_calibration/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/PyADM1ODE_calibration/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE_calibration/actions/workflows/tests.yml)
[![CodeQL](https://github.com/dgaida/PyADM1ODE_calibration/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE_calibration/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

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

## Key Features

### 🎯 Calibration Methods

- **Initial calibration** from 7-30 days of stable operation data
- **Online re-calibration** with variance-based triggering
- **Bounded parameter updates** to prevent unrealistic drift
- **Substrate-dependent parameters** (k_dis, k_hyd_*, k_m_*)
- **Yield coefficients and kinetic parameters** (Y_*, K_S_*)

### 🔧 Optimization Algorithms

| Algorithm | Type | Best For | Speed |
|-----------|------|----------|-------|
| Differential Evolution | Global | Initial calibration | Slow ⭐⭐ |
| Particle Swarm | Global | Alternative to DE | Medium ⭐⭐⭐ |
| Nelder-Mead | Local | Online re-calibration | Fast ⭐⭐⭐⭐ |
| L-BFGS-B | Gradient | Smooth problems | Very Fast ⭐⭐⭐⭐⭐ |

### 📊 Validation & Analysis

- **Goodness-of-fit**: RMSE, MAE, R², Nash-Sutcliffe Efficiency, PBIAS
- **Residual analysis**: Normality tests, autocorrelation, heteroscedasticity
- **Sensitivity analysis**: Local gradients, normalized indices, variance contribution
- **Identifiability**: Correlation matrices, confidence intervals, VIF
- **Cross-validation**: k-fold validation for generalization assessment

### 💾 Data Management

- **Measurement data**: CSV/PostgreSQL import with validation
- **Outlier detection**: Z-score, IQR, moving window methods
- **Gap filling**: Interpolation, forward/backward fill, mean/median
- **Data validation**: Range checks, missing data analysis, quality scoring
- **Database storage**: Plant configurations, calibration history, simulation results

## Installation

### Prerequisites

- Python 3.10 or higher
- [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) base package

### From GitHub

```bash
# Clone repository
git clone https://github.com/dgaida/PyADM1ODE_calibration.git
cd PyADM1ODE_calibration

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev,docs]"
```

### Using Conda

```bash
# Create environment
conda env create -f environment.yml
conda activate biogas

# Install package
pip install -e .
```

## Quick Start

### Basic Calibration Workflow

```python
from pyadm1ode_calibration.calibration import InitialCalibrator
from pyadm1ode_calibration.io import MeasurementData
from pyadm1ode import BiogasPlant

# 1. Load plant model
plant = BiogasPlant.from_json("plant.json", feedstock)

# 2. Load and validate measurement data
measurements = MeasurementData.from_csv(
    "plant_measurements.csv",
    timestamp_column="timestamp",
    resample="1H"
)

# Validate data quality
validation = measurements.validate()
print(f"Data quality: {validation.quality_score:.2f}")

# Clean data
measurements.remove_outliers(method="zscore", threshold=3.0)
measurements.fill_gaps(method="interpolate", limit=3)

# 3. Create calibrator and run calibration
calibrator = InitialCalibrator(plant, verbose=True)

result = calibrator.calibrate(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch", "Y_su"],
    bounds={"k_dis": (0.3, 0.8), "Y_su": (0.05, 0.15)},
    objectives=["Q_ch4", "pH"],
    weights={"Q_ch4": 0.8, "pH": 0.2},
    method="differential_evolution",
    validation_split=0.2,
    max_iterations=100
)

# 4. Check results
if result.success:
    print(f"Calibration successful!")
    print(f"Objective value: {result.objective_value:.4f}")
    print(f"Calibrated parameters:")
    for param, value in result.parameters.items():
        initial = result.initial_parameters[param]
        change = (value - initial) / initial * 100
        print(f"  {param}: {initial:.4f} → {value:.4f} ({change:+.1f}%)")

    # Apply to plant
    calibrator.apply_calibration(result)
```

### Online Re-Calibration

```python
from pyadm1ode_calibration.calibration import OnlineCalibrator

# Create online calibrator
calibrator = OnlineCalibrator(plant, verbose=True)

# Configure trigger conditions
calibrator.set_trigger(
    variance_threshold=0.15,      # Trigger at 15% variance
    time_threshold=24.0,          # Min 24h between calibrations
    consecutive_violations=3      # Require 3 consecutive violations
)

# Monitor and re-calibrate when needed
recent_data = MeasurementData.from_csv("recent_measurements.csv")

if calibrator.should_recalibrate(recent_data):
    result = calibrator.calibrate(
        measurements=recent_data,
        parameters=["k_dis", "Y_su"],
        max_parameter_change=0.10,  # Max 10% change
        time_window=7,               # Use last 7 days
        method="nelder_mead"
    )

    if result.success:
        calibrator.apply_calibration(result)
```

### Database Integration

```python
from pyadm1ode_calibration.io import Database

# Connect to PostgreSQL database
db = Database("postgresql://user:pass@localhost/biogas")
db.create_all_tables()

# Store measurement data
db.store_measurements(
    plant_id="plant1",
    data=measurements_df,
    source="SCADA"
)

# Load measurement data
data = db.load_measurements(
    plant_id="plant1",
    start_time="2024-01-01",
    end_time="2024-01-31"
)

# Store calibration results
db.store_calibration(
    plant_id="plant1",
    calibration_type="initial",
    method="differential_evolution",
    parameters=result.parameters,
    objective_value=result.objective_value,
    objectives=["Q_ch4", "pH"],
    validation_metrics=result.validation_metrics,
    success=result.success
)
```

## Project Structure

```
PyADM1ODE_calibration/
├── pyadm1ode_calibration/
│   ├── calibration/                 # Calibration framework
│   │   ├── __init__.py
│   │   ├── calibrator.py           # Main Calibrator class
│   │   ├── initial.py              # Initial calibration
│   │   ├── online.py               # Online re-calibration
│   │   ├── parameter_bounds.py     # Parameter bounds management
│   │   ├── validation.py           # Calibration validation
│   │   └── optimization/           # Optimization algorithms
│   │       ├── __init__.py
│   │       ├── optimizer.py        # Optimizer base classes
│   │       ├── objective.py        # Objective functions
│   │       └── constraints.py      # Parameter constraints
│   │
│   └── io/                         # Data input/output
│       ├── __init__.py
│       ├── csv_handler.py          # CSV import/export
│       ├── database.py             # PostgreSQL database interface
│       └── measurement_data.py     # Measurement data management
│
├── tests/                          # Test suite
│   ├── unit/
│   │   ├── test_calibration/
│   │   └── test_io/
│   ├── integration/
│   └── validation/
│
├── scripts/                        # Utility scripts
│   ├── calibration_example.py     # Complete example workflow
│   ├── generate_measurement_data.py
│   └── test_calibration.py
│
├── data/                           # Example data
│   ├── initial_states/
│   └── plant_measurements_csv.md
│
├── docs/                           # Documentation
│   ├── user_guide/
│   │   └── calibration.md
│   └── examples/
│
├── .github/
│   └── workflows/                  # GitHub Actions CI/CD
│
├── README.md
├── pyproject.toml
├── requirements.txt
└── environment.yml
```

## Documentation

### User Guide

- [Calibration Guide](docs/user_guide/calibration.md) - Complete calibration workflow
- [Example Workflow](docs/examples/calibration_workflow.md) - Step-by-step tutorial
- [API Reference](docs/api_reference/) - Detailed API documentation

### Example Scripts

- [`scripts/calibration_example.py`](scripts/calibration_example.py) - Complete calibration workflow
- [`scripts/generate_measurement_data.py`](scripts/generate_measurement_data.py) - Generate synthetic test data
- [`scripts/test_calibration.py`](scripts/test_calibration.py) - Test parameter bounds and data loading

## Calibration Parameters

### Substrate-Dependent Parameters

Most commonly calibrated for different substrates:

| Parameter | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| `k_dis` | Disintegration rate | 1/d | 0.3 - 0.8 |
| `k_hyd_ch` | Carbohydrate hydrolysis | 1/d | 5.0 - 15.0 |
| `k_hyd_pr` | Protein hydrolysis | 1/d | 5.0 - 15.0 |
| `k_hyd_li` | Lipid hydrolysis | 1/d | 5.0 - 15.0 |

### Yield Coefficients

| Parameter | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| `Y_su` | Sugar degrader yield | kg COD/kg COD | 0.05 - 0.15 |
| `Y_aa` | Amino acid degrader yield | kg COD/kg COD | 0.04 - 0.12 |
| `Y_fa` | LCFA degrader yield | kg COD/kg COD | 0.03 - 0.10 |

### Maximum Uptake Rates

| Parameter | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| `k_m_c4` | C4 uptake rate | 1/d | 15.0 - 30.0 |
| `k_m_pro` | Propionate uptake rate | 1/d | 8.0 - 18.0 |
| `k_m_ac` | Acetate uptake rate | 1/d | 4.0 - 12.0 |
| `k_m_h2` | Hydrogen uptake rate | 1/d | 25.0 - 45.0 |

## Data Requirements

### Minimum Requirements

- **Duration**: 7-30 days of stable operation
- **Frequency**: Hourly measurements (daily for lab analyses)
- **Quality**: < 10% missing data, validated ranges

### Required Measurements

| Measurement | Unit | Critical | Frequency |
|-------------|------|----------|-----------|
| Biogas production (Q_gas) | m³/d | ✓ | Hourly |
| Methane production (Q_ch4) | m³/d | ✓ | Hourly |
| pH | - | ✓ | Hourly |
| VFA | g HAc eq/L | ✓ | Daily |
| TAC | g CaCO₃ eq/L | | Daily |
| Temperature | K | ✓ | Hourly |
| Substrate feeds | m³/d | ✓ | Continuous |

### Optional Measurements

- CO₂ production (Q_co2)
- FOS/TAC ratio
- NH₄-N concentration
- Electrical power (P_el)
- Thermal power (P_th)

## Validation Metrics

### Goodness-of-Fit

| Metric | Excellent | Good | Fair | Poor |
|--------|-----------|------|------|------|
| **R²** | > 0.90 | 0.75-0.90 | 0.50-0.75 | < 0.50 |
| **NSE** | > 0.90 | 0.70-0.90 | 0.50-0.70 | < 0.50 |
| **PBIAS** | < ±5% | ±5-±10% | ±10-±25% | > ±25% |

### Interpretation

- **R² > 0.75**: Good model fit, parameters reliable
- **PBIAS < 10%**: Low systematic bias
- **NSE > 0.70**: Model better than mean prediction

## Performance

### Typical Calibration Times

| Configuration | Time | Notes |
|---------------|------|-------|
| 3 parameters, DE, 100 iter | 5-10 min | Initial calibration |
| 5 parameters, DE, 200 iter | 20-30 min | Comprehensive calibration |
| 2 parameters, NM, 50 iter | 1-2 min | Online re-calibration |

*Times based on 30 days of hourly data on standard desktop PC*

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/test_calibration/ -v
pytest tests/unit/test_io/ -v

# Run with coverage
pytest tests/ --cov=pyadm1ode_calibration --cov-report=html

# Run only fast tests
pytest tests/ -v -m "not slow"
```

## Contributing

We welcome contributions! Areas where we need help:

- Additional optimization algorithms
- More sophisticated online calibration strategies
- Bayesian calibration with uncertainty quantification
- Real plant validation data
- Performance optimization
- Documentation improvements

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use PyADM1ODE_calibration in your research, please cite:

```bibtex
@software{pyadm1_calibration,
  author = {Gaida, Daniel},
  title = {PyADM1ODE\_calibration: Parameter Calibration Framework for Biogas Plant Models},
  year = {2026},
  url = {https://github.com/dgaida/PyADM1ODE_calibration}
}

@phdthesis{gaida2014dynamic,
  title={Dynamic real-time substrate feed optimization of anaerobic co-digestion plants},
  author={Gaida, Daniel},
  year={2014},
  school={Universiteit Leiden}
}
```

## Related Publications

- **Gaida, D. (2014).** *Dynamic real-time substrate feed optimization of anaerobic co-digestion plants.* PhD thesis, Leiden University.
- **Batstone, D.J., et al. (2002).** *Anaerobic Digestion Model No. 1 (ADM1).* IWA Publishing, London.
- **Dochain, D. & Vanrolleghem, P. (2001).** *Dynamical Modelling & Estimation in Wastewater Treatment Processes.* IWA Publishing.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- ADM1 development by IWA Task Group
- SIMBA# implementation by ifak e.V.

## Contact

**Daniel Gaida**
- Email: daniel.gaida@th-koeln.de  
- GitHub: [@dgaida](https://github.com/dgaida)  
- Institution: TH Köln - University of Applied Sciences

---

**Note**: This package requires [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) to be installed.
