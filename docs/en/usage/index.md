# Usage

## The two workflows

**Initial calibration** fits a new model to a historical window, once. Global optimizer, train/test
split, sensitivity analysis. See [Calibration](calibration.md#initial-calibration).

**Online recalibration** keeps a running model on track. Local optimizer, capped step size, variance
trigger. See [Calibration](calibration.md#online-recalibration).

## Package layout

| Module | Contents |
|--------|----------|
| `pyadm1ode_calibration.calibration` | `Calibrator`, `InitialCalibrator`, `OnlineCalibrator`, `CalibrationResult`, bounds |
| `pyadm1ode_calibration.calibration.optimization` | optimizers, objective functions, constraints |
| `pyadm1ode_calibration.calibration.analysis` | sensitivity and identifiability |
| `pyadm1ode_calibration.io.loaders` | `MeasurementData`, `CSVHandler`, `PlantSchema`, `MeasurementBuilder` |
| `pyadm1ode_calibration.io.persistence` | `Database`, ORM models, repositories |
| `pyadm1ode_calibration.io.validation` | `DataValidator`, `OutlierDetector` |
| `pyadm1ode_calibration.plants` | plant topology builders |

The most-used names are re-exported at the top level, so
`from pyadm1ode_calibration import Calibrator, MeasurementData` works.

## Where to look next

- [Tutorials](../tutorials/index.md) for the notebook series.  
- [Examples](../examples/index.md) for runnable scripts.  
- [API Reference](../api/index.md) for signatures.  
