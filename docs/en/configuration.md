# Configuration

## Parameters and bounds

`create_default_bounds()` returns bounds for 41 ADM1 parameters. A **soft** bound is a weighted penalty the optimizer may cross, a **hard** bound makes the candidate infeasible.

| Parameter | Meaning | Unit | Bounds | Default | Type |
|-----------|---------|------|--------|---------|------|
| `k_dis` | Disintegration rate | 1/d | 0.1 - 1.0 | 0.5 | soft |
| `k_hyd_ch` | Hydrolysis, carbohydrates | 1/d | 1.0 - 15.0 | 4.0 | soft |
| `k_hyd_pr` | Hydrolysis, proteins | 1/d | 1.0 - 15.0 | 4.0 | soft |
| `k_hyd_li` | Hydrolysis, lipids | 1/d | 1.0 - 15.0 | 4.0 | soft |
| `k_m_ac` | Max. uptake rate, acetate | 1/d | 4.0 - 12.0 | 8.0 | soft |
| `Y_su` | Yield, sugar degraders | kg COD/kg COD | 0.05 - 0.15 | 0.1 | hard |

```python
from pyadm1ode_calibration import create_default_bounds

bounds = create_default_bounds()
print(bounds.get_bounds("k_dis"))
```

Which parameters belong in a run is a question the data answers, not the modeller.
`calibrate(..., check_identifiability=True)` screens the set before the optimizer starts and warns
when the data cannot resolve it, at the cost of `2 * len(parameters) + 1` simulations. See
[Calibration](usage/calibration.md#which-parameters-are-worth-fitting).

Pass narrower ranges per run instead of editing the defaults:

```python
result = calibrator.calibrate(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch"],
    bounds={"k_dis": (0.3, 0.8), "k_hyd_ch": (2.0, 8.0)},
)
```

## Optimizers

Pass one via `method=`. Hyphens, spaces and case are normalised, so `"Nelder-Mead"` and `"nelder_mead"` are the same.

| Name | Type | Use for |
|------|------|---------|
| `differential_evolution`, `de` | global | initial calibration (default) |
| `particle_swarm`, `pso` | global | alternative to DE |
| `nelder_mead`, `nm` | local | online recalibration (default there) |
| `powell` | local | derivative-free, no bounds needed |
| `l_bfgs_b` | gradient | smooth objectives |
| `slsqp` | gradient | smooth objectives with constraints |

Sizing matters more than the choice: differential evolution runs `population_size x len(parameters)`
simulations per generation. With three parameters, `population_size=10` and `max_iterations=50` that
came to 1217 simulations and 56 minutes, so budget roughly three seconds per simulated window.

## Objectives

```python
objectives = ["Q_ch4", "pH", "VFA"]
weights = {"Q_ch4": 0.7, "pH": 0.2, "VFA": 0.1}
```

Weights are normalised internally. A channel missing from the data is skipped rather than scored as zero.

## Data sources

A plant schema declares its sources and channels in one YAML file. Copy `configs/plants/_template.yaml` and adapt it:

```python
from pyadm1ode_calibration.io.loaders import MeasurementBuilder, PlantSchema

schema = PlantSchema.from_yaml("configs/plants/my_plant.yaml")
measurements = MeasurementBuilder(schema).build(variables=["Q_gas", "T_digester"])
```

## Feeds that change over time

By default a calibration collapses the whole substrate feed series of a window into
its mean and holds it constant. That is cheap and correct for a record at steady
load, but it hides every load step and substrate change, and those are exactly what
makes the kinetics identifiable. Pass `time_varying_feed=True` to apply the feeds
step by step:

```python
calibrator = InitialCalibrator(plant, time_varying_feed=True)
```

It costs roughly 20 % more run time, and it is available on `InitialCalibrator`,
`OnlineCalibrator`, `CalibrationValidator` and `PlantSimulator`. A window whose feed
really is constant takes the cheap path either way.

## Database

```python
from pyadm1ode_calibration import Database

db = Database(connection_string="postgresql://user:pw@localhost:5432/biogas")
db = Database.from_env()   # reads DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
```

Timestamps are stored as naive UTC. Use `pyadm1ode_calibration.timeutils.utc_now()` when writing your own rows.
