# Configuration

PyADM1ODE_calibration can be configured in several ways to adapt the calibration process to your plant.

## Parameter Bounds

Default bounds for ADM1 parameters are defined in `ParameterBounds`. You can adjust them globally or per calibration.

```python
from pyadm1ode_calibration.calibration import create_default_bounds

bounds = create_default_bounds()
bounds.add_bound("k_dis", lower=0.3, upper=0.8, default=0.5)
```

## Database Connection

The connection to the PostgreSQL database is established via environment variables or a configuration object.

### Environment Variables

| Variable | Description | Default |
|----------|--------------|----------|
| `DB_HOST` | Database host | `localhost` |
| `DB_PORT` | Port | `5432` |
| `DB_NAME` | Database name | - |
| `DB_USER` | Username | - |
| `DB_PASSWORD` | Password | - |

## Optimizer Settings

Each algorithm has specific parameters that are passed via `kwargs` to the `calibrate` method:

- **Differential Evolution**: `population_size`, `mutation`, `recombination`.
- **Nelder-Mead**: `adaptive`, `tolerance`.
- **L-BFGS-B**: `gtol`.

## Logging

The framework uses the standard Python logging module. You can set the verbosity level as follows:

```python
import logging
logging.getLogger("pyadm1ode_calibration").setLevel(logging.DEBUG)
```
