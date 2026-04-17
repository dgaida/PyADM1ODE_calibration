# Configuration

PyADM1ODE_calibration offers flexible configuration options for parameters, optimizers, and objective functions.

## Calibratable Parameters

The following ADM1 parameters are most commonly used for calibration:

| Parameter | Description | Unit | Typical Range |
|-----------|-------------|------|---------------|
| `k_dis` | Disintegration constant | 1/d | 0.3 - 0.8 |
| `k_hyd_ch` | Carbohydrate hydrolysis rate | 1/d | 5.0 - 15.0 |
| `k_m_ac` | Max. uptake rate acetate | 1/d | 4.0 - 12.0 |
| `Y_su` | Yield coefficient sugar degraders | kg COD/kg COD | 0.05 - 0.15 |

### Parameter Bounds

You can manually define the search ranges for the optimization:

```python
bounds = {
    "k_dis": (0.2, 1.0),
    "k_hyd_ch": (2.0, 20.0)
}
```

## Optimization Methods

Supported algorithms for the `calibrate` method:

- `differential_evolution` (default): Robust for global search.
- `nelder-mead`: Efficient for local search (good for online calibration).
- `l-bfgs-b`: Gradient-based, requires smooth objective functions.
- `particle_swarm`: Stochastic global search.

## Objective Functions

By default, methane production (`Q_ch4`) is optimized. However, you can weight multiple variables:

```python
objectives = ["Q_ch4", "pH", "VFA"]
weights = {
    "Q_ch4": 0.7,
    "pH": 0.2,
    "VFA": 0.1
}
```

## Database Connection

The database is configured via a connection URL (SQLAlchemy format):

```python
db_url = "postgresql://user:password@localhost:5432/biogas_db"
```
