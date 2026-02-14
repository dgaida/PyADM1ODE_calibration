# Konfiguration

PyADM1ODE_calibration kann über verschiedene Wege konfiguriert werden, um den Kalibrierungsprozess an Ihre Anlage anzupassen.

## Parameter-Grenzen

Die Standardgrenzen für ADM1-Parameter sind in `ParameterBounds` definiert. Sie können diese global oder pro Kalibrierung anpassen.

```python
from pyadm1ode_calibration.calibration import create_default_bounds

bounds = create_default_bounds()
bounds.add_bound("k_dis", lower=0.3, upper=0.8, default=0.5)
```

## Datenbank-Verbindung

Die Verbindung zur PostgreSQL-Datenbank erfolgt über Umgebungsvariablen oder ein Konfigurationsobjekt.

### Umgebungsvariablen

| Variable | Beschreibung | Standard |
|----------|--------------|----------|
| `DB_HOST` | Datenbank-Host | `localhost` |
| `DB_PORT` | Port | `5432` |
| `DB_NAME` | Datenbank-Name | - |
| `DB_USER` | Benutzername | - |
| `DB_PASSWORD` | Passwort | - |

## Optimierer-Einstellungen

Jeder Algorithmus hat spezifische Parameter, die über `kwargs` an die `calibrate`-Methode übergeben werden:

- **Differential Evolution**: `population_size`, `mutation`, `recombination`.
- **Nelder-Mead**: `adaptive`, `tolerance`.
- **L-BFGS-B**: `gtol`.

## Logging

Das Framework nutzt das Standard-Python-Logging-Modul. Sie können den Detailgrad wie folgt einstellen:

```python
import logging
logging.getLogger("pyadm1ode_calibration").setLevel(logging.DEBUG)
```
