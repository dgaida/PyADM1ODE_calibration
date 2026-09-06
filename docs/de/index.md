# PyADM1ODE Kalibrierung

[![PyPI version](https://img.shields.io/pypi/v/pyadm1ode-calibration.svg)](https://pypi.org/project/pyadm1ode-calibration/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyadm1ode-calibration.svg)](https://pypi.org/project/pyadm1ode-calibration/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://dgaida.github.io/PyADM1ODE_calibration/)
[![Interrogate](assets/interrogate.svg)](development/metrics.md)
[![Open In Colab](assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/PyADM1ODE_calibration/blob/main/notebooks/00_quickstart.ipynb)

**Parameterkalibrierung für [PyADM1ODE](https://github.com/dgaida/PyADM1ODE)-Biogasanlagenmodelle.**

Passt ADM1-Parameter an gemessene Anlagendaten an, einmalig aus einem historischen Zeitfenster oder fortlaufend im Betrieb.

## Was das Paket leistet

- **Initialkalibrierung**: globale Suche über ein historisches Zeitfenster, mit Train/Test-Aufteilung.
- **Online-Rekalibrierung**: begrenzte Parameteranpassungen, ausgelöst über die Prognosevarianz.
- **Analyse**: Sensitivität und Identifizierbarkeit, damit erkennbar ist, welche Parameter die Daten überhaupt auflösen.
- **Daten**: CSV- und Datenbankquellen hinter einer Schemadatei, dazu Validierung, Ausreißerentfernung und Lückenfüllung.

## Inhalt

| Seite | Inhalt |
|-------|--------|
| [Erste Schritte](getting-started.md) | Voraussetzungen und Grundbegriffe |
| [Installation](installation.md) | pip, conda, Docker |
| [Konfiguration](configuration.md) | Parameter, Grenzen, Optimierer, Datenbank |
| [Nutzung](usage/index.md) | Die beiden Workflows im Code |
| [Tutorials](tutorials/index.md) | Notebook-Reihe 00 bis 06 |
| [Beispiele](examples/index.md) | Ausführbare Skripte |
| [API-Referenz](api/index.md) | Klassen und Funktionen |
| [Architektur](architecture/index.md) | Module und Datenfluss |

## Quickstart

```python
from pyadm1ode_calibration import Calibrator, MeasurementData

measurements = MeasurementData.from_csv("plant_data.csv")
calibrator = Calibrator(plant)          # plant: a pyadm1 BiogasPlant

result = calibrator.run_initial_calibration(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch"],
    objectives=["Q_ch4", "pH"],
)

if result.success:
    calibrator.apply_calibration(result)
```

## Zitation

Wenn Sie PyADM1ODE_calibration in Ihrer Forschung verwenden, zitieren Sie bitte:

```bibtex
@software{pyadm1_calibration,
  author = {Gaida, Daniel},
  title = {PyADM1ODE\_calibration: Parameter Calibration Framework for Biogas Plant Models},
  year = {2026},
  url = {https://github.com/dgaida/PyADM1ODE_calibration}
}
```
