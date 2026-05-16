# PyADM1ODE Kalibrierung

[![PyPI version](https://img.shields.io/pypi/v/pyadm1ode-calibration.svg)](https://pypi.org/project/pyadm1ode-calibration/)
[![Python versions](https://img.shields.io/pypi/pyversions/pyadm1ode-calibration.svg)](https://pypi.org/project/pyadm1ode-calibration/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://dgaida.github.io/PyADM1ODE_calibration/)
[![Interrogate](assets/interrogate.svg)](development/metrics.md)
[![Open In Colab](assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/PyADM1ODE_calibration/blob/main/notebooks/calibration_tutorial.ipynb)

**Fortschrittliches Parameter-Kalibrierungs-Framework für PyADM1ODE Biogasanlagenmodelle.**

PyADM1ODE_calibration bietet eine vollständige Lösung für die Kalibrierung von [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) Modellen. Es ermöglicht die präzise Abstimmung komplexer ADM1-Parameter auf reale Anlagendaten durch modernste Optimierungsverfahren.

## Hauptmerkmale

- 🎯 **Präzision**: Hochgenaue Abstimmung von ADM1-Parametern auf reale Anlagendaten.  
- ⚡ **Effizienz**: Schnelle lokale Optimierer für den Online-Einsatz und robuste globale Optimierer für die Initialkalibrierung.  
- 📊 **Analyse**: Integrierte Sensitivitäts- und Identifizierbarkeitsanalyse zur Identifizierung kritischer Parameter.  
- 💾 **Integration**: Nahtlose Anbindung an PostgreSQL-Datenbanken und CSV-Workflows.  
- 🌍 **Mehrsprachig**: Dokumentation in Deutsch und Englisch verfügbar.  

## Inhaltsverzeichnis

- [Erste Schritte](getting-started.md) — Schneller Einstieg in das Projekt.  
- [Installation](installation.md) — Installationsanleitungen für verschiedene Umgebungen.  
- [Konfiguration](configuration.md) — Überblick über Konfigurationsoptionen und Parameter.  
- [Tutorials](tutorials/index.md) — Schritt-für-Schritt-Anleitungen (auch für Google Colab).  
- [API-Referenz](api/index.md) — Detaillierte Dokumentation der Klassen und Funktionen.  

## Quickstart

```python
from pyadm1ode_calibration.calibration import InitialCalibrator
from pyadm1ode_calibration.io.loaders import MeasurementData

# 1. Daten laden
measurements = MeasurementData.from_csv("plant_data.csv")

# 2. Kalibrator erstellen
calibrator = InitialCalibrator(plant_model)

# 3. Kalibrierung ausführen
result = calibrator.calibrate(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch"],
    objectives=["Q_ch4", "pH"]
)

# 4. Ergebnisse anwenden
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
