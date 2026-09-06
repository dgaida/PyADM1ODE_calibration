# Erste Schritte

## Voraussetzungen

- **Python** 3.10 oder neuer.  
- **[PyADM1ODE](https://github.com/dgaida/PyADM1ODE)**, das Anlagenmodell, das dieses Paket kalibriert.  
- **Messdaten** als Zeitreihe, aus CSV oder einer Datenbank. Brauchbare Kanäle sind Gasmenge, Methangehalt, pH-Wert und FOS.  

## Installation

```bash
pip install pyadm1ode-calibration
```

Für die Entwicklung siehe [Installation](installation.md).

## Grundbegriffe

**`MeasurementData`** kapselt eine zeitindizierte Tabelle. Jede Kalibrierung liest daraus, und die Vorverarbeitung steckt ebenfalls darin: `remove_outliers`, `fill_gaps`, `get_time_window`.

**`InitialCalibrator`** passt ein frisches Modell an ein historisches Zeitfenster an. Er teilt die Daten in Training und Validierung, führt einen globalen Optimierer aus und weist beide Güten aus.

**`OnlineCalibrator`** hält ein laufendes Modell nach. `should_recalibrate` liefert eine Entscheidung samt Begründung, und `calibrate` begrenzt, wie weit eine einzelne Anpassung einen Parameter verschieben darf.

**`Calibrator`** ist eine schmale Fassade über beide. Er und `OnlineCalibrator` schreiben ein Ergebnis mit `apply_calibration` in das Anlagenmodell zurück, `InitialCalibrator` nicht.

**Zielgrößen** sind die Messkanäle, gegen die die Anpassung bewertet wird. Mehrere lassen sich gewichtet kombinieren, etwa 80 % auf die Gasmenge und 20 % auf den pH-Wert.

## Nächste Schritte

- Die Notebooks durcharbeiten, beginnend bei [Tutorials](tutorials/index.md).  
- Parameter und Optimierer in [Konfiguration](configuration.md) nachschlagen.  
- Beide Workflows im Code unter [Nutzung](usage/index.md) ansehen.  
