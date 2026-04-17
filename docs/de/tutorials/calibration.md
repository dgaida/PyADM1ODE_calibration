# Kalibrierungs-Tutorial

Dieses Tutorial beschreibt die grundlegenden Schritte zur Kalibrierung eines ADM1-Modells.

## 1. Datenvorbereitung

Messdaten müssen als Zeitreihen vorliegen. Das `MeasurementData` Objekt erwartet Spaltennamen, die den ADM1-Zustandsvariablen entsprechen.

```python
from pyadm1ode_calibration.io.loaders import MeasurementData
measurements = MeasurementData.from_csv("daten.csv")
```

## 2. Initialisierung des Kalibrators

Der `InitialCalibrator` benötigt eine Instanz des Anlagenmodells.

```python
from pyadm1ode_calibration.calibration import InitialCalibrator
calibrator = InitialCalibrator(plant)
```

## 3. Durchführung der Kalibrierung

Wählen Sie die Parameter aus, die optimiert werden sollen. Typischerweise sind dies Hydrolysekonstanten oder Ertragskoeffizienten.

```python
result = calibrator.calibrate(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch"],
    method="differential_evolution"
)
```

## 4. Auswertung

Prüfen Sie `result.success` und die optimierten Werte in `result.parameters`.
