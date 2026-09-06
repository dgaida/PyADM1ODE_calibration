# Kalibrierungs-Tutorial

Die Kurzfassung der Notebooks 01 bis 03.

## 1. Daten laden und bereinigen

```python
from pyadm1ode_calibration import MeasurementData

measurements = MeasurementData.from_csv("data.csv")
measurements.remove_outliers(method="zscore", threshold=3.0)
measurements.fill_gaps(method="interpolate", limit=3)
```

Der Index muss ein Zeitstempel sein, und die Spalten tragen die Kanalnamen, auf die sich die
Zielgrößen beziehen (`Q_gas`, `Q_ch4`, `pH`, ...).

## 2. Parameter auswählen

Wenige anpassen. Am besten mit der Hydrolyserate der dominierenden Substratfraktion beginnen, denn
sie bewegt die Gasproduktion am unmittelbarsten. [Konfiguration](../configuration.md) listet die
üblichen Kandidaten samt Grenzen.

## 3. Anpassen

```python
from pyadm1ode_calibration import InitialCalibrator

calibrator = InitialCalibrator(plant)
result = calibrator.calibrate(
    measurements=measurements,
    parameters=["k_hyd_ch"],
    objectives=["Q_gas"],
    validation_split=0.2,
)
```

## 4. Ergebnis prüfen

`result.success` besagt nur, dass der Optimierer beendet wurde. Entscheidend ist, ob auch der Fehler
auf dem zurückgehaltenen Teil gesunken ist: `result.validation_metrics` gegen die Trainingswerte
halten. Eine große Lücke zwischen beiden bedeutet, dass die Anpassung dem Rauschen gefolgt ist und
nicht der Anlage.
