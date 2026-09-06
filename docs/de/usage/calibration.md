# Kalibrierung

## Initialkalibrierung

Einmalig auf einem historischen Zeitfenster stabilen Betriebs. Üblich sind sieben bis dreißig Tage.

```python
from pyadm1ode_calibration import InitialCalibrator, MeasurementData

measurements = MeasurementData.from_csv("hist_data.csv")
measurements.remove_outliers(method="zscore", threshold=3.0)
measurements.fill_gaps(method="interpolate", limit=3)

calibrator = InitialCalibrator(plant)
result = calibrator.calibrate(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch", "Y_su"],
    objectives=["Q_ch4", "pH"],
    weights={"Q_ch4": 0.8, "pH": 0.2},
    method="differential_evolution",
    validation_split=0.2,
    max_iterations=50,
    population_size=10,
)

print(result.success, result.parameters, result.validation_metrics)
```

`InitialCalibrator` schreibt nicht in das Anlagenmodell. Dafür die Fassade `Calibrator` verwenden
oder die Werte selbst setzen.

### Welche Parameter sich lohnen

```python
sensitivity = calibrator.sensitivity_analysis(result.parameters, measurements)
identifiability = calibrator.identifiability_analysis(
    result.parameters, measurements, correlation_threshold=0.8
)
```

`identifiability_analysis` unterscheidet nach Raue et al. 2009 zwei Arten des Scheiterns:

- **strukturell nicht identifizierbar**: Der Parameter bewegt keinen Messkanal, keine Datenmenge  
  ändert daran etwas. Auf der Beispielanlage trifft das `k_dis`, denn die Substrate treten als
  hydrolysierbare Fraktionen ein, die Desintegration hat also nie ein Substrat.  
- **praktisch nicht identifizierbar**: Er bewegt die Ausgänge, aber so schwach, dass das  
  Konfidenzintervall breiter bleibt als der Schätzwert selbst.

Ein Satz kann auch als Satz scheitern. `analyze_subset` wendet die beiden Maße von Brun et al. 2001
in ihrer Reihenfolge an: erst jeden Parameter für sich, dann den **Kollinearitätsindex** über die
Überlebenden. Er ist 1, wenn die Parameter in unabhängige Richtungen auf die Ausgänge wirken, und
wächst unbeschränkt, je austauschbarer sie werden. Die übliche Grenze liegt bei 20.

```python
verdict = calibrator.identifiability_analyzer.analyze_subset(parameters, measurements)
print(verdict.collinearity_index, verdict.reason)
```

Günstiger, als es hinterher zu merken: der Test kostet `2 * len(parameters) + 1` Simulationen
gegenüber den mehreren hundert einer globalen Suche. `calibrate(..., check_identifiability=True)`
führt ihn deshalb vorab aus und warnt, bevor der Optimierer startet.

## Online-Rekalibrierung

Die Auslöseschwellen liegen auf `calibrator.trigger`, die Schrittbegrenzung ist ein Argument von
`calibrate`.

```python
from pyadm1ode_calibration import OnlineCalibrator

calibrator = OnlineCalibrator(plant)
calibrator.trigger.variance_threshold = 0.15     # 15 % relative Abweichung
calibrator.trigger.time_threshold = 24.0         # Stunden seit dem letzten Lauf
calibrator.trigger.consecutive_violations = 2

needed, reason = calibrator.should_recalibrate(recent_data)
if needed:
    result = calibrator.calibrate(
        measurements=recent_data,
        parameters=["k_hyd_ch"],
        current_parameters={"k_hyd_ch": 2.0},
        max_parameter_change=0.20,               # höchstens 20 % je Lauf
        method="nelder_mead",
    )
    if result.success:
        calibrator.apply_calibration(result)
```

`should_recalibrate` liefert ein Paar, die Entscheidung und deren Begründung. Das Paar selbst
abzufragen ist immer wahr.

## Validierung

Mit einem `validation_split` landen die Kennzahlen des zurückgehaltenen Teils in
`result.validation_metrics`:

- **RMSE**, **MAE**: absoluter Fehler  
- **R2**: Bestimmtheitsmaß  
- **NSE**: Nash-Sutcliffe-Effizienz  
- **PBIAS**: systematische Über- oder Unterschätzung  

`CalibrationValidator` ergänzt Residuendiagnostik: Normalität, Autokorrelation und
Heteroskedastizität.
