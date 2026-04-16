# Kalibrierung

Die Kalibrierung ist das Herzstück dieses Frameworks. Sie unterteilt sich in die Initialkalibrierung und die Online-Rekalibrierung.

## Initialkalibrierung

Die Initialkalibrierung wird meist einmalig durchgeführt, wenn historische Daten für einen Zeitraum (z.B. 30 Tage) vorliegen.

```python
from pyadm1ode_calibration.calibration import InitialCalibrator
from pyadm1ode_calibration.io.loaders import MeasurementData

# 1. Messdaten laden
measurements = MeasurementData.from_csv("hist_data.csv")

# 2. Calibrator initialisieren
# plant ist eine Instanz von pyadm1ode.Plant
calibrator = InitialCalibrator(plant)

# 3. Kalibrierung starten
result = calibrator.calibrate(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch", "Y_su"],
    method="differential_evolution",
    validation_split=0.2
)

# 4. Ergebnisse auswerten
print(f"Bester Fit: {result.objective_value}")
print(f"Kalibrierte Parameter: {result.parameters}")
```

## Online-Rekalibrierung

Die Online-Rekalibrierung dient der Anpassung des Modells im laufenden Betrieb, falls die Vorhersagegüte nachlässt.

```python
from pyadm1ode_calibration.calibration import OnlineCalibrator

calibrator = OnlineCalibrator(plant)

# Trigger-Bedingungen definieren
calibrator.set_trigger(
    variance_threshold=0.15,  # 15% Abweichung
    time_threshold=24.0       # Mindestens 24h Abstand
)

# Prüfung, ob Rekalibrierung nötig ist
if calibrator.should_recalibrate(recent_data):
    result = calibrator.calibrate(
        measurements=recent_data,
        method="nelder-mead"
    )
    if result.success:
        calibrator.apply_calibration(result)
```

## Validierung

Nach jeder Kalibrierung wird automatisch eine Validierung durchgeführt (falls ein `validation_split` angegeben wurde). Die Ergebnisse finden Sie in `result.validation_metrics`:

- **RMSE**: Root Mean Square Error
- **R²**: Bestimmtheitsmaß
- **NSE**: Nash-Sutcliffe Efficiency
