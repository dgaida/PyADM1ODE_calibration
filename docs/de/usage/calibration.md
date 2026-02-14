# Handbuch zur Modellkalibrierung

Dieses Handbuch behandelt die Parameterkalibrierung in PyADM1ODE, einschließlich der Erstkalibrierung aus historischen Daten und der Online-Rekalibrierung während des Anlagenbetriebs.

## Übersicht

Die Modellkalibrierung ist für eine genaue Biogasanlagensimulation unerlässlich. PyADM1ODE bietet:

- **Erstkalibrierung**: Batch-Optimierung aus historischen Messdaten
- **Online-Rekalibrierung**: Adaptive Parameteranpassung während des Betriebs
- **Sensitivitätsanalyse**: Identifizierung einflussreicher Parameter
- **Identifizierbarkeitsbewertung**: Erkennung von Überparametrisierung
- **Validierungswerkzeuge**: Gütekriterien und Residualanalyse

## Schnelleinstieg

```python
from pyadm1ode_calibration.calibration import Calibrator
from pyadm1ode_calibration.io import MeasurementData
# Annahme: plant und feedstock sind von pyadm1ode verfügbar
from pyadm1ode import BiogasPlant

# Anlage und Messungen laden
plant = BiogasPlant.from_json("plant.json", feedstock)
measurements = MeasurementData.from_csv("plant_data.csv")

# Kalibrator erstellen
calibrator = Calibrator(plant)

# Parameter kalibrieren
result = calibrator.run_initial_calibration(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch", "Y_su"],
    objectives=["Q_ch4", "pH"],
    weights={"Q_ch4": 0.8, "pH": 0.2}
)

# Kalibrierte Parameter anwenden
if result.success:
    calibrator.apply_calibration(result)
```

## Erstkalibrierung

Die Erstkalibrierung wird typischerweise mit 7-30 Tagen stabilen Anlagendaten durchgeführt.

### Wann die Erstkalibrierung genutzt werden sollte

Nutzen Sie die Erstkalibrierung, wenn Sie:
- Historische Messdaten haben (≥ 2 Wochen empfohlen)
- Einen stabilen Anlagenbetrieb während des Zeitraums haben
- Zuverlässige Messungen der Zielgrößen (Gasproduktion, pH, VFA) haben
- Bekannte Substratzulaufmengen und -zusammensetzungen haben

## Online-Rekalibrierung

Die Online-Rekalibrierung ermöglicht die kontinuierliche Anpassung der Parameter während des Betriebs, um schleichende Änderungen der Substrateigenschaften oder Sensordrift auszugleichen.

### Nutzung des Online-Kalibrators

```python
# Online-Rekalibrierung mit begrenzten Änderungen
result = calibrator.run_online_calibration(
    measurements=recent_data,
    parameters=["k_dis", "Y_su"],
    max_parameter_change=0.10,  # Max 10% Änderung
    time_window=7,               # Nutze die letzten 7 Tage
    method="nelder_mead"         # Schneller lokaler Optimierer
)

if result.success:
    calibrator.apply_calibration(result)
```

## Validierung und Qualitätsprüfung

### Gütekriterien

Der `CalibrationValidator` bietet umfassende Metriken zur Bewertung der Kalibrierungsqualität.

| Metrik                          | Exzellent | Gut | Ausreichend | Mangelhaft |
|---------------------------------|-----------|------|------|------|
| R²                              | > 0.90 | 0.75-0.90 | 0.50-0.75 | < 0.50 |
| NSE (Nash-Sutcliffe Efficiency) | > 0.90 | 0.70-0.90 | 0.50-0.70 | < 0.50 |
| PBIAS (Percent Bias)            | < ±5% | ±5-±10% | ±10-±25% | > ±25% |

## Referenzen

1. **Batstone et al. (2002)**: Anaerobic Digestion Model No. 1 (ADM1). IWA Publishing.
2. **Gaida (2014)**: Dynamic real-time substrate feed optimization of anaerobic co-digestion plants. PhD thesis.
3. **Dochain & Vanrolleghem (2001)**: Dynamical Modelling & Estimation in Wastewater Treatment Processes.
