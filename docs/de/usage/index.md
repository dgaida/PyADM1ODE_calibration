# Nutzung

## Die beiden Workflows

**Initialkalibrierung** passt ein neues Modell einmalig an ein historisches Zeitfenster an. Globaler
Optimierer, Train/Test-Aufteilung, Sensitivitätsanalyse. Siehe
[Kalibrierung](calibration.md#initialkalibrierung).

**Online-Rekalibrierung** hält ein laufendes Modell nach. Lokaler Optimierer, begrenzte
Schrittweite, Varianzauslöser. Siehe [Kalibrierung](calibration.md#online-rekalibrierung).

## Paketaufbau

| Modul | Inhalt |
|-------|--------|
| `pyadm1ode_calibration.calibration` | `Calibrator`, `InitialCalibrator`, `OnlineCalibrator`, `CalibrationResult`, Grenzen |
| `pyadm1ode_calibration.calibration.optimization` | Optimierer, Zielfunktionen, Nebenbedingungen |
| `pyadm1ode_calibration.calibration.analysis` | Sensitivität und Identifizierbarkeit |
| `pyadm1ode_calibration.io.loaders` | `MeasurementData`, `CSVHandler`, `PlantSchema`, `MeasurementBuilder` |
| `pyadm1ode_calibration.io.persistence` | `Database`, ORM-Modelle, Repositories |
| `pyadm1ode_calibration.io.validation` | `DataValidator`, `OutlierDetector` |
| `pyadm1ode_calibration.plants` | Aufbauroutinen für Anlagentopologien |

Die gebräuchlichsten Namen sind auf oberster Ebene erneut exportiert,
`from pyadm1ode_calibration import Calibrator, MeasurementData` funktioniert also.

## Weiterlesen

- [Tutorials](../tutorials/index.md) für die Notebook-Reihe.  
- [Beispiele](../examples/index.md) für ausführbare Skripte.  
- [API-Referenz](../api/index.md) für Signaturen.  
