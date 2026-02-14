# Nutzung

Dieser Bereich enthält detaillierte Anleitungen zur Verwendung von PyADM1ODE_calibration.

## Kern-Workflows

- [Batch-Kalibrierung](calibration.md): Initialkalibrierung basierend auf historischen Daten.
- [Online-Rekalibrierung](calibration.md#online-rekalibrierung): Dynamische Anpassung im laufenden Betrieb.

## Beispiele

- [Vollständiger Workflow](../examples/calibration_workflow.md): Ein Schritt-für-Schritt Beispiel von der Datenaufbereitung bis zur Ergebnisanalyse.

## Datenformate

Das Framework erwartet Messdaten als Zeitreihen. Stellen Sie sicher, dass Ihre CSV-Dateien eine `timestamp`-Spalte und entsprechende Spalten für die Zielgrößen (z.B. `Q_ch4`, `pH`) enthalten.
