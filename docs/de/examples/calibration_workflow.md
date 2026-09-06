# Kalibrierungs-Workflow

Die acht Schritte, die `examples/calibration_workflow_complete.py` der Reihe nach durchläuft.

| Schritt | Was passiert | API |
|---------|--------------|-----|
| 1 | Anlage aufbauen, dann eine Twin-Aufzeichnung daraus erzeugen | `create_twin_measurements` |
| 2 | Validieren, Ausreißer entfernen, Lücken füllen | `measurements.validate`, `remove_outliers`, `fill_gaps` |
| 3 | Anpassen, mit Train/Test-Aufteilung | `InitialCalibrator.calibrate` |
| 4 | Kennzahlen beider Teile ablesen | `result.validation_metrics` |
| 5 | Parameter nach Einfluss ordnen | `sensitivity_analysis` |
| 6 | Jeden Parameter prüfen, dann den Satz als Ganzes | `identifiability_analysis`, `analyze_subset` |
| 7 | Werte in die Anlage schreiben | `digester.apply_calibration_parameters` |
| 8 | Zusammenfassung ausgeben | - |

Die Schritte 5 und 6 gehören zusammen. Die Sensitivität sagt, ob ein Parameter die Ausgabe überhaupt
bewegt, die Identifizierbarkeit sagt, ob sich seine Wirkung von der eines anderen Parameters trennen
lässt. Ein Parameter, der an einem der beiden scheitert, hätte in Schritt 3 nichts zu suchen, und
genau deshalb stellt `check_identifiability=True` dieselbe Frage vor den Optimierer.

Die Aufzeichnung in Schritt 1 entsteht durch Simulation der Anlage selbst mit bekannten
Parameterwerten. Schritt 4 kann den Fehler daher gegen die Wahrheit ausweisen statt nur „konvergiert"
zu melden. Ihr Fütterungsprofil fährt die Raumbelastung hoch und herunter und verschiebt den
Substratmix, denn ein bei konstanter Last gehaltener Fermenter läuft in einen Zustand, in dem
mehrere Parametersätze dieselbe Gaskurve erzeugen.

Das Skript speichert nichts dauerhaft. Wer einen Lauf behalten will, ergänzt nach Schritt 7
`Database.store_calibration`, so wie es `examples/scheduled_recalibration.py` tut.
