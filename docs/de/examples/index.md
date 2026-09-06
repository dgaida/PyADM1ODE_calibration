# Beispiele

Drei ausführbare Skripte in `examples/`. Anders als die Notebooks sind sie als Programme geschrieben,
nicht als Erklärung.

| Skript | Zweck | Laufzeit |
|--------|-------|----------|
| `calibration_example.py` | Kompakter Durchgang durch eine Kalibrierung | Minuten |
| `calibration_workflow_complete.py` | Alle Schritte in einem Lauf gegen eine Twin-Aufzeichnung mit bekannter Lösung | rund eine Stunde, siehe unten |
| `scheduled_recalibration.py` | Unbeaufsichtigter Auftrag für einen Scheduler | Minuten |

## calibration_workflow_complete.py

Die Referenz dafür, was das Paket kann, und das Langsamste im Repository. Die Standardeinstellungen
kosten rund 1200 Anlagensimulationen, gemessen 56 Minuten auf einem Desktop-Rechner, siehe
[Konfiguration](../configuration.md#optimierer). Vor einem interaktiven Lauf `population_size` und
`max_iterations` verkleinern.

## scheduled_recalibration.py

Das Skript liest ein Zeitfenster, schlägt die verwendeten Parameter nach, rekalibriert, verwirft das Ergebnis, wenn Daten oder
Anpassung nicht vertrauenswürdig aussehen, und speichert das Angenommene. Der Scheduler muss nur den
Rückgabewert auswerten:

```bash
python examples/scheduled_recalibration.py --db sqlite:///calibration.db
```

| Code | Bedeutung |
|------|-----------|
| 0 | angenommen und gespeichert |
| 1 | von einer Schutzregel abgelehnt, bisherige Parameter bleiben |
| 2 | Eingabe unbrauchbar, zu kurz oder zu viele Lücken |
| 3 | unerwarteter Fehler, siehe Log |

Den Ablauf der Schritte zeigt [Kalibrierungs-Workflow](calibration_workflow.md).
