# Konfiguration

## Parameter und Grenzen

`create_default_bounds()` liefert Grenzen für 41 ADM1-Parameter. Eine **soft**-Grenze ist ein gewichteter Strafterm, den der Optimierer überschreiten darf, eine **hard**-Grenze macht den Kandidaten unzulässig.

| Parameter | Bedeutung | Einheit | Grenzen | Standard | Typ |
|-----------|-----------|---------|---------|----------|-----|
| `k_dis` | Desintegrationsrate | 1/d | 0,1 - 1,0 | 0,5 | soft |
| `k_hyd_ch` | Hydrolyse, Kohlenhydrate | 1/d | 1,0 - 15,0 | 4,0 | soft |
| `k_hyd_pr` | Hydrolyse, Proteine | 1/d | 1,0 - 15,0 | 4,0 | soft |
| `k_hyd_li` | Hydrolyse, Fette | 1/d | 1,0 - 15,0 | 4,0 | soft |
| `k_m_ac` | Max. Aufnahmerate, Acetat | 1/d | 4,0 - 12,0 | 8,0 | soft |
| `Y_su` | Ertrag, Zuckerverwerter | kg CSB/kg CSB | 0,05 - 0,15 | 0,1 | hard |

```python
from pyadm1ode_calibration import create_default_bounds

bounds = create_default_bounds()
print(bounds.get_bounds("k_dis"))
```

Welche Parameter in einen Lauf gehören, beantworten die Daten, nicht der Modellierer.
`calibrate(..., check_identifiability=True)` prüft den Satz, bevor der Optimierer startet, und warnt,
wenn die Daten ihn nicht auflösen. Kosten: `2 * len(parameters) + 1` Simulationen. Siehe
[Kalibrierung](usage/calibration.md#welche-parameter-sich-lohnen).

Engere Bereiche pro Lauf übergeben, statt die Standardwerte zu ändern:

```python
result = calibrator.calibrate(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch"],
    bounds={"k_dis": (0.3, 0.8), "k_hyd_ch": (2.0, 8.0)},
)
```

## Optimierer

Auswahl über `method=`. Bindestriche, Leerzeichen und Groß-/Kleinschreibung werden vereinheitlicht, `"Nelder-Mead"` und `"nelder_mead"` sind also dasselbe.

| Name | Typ | Einsatz |
|------|-----|---------|
| `differential_evolution`, `de` | global | Initialkalibrierung (Standard) |
| `particle_swarm`, `pso` | global | Alternative zu DE |
| `nelder_mead`, `nm` | lokal | Online-Rekalibrierung (dort Standard) |
| `powell` | lokal | ableitungsfrei, ohne Grenzen nutzbar |
| `l_bfgs_b` | Gradient | glatte Zielfunktionen |
| `slsqp` | Gradient | glatte Zielfunktionen mit Nebenbedingungen |

Wichtiger als die Wahl ist die Dimensionierung: Differential Evolution rechnet
`population_size x len(parameters)` Simulationen je Generation. Mit drei Parametern,
`population_size=10` und `max_iterations=50` waren das 1217 Simulationen und 56 Minuten, also grob
drei Sekunden je simuliertem Zeitfenster.

## Zielgrößen

```python
objectives = ["Q_ch4", "pH", "VFA"]
weights = {"Q_ch4": 0.7, "pH": 0.2, "VFA": 0.1}
```

Die Gewichte werden intern normiert. Ein in den Daten fehlender Kanal wird übersprungen, nicht als null bewertet.

## Datenquellen

Ein Anlagenschema deklariert Quellen und Kanäle in einer YAML-Datei. `configs/plants/_template.yaml` kopieren und anpassen:

```python
from pyadm1ode_calibration.io.loaders import MeasurementBuilder, PlantSchema

schema = PlantSchema.from_yaml("configs/plants/my_plant.yaml")
measurements = MeasurementBuilder(schema).build(variables=["Q_gas", "T_digester"])
```

## Zeitlich veränderliche Fütterung

Eine Kalibrierung fasst die gesamte Substratreihe eines Fensters standardmäßig zu
ihrem Mittelwert zusammen und hält ihn konstant. Das ist günstig und für eine
Aufzeichnung bei gleichbleibender Last richtig, verbirgt aber jeden Lastsprung und
jeden Substratwechsel, und genau die machen die Kinetiken identifizierbar. Mit
`time_varying_feed=True` wird die Fütterung schrittweise angelegt:

```python
calibrator = InitialCalibrator(plant, time_varying_feed=True)
```

Das kostet grob 20 % mehr Laufzeit und steht auf `InitialCalibrator`,
`OnlineCalibrator`, `CalibrationValidator` und `PlantSimulator` zur Verfügung. Ein
Fenster mit tatsächlich konstanter Fütterung nimmt ohnehin den günstigen Weg.

## Datenbank

```python
from pyadm1ode_calibration import Database

db = Database(connection_string="postgresql://user:pw@localhost:5432/biogas")
db = Database.from_env()   # liest DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
```

Zeitstempel werden als naive UTC gespeichert. Für eigene Datensätze `pyadm1ode_calibration.timeutils.utc_now()` verwenden.
