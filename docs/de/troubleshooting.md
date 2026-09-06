# Fehlerbehebung

## Die Kalibrierung läuft stundenlang

Differential Evolution kostet `population_size x len(parameters)` Simulationen je Generation, siehe
[Konfiguration](configuration.md#optimierer). Zuerst die Parameterliste kürzen, dann die Population,
dann die Generationen.

## `result.success` ist `False`

Der Optimierer hat ohne Konvergenz abgebrochen. Meist sind die Grenzen zu weit, die Parameterliste zu
lang oder das Zeitfenster zu kurz, um den angepassten Effekt zu enthalten. `result.message` enthält
die Begründung von SciPy.

## Die Anpassung ist gut, die Werte sind unplausibel

Die Parameter gleichen einander aus. `analyze_subset` beziffert das: ein Kollinearitätsindex über 20
heißt, dass der Satz nicht auflösbar ist, so gut jedes Mitglied für sich auch dasteht. Den Parameter
streichen, der weniger wiegt, oder ihn auf einen Literaturwert festsetzen. `use_constraints=True`
nimmt die Grenzstrafen in die Zielfunktion auf.

## Validierungsfehler viel größer als Trainingsfehler

Die Anpassung ist dem Rauschen gefolgt. Parameterliste kürzen, Zeitfenster verlängern oder die Daten
vor der Anpassung gründlicher bereinigen: `remove_outliers` und `fill_gaps`.

## Ein Lastwechsel in den Daten wirkt sich nicht auf die Anpassung aus

Standardmäßig wird die Fütterungsreihe auf ihren Mittelwert zusammengefasst, Stufen
sind für das Modell also unsichtbar. Dem Kalibrator `time_varying_feed=True`
übergeben, siehe
[Konfiguration](configuration.md#zeitlich-veranderliche-futterung).

## `ValueError: Column 'Q_ch4' not found`

Der Kanalname in `objectives` kommt in der Messtabelle nicht vor. `measurements.data.columns` zeigt,
was tatsächlich vorhanden ist. `CSVHandler` bildet gängige deutsche Spaltennamen ab, ein
`PlantSchema` bildet beliebige Tags auf Kanalnamen ab.

## Die Online-Rekalibrierung löst nie aus

`should_recalibrate` liefert ein Paar. `if calibrator.should_recalibrate(data):` ist immer wahr und
sagt nichts aus, das Paar muss entpackt werden: `needed, reason = ...`. Die Begründung benennt die
Bedingung, die es zurückgehalten hat, meist `time_threshold` oder `consecutive_violations`.

## Etwas anderes

Bitte ein [Issue auf GitHub](https://github.com/dgaida/PyADM1ODE_calibration/issues) anlegen.
