# Tutorials

Eine Reihe von sieben Notebooks in `notebooks/`, gedacht zum Durcharbeiten in dieser Reihenfolge.
Jedes läuft für sich und erzeugt seine Daten selbst, ein Anlagenzugang ist also nicht nötig.

| Notebook | Frage, die es beantwortet |
|----------|---------------------------|
| [`00_quickstart`](https://colab.research.google.com/github/dgaida/PyADM1ODE_calibration/blob/main/notebooks/00_quickstart.ipynb) | Der gesamte Ablauf in einem Durchgang |
| `01_explore_measurements` | Was steckt in den Daten, und wo sind sie defekt? |
| `02_model_vs_measurement` | Wie weit liegt ein unkalibriertes Modell daneben? |
| `03_first_calibration` | Einen Parameter anpassen, bis die Kurven zusammenfallen |
| `04_which_parameters_matter` | Welche Parameter lohnen sich überhaupt? |
| `05_train_test_and_residuals` | Hält die Anpassung auf ungesehenen Daten? |
| `06_online_recalibration` | Einer driftenden Anlage folgen |

Notebook 00 läuft ohne lokale Installation in **Google Colab**:

[![Open In Colab](../assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/PyADM1ODE_calibration/blob/main/notebooks/00_quickstart.ipynb)

Die übrigen setzen eine lokale Installation voraus. Sie teilen sich `notebooks/demo_plant.py`, das
die kleine Anlage aufbaut und die Messreihen erzeugt.

- [Kalibrierungs-Tutorial](calibration.md): derselbe Weg als Textfassung.
