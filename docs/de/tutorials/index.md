# Tutorials

Eine Reihe von sieben Notebooks in [`notebooks/`](https://github.com/dgaida/PyADM1ODE_calibration/tree/main/notebooks) auf GitHub, gedacht zum Durcharbeiten in dieser Reihenfolge.
Jedes läuft für sich und erzeugt seine Daten selbst, ein Anlagenzugang ist also nicht nötig.

| Notebook | Frage, die es beantwortet |
|----------|---------------------------|
| [`00_quickstart`](https://colab.research.google.com/github/dgaida/PyADM1ODE_calibration/blob/main/notebooks/00_quickstart.ipynb) | Der gesamte Ablauf in einem Durchgang |
| [`01_explore_measurements`](https://github.com/dgaida/PyADM1ODE_calibration/blob/main/notebooks/01_explore_measurements.ipynb) | Was steckt in den Daten, und wo sind sie defekt? |
| [`02_model_vs_measurement`](https://github.com/dgaida/PyADM1ODE_calibration/blob/main/notebooks/02_model_vs_measurement.ipynb) | Wie weit liegt ein unkalibriertes Modell daneben? |
| [`03_first_calibration`](https://github.com/dgaida/PyADM1ODE_calibration/blob/main/notebooks/03_first_calibration.ipynb) | Einen Parameter anpassen, bis die Kurven zusammenfallen |
| [`04_which_parameters_matter`](https://github.com/dgaida/PyADM1ODE_calibration/blob/main/notebooks/04_which_parameters_matter.ipynb) | Welche Parameter lohnen sich überhaupt? |
| [`05_train_test_and_residuals`](https://github.com/dgaida/PyADM1ODE_calibration/blob/main/notebooks/05_train_test_and_residuals.ipynb) | Hält die Anpassung auf ungesehenen Daten? |
| [`06_online_recalibration`](https://github.com/dgaida/PyADM1ODE_calibration/blob/main/notebooks/06_online_recalibration.ipynb) | Einer driftenden Anlage folgen |

Notebook 00 läuft ohne lokale Installation in **Google Colab**:

[![Open In Colab](../assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/PyADM1ODE_calibration/blob/main/notebooks/00_quickstart.ipynb)

Die übrigen setzen eine lokale Installation voraus. Sie teilen sich [`notebooks/demo_plant.py`](https://github.com/dgaida/PyADM1ODE_calibration/blob/main/notebooks/demo_plant.py), das
die kleine Anlage aufbaut und die Messreihen erzeugt.

- [Kalibrierungs-Tutorial](calibration.md): derselbe Weg als Textfassung.  
