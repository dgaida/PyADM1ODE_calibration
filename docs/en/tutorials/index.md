# Tutorials

A series of seven notebooks in `notebooks/`, meant to be read in order. Each one runs on its own and
builds its own data, so no plant access is needed.

| Notebook | Question it answers |
|----------|---------------------|
| [`00_quickstart`](https://colab.research.google.com/github/dgaida/PyADM1ODE_calibration/blob/main/notebooks/00_quickstart.ipynb) | The whole workflow in one pass |
| `01_explore_measurements` | What is in the data, and where is it broken? |
| `02_model_vs_measurement` | How far off is an uncalibrated model? |
| `03_first_calibration` | Fit one parameter until the curves meet |
| `04_which_parameters_matter` | Which parameters is it worth fitting at all? |
| `05_train_test_and_residuals` | Does the fit hold on data it has not seen? |
| `06_online_recalibration` | Following a plant that drifts |

Notebook 00 runs in **Google Colab** without a local installation:

[![Open In Colab](../assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/PyADM1ODE_calibration/blob/main/notebooks/00_quickstart.ipynb)

The others expect a local install. They share `notebooks/demo_plant.py`, which builds the small
plant and generates the measurement records.

- [Calibration Tutorial](calibration.md): the same path as a text summary.
