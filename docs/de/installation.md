# Installation

## Standard-Installation

Die stabilste Version von PyADM1ODE_calibration kann direkt über den Python Package Index (PyPI) installiert werden:

```bash
pip install pyadm1ode-calibration
```

## Installation aus dem Quellcode

Für die neuesten Features oder zur aktiven Entwicklung können Sie das Repository klonen:

```bash
git clone https://github.com/dgaida/PyADM1ODE_calibration.git
cd PyADM1ODE_calibration
pip install .
```

### Entwicklungs-Modus

Wenn Sie Änderungen am Code vornehmen möchten, installieren Sie das Paket im Editier-Modus mit allen Entwicklungs-Abhängigkeiten:

```bash
pip install -e ".[dev]"
```

## Installation als Anaconda-Umgebung

Wenn Sie Conda verwenden, können Sie die Umgebung mit der bereitgestellten `environment.yml` Datei erstellen:

```bash
conda env create -f environment.yml
conda activate biogas
```

Anschließend installieren Sie dieses Paket im Editier-Modus:

```bash
pip install -e .
```

## System-Voraussetzungen (Linux)

Da PyADM1ODE auf .NET-Komponenten basiert (siehe [biogas_c#](https://github.com/dgaida/matlab_toolboxes/tree/master/biogas_c%23)), muss unter Linux das Paket `mono-complete` installiert sein:

```bash
sudo apt-get update
sudo apt-get install mono-complete
```

## Dokumentations-Tools

Um die Dokumentation lokal zu bauen, installieren Sie die `docs` Extras:

```bash
pip install -e ".[docs]"
```

Danach können Sie die Dokumentation mit MkDocs betrachten:

```bash
mkdocs serve
```

## Docker

Ein Dockerfile ist im Repository verfügbar, um eine konsistente Umgebung bereitzustellen, die bereits alle .NET/Mono Abhängigkeiten enthält.

```bash
docker build -t pyadm1-calibration .
docker run -it pyadm1-calibration
```
