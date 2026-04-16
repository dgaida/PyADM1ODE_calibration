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

## System-Voraussetzungen (Linux)

Da PyADM1ODE auf .NET-Komponenten (SIMBA# Kern) basiert, muss unter Linux das Paket `mono-complete` installiert sein:

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

## Docker (Optional)

Ein Dockerfile ist im Repository verfügbar, um eine konsistente Umgebung bereitzustellen, die bereits alle .NET/Mono Abhängigkeiten enthält.
