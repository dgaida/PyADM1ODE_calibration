# Installation

## Aus PyPI

```bash
pip install pyadm1ode-calibration
```

## Aus dem Quellcode

```bash
git clone https://github.com/dgaida/PyADM1ODE_calibration.git
cd PyADM1ODE_calibration
pip install -e ".[dev]"
```

`[dev]` ergänzt pytest, ruff und black. Für MkDocs `.[docs]` verwenden, für beides `.[dev,docs]`.

## Conda

```bash
conda env create -f environment.yml
conda activate biogas
pip install -e .
```

## Docker

Das Repository enthält ein Dockerfile mit festgelegten Abhängigkeiten:

```bash
docker build -t pyadm1-calibration .
docker run -it pyadm1-calibration
```

## Dokumentation bauen

```bash
pip install -e ".[docs]"
mkdocs serve
```

Die Seite ist zweisprachig. `docs/de/` ist die Standardsprache, `docs/en/` spiegelt sie Datei für Datei.
