# Installation

## From PyPI

```bash
pip install pyadm1ode-calibration
```

## From source

```bash
git clone https://github.com/dgaida/PyADM1ODE_calibration.git
cd PyADM1ODE_calibration
pip install -e ".[dev]"
```

`[dev]` adds pytest, ruff and black. Use `.[docs]` for MkDocs, or `.[dev,docs]` for both.

## Conda

```bash
conda env create -f environment.yml
conda activate biogas
pip install -e .
```

## Docker

The repository ships a Dockerfile with all dependencies pinned:

```bash
docker build -t pyadm1-calibration .
docker run -it pyadm1-calibration
```

## Building the docs

```bash
pip install -e ".[docs]"
mkdocs serve
```

The site is bilingual. `docs/de/` is the default locale and `docs/en/` mirrors it file by file.
