# Installation

## Standard Installation

The most stable version of PyADM1ODE_calibration can be installed directly via the Python Package Index (PyPI):

```bash
pip install pyadm1ode-calibration
```

## Installation from Source

For the latest features or active development, you can clone the repository:

```bash
git clone https://github.com/dgaida/PyADM1ODE_calibration.git
cd PyADM1ODE_calibration
pip install .
```

### Development Mode

If you want to make changes to the code, install the package in editable mode with all development dependencies:

```bash
pip install -e ".[dev]"
```

## System Requirements (Linux)

Since PyADM1ODE is based on .NET components (SIMBA# core), the `mono-complete` package must be installed on Linux:

```bash
sudo apt-get update
sudo apt-get install mono-complete
```

## Documentation Tools

To build the documentation locally, install the `docs` extras:

```bash
pip install -e ".[docs]"
```

Then you can view the documentation using MkDocs:

```bash
mkdocs serve
```

## Docker (Optional)

A Dockerfile is available in the repository to provide a consistent environment that already includes all .NET/Mono dependencies.
