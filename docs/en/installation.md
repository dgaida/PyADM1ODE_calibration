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

## Installation as Anaconda Environment

If you are using Conda, you can create the environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate biogas
```

Then install this package in editable mode:

```bash
pip install -e .
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

## Docker

A Dockerfile is available in the repository to provide a consistent environment that already includes all dependencies.

```bash
docker build -t pyadm1-calibration .
docker run -it pyadm1-calibration
```
