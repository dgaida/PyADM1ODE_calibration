# Installation

Detailed instructions for installing PyADM1ODE_calibration in various environments.

## Pip Installation

### Standard Installation

```bash
pip install git+https://github.com/dgaida/PyADM1ODE_calibration.git
```

### With Optional Dependencies

```bash
# For development (tests, linting)
pip install "pyadm1ode_calibration[dev] @ git+https://github.com/dgaida/PyADM1ODE_calibration.git"

# For documentation
pip install "pyadm1ode_calibration[docs] @ git+https://github.com/dgaida/PyADM1ODE_calibration.git"
```

## Conda / Mamba

We recommend using a dedicated environment:

```bash
# Create environment
conda env create -f environment.yml
conda activate biogas

# Install package in development mode
pip install -e .
```

## Database Dependencies

If you plan to use the PostgreSQL integration, ensure that the appropriate client libraries are installed on your system:

- **Linux (Debian/Ubuntu)**: `sudo apt-get install libpq-dev`
- **macOS**: `brew install postgresql`

## Verification

After installation, you can verify it is working correctly:

```python
import pyadm1ode_calibration
print(pyadm1ode_calibration.__version__)
```
