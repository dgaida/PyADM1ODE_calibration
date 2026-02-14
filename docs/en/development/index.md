# Development

Information for contributors and developers of PyADM1ODE_calibration.

## Development Setup

1. Clone the repository.
2. Install dependencies: `pip install -e ".[dev,docs]"`.
3. Install pre-commit hooks: `pre-commit install`.

## Code Style

We use:
- **Black** for formatting.
- **Ruff** for linting.
- **Google-style docstrings** for documentation.

## Tests

Run tests with pytest:

```bash
pytest
```

With coverage report:

```bash
pytest --cov=pyadm1ode_calibration
```

## Building Documentation

```bash
mkdocs build
```

To view the documentation locally with live preview:

```bash
mkdocs serve
```
