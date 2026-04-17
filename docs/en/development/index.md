# Development

Welcome to the developer area of PyADM1ODE_calibration. We welcome contributions!

## Development Setup

1. Clone the repository:  
   ```bash
   git clone https://github.com/dgaida/PyADM1ODE_calibration.git
   cd PyADM1ODE_calibration
   ```

2. Create a virtual environment and install dependencies:  
   ```bash
   pip install -e ".[dev,docs]"
   ```

## Guidelines

- **Code Style**: We use `ruff` for linting and `black` for formatting.  
- **Tests**: New features must be covered by tests in `tests/`. We aim for > 90% coverage.  
- **Docstrings**: All public functions must have Google-style docstrings.  

## Workflow

1. Create a feature branch.  
2. Implement your changes.  
3. Run tests and linter:  
   ```bash
   ruff check .
   black --check .
   python3 -m pytest tests/
   ```
4. Create a pull request.  

## Documentation

The documentation is built with MkDocs. To view it locally:
```bash
mkdocs serve
```
