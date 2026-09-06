# Development

## Setup

```bash
git clone https://github.com/dgaida/PyADM1ODE_calibration.git
cd PyADM1ODE_calibration
pip install -e ".[dev,docs]"
```

## Before every commit

```bash
ruff check .
black --check .
python -m pytest
```

CI runs all three and fails on any of them. `.pre-commit-config.yaml` runs the same tools on staged
files, install it with `pre-commit install`.

## Conventions

- **Formatting**: black, line length 127. Ruff uses the same limit.  
- **Lint**: the repository is clean under ruff's default rule set. Where a rule is deliberately not  
  followed, the reason stands next to it, either as `# noqa: <RULE> - reason` or as an entry under
  `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml`.  
- **Tests**: new behaviour needs a test in `tests/`. Coverage is around 90 %.  
- **Docstrings**: Google style, see the [Docstring Guide](docstring-guide.md).  
- **Language**: code, comments and docstrings in English. The documentation is bilingual.  

## Documentation

```bash
mkdocs serve
```

`docs/de/` and `docs/en/` mirror each other file by file. Change both in the same commit, otherwise
the language switch drops readers onto a stale page.
