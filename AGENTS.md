# Agent Instructions

## Development Environment
- To install the project and its development dependencies, use: `pip install -e ".[dev]"`
- The project requires `mono-complete` to be installed on Linux systems to support .NET integration used by the underlying `PyADM1` package.

## Testing and Verification
- Always run tests before submitting. Use: `python3 -m pytest -c /dev/null tests/` if the default `pyproject.toml` configuration causes issues.
- The repository targets a code coverage of over 90%.
- Ensure all files pass linting and formatting:
  - `ruff check .`
  - `black --check .`

## Workflow and Housekeeping
- **Temporary Files:** Always delete any temporary files created during your work (e.g., `.diff` files, log files, temporary scripts) before submitting.
- **Versioning:** Project versioning follows a policy of incrementing by 0.0.1 on every pull request merge to `main`, automated via GitHub Actions.
- **Documentation:** The project uses MkDocs with the Material theme. Documentation is multi-language (German/English) via `mkdocs-static-i18n`.
- **Git Hooks:** Pre-commit checks are highly recommended.

## Troubleshooting
- If Codecov uploads fail with "Token required", ensure `use_oidc: true` and proper permissions are set in the workflow.
- Always use lowercase pandas frequency strings (e.g., '1h' instead of '1H', '1d' instead of '1D') to maintain compatibility with recent pandas versions.
