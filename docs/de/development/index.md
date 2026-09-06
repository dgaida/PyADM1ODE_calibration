# Entwicklung

## Einrichtung

```bash
git clone https://github.com/dgaida/PyADM1ODE_calibration.git
cd PyADM1ODE_calibration
pip install -e ".[dev,docs]"
```

## Vor jedem Commit

```bash
ruff check .
black --check .
python -m pytest
```

CI führt alle drei aus und schlägt bei jedem davon fehl. `.pre-commit-config.yaml` führt dieselben
Werkzeuge auf den vorgemerkten Dateien aus, Einrichtung mit `pre-commit install`.

## Konventionen

- **Formatierung**: black, Zeilenlänge 127. Ruff nutzt dieselbe Grenze.
- **Lint**: Das Repository ist unter dem Standardregelsatz von ruff sauber. Wo eine Regel bewusst
  nicht befolgt wird, steht die Begründung daneben, entweder als `# noqa: <REGEL> - Begründung` oder
  als Eintrag unter `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml`.
- **Tests**: Neues Verhalten braucht einen Test in `tests/`. Die Abdeckung liegt bei rund 90 %.
- **Docstrings**: Google-Stil, siehe [Docstring-Guide](docstring-guide.md).
- **Sprache**: Code, Kommentare und Docstrings auf Englisch. Die Dokumentation ist zweisprachig.

## Dokumentation

```bash
mkdocs serve
```

`docs/de/` und `docs/en/` spiegeln einander Datei für Datei. Beide im selben Commit ändern, sonst
landet der Sprachumschalter auf einer veralteten Seite.
