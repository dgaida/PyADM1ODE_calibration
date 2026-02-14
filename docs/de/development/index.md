# Entwicklung

Informationen für Mitwirkende und Entwickler von PyADM1ODE_calibration.

## Entwicklungs-Setup

1. Repository klonen.
2. Abhängigkeiten installieren: `pip install -e ".[dev,docs]"`.
3. Pre-commit Hooks installieren: `pre-commit install`.

## Code-Style

Wir verwenden:
- **Black** für die Formatierung.
- **Ruff** für das Linting.
- **Google-Style Docstrings** für die Dokumentation.

## Tests

Führen Sie Tests mit pytest aus:

```bash
pytest
```

Mit Coverage-Bericht:

```bash
pytest --cov=pyadm1ode_calibration
```

## Dokumentation bauen

```bash
mkdocs build
```

Um die Dokumentation lokal mit Live-Vorschau zu sehen:

```bash
mkdocs serve
```
