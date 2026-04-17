# Entwicklung

Willkommen im Entwicklerbereich von PyADM1ODE_calibration. Wir freuen uns über Beiträge!

## Entwicklungs-Setup

1. Repository klonen:
   ```bash
   git clone https://github.com/dgaida/PyADM1ODE_calibration.git
   cd PyADM1ODE_calibration
   ```

2. Virtuelle Umgebung erstellen und Abhängigkeiten installieren:
   ```bash
   pip install -e ".[dev,docs]"
   ```

## Richtlinien

- **Code-Stil**: Wir verwenden `ruff` für das Linting und `black` für die Formatierung.
- **Tests**: Neue Features müssen durch Tests in `tests/` abgedeckt werden. Wir streben eine Abdeckung von > 90% an.
- **Docstrings**: Alle öffentlichen Funktionen müssen Google-Style Docstrings haben.

## Workflow

1. Erstellen Sie einen Feature-Branch.
2. Implementieren Sie Ihre Änderungen.
3. Führen Sie Tests und Linter aus:
   ```bash
   ruff check .
   black --check .
   python3 -m pytest tests/
   ```
4. Erstellen Sie einen Pull Request.

## Dokumentation

Die Dokumentation wird mit MkDocs erstellt. Um sie lokal zu betrachten:
```bash
mkdocs serve
```
