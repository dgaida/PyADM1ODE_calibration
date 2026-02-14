# Installation

Detaillierte Anweisungen zur Installation von PyADM1ODE_calibration in verschiedenen Umgebungen.

## Pip Installation

### Standard Installation

```bash
pip install git+https://github.com/dgaida/PyADM1ODE_calibration.git
```

### Mit optionalen Abhängigkeiten

```bash
# Für Entwicklung (Tests, Linting)
pip install "pyadm1ode_calibration[dev] @ git+https://github.com/dgaida/PyADM1ODE_calibration.git"

# Für Dokumentation
pip install "pyadm1ode_calibration[docs] @ git+https://github.com/dgaida/PyADM1ODE_calibration.git"
```

## Conda / Mamba

Wir empfehlen die Verwendung einer dedizierten Umgebung:

```bash
# Umgebung erstellen
conda env create -f environment.yml
conda activate biogas

# Paket im Entwicklungsmodus installieren
pip install -e .
```

## Datenbank-Abhängigkeiten

Wenn Sie die PostgreSQL-Integration nutzen möchten, stellen Sie sicher, dass die entsprechenden Client-Bibliotheken auf Ihrem System installiert sind:

- **Linux (Debian/Ubuntu)**: `sudo apt-get install libpq-dev`
- **macOS**: `brew install postgresql`

## Verifizierung

Nach der Installation können Sie die korrekte Funktion wie folgt prüfen:

```python
import pyadm1ode_calibration
print(pyadm1ode_calibration.__version__)
```
