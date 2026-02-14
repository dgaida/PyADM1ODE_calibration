# Erste Schritte

Diese Anleitung hilft Ihnen bei den ersten Schritten mit PyADM1ODE_calibration.

## Voraussetzungen

- **Python 3.10** oder höher.
- **PyADM1ODE**: Das Basispaket muss installiert sein.
- **Daten**: Historische Messdaten der Biogasanlage im CSV-Format (empfohlen).

## Installation

Die einfachste Methode ist die Installation via pip direkt aus dem Repository:

```bash
pip install git+https://github.com/dgaida/PyADM1ODE_calibration.git
```

Für Entwickler empfehlen wir die Installation im Editier-Modus:

```bash
git clone https://github.com/dgaida/PyADM1ODE_calibration.git
cd PyADM1ODE_calibration
pip install -e .
```

## Schnelldurchlauf

Ein typischer Kalibrierungs-Workflow besteht aus vier Schritten:

1. **Daten laden**: Importieren und Validieren Ihrer Messdaten.
2. **Modell vorbereiten**: Konfiguration Ihres PyADM1ODE-Anlagenmodells.
3. **Kalibrierung ausführen**: Wahl des Algorithmus und Start der Optimierung.
4. **Ergebnisse prüfen**: Validierung der Parameter anhand von Gütekriterien.

Sehen Sie sich das [Kalibrierungs-Beispiel](usage/calibration.md) für einen detaillierten Code-Durchlauf an.
