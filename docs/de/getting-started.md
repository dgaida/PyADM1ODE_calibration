# Erste Schritte

Diese Anleitung hilft Ihnen bei den ersten Schritten mit **PyADM1ODE_calibration**.

## Voraussetzungen

Bevor Sie beginnen, stellen Sie sicher, dass Ihre Umgebung folgende Anforderungen erfüllt:

- **Python**: 3.10 oder höher.
- **PyADM1ODE**: Das Basispaket für die Simulation von Biogasanlagen.
- **Daten**: Historische Messdaten der Anlage (z.B. CH₄-Produktion, pH-Wert) im CSV-Format oder in einer PostgreSQL-Datenbank.

## Installation

Installieren Sie das Paket direkt via pip:

```bash
pip install pyadm1ode-calibration
```

Oder für die Entwicklung:

```bash
git clone https://github.com/dgaida/PyADM1ODE_calibration.git
cd PyADM1ODE_calibration
pip install -e ".[dev]"
```

## Kernkonzepte

### 1. Daten-Loading (`MeasurementData`)
Alle Kalibrierungen basieren auf dem `MeasurementData` Objekt. Es verwaltet Zeitreihen von Messwerten und bietet Funktionen zur Validierung und Vorverarbeitung.

### 2. Kalibratoren
- **InitialCalibrator**: Wird für die erstmalige Abstimmung des Modells auf einen historischen Datensatz verwendet (Batch-Optimierung).
- **OnlineCalibrator**: Ermöglicht die kontinuierliche Anpassung von Parametern während des laufenden Betriebs, um auf Änderungen in der Substratqualität oder Biologie zu reagieren.

### 3. Zielfunktionen
Sie können mehrere Zielgrößen (Objectives) definieren und gewichten, z.B. 80% Gewicht auf Methanproduktion und 20% auf den pH-Wert.

## Nächste Schritte

- Folgen Sie dem [Tutorial für die Initialkalibrierung](tutorials/calibration.md).
- Erfahren Sie mehr über die [Konfiguration von Parametern](configuration.md).
- Prüfen Sie die [API-Referenz](api/index.md) für detaillierte Informationen.
