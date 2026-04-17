# Nutzung

Willkommen im Nutzungs-Leitfaden für PyADM1ODE_calibration. Hier erfahren Sie, wie Sie die Kernfunktionalitäten des Pakets in Ihren Workflow integrieren.

## Typische Workflows

### 1. Initialkalibrierung
Wird verwendet, um ein neues Anlagenmodell an historische Daten anzupassen.
[Mehr erfahren](calibration.md#initialkalibrierung)

### 2. Online-Monitoring & Rekalibrierung
Kontinuierliche Überwachung der Modellgüte und automatische Parameteranpassung.
[Mehr erfahren](calibration.md#online-rekalibrierung)

### 3. Datenmanagement
Effizientes Laden, Validieren und Speichern von Messwerten.
[API Referenz zu IO](../api/io.md)

## Code-Struktur

- **`calibration`**: Enthält die Optimierungslogik und Kalibratoren.  
- **`io`**: Behandelt den Datenimport (CSV, Datenbank) und Validierung.  
- **`optimization`**: Implementiert verschiedene Optimierungsalgorithmen.  

## Beispiele

Praktische Beispiele finden Sie im [Beispiele-Bereich](../examples/index.md) oder in den interaktiven [Tutorials](../tutorials/index.md).
