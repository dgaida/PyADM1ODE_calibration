# PyADM1ODE_Kalibrierung

> **Hinweis**: Dies ist ein Framework für die Parameterkalibrierung, das nur zusammen mit [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) funktioniert.

**Fortgeschrittenes Framework zur Parameterkalibrierung für PyADM1ODE Biogasanlagenmodelle**

Automatisierte Kalibrierung und Rekalibrierung von Parametern des Anaerobic Digestion Model No. 1 (ADM1) unter Verwendung von realen Messdaten mit mehreren Optimierungsalgorithmen, umfassender Validierung und Online-Anpassungsfähigkeiten.

## Übersicht

PyADM1ODE_calibration bietet ein vollständiges Kalibrierungs-Framework für [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) Biogasanlagenmodelle:

- **Erstkalibrierung**: Batch-Optimierung aus historischen Messdaten
- **Online-Rekalibrierung**: Echtzeit-Parameteranpassung während des Anlagenbetriebs
- **Mehrere Optimierungsalgorithmen**: Differential Evolution, Nelder-Mead, L-BFGS-B, Particle Swarm
- **Multi-Objective Optimierung**: Ausgleich mehrerer Ausgänge (CH₄, pH, VFA) mit gewichteten Zielvorgaben
- **Umfassende Validierung**: Gütekriterien, Residualanalyse, Kreuzvalidierung
- **Parameter-Identifizierbarkeit**: Sensitivitätsanalyse und Korrelationserkennung
- **Datenmanagement**: CSV/Datenbank-Import, Validierung, Ausreißererkennung, Lückenfüllung
