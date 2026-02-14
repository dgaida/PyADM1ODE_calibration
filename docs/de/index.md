# PyADM1ODE Kalibrierung

**Fortschrittliches Parameter-Kalibrierungs-Framework für PyADM1ODE Biogasanlagenmodelle.**

PyADM1ODE_calibration bietet eine vollständige Lösung für die Kalibrierung von [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) Modellen:

- **Initialkalibrierung**: Batch-Optimierung basierend auf historischen Messdaten.
- **Online-Rekalibrierung**: Echtzeit-Parameteranpassung während des Anlagenbetriebs.
- **Mehrere Optimierungsalgorithmen**: Differential Evolution, Nelder-Mead, L-BFGS-B, Particle Swarm.
- **Multikriterielle Optimierung**: Ausgleich mehrerer Zielgrößen (CH₄, pH, VFA) mit gewichteten Zielfunktionen.
- **Umfassende Validierung**: Gütekriterien, Residuenanalyse, Kreuzvalidierung.
- **Datenmanagement**: CSV/Datenbank-Import, Validierung, Ausreißererkennung, Lückenfüllung.

## Hauptmerkmale

- 🎯 **Präzision**: Hochgenaue Abstimmung von ADM1-Parametern auf reale Anlagendaten.
- ⚡ **Effizienz**: Schnelle lokale Optimierer für den Online-Einsatz.
- 📊 **Analyse**: Integrierte Sensitivitäts- und Identifizierbarkeitsanalyse.
- 💾 **Integration**: Nahtlose Anbindung an PostgreSQL-Datenbanken und CSV-Workflows.

## Erste Schritte

Beginnen Sie mit der [Installation](getting-started.md) und folgen Sie unserem [Quickstart-Guide](usage/index.md).
