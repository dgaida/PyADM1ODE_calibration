# Fehlerbehebung

Häufige Probleme und deren Lösungen.

## Simulation konvergiert nicht

- **Ursache**: Zu aggressive Parameteränderungen oder instabile Anfangswerte.
- **Lösung**: Verkleinern Sie die Parameter-Grenzen (`bounds`) oder nutzen Sie `max_parameter_change` bei der Online-Kalibrierung.

## Datenbank-Verbindungsfehler

- **Ursache**: Falsche Umgebungsvariablen oder fehlende Berechtigungen.
- **Lösung**: Prüfen Sie `DB_HOST`, `DB_NAME` etc. Stellen Sie sicher, dass PostgreSQL läuft und Verbindungen akzeptiert.

## Langsame Kalibrierung

- **Ursache**: Zu viele Parameter oder zu viele Iterationen bei der Differential Evolution.
- **Lösung**: Führen Sie zuerst eine Sensitivitätsanalyse durch, um die wichtigsten Parameter zu identifizieren. Nutzen Sie lokale Optimierer (Nelder-Mead) für die Feinabstimmung.
