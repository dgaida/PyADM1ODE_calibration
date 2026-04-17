# Fehlerbehebung

Häufig auftretende Probleme und deren Lösungen.

## Installation

### Fehler: `ImportError: libdl.so.2: cannot open shared object file`
**Ursache**: .NET/Mono ist nicht korrekt auf dem Linux-System installiert.
**Lösung**: Installieren Sie `mono-complete`:
```bash
sudo apt-get install mono-complete
```

## Kalibrierung

### Fehler: `CalibrationResult.success` ist `False`
**Ursache**: Der Optimierer konnte kein Minimum finden oder die maximale Iterationszahl wurde erreicht.
**Lösung**:  
- Erhöhen Sie `max_iterations`.  
- Überprüfen Sie die `bounds`. Sind sie zu eng oder zu weit gefasst?  
- Prüfen Sie die Datenqualität der Eingangsdaten.  

### Unrealistische Parameterwerte
**Ursache**: Overfitting oder schlecht gewählte Startwerte/Grenzen.
**Lösung**:  
- Nutzen Sie `use_constraints=True` in der `calibrate` Methode.  
- Führen Sie eine Sensitivitätsanalyse durch, um nicht-identifizierbare Parameter auszuschließen.  

## Daten-Import

### Fehler: `Column 'Q_ch4' not found`
**Ursache**: Die CSV-Datei hat falsche Spaltenüberschriften.
**Lösung**: Benennen Sie die Spalten in Ihrer CSV entsprechend den ADM1-Standards um oder nutzen Sie ein Mapping-Skript.

## Weitere Hilfe
Falls Ihr Problem hier nicht gelistet ist, erstellen Sie bitte ein [Issue auf GitHub](https://github.com/dgaida/PyADM1ODE_calibration/issues).
