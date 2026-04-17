# Docstring Guide

In diesem Projekt verwenden wir den **Google-Style** für alle Python-Docstrings. Dies ermöglicht eine automatische Dokumentationsgenerierung mittels `mkdocstrings`.

## Grundstruktur

Jeder Docstring sollte eine kurze Zusammenfassung, gefolgt von einer optionalen detaillierten Beschreibung, den Argumenten, Rückgabewerten und Ausnahmen enthalten.

```python
def calibrate(self, measurements: MeasurementData, parameters: List[str]) -> CalibrationResult:
    """Führt den Kalibrierungs-Workflow aus.

    Diese Methode nutzt den konfigurierten Optimierungsalgorithmus, um die
    bestmöglichen Parameter für das gegebene Anlagenmodell zu finden.

    Args:
        measurements: Die historischen Messdaten der Anlage.
        parameters: Eine Liste der zu kalibrierenden Parameternamen.

    Returns:
        Ein Objekt mit den optimierten Parametern und Validierungsmetriken.

    Raises:
        ValueError: Wenn keine Parameter angegeben wurden.
    """
```

## Typhinweise (Type Hints)

Wir verwenden Python Type Hints sowohl in der Funktionssignatur als auch im Docstring (falls nötig). In der Dokumentation werden die Typen automatisch aus der Signatur extrahiert.

## Beispiele

Fügen Sie nach Möglichkeit ein `Example`-Abschnitt hinzu, um die Nutzung zu verdeutlichen:

```python
    Example:
        >>> result = calibrator.calibrate(data, ["k_dis"])
        >>> print(result.success)
        True
```
