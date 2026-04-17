# Versionierung der Dokumentation

Wir verwenden `mike`, um mehrere Versionen der Dokumentation gleichzeitig zu verwalten. Dies ermöglicht es Benutzern, zwischen der Dokumentation für verschiedene Versionen des Pakets zu wechseln.

## Workflow

### Neue Version veröffentlichen
Wenn ein neues Release des Pakets (z.B. `v1.1.0`) erstellt wird, sollte auch die Dokumentation versioniert werden:

```bash
mike deploy --push --update-aliases 1.1 latest
mike set-default --push latest
```

### Patches für alte Versionen
Änderungen an der Dokumentation für eine ältere Version:

```bash
mike deploy --push 1.0
```

## Versions-Switcher
Der Switcher befindet sich oben rechts in der Navigationsleiste (neben der Sprachauswahl).

## Lokale Vorschau
Um eine bestimmte Version lokal zu betrachten:
```bash
mike serve
```
