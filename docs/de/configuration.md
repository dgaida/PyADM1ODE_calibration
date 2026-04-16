# Konfiguration

PyADM1ODE_calibration bietet flexible Konfigurationsmöglichkeiten für Parameter, Optimierer und Zielfunktionen.

## Kalibrierbare Parameter

Die folgenden ADM1-Parameter werden am häufigsten für die Kalibrierung verwendet:

| Parameter | Beschreibung | Einheit | Standardbereich |
|-----------|-------------|---------|-----------------|
| `k_dis` | Desintegrationskonstante | 1/d | 0.3 - 0.8 |
| `k_hyd_ch` | Hydrolysekonstante Kohlenhydrate | 1/d | 5.0 - 15.0 |
| `k_m_ac` | Max. Aufnahmerate Acetat | 1/d | 4.0 - 12.0 |
| `Y_su` | Ertragskoeffizient Zuckerabbauer | kg COD/kg COD | 0.05 - 0.15 |

### Parameter-Grenzen (Bounds)

Sie können die Suchbereiche für die Optimierung manuell definieren:

```python
bounds = {
    "k_dis": (0.2, 1.0),
    "k_hyd_ch": (2.0, 20.0)
}
```

## Optimierungs-Methoden

Unterstützte Algorithmen für die `calibrate` Methode:

- `differential_evolution` (Standard): Robust für globale Suche.
- `nelder-mead`: Effizient für lokale Suche (gut für Online-Kalibrierung).
- `l-bfgs-b`: Gradienten-basiert, erfordert glatte Zielfunktionen.
- `particle_swarm`: Stochastische globale Suche.

## Zielfunktionen (Objectives)

Standardmäßig wird die Methanproduktion (`Q_ch4`) optimiert. Sie können jedoch mehrere Variablen gewichten:

```python
objectives = ["Q_ch4", "pH", "VFA"]
weights = {
    "Q_ch4": 0.7,
    "pH": 0.2,
    "VFA": 0.1
}
```

## Datenbank-Verbindung

Die Konfiguration der Datenbank erfolgt über eine Verbindungs-URL (SQLAlchemy-Format):

```python
db_url = "postgresql://user:password@localhost:5432/biogas_db"
```
