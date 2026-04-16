# Dokumentations-Metriken

Dieses Dashboard zeigt die Qualität und Vollständigkeit der Projektdokumentation.

## API-Abdeckung
Die API-Abdeckung wird mit `interrogate` gemessen. Sie gibt an, wie viele der öffentlichen Klassen und Funktionen über einen Docstring verfügen.

![Interrogate Badge](../assets/interrogate.svg)

## Detaillierte Qualitätsmetriken

<div id="metrics-dashboard">
  Lädt Metriken...
</div>

<script>
fetch('../assets/metrics.json')
  .then(response => response.json())
  .then(data => {
    const container = document.getElementById('metrics-dashboard');
    let html = '<table style="width:100%">';
    html += '<tr><th>Metrik</th><th>Wert</th><th>Status</th></tr>';

    for (const [key, val] of Object.entries(data)) {
      let status = '✅ OK';
      if (val.status === 'warning') status = '⚠️ Warnung';
      if (val.status === 'error') status = '❌ Fehler';

      html += \`<tr><td>\${val.label}</td><td>\${val.value}</td><td>\${status}</td></tr>\`;
    }

    html += '</table>';
    container.innerHTML = html;
  })
  .catch(err => {
    document.getElementById('metrics-dashboard').innerHTML = 'Metriken aktuell nicht verfügbar.';
  });
</script>

---
*Die Metriken werden bei jedem CI-Lauf aktualisiert.*
