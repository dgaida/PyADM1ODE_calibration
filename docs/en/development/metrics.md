# Documentation Metrics

This dashboard shows the quality and completeness of the project documentation.

## API Coverage
API coverage is measured with `interrogate`. It indicates how many of the public classes and functions have a docstring.

![Interrogate Badge](../assets/interrogate.svg)

## Detailed Quality Metrics

<div id="metrics-dashboard">
  Loading metrics...
</div>

<script>
fetch('../assets/metrics.json')
  .then(response => response.json())
  .then(data => {
    const container = document.getElementById('metrics-dashboard');
    let html = '<table style="width:100%">';
    html += '<tr><th>Metric</th><th>Value</th><th>Status</th></tr>';

    for (const [key, val] of Object.entries(data)) {
      let status = '✅ OK';
      if (val.status === 'warning') status = '⚠️ Warning';
      if (val.status === 'error') status = '❌ Error';

      html += \`<tr><td>\${val.label}</td><td>\${val.value}</td><td>\${status}</td></tr>\`;
    }

    html += '</table>';
    container.innerHTML = html;
  })
  .catch(err => {
    document.getElementById('metrics-dashboard').innerHTML = 'Metrics currently unavailable.';
  });
</script>

---
*Metrics are updated with every CI run.*
