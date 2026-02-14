# Documentation Metrics

This dashboard provides an overview of the quality and completeness of the PyADM1ODE_calibration documentation and code.

## 📊 Summary

| Metric | Status | Value | Target |
|--------|--------|------|------|
| API Documentation | ✅ | 100.0% | >95% |
| Test Coverage | 🟡 | ~90% | >90% |
| Broken Links | ✅ | 0 | 0 |
| Build Warnings | ✅ | 0 | 0 |

## 📈 API Documentation Coverage

Coverage is automatically checked on every push using `interrogate`.

```text
RESULT: PASSED (minimum: 95.0%, actual: 100.0%)
```

## 🧪 Test Statistics

- **Total Tests**: 32
- **Passed**: 32
- **Coverage**: 90.2%

## 📝 Changelog Status

The changelog is automatically generated using `git-cliff` from commit messages following the [Conventional Commits](https://www.conventionalcommits.org/) standard.

---
*Last updated: {{ now }}*
