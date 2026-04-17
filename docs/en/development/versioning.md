# Documentation Versioning

We use `mike` to manage multiple versions of the documentation simultaneously. This allows users to switch between documentation for different versions of the package.

## Workflow

### Publishing a New Version
When a new release of the package (e.g., `v1.1.0`) is created, the documentation should also be versioned:

```bash
mike deploy --push --update-aliases 1.1 latest
mike set-default --push latest
```

### Patches for Old Versions
Changes to the documentation for an older version:

```bash
mike deploy --push 1.0
```

## Version Switcher
The switcher is located at the top right of the navigation bar (next to the language selection).

## Local Preview
To view a specific version locally:
```bash
mike serve
```
