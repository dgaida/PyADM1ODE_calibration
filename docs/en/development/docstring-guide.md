# Docstring Guide

In this project, we use the **Google Style** for all Python docstrings. This enables automatic documentation generation via `mkdocstrings`.

## Basic Structure

Each docstring should contain a short summary, followed by an optional detailed description, arguments, return values, and exceptions.

```python
def calibrate(self, measurements: MeasurementData, parameters: List[str]) -> CalibrationResult:
    """Executes the calibration workflow.

    This method uses the configured optimization algorithm to find the
    best possible parameters for the given plant model.

    Args:
        measurements: The historical measurement data of the plant.
        parameters: A list of parameter names to be calibrated.

    Returns:
        An object with the optimized parameters and validation metrics.

    Raises:
        ValueError: If no parameters were specified.
    """
```

## Type Hints

We use Python type hints in both the function signature and the docstring (if necessary). In the documentation, types are automatically extracted from the signature.

## Examples

If possible, add an `Example` section to illustrate usage:

```python
    Example:
        >>> result = calibrator.calibrate(data, ["k_dis"])
        >>> print(result.success)
        True
```
