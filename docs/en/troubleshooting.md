# Troubleshooting

Commonly occurring problems and their solutions.

## Installation

## Calibration

### Error: `CalibrationResult.success` is `False`
**Cause**: The optimizer could not find a minimum or the maximum number of iterations was reached.
**Solution**:  
- Increase `max_iterations`.  
- Check the `bounds`. Are they too narrow or too wide?  
- Check the quality of the input data.  

### Unrealistic Parameter Values
**Cause**: Overfitting or poorly chosen starting values/bounds.
**Solution**:  
- Use `use_constraints=True` in the `calibrate` method.  
- Perform a sensitivity analysis to exclude non-identifiable parameters.  

## Data Import

### Error: `Column 'Q_ch4' not found`
**Cause**: The CSV file has incorrect column headers.
**Solution**: Rename the columns in your CSV according to ADM1 standards or use a mapping script.

## Further Help
If your problem is not listed here, please create an [issue on GitHub](https://github.com/dgaida/PyADM1ODE_calibration/issues).
