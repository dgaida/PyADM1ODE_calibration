# Troubleshooting

Common issues and their solutions.

## Simulation does not converge

- **Cause**: Too aggressive parameter changes or unstable initial values.
- **Solution**: Narrow the parameter `bounds` or use `max_parameter_change` during online calibration.

## Database Connection Errors

- **Cause**: Incorrect environment variables or missing permissions.
- **Solution**: Check `DB_HOST`, `DB_NAME` etc. Ensure PostgreSQL is running and accepting connections.

## Slow Calibration

- **Cause**: Too many parameters or too many iterations in Differential Evolution.
- **Solution**: Perform a sensitivity analysis first to identify the most important parameters. Use local optimizers (Nelder-Mead) for fine-tuning.
