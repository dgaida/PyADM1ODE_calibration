# Calibration Workflow

The eight steps `examples/calibration_workflow_complete.py` runs, in order.

| Step | What happens | API |
|------|--------------|-----|
| 1 | Build the plant, then generate a twin record from it | `create_twin_measurements` |
| 2 | Validate, remove outliers, fill gaps | `measurements.validate`, `remove_outliers`, `fill_gaps` |
| 3 | Fit, with a train/test split | `InitialCalibrator.calibrate` |
| 4 | Read the metrics of both parts | `result.validation_metrics` |
| 5 | Rank the parameters by influence | `sensitivity_analysis` |
| 6 | Check each parameter, then the set as a whole | `identifiability_analysis`, `analyze_subset` |
| 7 | Write the values into the plant | `digester.apply_calibration_parameters` |
| 8 | Print the summary | - |

Steps 5 and 6 belong together. Sensitivity says whether a parameter moves the output at all,
identifiability says whether its effect can be told apart from that of another parameter. A
parameter that fails either test should not have been in step 3, which is why
`check_identifiability=True` moves the same question in front of the optimizer.

The record in step 1 is produced by simulating the plant itself with known parameter values, so
step 4 can report the error against the truth rather than only "it converged". Its feed profile steps
the organic load up and down and shifts the substrate mix, because a digester held at constant load
settles into a state where several parameter sets produce the same gas curve.

The script does not persist anything. To keep a run, add `Database.store_calibration` after step 7,
the way `examples/scheduled_recalibration.py` does it.
