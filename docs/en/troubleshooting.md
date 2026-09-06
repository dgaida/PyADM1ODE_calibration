# Troubleshooting

## The calibration runs for hours

Differential evolution costs `population_size x len(parameters)` simulations per generation, see
[Configuration](configuration.md#optimizers). Cut the parameter list first, then the population,
then the generations.

## `result.success` is `False`

The optimizer stopped without converging. Usually the bounds are too wide, the parameter list too
long, or the window too short to contain the effect being fitted. `result.message` carries the
reason from SciPy.

## The fit is good, the values are implausible

The parameters are compensating for each other. `analyze_subset` puts a number on it: a collinearity
index above 20 means the set cannot be resolved however good each member looks alone. Drop whichever
member matters less, or fix it to a literature value. `use_constraints=True` adds the bound penalties
to the objective.

## Validation error much larger than training error

The fit followed noise. Shorten the parameter list, widen the window, or clean the data harder
before fitting: `remove_outliers` and `fill_gaps`.

## A load change in the data has no effect on the fit

The default collapses the feed series to its mean, so steps are invisible to the
model. Pass `time_varying_feed=True` to the calibrator, see
[Configuration](configuration.md#feeds-that-change-over-time).

## `ValueError: Column 'Q_ch4' not found`

The channel name in `objectives` does not appear in the measurement frame. `measurements.data.columns`
shows what is actually there. `CSVHandler` maps common German column names, and a `PlantSchema`
maps arbitrary tags to channel names.

## Online recalibration never triggers

`should_recalibrate` returns a pair. `if calibrator.should_recalibrate(data):` is always true and
tells you nothing, unpack it: `needed, reason = ...`. The reason names the condition that held it
back, most often `time_threshold` or `consecutive_violations`.

## Something else

Please open an [issue on GitHub](https://github.com/dgaida/PyADM1ODE_calibration/issues).
