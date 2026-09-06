# Examples

Three runnable scripts in `examples/`. Unlike the notebooks they are written as programs, not as
explanations.

| Script | Purpose | Runtime |
|--------|---------|---------|
| `calibration_example.py` | Compact walk through a single calibration | minutes |
| `calibration_workflow_complete.py` | Every step in one run against a twin record with a known answer | about an hour, see below |
| `scheduled_recalibration.py` | Unattended job for a scheduler | minutes |

## calibration_workflow_complete.py

The reference for what the package can do, and the slowest thing in the repository. Its default
settings cost about 1200 plant simulations, measured at 56 minutes on a desktop machine, see
[Configuration](../configuration.md#optimizers). Reduce `population_size` and `max_iterations`
before running it interactively.

## scheduled_recalibration.py

Built to be called by cron or the Windows task scheduler. It reads a window, looks up the parameters
in use, recalibrates, refuses the result if the data or the fit look untrustworthy, and stores what
it accepted. The scheduler only has to read the exit code:

```bash
python examples/scheduled_recalibration.py --db sqlite:///calibration.db
```

| Code | Meaning |
|------|---------|
| 0 | accepted and stored |
| 1 | rejected by a guardrail, previous parameters kept |
| 2 | input unusable, too short or too many gaps |
| 3 | unexpected error, see the log |

See [Calibration Workflow](calibration_workflow.md) for the sequence of steps.
