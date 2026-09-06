"""Guards for the claims the kinetics-calibration study rests on.

If any of these break, the study's conclusions are void — they are what make the
kinetics the *only* error source, so a failed recovery can be blamed on
identifiability rather than on a harness bug.

Needs the benchmark dataset from ``PyADM1ODE_estimate``; it is located
automatically, see :mod:`paths`. Run with the ``biogas_torch`` environment::

    python -m pytest experiments/kinetics_calibration/test_fastsim.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import paths

_DATASET = paths.add_dataset_to_path()

from fastsim import KINETIC_KEYS, ForwardModel, build_plant, sensor_sigma
from loader import load_test
from problem import CalibrationProblem, start_point


@pytest.fixture(scope="module")
def series():
    return load_test()[0]


@pytest.fixture(scope="module")
def model():
    return ForwardModel.build(prune=True)


def test_kinetic_keys_match_dataset():
    """The study's parameter order must be the dataset's, or every factor is
    silently attached to the wrong kinetic."""
    npz = np.load(_DATASET / "test.npz", allow_pickle=True)
    assert list(npz["kinetic_keys"]) == list(KINETIC_KEYS)


def test_pruning_keeps_only_the_primary_digester():
    plant = build_plant(prune=True)
    assert set(plant.components) == {"primary", "primary_storage"}


def test_true_kinetics_reproduce_the_dataset_states(series, model):
    """The headline claim: truth in, dataset out — exactly.

    The pruned model is not an approximation of the generator, it *is* the
    generator restricted to the components that can affect the sensors.
    """
    n_steps = 5 * 24
    model.set_log_factors(np.log(np.asarray(series["kinetic_factors"], float)))
    states, _ = model.simulate(series["states"][0], series["feed_true"], n_steps)
    model.reset_kinetics()
    np.testing.assert_allclose(states, np.asarray(series["states"], float)[: n_steps + 1], rtol=1e-10, atol=1e-12)


def test_chi_square_at_truth_is_about_one(series, model):
    """The objective is normalised so 1.0 means 'fits inside sensor noise'."""
    n_steps = 5 * 24
    model.set_log_factors(np.log(np.asarray(series["kinetic_factors"], float)))
    _, obs = model.simulate(series["states"][0], series["feed_true"], n_steps)
    model.reset_kinetics()
    meas = np.asarray(series["measurements"], float)[: n_steps + 1]
    chi2 = float(np.mean(((meas - obs) / sensor_sigma(obs)) ** 2))
    assert 0.7 < chi2 < 1.5, f"chi2 at the true kinetics is {chi2}, expected ~1"


def test_set_log_factors_is_multiplicative_and_reversible(model):
    nominal = dict(model.nominal_kinetics)
    model.set_log_factors(np.full(len(KINETIC_KEYS), np.log(2.0)))
    kin = model.plant.components["primary"].adm1._kinetic
    for key in KINETIC_KEYS:
        assert kin[key] == pytest.approx(2.0 * nominal[key])
    model.reset_kinetics()
    for key in KINETIC_KEYS:
        assert kin[key] == pytest.approx(nominal[key])


def test_zero_log_factors_are_the_nominal_values(model):
    model.set_log_factors(np.zeros(len(KINETIC_KEYS)))
    kin = model.plant.components["primary"].adm1._kinetic
    for key in KINETIC_KEYS:
        assert kin[key] == pytest.approx(model.nominal_kinetics[key])


def test_near_start_is_nominal_and_far_start_is_further(series):
    theta_true = np.log(np.asarray(series["kinetic_factors"], float))
    near = start_point("near", seed=0)
    far = start_point("far", seed=1000)
    assert np.all(near == 0.0)
    assert np.median(np.abs(far - theta_true)) > np.median(np.abs(near - theta_true))


def test_far_start_does_not_depend_on_the_truth():
    """The far start must be reproducible from its seed alone — if it were built
    from the true values it would leak the answer into the experiment."""
    a = start_point("far", seed=42)
    b = start_point("far", seed=42)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, start_point("far", seed=43))


def test_objective_at_truth_beats_a_wrong_parameter_set(series, model):
    """Sanity: the objective must actually prefer the true kinetics."""
    problem = CalibrationProblem(series=series, n_days=3, active=np.arange(len(KINETIC_KEYS)), model=model)
    problem.set_base(np.zeros(len(KINETIC_KEYS)))
    at_truth = problem.objective(problem.theta_true)
    at_wrong = problem.objective(problem.theta_true + 0.5)
    assert at_truth < at_wrong


def test_budget_is_enforced(series, model):
    problem = CalibrationProblem(series=series, n_days=1, active=np.arange(3), model=model, budget=2)
    problem.set_base(np.zeros(len(KINETIC_KEYS)))
    problem.objective(np.zeros(3))
    problem.objective(np.zeros(3))
    from problem import BudgetExhausted

    with pytest.raises(BudgetExhausted):
        problem.objective(np.zeros(3))


def test_fostac_is_stored_hourly(series):
    """Every hour carries a titration, so the sampling frequency is an
    experiment's choice rather than a property of the files."""
    lab = np.asarray(series["fostac"], dtype=float)
    assert lab.shape == (1441, 2)
    assert np.all(np.isfinite(lab))
    assert np.asarray(series["fostac_true"], dtype=float).shape == (1441, 2)


@pytest.mark.parametrize("every_days, expected", [(7.0, 9), (1.0, 61), (0.5, 121), (1 / 24, 1441)])
def test_fostac_subsampling_picks_any_frequency(series, every_days, expected):
    from fostac import subsample

    assert subsample(series["fostac"], every_days=every_days).shape == (expected, 2)


def test_fostac_subsample_mask_matches_compact_form(series):
    """The NaN-masked view and the compact array must carry the same values, or
    a filter and a scorer would silently disagree."""
    from fostac import subsample

    compact = subsample(series["fostac"], every_days=7)
    masked = subsample(series["fostac"], every_days=7, as_mask=True)
    rows = np.flatnonzero(np.isfinite(masked[:, 0]))
    assert len(rows) == 9
    np.testing.assert_array_equal(np.diff(rows), np.full(8, 7 * 24))
    np.testing.assert_allclose(masked[rows], compact)


def test_fostac_noise_is_independent_per_hour(series):
    """Subsampling is only exact if each hour is an independent draw. Adjacent
    residuals must therefore be uncorrelated, unlike the smooth truth."""
    lab = np.asarray(series["fostac"], dtype=float)
    truth = np.asarray(series["fostac_true"], dtype=float)
    resid = (lab[:, 1] - truth[:, 1]) / np.maximum(truth[:, 1], 1e-9)
    assert abs(np.corrcoef(resid[:-1], resid[1:])[0, 1]) < 0.1


def test_fostac_noise_matches_the_literature_asymmetry(series):
    """TAC is precise and roughly constant; FOS is worse and gets worse the lower
    the value, because its titration leg is short and shrinks with the reading.

    The exact FOS/TAC precision ratio is therefore not a fixed number — it
    depends on where the series sits — so the invariants tested are the ones the
    titration geometry actually guarantees.
    """
    from fastsim import fostac_sigma
    from fostac import sample_indices

    lab = np.asarray(series["fostac"], dtype=float)
    rows = sample_indices(len(lab), every_days=7.0)
    rows = rows[lab[rows, 0] > 0.0]
    sig = fostac_sigma(lab[rows])
    rel_fos = sig[:, 0] / lab[rows, 0]
    rel_tac = sig[:, 1] / lab[rows, 1]

    # TAC: literature ~1.5 %, here ~2 % and near-constant.
    assert 0.01 < np.median(rel_tac) < 0.04, np.median(rel_tac)
    assert rel_tac.max() - rel_tac.min() < 0.01
    # FOS is always the worse of the two, sample by sample.
    assert np.all(rel_fos > rel_tac)
    # ... and its relative error falls monotonically as the reading rises. The
    # relation is hyperbolic (it decays towards the 2 % sample error), so
    # monotonicity is the invariant, not a linear correlation.
    order = np.argsort(lab[rows, 0])
    assert np.all(np.diff(rel_fos[order]) < 0), rel_fos[order]


def test_fostac_relative_error_matches_literature_at_typical_level():
    """At a typical FOS of ~2000 mg/L the model must land near the literature's
    6.7 %, which is the value it was calibrated against."""
    from fastsim import fostac_sigma

    lab = np.array([[2000.0, 10000.0]])
    sig = fostac_sigma(lab)
    assert 0.05 < sig[0, 0] / 2000.0 < 0.09, sig[0, 0] / 2000.0


def test_fostac_truth_reproduces_the_stored_measurement(series, model):
    """The stored titration must sit within a few sigma of what the true states
    imply — otherwise measurement and truth were generated from different runs."""
    from fastsim import adm1_torch_params, fostac_from_states, fostac_sigma
    from fostac import sample_indices

    lab = np.asarray(series["fostac"], dtype=float)
    rows = sample_indices(len(lab), every_days=7.0)
    rows = rows[lab[rows, 0] > 0.0]
    truth = fostac_from_states(np.asarray(series["states"], float)[rows], adm1_torch_params(model.plant))
    z = (lab[rows] - truth) / fostac_sigma(lab[rows])
    assert np.abs(z).max() < 5.0, np.abs(z).max()


def test_objective_with_fostac_is_still_about_one_at_truth(series, model):
    """Adding the lab channel must not shift the chi-square scale."""
    problem = CalibrationProblem(
        series=series,
        n_days=30,
        active=np.arange(len(KINETIC_KEYS)),
        model=model,
        use_fostac=True,
    )
    problem.set_base(np.zeros(len(KINETIC_KEYS)))
    assert len(problem._fostac_idx) == 5  # days 0, 7, 14, 21, 28
    assert 0.7 < problem.objective(problem.theta_true) < 1.5


def test_inactive_parameters_stay_at_the_base(series, model):
    """A reduced parameter set must not secretly move the parameters it excluded."""
    problem = CalibrationProblem(series=series, n_days=1, active=np.array([0, 5]), model=model)
    base = np.full(len(KINETIC_KEYS), 0.3)
    problem.set_base(base)
    full = problem.expand([1.0, -1.0])
    assert full[0] == 1.0 and full[5] == -1.0
    others = [i for i in range(len(KINETIC_KEYS)) if i not in (0, 5)]
    np.testing.assert_allclose(full[others], 0.3)
