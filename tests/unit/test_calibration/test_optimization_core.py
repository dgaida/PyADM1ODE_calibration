from unittest.mock import MagicMock

import numpy as np
import pytest

from pyadm1ode_calibration.calibration.optimization.constraints import (
    BarrierPenalty,
    BoxConstraint,
    ExponentialPenalty,
    LinearConstraint,
    LinearPenalty,
    LogarithmicPenalty,
    NonlinearConstraint,
    ParameterConstraints,
    QuadraticPenalty,
    create_penalty_function,
)
from pyadm1ode_calibration.calibration.optimization.objective import (
    CustomObjective,
    ErrorMetrics,
    LikelihoodObjective,
    MultiObjectiveFunction,
    SingleObjective,
    WeightedSumObjective,
    create_objective,
)
from pyadm1ode_calibration.calibration.optimization.optimizer import (
    DifferentialEvolutionOptimizer,
    LBFGSBOptimizer,
    NelderMeadOptimizer,
    PowellOptimizer,
    SLSQPOptimizer,
    create_optimizer,
)


class TestErrorMetrics:
    def test_compute(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 1.9, 3.2])
        metrics = ErrorMetrics.compute(obs, pred)
        assert metrics.rmse > 0
        assert metrics.r2 > 0

    def test_compute_with_nans(self):
        obs = np.array([1.0, np.nan, 3.0])
        pred = np.array([1.1, 2.0, np.nan])
        metrics = ErrorMetrics.compute(obs, pred)
        assert metrics.mse == pytest.approx((1.0 - 1.1) ** 2)

    def test_compute_all_nans(self):
        obs = np.array([np.nan, np.nan])
        pred = np.array([1.0, 2.0])
        metrics = ErrorMetrics.compute(obs, pred)
        assert metrics.mse == float("inf")


class TestObjectives:
    def test_single_objective(self):
        def simulator(p):
            return {"Q_ch4": np.array([p["k_dis"] * 2])}

        measurements = np.array([1.0])
        obj = SingleObjective(simulator, measurements, "Q_ch4", ["k_dis"], error_metric="mse")
        assert obj(np.array([0.5])) == 0.0
        assert obj(np.array([0.6])) > 0

        obj.error_metric = "mae"
        assert obj(np.array([0.6])) > 0
        obj.error_metric = "mape"
        assert obj(np.array([0.6])) > 0
        obj.error_metric = "nse"
        assert obj(np.array([0.6])) is not None
        obj.error_metric = "r2"
        assert obj(np.array([0.6])) is not None

    def test_single_objective_errors(self):
        def simulator(p):
            return {}  # Missing output

        measurements = np.array([1.0])
        obj = SingleObjective(simulator, measurements, "Q_ch4", ["k_dis"])
        assert obj(np.array([0.5])) == 1e10

        def simulator_err(p):
            return 1 / 0  # Error

        obj_err = SingleObjective(simulator_err, measurements, "Q_ch4", ["k_dis"])
        assert obj_err(np.array([0.5])) == 1e10

    def test_multi_objective(self):
        def simulator(p):
            return {"Q_ch4": np.array([p["k_dis"]]), "pH": np.array([p["Y_su"]])}

        meas_dict = {"Q_ch4": np.array([0.5]), "pH": np.array([0.1])}
        obj = MultiObjectiveFunction(
            simulator,
            meas_dict,
            ["Q_ch4", "pH"],
            {"Q_ch4": 0.5, "pH": 0.5},
            ["k_dis", "Y_su"],
            error_metric="rmse",
            normalize=True,
        )
        assert obj(np.array([0.5, 0.1])) == 0.0

        # Test weighted sum alias
        obj2 = WeightedSumObjective(simulator, meas_dict, ["Q_ch4", "pH"], ["k_dis", "Y_su"])
        assert obj2(np.array([0.5, 0.1])) == 0.0

    def test_likelihood_objective(self):
        def simulator(p):
            return {"Q_ch4": np.array([p["k_dis"]])}

        meas_dict = {"Q_ch4": np.array([0.5, 0.5, 0.5])}
        obj = LikelihoodObjective(simulator, meas_dict, ["Q_ch4"], ["k_dis"])
        val = obj(np.array([0.5]))
        assert val is not None

    def test_custom_objective(self):
        def simulator(p):
            return {"Q_ch4": np.array([p["k_dis"]])}

        meas_dict = {"Q_ch4": np.array([0.5])}

        def custom_func(sim, meas):
            return np.sum(np.abs(sim - meas))

        obj = CustomObjective(simulator, meas_dict, ["Q_ch4"], ["k_dis"], custom_func)
        assert obj(np.array([0.6])) == pytest.approx(0.1)

    def test_create_objective(self):
        simulator = MagicMock()
        meas_dict = {"Q_ch4": np.array([1.0])}
        obj = create_objective("single", simulator, meas_dict, ["Q_ch4"], ["k_dis"])
        assert isinstance(obj, SingleObjective)

        obj = create_objective("multi", simulator, meas_dict, ["Q_ch4"], ["k_dis"])
        assert isinstance(obj, MultiObjectiveFunction)


class TestConstraints:
    def test_box_constraint(self):
        c = BoxConstraint("p1", 0.0, 1.0)
        assert c.is_feasible(0.5)
        assert not c.is_feasible(1.5)
        assert c.project(1.5) == 1.0
        assert c.violation(1.5) == 0.5

    def test_linear_constraint(self):
        c = LinearConstraint({"p1": 1.0, "p2": 1.0}, upper_bound=1.0)
        assert c.is_feasible({"p1": 0.4, "p2": 0.4})
        assert not c.is_feasible({"p1": 0.6, "p2": 0.6})
        assert c.violation({"p1": 0.6, "p2": 0.6}) == pytest.approx(0.2)

    def test_nonlinear_constraint(self):
        def func(p):
            return p["p1"] ** 2 - 0.25

        c = NonlinearConstraint("nl", func, constraint_type="inequality")  # g(x) <= 0
        assert c.is_feasible({"p1": 0.4})  # 0.16 - 0.25 = -0.09 <= 0
        assert not c.is_feasible({"p1": 0.6})  # 0.36 - 0.25 = 0.11 > 0
        assert c.violation({"p1": 0.6}) == pytest.approx(0.11)

        c_eq = NonlinearConstraint("nl_eq", func, constraint_type="equality")
        assert not c_eq.is_feasible({"p1": 0.6})
        assert c_eq.violation({"p1": 0.6}) == pytest.approx(0.11)

    def test_parameter_constraints(self):
        pc = ParameterConstraints()
        pc.add_box_constraint("p1", 0, 1, hard=True)
        pc.add_box_constraint("p2", 0, 1, hard=False, weight=2.0)
        pc.add_linear_inequality({"p1": 1}, upper_bound=0.5)
        pc.add_linear_equality({"p1": 1, "p2": 1}, target=1.0)

        def nl_constraint(p):
            return p["p1"] - 0.2

        pc.add_nonlinear_constraint("nl", nl_constraint)

        assert pc.is_feasible({"p1": 0.2, "p2": 0.8})
        # Violate hard box
        assert not pc.is_feasible({"p1": 1.2, "p2": 0.8})

        # Soft box violation
        pc2 = ParameterConstraints()
        pc2.add_box_constraint("p1", 0, 1, hard=False, weight=1.0)
        assert pc2.is_feasible({"p1": 1.2})
        assert pc2.calculate_penalty({"p1": 1.2}) > 0

        # Project
        projected = pc.project_to_feasible({"p1": 1.2, "p2": 1.2})
        assert projected["p1"] == 1.0
        assert projected["p2"] == 1.0

        # Scipy format
        scipy_cons = pc.get_scipy_constraints(["p1", "p2"])
        assert len(scipy_cons) >= 2

        # Validation
        valid, errors = pc.validate_parameters({"p1": 0.6})
        assert not valid
        assert len(errors) > 0

    def test_parameter_constraints_hard_violation(self):
        pc = ParameterConstraints()
        pc.add_box_constraint("p1", 0, 1, hard=True)
        assert pc.calculate_penalty({"p1": 1.5}) == float("inf")

    def test_linear_constraint_types(self):
        # Inequality with lower bound
        c = LinearConstraint({"p1": 1}, lower_bound=0.5, constraint_type="inequality")
        assert c.is_feasible({"p1": 0.6})
        assert not c.is_feasible({"p1": 0.4})
        assert c.violation({"p1": 0.4}) == pytest.approx(0.1)

        # Equality
        c_eq = LinearConstraint({"p1": 1}, upper_bound=0.5, constraint_type="equality")
        assert c_eq.is_feasible({"p1": 0.5})
        assert not c_eq.is_feasible({"p1": 0.6})
        assert c_eq.violation({"p1": 0.6}) == pytest.approx(0.1)

    def test_scipy_constraints_generation(self):
        pc = ParameterConstraints()
        pc.add_linear_inequality({"p1": 1}, lower_bound=0.1, upper_bound=0.9)
        pc.add_linear_equality({"p1": 1, "p2": 1}, target=1.0)

        def nl_func(p):
            return p["p1"] ** 2

        def nl2_func(p):
            return p["p1"] - 0.5

        pc.add_nonlinear_constraint("nl", nl_func, constraint_type="equality")
        pc.add_nonlinear_constraint("nl2", nl2_func, constraint_type="inequality")

        cons = pc.get_scipy_constraints(["p1", "p2"])
        assert len(cons) == 5

        x = np.array([0.5, 0.5])
        for c in cons:
            assert isinstance(c["fun"](x), (float, np.float64))

    def test_create_penalty_function(self):
        for ptype in ["quadratic", "linear", "logarithmic", "exponential", "barrier"]:
            assert create_penalty_function(ptype) is not None
        with pytest.raises(ValueError):
            create_penalty_function("invalid")

    def test_penalty_functions(self):
        assert QuadraticPenalty()(0.5, 2.0) == 0.5
        assert LinearPenalty()(0.5, 2.0) == 1.0
        assert ExponentialPenalty()(0.5, 1.0) == pytest.approx(np.exp(0.5) - 1.0)
        assert BarrierPenalty()(0.5, 1.0) == 2.0
        assert LogarithmicPenalty()(0.5, 1.0) == -np.log(0.5)


class TestOptimizers:
    def test_differential_evolution(self):
        def obj(x):
            return float(np.sum(x**2))

        opt = DifferentialEvolutionOptimizer({"x": (-1, 1), "y": (-1, 1)}, max_iterations=2, verbose=False)
        res = opt.optimize(obj)
        assert len(res.x) == 2
        assert "x" in res.parameter_dict

    def test_nelder_mead(self):
        def obj(x):
            return float(np.sum(x**2))

        opt = NelderMeadOptimizer({"x": (-1, 1), "y": (-1, 1)}, max_iterations=10, verbose=False)
        res = opt.optimize(obj, initial_guess=np.array([0.5, 0.5]))
        assert len(res.x) == 2

    def test_lbfgsb(self):
        def obj(x):
            return float(np.sum(x**2))

        opt = LBFGSBOptimizer({"x": (-1, 1)}, verbose=False)
        res = opt.optimize(obj)
        assert len(res.x) == 1

    def test_powell(self):
        def obj(x):
            return float(np.sum(x**2))

        opt = PowellOptimizer({"x": (-1, 1)}, max_iterations=5, verbose=False)
        res = opt.optimize(obj)
        assert len(res.x) == 1

    def test_slsqp(self):
        def obj(x):
            return float(np.sum(x**2))

        opt = SLSQPOptimizer({"x": (-1, 1)}, max_iterations=5, verbose=False)
        res = opt.optimize(obj)
        assert len(res.x) == 1

    def test_pso(self):
        try:
            import pyswarm  # noqa: F401
        except ImportError:
            pytest.skip("pyswarm package not installed")

        def obj(x):
            return float(np.sum(x**2))

        opt = create_optimizer("pso", {"x": (-1, 1)}, max_iterations=2, verbose=False)
        res = opt.optimize(obj)
        assert len(res.x) == 1

    def test_optimizer_base_methods(self):
        opt = NelderMeadOptimizer({"x": (0, 1)}, verbose=True)
        assert opt._check_bounds(np.array([0.5]))
        assert not opt._check_bounds(np.array([1.5]))
        assert opt._project_to_bounds(np.array([1.5])) == 1.0

        opt._reset_tracking()
        assert opt._n_evaluations == 0

    def test_create_optimizer(self):
        opt = create_optimizer("de", {"x": (0, 1)})
        assert isinstance(opt, DifferentialEvolutionOptimizer)

        opt = create_optimizer("nm", {"x": (0, 1)})
        assert isinstance(opt, NelderMeadOptimizer)

    def test_create_optimizer_invalid(self):
        with pytest.raises(ValueError, match="Unknown optimization method"):
            create_optimizer("invalid", {"x": (0, 1)})


def _quadratic(x):
    """Shifted bowl with minimum at (0.3, -0.2)."""
    return float((x[0] - 0.3) ** 2 + (x[1] + 0.2) ** 2)


def _residuals(x):
    return np.array([x[0] - 0.3, x[1] + 0.2])


BOUNDS_2D = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}


class TestNewOptimizers:
    """The optimizers added for the kinetics-calibration study."""

    def test_de_init_around_start(self):
        opt = DifferentialEvolutionOptimizer(
            BOUNDS_2D, max_iterations=3, population_size=4, init_sigma=0.1, polish=False, seed=1, verbose=False
        )
        res = opt.optimize(_quadratic, initial_guess=np.array([0.3, -0.2]))
        # The start point is the first member, so the best value can never exceed f(x0) = 0.
        assert res.fun <= 1e-12
        assert all(abs(h["parameters"]["x"]) <= 1.0 for h in res.history)

    def test_de_population_is_clipped(self):
        opt = DifferentialEvolutionOptimizer(BOUNDS_2D, population_size=3, init_sigma=5.0, seed=0, verbose=False)
        pop = opt._initial_population(np.array([0.9, -0.9]))
        assert pop.shape == (6, 2)
        assert np.all(pop >= -1.0) and np.all(pop <= 1.0)
        assert np.allclose(pop[0], [0.9, -0.9])

    def test_nelder_mead_initial_simplex(self):
        opt = NelderMeadOptimizer(BOUNDS_2D, max_iterations=50, initial_simplex_scale=0.25, verbose=False)
        simplex = opt._initial_simplex(np.array([0.9, 0.0]))
        assert simplex.shape == (3, 2)
        # Reflected inward because 0.9 + 0.25 would leave the box.
        assert simplex[1, 0] == pytest.approx(0.65)
        assert simplex[2, 1] == pytest.approx(0.25)
        res = opt.optimize(_quadratic, initial_guess=np.array([0.0, 0.0]))
        assert res.fun < 1e-3

    def test_lbfgsb_eps_and_jac(self):
        opt = LBFGSBOptimizer(BOUNDS_2D, eps=1e-3, verbose=False)
        res = opt.optimize(_quadratic, initial_guess=np.zeros(2))
        assert res.fun < 1e-6

        def grad(x):
            return np.array([2 * (x[0] - 0.3), 2 * (x[1] + 0.2)])

        opt = LBFGSBOptimizer(BOUNDS_2D, jac=grad, verbose=False)
        res = opt.optimize(_quadratic, initial_guess=np.zeros(2))
        assert res.fun < 1e-10

    def test_slsqp_eps(self):
        opt = SLSQPOptimizer(BOUNDS_2D, eps=1e-4, verbose=False)
        res = opt.optimize(_quadratic, initial_guess=np.zeros(2))
        assert res.fun < 1e-6

    def test_cmaes(self):
        pytest.importorskip("cma")
        opt = create_optimizer("cmaes", BOUNDS_2D, max_iterations=60, sigma0=0.3, seed=1, verbose=False)
        res = opt.optimize(_quadratic, initial_guess=np.zeros(2))
        assert res.fun < 1e-4
        assert res.nfev == len(res.history)

    def test_bobyqa(self):
        pytest.importorskip("pybobyqa")
        opt = create_optimizer("bobyqa", BOUNDS_2D, max_evaluations=80, rhobeg=0.2, verbose=False)
        res = opt.optimize(_quadratic, initial_guess=np.zeros(2))
        assert res.fun < 1e-6
        assert res.nfev <= 80

    def test_dfols(self):
        pytest.importorskip("dfols")
        opt = create_optimizer("dfols", BOUNDS_2D, max_evaluations=60, rhobeg=0.2, noisy=False, verbose=False)
        res = opt.optimize(_residuals, initial_guess=np.zeros(2))
        assert res.fun < 1e-8
        assert res.history[0]["objective"] == pytest.approx(0.3**2 + 0.2**2)

    def test_least_squares_with_jacobian(self):
        def jac(x):
            return np.eye(2)

        opt = create_optimizer("least_squares", BOUNDS_2D, jac=jac, verbose=False)
        res = opt.optimize(_residuals, initial_guess=np.zeros(2))
        assert res.fun < 1e-12
        assert res.x == pytest.approx([0.3, -0.2])

    def test_hybrid_de_then_powell(self):
        opt = create_optimizer(
            "de_powell",
            BOUNDS_2D,
            global_max_iterations=2,
            local_max_iterations=20,
            global_kwargs={"population_size": 3, "init_sigma": 0.2, "polish": False, "seed": 0},
            verbose=False,
        )
        res = opt.optimize(_quadratic, initial_guess=np.zeros(2))
        assert res.fun < 1e-4
        iterations = [h["iteration"] for h in res.history]
        assert iterations == list(range(1, len(iterations) + 1))
        assert "global:" in res.message and "local:" in res.message

    def test_multistart(self):
        opt = create_optimizer(
            "multistart", BOUNDS_2D, local_method="powell", n_starts=3, spread=0.3, seed=0, max_iterations=20, verbose=False
        )
        res = opt.optimize(_quadratic, initial_guess=np.array([0.8, 0.8]))
        assert res.fun < 1e-4
        assert "best of 3 starts" in res.message

    def test_multistart_lhs_points_are_feasible(self):
        opt = create_optimizer("multistart", BOUNDS_2D, n_starts=4, seed=0, verbose=False)
        starts = opt._start_points(np.zeros(2))
        assert starts.shape == (4, 2)
        assert np.all(starts >= -1.0) and np.all(starts <= 1.0)
        assert np.allclose(starts[0], 0.0)

    def test_create_optimizer_aliases(self):
        from pyadm1ode_calibration.calibration.optimization.optimizer import (
            BOBYQAOptimizer,
            CMAESOptimizer,
            DFOLSOptimizer,
            HybridOptimizer,
            LeastSquaresOptimizer,
            MultiStartOptimizer,
        )

        assert isinstance(create_optimizer("cma", BOUNDS_2D), CMAESOptimizer)
        assert isinstance(create_optimizer("bobyqa", BOUNDS_2D), BOBYQAOptimizer)
        assert isinstance(create_optimizer("dfo_ls", BOUNDS_2D), DFOLSOptimizer)
        assert isinstance(create_optimizer("trf", BOUNDS_2D), LeastSquaresOptimizer)
        assert isinstance(create_optimizer("hybrid", BOUNDS_2D), HybridOptimizer)
        assert isinstance(create_optimizer("multi_start", BOUNDS_2D), MultiStartOptimizer)

    def test_multistart_splits_budget_between_starts(self):
        calls = []

        def counting(x):
            calls.append(1)
            return _quadratic(x)

        opt = create_optimizer(
            "multistart",
            BOUNDS_2D,
            local_method="powell",
            local_kwargs={"max_evaluations": 30},
            n_starts=3,
            spread=0.3,
            seed=0,
            verbose=False,
        )
        res = opt.optimize(counting, initial_guess=np.array([0.8, 0.8]))
        assert "best of 3 starts" in res.message
        # Powell may overshoot maxfev by one line search, never by a whole run.
        assert 60 < len(calls) <= 3 * 30 + 3 * 15
