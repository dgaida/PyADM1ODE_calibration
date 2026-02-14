import pytest
import numpy as np
from pyadm1ode_calibration.calibration.parameter_bounds import (
    ParameterBound, ParameterBounds, BoundType, create_default_bounds
)

class TestParameterBound:
    def test_calculate_penalty_soft(self):
        pb = ParameterBound("test", 10.0, 20.0, 15.0, bound_type=BoundType.SOFT, penalty_weight=2.0)
        assert pb.calculate_penalty(15.0) == 0.0
        # Quadratic: 2 * (10-5)^2 = 2 * 25 = 50
        assert pb.calculate_penalty(5.0, penalty_type="quadratic") == 50.0
        # Linear: 2 * 5 = 10
        assert pb.calculate_penalty(5.0, penalty_type="linear") == 10.0
        # Logarithmic (note: formula is -log(distance), so distance must be < 1.0 for positive penalty)
        assert pb.calculate_penalty(9.9, penalty_type="logarithmic") > 0
        # Barrier
        assert pb.calculate_penalty(5.0, penalty_type="barrier") > 0

        with pytest.raises(ValueError, match="Unknown penalty type"):
            pb.calculate_penalty(5.0, penalty_type="invalid")

    def test_calculate_penalty_hard(self):
        pb = ParameterBound("test", 10.0, 20.0, 15.0, bound_type=BoundType.HARD)
        assert pb.calculate_penalty(15.0) == 0.0
        assert pb.calculate_penalty(5.0) == np.inf

    def test_calculate_penalty_fixed(self):
        pb = ParameterBound("test", 10.0, 20.0, 15.0, bound_type=BoundType.FIXED)
        assert pb.calculate_penalty(5.0) == 0.0

    def test_get_relative_position(self):
        pb = ParameterBound("test", 10.0, 20.0, 15.0)
        assert pb.get_relative_position(15.0) == 0.5

        pb_fixed = ParameterBound("test", 10.0, 10.0, 10.0)
        assert pb_fixed.get_relative_position(10.0) == 0.5

class TestParameterBoundsManager:
    def test_manager_methods_missing_bound(self):
        pm = ParameterBounds()
        assert pm.get_bounds_tuple("missing") is None
        assert pm.is_within_bounds("missing", 100.0) is True
        assert pm.clip_to_bounds("missing", 100.0) == 100.0
        assert pm.calculate_penalty("missing", 100.0) == 0.0
        assert pm.scale_to_unit_interval("missing", 10.0) == 10.0
        assert pm.unscale_from_unit_interval("missing", 0.5) == 0.5

    def test_calculate_total_penalty_inf(self):
        pm = ParameterBounds()
        pm.add_bound("p1", 0, 1, 0.5, bound_type=BoundType.HARD)
        assert pm.calculate_total_penalty({"p1": 1.5}) == np.inf

    def test_validate_parameters_raise(self):
        pm = ParameterBounds()
        pm.add_bound("p1", 0, 1, 0.5)
        with pytest.raises(ValueError, match="outside bounds"):
            pm.validate_parameters({"p1": 1.5}, raise_on_invalid=True)

    def test_scale_same_bounds(self):
        pm = ParameterBounds()
        pm.add_bound("p1", 10, 10, 10)
        assert pm.scale_to_unit_interval("p1", 10) == 0.5

    def test_get_default_values(self):
        pm = ParameterBounds()
        pm.add_bound("p1", 0, 1, 0.5)
        defaults = pm.get_default_values(["p1", "p2"])
        assert defaults == {"p1": 0.5}

def test_default_bounds_creation():
    bounds = create_default_bounds()
    assert "k_dis" in bounds.bounds
    assert "Y_su" in bounds.bounds
