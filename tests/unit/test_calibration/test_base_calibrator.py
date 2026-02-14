import pytest
from unittest.mock import MagicMock
from pyadm1ode_calibration.calibration.core.base_calibrator import BaseCalibrator


class ConcreteCalibrator(BaseCalibrator):
    def calibrate(self, measurements, parameters, **kwargs):
        pass


@pytest.fixture
def mock_plant():
    plant = MagicMock()
    comp = MagicMock()
    comp.component_type.value = "digester"
    comp._calibration_params = {"k": 1.0}
    plant.components = {"c1": comp}
    return plant


def test_base_calibrator_methods(mock_plant):
    cal = ConcreteCalibrator(mock_plant, verbose=False)

    # test _get_current_parameters
    params = cal._get_current_parameters()
    assert params == {"k": 1.0}

    # test _apply_parameters_to_plant
    cal._apply_parameters_to_plant({"k": 2.0})
    assert mock_plant.components["c1"]._calibration_params["k"] == 2.0
