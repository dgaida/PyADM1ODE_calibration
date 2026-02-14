import pytest
from unittest.mock import MagicMock
from pyadm1ode_calibration.calibration import Calibrator, CalibrationResult


@pytest.fixture
def mock_plant():
    plant = MagicMock()
    return plant


def test_calibrator_orchestrator(mock_plant):
    cal = Calibrator(mock_plant, verbose=False)
    assert cal.plant == mock_plant

    # Mock internal calibrators
    cal.initial_calibrator = MagicMock()
    cal.online_calibrator = MagicMock()

    # run_initial
    cal.run_initial_calibration("meas", ["p1"])
    cal.initial_calibrator.calibrate.assert_called_once()

    # run_online
    cal.run_online_calibration("meas", ["p1"])
    cal.online_calibrator.calibrate.assert_called_once()

    # apply
    res = CalibrationResult(True, {}, {}, 0, 0, 0, "m", "msg")
    cal.apply_calibration(res)
    cal.online_calibrator.apply_calibration.assert_called_once_with(res)
