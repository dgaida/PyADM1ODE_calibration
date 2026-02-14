import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from pyadm1ode_calibration.calibration.core.simulator import PlantSimulator
from pyadm1ode_calibration.io.loaders.measurement_data import MeasurementData

@pytest.fixture
def mock_plant():
    plant = MagicMock()
    comp = MagicMock()
    comp.component_type.value = "digester"
    # No _calibration_params initially
    plant.components = {"d1": comp}
    plant.simulate.return_value = [
        {"components": {"d1": {"Q_ch4": 1.0, "pH": 7.0, "VFA": 1.0, "TAC": 1.0}}}
    ]
    return plant

class TestPlantSimulator:
    def test_simulate_with_parameters_full(self, mock_plant):
        sim = PlantSimulator(mock_plant, verbose=False)
        meas = MeasurementData(pd.DataFrame({
            "Q_sub1": [10],
            "Q_ch4": [1.0]
        }, index=pd.date_range("2024-01-01", periods=1, freq="h")))

        out = sim.simulate_with_parameters({"k_dis": 0.6}, meas)
        assert "Q_ch4" in out
        assert out["pH"][0] == 7.0

    def test_simulate_without_components_output(self, mock_plant):
        mock_plant.simulate.return_value = [{"components": {"d1": {}}}] # Empty results
        sim = PlantSimulator(mock_plant, verbose=False)
        meas = MeasurementData(pd.DataFrame({"Q_sub1": [10]}, index=pd.date_range("2024-01-01", periods=1, freq="h")))
        out = sim.simulate_with_parameters({}, meas)
        assert out["pH"][0] == 7.0 # Default value in code
