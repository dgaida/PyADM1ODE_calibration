import pytest
from unittest.mock import MagicMock
from pyadm1ode_calibration.io.persistence.repositories import PlantRepository
from pyadm1ode_calibration.io.persistence.models import Plant
from pyadm1ode_calibration.exceptions import DatabaseError

def test_plant_repository():
    session = MagicMock()
    factory = lambda: session
    repo = PlantRepository(factory)

    # Create plant
    mock_plant = Plant(id="p1", name="Plant 1")
    session.add.return_value = None
    session.commit.return_value = None
    session.refresh.return_value = None

    res = repo.create_plant(id="p1", name="Plant 1")
    assert res.id == "p1"

    # Get plant
    session.query.return_value.filter.return_value.first.return_value = mock_plant
    res = repo.get_plant("p1")
    assert res.id == "p1"

    # Not found
    session.query.return_value.filter.return_value.first.return_value = None
    with pytest.raises(ValueError, match="not found"):
        repo.get_plant("p2")
