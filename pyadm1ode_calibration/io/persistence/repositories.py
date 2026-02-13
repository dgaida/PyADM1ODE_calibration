import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from .models import Plant, Measurement, Simulation, SimulationTimeSeries, Calibration, Substrate, Base
from .connection import ConnectionManager
from ...exceptions import DatabaseError

class PlantRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    def get_plant(self, plant_id: str) -> Plant:
        session = self.session_factory()
        try:
            plant = session.query(Plant).filter(Plant.id == plant_id).first()
            if plant is None:
                raise ValueError(f"Plant '{plant_id}' not found")
            session.expunge(plant)
            return plant
        except SQLAlchemyError as e:
            raise DatabaseError(f"Failed to retrieve plant '{plant_id}': {e}")
        finally:
            session.close()

    def create_plant(self, **kwargs) -> Plant:
        session = self.session_factory()
        try:
            plant = Plant(**kwargs)
            session.add(plant)
            session.commit()
            session.refresh(plant)
            session.expunge(plant)
            return plant
        except SQLAlchemyError as e:
            session.rollback()
            raise DatabaseError(f"Failed to create plant: {e}")
        finally:
            session.close()
