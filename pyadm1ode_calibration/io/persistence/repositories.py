"""
Data Repositories.

Implements the Repository pattern to decouple high-level database operations
from ORM-specific logic for key entities like Plants.
"""

from sqlalchemy.exc import SQLAlchemyError
from .models import Plant
from ...exceptions import DatabaseError


class PlantRepository:
    """
    Repository for Plant entity operations.

    Args:
        session_factory: A callable that returns a SQLAlchemy Session.
    """

    def __init__(self, session_factory):
        self.session_factory = session_factory

    def get_plant(self, plant_id: str) -> Plant:
        """
        Retrieve a plant by ID.

        Args:
            plant_id (str): The plant identifier.

        Returns:
            Plant: The plant ORM instance.
        """
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
        """
        Create and persist a new plant.

        Args:
            **kwargs: Attributes for the Plant model.

        Returns:
            Plant: The newly created plant.
        """
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
